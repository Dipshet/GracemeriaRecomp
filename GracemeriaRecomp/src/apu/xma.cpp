// XMA decoder hardware.
//
// The console decodes XMA (a WMA Pro variant) in hardware: the game's audio engine fills 64-byte context
// blocks in physical memory (input packet buffers, an output ring of 256-byte blocks, loop points) and
// starts them by writing context bitmasks to the APU's registers (kick 0x650, lock 0x690, clear 0x6A0 at
// 0x7FEA0000 + register * 4). Those registers are plain memory here: the three engine functions that
// write them (sub_823AEFA0 kick, sub_823AED08 lock, sub_823AEC38 clear) are wrapped instead, and they
// name their contexts the same way, through the engine's voice records (count at +0, 96-byte records at
// the pointer in +8, each holding a copy of its context at +0, the context's address at +64 and its
// hardware index at +80).
//
// A kick decodes synchronously, as far as the context's output ring has room. Each context is one XMA
// stream (mono or stereo, a 5.1 stream is three contexts reading the same interleaved packets): its 2 KB
// packets go one by one to FFmpeg's XMA2 decoder, the next one of the stream found through each packet
// header's skip count (zeroed in the copy given to FFmpeg, which would otherwise apply it again). Decoded
// samples queue per context and leave as 16-bit big-endian PCM in whole 256-byte blocks. Running past
// the end of an input buffer invalidates it and continues in the other one, as the hardware does. The
// engine refills the invalid one. The context's input state follows what has left for the ring, not what
// has been decoded (see Decoder). Decoded frames are matched to their bit positions, for resets at a read
// offset the engine sets itself (a new sound, a seek) and for loop points. FFmpeg's XMA path carries
// three fixes (tools/patches/ffmpeg-xma.patch).
#include "apu/xma.h"

#include "apu/audio_driver.h"

#include "endian.h"
#include "heap.h"
#include "kernel/function.h"
#include "log.h"
#include "memory.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <string>
#include <vector>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/channel_layout.h>
#include <libavutil/log.h>
#include <libavutil/mem.h>
}

namespace ac6::apu {

namespace {

constexpr uint32_t kContextCount = 320;
constexpr uint32_t kContextSize = 64;
constexpr uint32_t kRegisterBase = 0x7FEA0000;
constexpr uint32_t kRegContextArray = 0x600;
constexpr uint32_t kPacketBytes = 2048;
constexpr uint32_t kPacketBits = kPacketBytes * 8;
constexpr uint32_t kHeaderBits = 32;
constexpr uint32_t kBlockBytes = 256;
constexpr std::array<int, 4> kSampleRates = { 24000, 32000, 44100, 48000 };

bool LogEnabled()
{
    static const bool enabled = [] { const char* v = std::getenv("AC6_XMA_LOG"); return v && *v == '1'; }();
    return enabled;
}

uint32_t Bits(uint32_t v, uint32_t at, uint32_t count) { return (v >> at) & ((1u << count) - 1); }
void SetBits(uint32_t& v, uint32_t at, uint32_t count, uint32_t value)
{
    const uint32_t mask = ((1u << count) - 1) << at;
    v = (v & ~mask) | ((value << at) & mask);
}

// The context as host-order dwords. Fields (bit positions within each dword):
//   0: input 0 packet count 0-11, loop count 12-19, input 0 valid 20, input 1 valid 21,
//      output block count 22-26, output write offset 27-31
//   1: input 1 packet count 0-11, loop subframe end 14-16, loop subframe skip 17-19, subframe decode
//      count 20-23, sample rate 27-28, stereo 29, output valid 31
//   2: input read offset (bits) 0-25, error status 26-30
//   3: loop start (bits) 0-25,  4: loop end (bits) 0-25, current input buffer 31
//   5, 6: input buffer 0 / 1 (physical),  7: output buffer (physical),  8: work buffer
//   9: output read offset 0-4
struct Context
{
    uint32_t w[16];

    uint32_t PacketCount(uint32_t buffer) const { return Bits(w[buffer ? 1 : 0], 0, 12); }
    bool InputValid(uint32_t buffer) const { return Bits(w[0], 20 + buffer, 1); }
    void SetInputValid(uint32_t buffer, bool v) { SetBits(w[0], 20 + buffer, 1, v); }
    uint32_t InputAddress(uint32_t buffer) const { return w[5 + buffer]; }
    uint32_t CurrentBuffer() const { return Bits(w[4], 31, 1); }
    void SetCurrentBuffer(uint32_t b) { SetBits(w[4], 31, 1, b); }
    uint32_t ReadOffset() const { return Bits(w[2], 0, 26); }
    void SetReadOffset(uint32_t v) { SetBits(w[2], 0, 26, v); }
    uint32_t BlockCount() const { return Bits(w[0], 22, 5); }
    uint32_t WriteOffset() const { return Bits(w[0], 27, 5); }
    void SetWriteOffset(uint32_t v) { SetBits(w[0], 27, 5, v); }
    uint32_t OutputReadOffset() const { return Bits(w[9], 0, 5); }
    bool OutputValid() const { return Bits(w[1], 31, 1); }
    void SetOutputValid(bool v) { SetBits(w[1], 31, 1, v); }
    uint32_t Channels() const { return Bits(w[1], 29, 1) ? 2 : 1; }
    int SampleRate() const { return kSampleRates[Bits(w[1], 27, 2)]; }
    uint32_t LoopCount() const { return Bits(w[0], 12, 8); }   // 255: forever
    void SetLoopCount(uint32_t v) { SetBits(w[0], 12, 8, v); }
    uint32_t LoopStart() const { return Bits(w[3], 0, 26); }   // bit offsets of frames in the buffer
    uint32_t LoopEnd() const { return Bits(w[4], 0, 26); }
    uint32_t LoopSubframeEnd() const { return Bits(w[1], 14, 3); }
    uint32_t LoopSubframeSkip() const { return Bits(w[1], 17, 3); }
};

Context Load(uint32_t guest)
{
    Context c;
    for (uint32_t i = 0; i < 16; i++)
        c.w[i] = ByteSwapValue(*g_memory.As<uint32_t>(guest + i * 4));
    return c;
}

void Store(uint32_t guest, const Context& c)
{
    for (uint32_t i = 0; i < 16; i++)
        *g_memory.As<uint32_t>(guest + i * 4) = ByteSwapValue(c.w[i]);
}

void LogContext(const char* what, uint32_t index, const Context& c)
{
    log::Info("XMA {} ctx {}: in0 {:08X} x{} {} in1 {:08X} x{} {} cur {} read {} | out {:08X} blocks {} w{} r{} valid {} | {} Hz x{} "
              "subframes {} | loop {}..{} x{} end {} skip {}",
        what, index, c.w[5], c.PacketCount(0), int(c.InputValid(0)), c.w[6], c.PacketCount(1), int(c.InputValid(1)), c.CurrentBuffer(),
        c.ReadOffset(), c.w[7], c.BlockCount(), c.WriteOffset(), c.OutputReadOffset(), int(c.OutputValid()), c.SampleRate(), c.Channels(),
        Bits(c.w[1], 20, 4), Bits(c.w[3], 0, 26), Bits(c.w[4], 0, 26), Bits(c.w[0], 12, 8), c.LoopSubframeEnd(), c.LoopSubframeSkip());
}

// FFmpeg's messages (warnings and errors) go to the log, a limited number of them.
void FfmpegLog(void*, int level, const char* fmt, va_list args)
{
    static std::atomic<uint32_t> count{ 0 };
    if (level > AV_LOG_WARNING || (count++ >= 32 && !LogEnabled()))
        return;
    char text[512];
    std::vsnprintf(text, sizeof(text), fmt, args);
    std::string line(text);
    while (!line.empty() && (line.back() == '\n' || line.back() == '\r'))
        line.pop_back();
    log::Warn("XMA: FFmpeg: {}", line);
}

constexpr uint32_t kFrameSamples = 512;   // per channel

uint32_t PacketBits(const uint8_t* p, uint32_t at, uint32_t count)
{
    uint32_t v = 0;
    for (uint32_t i = 0; i < count; i++)
        v = (v << 1) | ((p[(at + i) >> 3] >> (7 - ((at + i) & 7))) & 1);
    return v;
}

// Calls f(bit position in the packet) for each frame that starts in it: from the first frame offset (header
// bits 6-20, past the 32-bit header), each frame led by its 15-bit length (0 and 0x7FFF end the chain)
// and ended by a "more frames" bit. A frame whose length field itself crosses into the next packet still
// starts here.
template <typename F>
void ForEachFrameStart(const uint8_t* p, F&& f)
{
    uint32_t pos = kHeaderBits + PacketBits(p, 6, 15);
    while (pos < kPacketBits)
    {
        f(pos);
        if (pos + 15 > kPacketBits)
            break;
        const uint32_t length = PacketBits(p, pos, 15);
        if (length == 0 || length == 0x7FFF)
            break;
        pos += length;
        if (pos > kPacketBits || !PacketBits(p, pos - 1, 1))
            break;
    }
}

// Host side of one hardware context. The decoder runs up to one packet ahead of the output ring. The
// context's input state (read offset, current buffer, buffer valid bits) only moves once a packet's
// samples have left for the ring, as on the hardware, which consumes input frame by frame as the output
// has room. Until then the engine must not see the input as used up: it would refill or restart it and
// the samples still queued here would be lost (the engine loops sounds by restarting them at packet 0).
struct Decoder
{
    AVCodecContext* codec = nullptr;
    AVPacket* packet = nullptr;
    AVFrame* frame = nullptr;
    int channels = 0, rate = 0;
    std::vector<int16_t> pcm;   // decoded, interleaved, not yet in the output ring
    size_t pcmRead = 0;
    uint64_t produced = 0, consumed = 0;   // int16 values into / out of the queue since the reset
    // Where decoding continues (ahead of the context's committed position).
    uint32_t buffer = 0, packetIndex = 0;
    // Positions to commit once the queue has been read up to `end`.
    struct Mark { uint64_t end; uint32_t buffer, packetIndex; };
    std::vector<Mark> marks;
    uint32_t expectedReadOffset = UINT32_MAX;   // the read offset the last kick left (another: the engine moved it)
    uint32_t expectedBuffer = 0;
    // The frames handed to FFmpeg that have not come out yet, in order (FFmpeg outputs 512 samples per
    // frame), each by its input buffer and bit offset in it, the units of the read offset and loop points.
    struct FrameStart { uint32_t buffer, offset; };
    std::deque<FrameStart> starts;
    // After a reset or a loop jump, decoding restarts at the packet holding the target frame: the frames
    // before it in that packet are decoded but not played, and the first played one may skip subframes.
    bool trimming = false;
    FrameStart trimTo{};
    uint32_t skipSubframes = 0;

    // Starts decoding at a read offset (bits) in an input buffer.
    void Reset(uint32_t startBuffer, uint32_t readOffset)
    {
        if (codec)
            avcodec_flush_buffers(codec);
        pcm.clear();
        pcmRead = 0;
        produced = consumed = 0;
        marks.clear();
        starts.clear();
        Seek(startBuffer, readOffset, 0);
    }

    void Seek(uint32_t startBuffer, uint32_t readOffset, uint32_t skip)
    {
        buffer = startBuffer;
        packetIndex = readOffset / kPacketBits;
        trimming = true;
        trimTo = { startBuffer, readOffset };
        skipSubframes = skip;
    }

    void Close()
    {
        avcodec_free_context(&codec);
        av_packet_free(&packet);
        av_frame_free(&frame);
        channels = rate = 0;
        Reset(0, kHeaderBits);
        expectedReadOffset = UINT32_MAX;
    }

    bool Open(int ch, int sampleRate)
    {
        Close();
        static const AVCodec* xma2 = avcodec_find_decoder(AV_CODEC_ID_XMA2);
        if (!xma2)
        {
            static bool warned = false;
            if (!warned)
                log::Error("XMA: FFmpeg has no XMA2 decoder");
            warned = true;
            return false;
        }
        codec = avcodec_alloc_context3(xma2);
        packet = av_packet_alloc();
        frame = av_frame_alloc();
        if (!codec || !packet || !frame)
            return false;
        codec->sample_rate = sampleRate;
        av_channel_layout_default(&codec->ch_layout, ch);
        // A 34-byte XMA2WAVEFORMATEX extension tells the decoder "one stream of `ch` channels": NumStreams
        // (u16) and ChannelMask (u32), little-endian, first.
        codec->extradata = static_cast<uint8_t*>(av_mallocz(34 + AV_INPUT_BUFFER_PADDING_SIZE));
        codec->extradata_size = 34;
        codec->extradata[0] = 1;
        codec->extradata[2] = ch == 2 ? 0x03 : 0x04;   // front left + right, or front centre
        // FFmpeg would drop the first 64 samples of every stream (its decoder-delay convention). The
        // hardware outputs them, and every frame is 512 samples.
        codec->flags2 |= AV_CODEC_FLAG2_SKIP_MANUAL;
        if (avcodec_open2(codec, xma2, nullptr) < 0)
        {
            log::Error("XMA: could not open the decoder ({} channels, {} Hz)", ch, sampleRate);
            Close();
            return false;
        }
        channels = ch;
        rate = sampleRate;
        return true;
    }

    // Appends n samples per channel (silence when planes is null).
    void Push(const float* const* planes, int planeCount, int n)
    {
        if (pcmRead && pcmRead == pcm.size())
        {
            pcm.clear();
            pcmRead = 0;
        }
        const size_t at = pcm.size();
        pcm.resize(at + size_t(n) * channels);
        for (int c = 0; c < channels; c++)
        {
            const float* in = planes ? planes[std::min(c, planeCount - 1)] : nullptr;
            for (int i = 0; i < n; i++)
                pcm[at + size_t(i) * channels + c] = in ? int16_t(std::lround(std::clamp(in[i], -1.0f, 1.0f) * 32767.0f)) : 0;
        }
        produced += size_t(n) * channels;
    }

    // Queues one decoded frame (or part of one), applying the trim after a seek and the context's loop:
    // the frame at the loop end plays up to its loop subframe end, then decoding jumps back to the loop
    // start. Returns true on a jump.
    bool Output(const float* const* planes, int planeCount, int n, FrameStart at, Context& c)
    {
        if (trimming)
        {
            if (at.buffer == trimTo.buffer && at.offset < trimTo.offset)
                return false;
            trimming = false;
        }
        constexpr int kSubframe = 128;
        int from = std::min(n, int(skipSubframes) * kSubframe), to = n;
        skipSubframes = 0;
        // A loop needs a window: the engine leaves count 255 with both at 0 on sounds it repeats itself, which
        // must not loop at the stream's first frame. The start may come after the end: the streamed BGM puts the
        // song's end and then its loop start in one buffer and loops forward from one to the other (the menu's:
        // 397871 -> 486476, 3 subframes skipped). Taken as no loop, the bridge between them played too, a stutter
        // at every loop of the music.
        const bool loopEnd = c.LoopCount() > 0 && c.LoopStart() != c.LoopEnd() && at.offset == c.LoopEnd();
        if (loopEnd)
            to = std::min(n, int(c.LoopSubframeEnd() + 1) * kSubframe);
        if (to > from)
        {
            const float* shifted[2] = {};
            for (int i = 0; planes && i < std::min(planeCount, 2); i++)
                shifted[i] = planes[i] + from;
            Push(planes ? shifted : nullptr, std::min(planeCount, 2), to - from);
        }
        if (!loopEnd)
            return false;
        if (LogEnabled())
            log::Info("XMA loop: buffer {} offset {} -> {} (skip {} subframes, count {})", at.buffer, at.offset, c.LoopStart(),
                c.LoopSubframeSkip(), c.LoopCount());
        if (c.LoopCount() != 255)
            c.SetLoopCount(c.LoopCount() - 1);
        avcodec_flush_buffers(codec);
        starts.clear();
        Seek(at.buffer, c.LoopStart(), c.LoopSubframeSkip());
        return true;
    }

    // Decodes one packet (a copy, skip count cleared) at `buffer`/`packetIndex` into the queue. Returns
    // true when a loop jumped back (the position is already set).
    bool Decode(const uint8_t* data, uint32_t contextIndex, Context& c)
    {
        ForEachFrameStart(data, [&](uint32_t pos) { starts.push_back({ buffer, packetIndex * kPacketBits + pos }); });
        uint8_t copy[kPacketBytes + AV_INPUT_BUFFER_PADDING_SIZE] = {};
        std::memcpy(copy, data, kPacketBytes);
        copy[3] = 0;
        packet->data = copy;
        packet->size = int(kPacketBytes);
        const int err = avcodec_send_packet(codec, packet);
        while (avcodec_receive_frame(codec, frame) == 0)
        {
            const auto* planes = reinterpret_cast<const float* const*>(frame->extended_data);
            const int planeCount = frame->ch_layout.nb_channels;
            for (int at = 0; at < frame->nb_samples; at += int(kFrameSamples))
            {
                const float* chunk[2] = { planes[0] + at, planes[std::min(planeCount, 2) - 1] + at };
                FrameStart start{ UINT32_MAX, UINT32_MAX };
                if (!starts.empty())
                {
                    start = starts.front();
                    starts.pop_front();
                }
                else
                {
                    static uint32_t warned = 0;
                    if (warned++ < 4)
                        log::Warn("XMA: ctx {} decoded more frames than its packets start", contextIndex);
                }
                if (Output(chunk, std::min(planeCount, 2), std::min(int(kFrameSamples), frame->nb_samples - at), start, c))
                {
                    av_frame_unref(frame);
                    return true;
                }
            }
            av_frame_unref(frame);
        }
        if (err < 0)
        {
            // A packet the decoder rejects still occupies its frames in the stream's timeline: the frames
            // owed so far come out as silence, so the streams of a multichannel sound stay in step.
            static uint32_t warned = 0;
            if (warned++ < 8 || LogEnabled())
                log::Warn("XMA: ctx {} packet rejected by the decoder ({}), header {:02X} {:02X} {:02X} {:02X}; {} frame(s) of silence",
                    contextIndex, err, data[0], data[1], data[2], data[3], starts.size());
            avcodec_flush_buffers(codec);
            std::deque<FrameStart> owed;
            owed.swap(starts);
            for (const FrameStart& f : owed)
                if (Output(nullptr, 0, int(kFrameSamples), f, c))
                    return true;
        }
        return false;
    }

    size_t Queued() const { return pcm.size() - pcmRead; }
};

std::mutex s_mutex;
std::atomic<uint32_t> s_multichannelStarts{ 0 };
std::atomic<uint64_t> s_multichannelStartSamples{ 0 };
uint32_t s_arrayGuest = 0;      // guest virtual address of context 0
uint32_t s_arrayPhysical = 0;   // its physical address (register 0x600)
std::array<bool, kContextCount> s_allocated{};
std::array<Decoder, kContextCount> s_decoders;

// Runs one context: decodes into its output ring until the ring is full or the input runs out. Returns true
// when the context started a new sound (its first kick since it was created or cleared).
bool Run(uint32_t index)
{
    const uint32_t guest = s_arrayGuest + index * kContextSize;
    Context c = Load(guest);
    Decoder& d = s_decoders[index];
    if (LogEnabled())
        LogContext("kick", index, c);
    if (!c.OutputValid() || c.BlockCount() == 0)
        return false;

    const int channels = int(c.Channels()), rate = c.SampleRate();
    if (d.channels != channels || d.rate != rate || !d.codec)
        if (!d.Open(channels, rate))
            return false;
    // The engine started a sound or moved the stream: decode from there.
    const bool started = d.expectedReadOffset == UINT32_MAX;
    if (c.ReadOffset() != d.expectedReadOffset)
    {
        if (LogEnabled())
            log::Info("XMA reset ctx {}: read {} (expected {}), buffer {} (expected {}), {} queued", index, c.ReadOffset(),
                d.expectedReadOffset, c.CurrentBuffer(), d.expectedBuffer, d.Queued());
        d.Reset(c.CurrentBuffer(), std::max(c.ReadOffset(), kHeaderBits));
    }
    else if (c.CurrentBuffer() != d.expectedBuffer)
    {
        // Streaming through one buffer: when a chunk runs out (the switch to the other, empty buffer), the
        // engine loads the next chunk into the same buffer and points the context back at it. The stream
        // continues, and so does the decoder (the hardware keeps its state): a flush here would break the
        // overlap between frames at every chunk boundary.
        d.buffer = c.CurrentBuffer();
        d.packetIndex = c.ReadOffset() / kPacketBits;
        d.marks.clear();
    }

    const uint32_t blocks = c.BlockCount();
    uint32_t write = c.WriteOffset() % blocks;
    const uint32_t read = c.OutputReadOffset() % blocks;
    uint32_t freeBlocks = write == read ? blocks : (read + blocks - write) % blocks;
    const size_t samplesPerBlock = kBlockBytes / 2;   // int16 values (both channels counted)
    uint8_t* out = g_memory.As<uint8_t>(PhysicalToGuest(c.w[7]));

    // Commits the decoder's marks up to what has left the queue: the context's read position, and the
    // buffers it has finished (invalidated for the engine to refill).
    auto commit = [&] {
        size_t done = 0;
        for (const Decoder::Mark& m : d.marks)
        {
            if (m.end > d.consumed)
                break;
            if (m.buffer != c.CurrentBuffer())
            {
                c.SetInputValid(c.CurrentBuffer(), false);
                c.SetCurrentBuffer(m.buffer);
            }
            c.SetReadOffset(m.packetIndex * kPacketBits + kHeaderBits);
            done++;
        }
        d.marks.erase(d.marks.begin(), d.marks.begin() + done);
    };

    while (freeBlocks > 0)
    {
        // Enough decoded for one block? Decoding may run ahead into the other input buffer, never past it.
        while (d.Queued() < samplesPerBlock)
        {
            if (!c.InputValid(d.buffer) || !c.InputAddress(d.buffer))
                break;
            const uint32_t count = c.PacketCount(d.buffer);
            if (d.packetIndex >= count)
            {
                // This buffer is used up: the stream continues in the other one.
                if (d.buffer != c.CurrentBuffer())
                    break;   // the other one is used up too, ahead of the output
                d.packetIndex -= count;
                d.buffer ^= 1;
                d.marks.push_back({ d.produced, d.buffer, d.packetIndex });
                continue;
            }
            const uint8_t* packetData = g_memory.As<uint8_t>(PhysicalToGuest(c.InputAddress(d.buffer))) + size_t(d.packetIndex) * kPacketBytes;
            const uint32_t skip = packetData[3];
            if (LogEnabled())
                log::Info("XMA ctx {} decode buffer {} packet {} (skip {}, header {:02X} {:02X} {:02X})", index, d.buffer, d.packetIndex, skip,
                    packetData[0], packetData[1], packetData[2]);
            if (!d.Decode(packetData, index, c))
            {
                d.packetIndex += 1 + (skip == 0xFF ? 0 : skip);
                if (d.packetIndex >= count && d.buffer == c.CurrentBuffer())
                {
                    d.packetIndex -= count;
                    d.buffer ^= 1;
                }
            }
            d.marks.push_back({ d.produced, d.buffer, d.packetIndex });
        }
        if (d.Queued() < samplesPerBlock)
            break;
        // One block out, big-endian.
        auto* dst = reinterpret_cast<uint16_t*>(out + size_t(write) * kBlockBytes);
        for (size_t i = 0; i < samplesPerBlock; i++)
            dst[i] = ByteSwapValue(uint16_t(d.pcm[d.pcmRead + i]));
        d.pcmRead += samplesPerBlock;
        d.consumed += samplesPerBlock;
        write = (write + 1) % blocks;
        freeBlocks--;
        commit();
    }
    commit();

    c.SetWriteOffset(write);
    if (write == read)
        c.SetOutputValid(false);   // the ring is full until the engine consumes
    d.expectedReadOffset = c.ReadOffset();
    d.expectedBuffer = c.CurrentBuffer();
    Store(guest, c);
    if (LogEnabled())
    {
        LogContext("done", index, c);
        log::Info("XMA ctx {}: {} frame(s) in the decoder, {} samples queued", index, d.starts.size(), d.Queued());
    }
    return started;
}

uint32_t RecordIndex(uint32_t record) { return *g_memory.As<be<uint16_t>>(record + 80); }

template <typename F>
void ForEachRecord(uint32_t voices, F&& f)
{
    const uint32_t count = *g_memory.As<be<uint32_t>>(voices);
    const uint32_t records = *g_memory.As<be<uint32_t>>(voices + 8);
    for (uint32_t i = 0; i < count && i < 64; i++)
        if (const uint32_t index = RecordIndex(records + i * 96); index < kContextCount)
            f(index);
}

} // namespace

MultichannelStart LastMultichannelStart()
{
    const uint32_t count = s_multichannelStarts.load(std::memory_order_acquire);
    return { count, s_multichannelStartSamples.load(std::memory_order_relaxed) };
}

void XmaInit()
{
    void* p = g_heap.AllocPhysical(kContextCount * kContextSize, 4096);
    if (!p)
    {
        log::Error("XMA: no memory for the context array");
        return;
    }
    std::memset(p, 0, kContextCount * kContextSize);
    s_arrayGuest = g_memory.MapVirtual(p);
    s_arrayPhysical = s_arrayGuest & 0x1FFFFFFF;
    // The registers are little-endian (the engine reads them with lwbrx).
    *g_memory.As<uint32_t>(kRegisterBase + kRegContextArray * 4) = s_arrayPhysical;
    av_log_set_callback(FfmpegLog);
    log::Info("XMA: {} contexts at {:08X} (physical {:08X}); decoder {}", kContextCount, s_arrayGuest, s_arrayPhysical,
        avcodec_find_decoder(AV_CODEC_ID_XMA2) ? "FFmpeg XMA2" : "missing");
}

uint32_t XmaAllocateContext()
{
    std::lock_guard lock(s_mutex);
    for (uint32_t i = 0; i < kContextCount; i++)
        if (!s_allocated[i])
        {
            s_allocated[i] = true;
            std::memset(g_memory.Translate(s_arrayGuest + i * kContextSize), 0, kContextSize);
            s_decoders[i].Close();
            return s_arrayGuest + i * kContextSize;
        }
    return 0;
}

void XmaReleaseContext(uint32_t context)
{
    std::lock_guard lock(s_mutex);
    if (context >= s_arrayGuest && context < s_arrayGuest + kContextCount * kContextSize)
    {
        const uint32_t index = (context - s_arrayGuest) / kContextSize;
        s_allocated[index] = false;
        s_decoders[index].Close();
    }
}

void XmaKick(uint32_t voices)
{
    std::lock_guard lock(s_mutex);
    uint32_t started = 0;
    ForEachRecord(voices, [&](uint32_t index) { started += Run(index); });
    // A voice of several contexts (a multichannel stream, like a cutscene's 5.1 soundtrack) starting.
    if (started >= 2)
    {
        s_multichannelStartSamples.store(RenderedSamples(), std::memory_order_relaxed);
        s_multichannelStarts.fetch_add(1, std::memory_order_release);
    }
}

// Lock: the engine is about to rewrite the contexts. Decoding is synchronous, so nothing is in flight.
void XmaLock(uint32_t voices) {}

// Clear: the contexts start over (new sounds).
void XmaClear(uint32_t voices)
{
    std::lock_guard lock(s_mutex);
    ForEachRecord(voices, [](uint32_t index) {
        Decoder& d = s_decoders[index];
        d.Reset(0, kHeaderBits);
        d.expectedReadOffset = UINT32_MAX;
    });
}

} // namespace ac6::apu

// The engine's register writers, wrapped: the original runs (its register stores land in plain memory),
// then the hardware's response to them.
extern "C" PPC_FUNC(__imp__sub_823AEFA0);   // copy the voices' contexts to the hardware, kick them
extern "C" PPC_FUNC(__imp__sub_823AED08);   // lock the voices' contexts
extern "C" PPC_FUNC(__imp__sub_823AEC38);   // create and clear the voices' contexts

PPC_FUNC(sub_823AEFA0)
{
    const uint32_t voices = ctx.r3.u32;
    __imp__sub_823AEFA0(ctx, base);
    ac6::apu::XmaKick(voices);
}

PPC_FUNC(sub_823AED08)
{
    const uint32_t voices = ctx.r3.u32;
    // The original does nothing when the voices are already locked (bits 16-17 of the flags word at +4).
    const bool skipped = (*ac6::g_memory.As<ac6::be<uint32_t>>(voices + 4) & 0x30000) != 0;
    __imp__sub_823AED08(ctx, base);
    if (!skipped)
        ac6::apu::XmaLock(voices);
}

PPC_FUNC(sub_823AEC38)
{
    const uint32_t voices = ctx.r3.u32;
    __imp__sub_823AEC38(ctx, base);
    ac6::apu::XmaClear(voices);
}
