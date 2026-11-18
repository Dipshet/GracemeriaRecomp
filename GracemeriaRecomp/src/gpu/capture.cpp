// Frame capture and replay, see capture.h.
//
// File layout (little-endian): "AC6CAP02", u32 frames, u64 event bytes, events, u64 shader-state bytes,
// shader state (shaders.cpp), u64 resolve-target bytes, the resolve targets as the frames started
// (renderer.cpp, RendererRestoreResolved), u32 region count, then per region u32 guest address, u32 size,
// bytes.
// Events are { u32 type, u32 payload bytes, payload }. A gamma ramp event (the display ramp)
// comes first when the game has set one. Memory events carry the guest bytes a call reads
// that can change within a frame (the device state block, index data, shader microcode, the resolve
// destination's texture object). They are recorded only when different from the last recorded copy.
// The snapshot is taken at the last present, which is also when the renderer's deferred vertex copies
// read guest memory.
//
// The pipeline ring (AC6_PIPELINE_RING=<seconds>) records every frame into memory and keeps the
// last seconds. F12 writes them in the same format, so a transient bug can be replayed after it was seen. Each frame
// is self-contained in its calls' state (the device block and shaders are recorded at their first use in every
// frame). The guest data the renderer reads (vertices, textures) comes from the memory snapshot taken at the end,
// corrected three ways: pages the renderer starts watching (what it is about to read) are copied into the frame and
// applied before its calls, data it reads unwatched every frame (vertex copies at submission, textures hashed every
// frame) likewise, and a watched page's first write copies what it held, so the oldest such copy of each page, applied
// before the first frame, puts back what the window started with. The resolve targets carried into the first frame
// aren't kept (a GPU read-back every frame): replays need a few frames to settle.
#include "gpu/capture.h"
#include "platform/platform.h"

#include "endian.h"
#include "gpu/renderer.h"
#include "gpu/shaders.h"
#include "log.h"
#include "memory.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <deque>
#include <filesystem>
#include <format>
#include <fstream>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace ac6::gpu::capture {

void GammaRampEvent();

namespace {

enum EventType : uint32_t
{
    kMemory = 1,
    kClear,
    kDraw,
    kDrawIndexed,
    kResolve,
    kBeginTiling,
    kEndTiling,
    kPresent,
    kVsVariant,
    kVsLoad,
    kVsInPlace,
    kGammaRamp,   // the display gamma ramp as the capture started (1024 packed entries, renderer.cpp)
    kSeedFrame,   // u64: the live frame number of the frame starting
};

// Guest layout the renderer reads (gpu/renderer.cpp, gpu/shaders.cpp).
constexpr uint32_t kDeviceBlock = 0x3200;           // device state up to the shader/surface pointers
constexpr uint32_t kDevVertexShader = 0x3190, kDevPixelShader = 0x318C, kDevIndexBuffer = 0x308C;
constexpr uint32_t kVsMicrocode = 0x20, kVsContainer = 0x368, kPsMicrocode = 0x18, kPsContainer = 0x28;
constexpr uint32_t kIndexBufferAddress = 0x18;
constexpr uint32_t kTextureObjectBytes = 0x40;

std::mutex s_mutex;
bool s_recording = false;
uint64_t s_firstFrame = UINT64_MAX, s_frameCount = 1, s_framesDone = 0;
std::vector<uint8_t> s_events;
std::vector<uint8_t> s_shaderState;
std::vector<uint8_t> s_resolvedImages;
std::unordered_map<uint32_t, uint64_t> s_lastHash;            // memory range start -> hash recorded last
std::unordered_map<uint32_t, std::vector<uint8_t>> s_lastDevice;
std::vector<uint8_t>* s_out = &s_events;   // where events go: the capture, or the ring's current frame
std::unordered_set<uint32_t> s_frameShaders;   // the ring: shader objects recorded in the current frame

// The pipeline ring: one entry per frame, oldest first.
struct RingFrame
{
    uint64_t frame = 0;
    std::chrono::steady_clock::time_point start;
    std::vector<uint8_t> reads;    // memory events: what the renderer read in the frame (applied before its calls)
    std::vector<uint8_t> events;   // the frame's calls
    std::vector<uint8_t> undo;     // {u32 guest, 4 KB}: watched pages as they were before their first write
    std::set<std::pair<uint32_t, uint32_t>> readRanges;   // unwatched reads recorded in the frame
    std::shared_ptr<const std::vector<uint8_t>> shaderState;   // the vertex shader state as the frame started
    std::vector<uint8_t> resolved;   // the small resolve targets and surfaces as it started (exposure, luminance)
    size_t Bytes() const { return reads.capacity() + events.capacity() + undo.capacity() + resolved.capacity(); }
    size_t counted = 0;   // its part of s_ringBytes
};
std::mutex s_ringMutex;   // the ring's frames (page copies come from any thread, under the watch lock)
std::deque<RingFrame> s_ring;
size_t s_ringBytes = 0;   // of the frames before the current one
std::atomic<bool> s_ringDumpRequested{ false };
uint64_t s_ringShaderEpoch = UINT64_MAX;
std::shared_ptr<const std::vector<uint8_t>> s_ringShaderState;

double RingSeconds()
{
    // (Nothing kept with the captures' switch off at the start.)
    static const double seconds = [] {
        const char* on = std::getenv("AC6_CAPTURES");
        if (!on || (std::string_view(on) != "1" && std::string_view(on) != "true"))
            return 0.0;
        const char* v = std::getenv("AC6_PIPELINE_RING");
        return v ? std::max(0.0, std::atof(v)) : 0.0;
    }();
    return seconds;
}

uint32_t U32(uint32_t guest) { return *g_memory.As<be<uint32_t>>(guest); }

uint64_t Hash(const uint8_t* p, size_t n)
{
    uint64_t h = 0xCBF29CE484222325ull ^ n;
    size_t i = 0;
    for (; i + 8 <= n; i += 8)
    {
        uint64_t v;
        std::memcpy(&v, p + i, 8);
        h = (h ^ v) * 0x100000001B3ull;
        h ^= h >> 29;
    }
    for (; i < n; i++)
        h = (h ^ p[i]) * 0x100000001B3ull;
    return h;
}

void Put(const void* p, size_t n)
{
    auto* b = static_cast<const uint8_t*>(p);
    s_out->insert(s_out->end(), b, b + n);
}

// A memory event into `out` (the ring's page copies and reads).
void PutMemory(std::vector<uint8_t>& out, uint32_t guest, const uint8_t* bytes, uint32_t size)
{
    const uint32_t header[4] = { kMemory, 8 + size, guest, size };
    out.insert(out.end(), reinterpret_cast<const uint8_t*>(header), reinterpret_cast<const uint8_t*>(header) + 16);
    out.insert(out.end(), bytes, bytes + size);
}

template<typename... Words> void Event(uint32_t type, const Words&... words)
{
    const uint32_t payload = uint32_t((sizeof(words) + ... + 0));
    Put(&type, 4);
    Put(&payload, 4);
    (Put(&words, sizeof(words)), ...);
}

void Memory(uint32_t guest, uint32_t size)
{
    const uint32_t type = kMemory, payload = 8 + size;
    Put(&type, 4);
    Put(&payload, 4);
    Put(&guest, 4);
    Put(&size, 4);
    Put(g_memory.Translate(guest), size);
}

// Records [guest, guest + size) unless it holds what was recorded for that range last time.
void MemoryIfChanged(uint32_t guest, uint32_t size)
{
    if (!guest || !size || size > (16u << 20) || !g_memory.IsBacked(guest) || !g_memory.IsBacked(guest + size - 1))
        return;
    const uint64_t h = Hash(g_memory.As<uint8_t>(guest), size) ^ size;
    auto [it, inserted] = s_lastHash.try_emplace(guest, h);
    if (!inserted && it->second == h)
        return;
    it->second = h;
    Memory(guest, size);
}

// The device block, as runs of changed 64-byte chunks.
void Device(uint32_t device)
{
    auto& last = s_lastDevice[device];
    const uint8_t* now = g_memory.As<uint8_t>(device);
    if (last.empty())
    {
        Memory(device, kDeviceBlock);
        last.assign(now, now + kDeviceBlock);
        return;
    }
    for (uint32_t at = 0; at < kDeviceBlock;)
    {
        if (std::memcmp(&last[at], now + at, 64) == 0)
        {
            at += 64;
            continue;
        }
        uint32_t end = at + 64;
        while (end < kDeviceBlock && std::memcmp(&last[end], now + end, 64) != 0)
            end += 64;
        Memory(device + at, end - at);
        std::memcpy(&last[at], now + at, end - at);
        at = end;
    }
}

void Shader(uint32_t object, bool pixel)
{
    if (!object || !g_memory.IsBacked(object))
        return;
    const uint32_t virt = object + (pixel ? kPsContainer : kVsContainer);
    const uint32_t virtualSize = U32(virt + 4), physicalSize = U32(virt + 8);
    const uint32_t microcode = U32(object + (pixel ? kPsMicrocode : kVsMicrocode));
    if (virtualSize > 0x10000 || physicalSize > 0x40000)
        return;
    // The ring: once a frame (the renderer compares a shader's memory once a frame too, gpu/shaders.cpp).
    if (RingSeconds() > 0.0)
    {
        if (!s_frameShaders.insert(object).second)
            return;
        Memory(object, (pixel ? kPsContainer : kVsContainer) + virtualSize);
        if (microcode && physicalSize && g_memory.IsBacked(microcode) && g_memory.IsBacked(microcode + physicalSize - 1))
            Memory(microcode, physicalSize);
        return;
    }
    MemoryIfChanged(object, (pixel ? kPsContainer : kVsContainer) + virtualSize);
    MemoryIfChanged(microcode, physicalSize);
}

void DrawState(uint32_t device)
{
    Device(device);
    Shader(U32(device + kDevVertexShader), false);
    Shader(U32(device + kDevPixelShader), true);
}

// Writes a capture: the events (in parts, in order), the shader state and resolve targets, then guest memory as it is.
void WriteFile(const std::filesystem::path& path, uint32_t frames, const std::vector<std::pair<const uint8_t*, size_t>>& eventParts,
    const std::vector<uint8_t>& shaderState, const std::vector<uint8_t>& resolvedImages)
{
    // Written under a temporary name and renamed when complete: an exit mid-write leaves no capture.
    auto partial = path;
    partial += ".partial";
    std::ofstream f(partial, std::ios::binary);
    f.write("AC6CAP02", 8);
    f.write(reinterpret_cast<const char*>(&frames), 4);
    uint64_t eventBytes = 0;
    for (const auto& part : eventParts)
        eventBytes += part.second;
    f.write(reinterpret_cast<const char*>(&eventBytes), 8);
    for (const auto& [data, size] : eventParts)
        f.write(reinterpret_cast<const char*>(data), std::streamsize(size));
    const uint64_t stateBytes = shaderState.size();
    f.write(reinterpret_cast<const char*>(&stateBytes), 8);
    f.write(reinterpret_cast<const char*>(shaderState.data()), std::streamsize(stateBytes));
    const uint64_t resolvedBytes = resolvedImages.size();
    f.write(reinterpret_cast<const char*>(&resolvedBytes), 8);
    f.write(reinterpret_cast<const char*>(resolvedImages.data()), std::streamsize(resolvedBytes));

    // Committed guest memory below the physical alias, in 64 KB chunks that aren't all zero.
    std::vector<std::pair<uint32_t, uint32_t>> regions;
    g_memory.ForEachCommitted(kPhysHeapEnd, [&](uint64_t begin, uint64_t end) {
        for (uint64_t chunk = begin; chunk < end; chunk += 0x10000)
        {
            const uint32_t size = uint32_t(std::min<uint64_t>(0x10000, end - chunk));
            const uint8_t* p = g_memory.As<uint8_t>(uint32_t(chunk));
            bool zero = true;
            for (uint32_t i = 0; i < size && zero; i += 8)
            {
                uint64_t v;
                std::memcpy(&v, p + i, 8);
                zero = v == 0;
            }
            if (zero)
                continue;
            if (!regions.empty() && regions.back().first + regions.back().second == chunk)
                regions.back().second += size;
            else
                regions.push_back({ uint32_t(chunk), size });
        }
    });
    uint64_t total = 0;
    const uint32_t count = uint32_t(regions.size());
    f.write(reinterpret_cast<const char*>(&count), 4);
    for (auto [guest, size] : regions)
    {
        f.write(reinterpret_cast<const char*>(&guest), 4);
        f.write(reinterpret_cast<const char*>(&size), 4);
        f.write(reinterpret_cast<const char*>(g_memory.Translate(guest)), size);
        total += size;
    }
    f.close();
    std::error_code error;
    std::filesystem::rename(partial, path, error);
    if (error)
        log::Error("Capture: could not rename {}: {}", platform::PathUtf8(partial), error.message());
    log::Info("Capture written: {} ({} frames, {:.1f} MB of events, {:.0f} MB of memory in {} regions)", platform::PathUtf8(path), frames,
        eventBytes / 1048576.0, total / 1048576.0, count);
}

void Write()
{
    WriteFile(std::filesystem::path(log::Directory()) / std::format("capture_{}.ac6cap", s_firstFrame), uint32_t(s_frameCount),
        { { s_events.data(), s_events.size() } }, s_shaderState, s_resolvedImages);
}

// The ring's page copies (GuestMemory::SetPageCopyHook): into the current frame.
void RingPageCopy(uint32_t guest, const uint8_t* page, bool written)
{
    std::lock_guard lock(s_ringMutex);
    if (s_ring.empty())
        return;
    RingFrame& f = s_ring.back();
    if (written)
    {
        f.undo.insert(f.undo.end(), reinterpret_cast<const uint8_t*>(&guest), reinterpret_cast<const uint8_t*>(&guest) + 4);
        f.undo.insert(f.undo.end(), page, page + 4096);
    }
    else
        PutMemory(f.reads, guest, page, 4096);
}

// A new frame in the ring (on the renderer's worker at the end of a present: the game waits for it then), and the
// frames older than the ring's length dropped. The ring's lock is never held while guest memory or another lock is
// touched: the page copies take it inside the watch lock, from any thread.
void RingNextFrame(uint64_t frame)
{
    static const size_t maxBytes = [] {
        const char* v = std::getenv("AC6_PIPELINE_RING_MB");
        return size_t(v ? std::max(64.0, std::atof(v)) : 8192.0) << 20;
    }();
    const auto now = std::chrono::steady_clock::now();
    if (const uint64_t epoch = VertexShaderStateEpoch(); epoch != s_ringShaderEpoch || !s_ringShaderState)
    {
        s_ringShaderState = std::make_shared<const std::vector<uint8_t>>(ExportVertexShaderState());
        s_ringShaderEpoch = epoch;
    }
    std::deque<RingFrame> dropped;
    {
        std::lock_guard lock(s_ringMutex);
        size_t reserve = 0;
        // A frame without calls (begun while the ring was written) becomes this one.
        if (!s_ring.empty() && s_ring.back().events.empty())
            s_ring.back().counted = 0;
        else
        {
            if (!s_ring.empty())
            {
                s_ring.back().counted = s_ring.back().Bytes();
                s_ringBytes += s_ring.back().counted;
                reserve = s_ring.back().events.size();
            }
            while (!s_ring.empty() &&
                   (now - s_ring.front().start > std::chrono::duration<double>(RingSeconds()) || s_ringBytes > maxBytes))
            {
                s_ringBytes -= s_ring.front().counted;
                dropped.push_back(std::move(s_ring.front()));
                s_ring.pop_front();
            }
            s_ring.emplace_back().events.reserve(reserve + reserve / 8);
        }
        RingFrame& f = s_ring.back();
        f.frame = frame;
        f.start = now;
        f.shaderState = s_ringShaderState;
        s_out = &f.events;
    }
    dropped.clear();   // freed outside the lock
    // The small images carried from frame to frame (the exposure adaptation's: without them a replay starts washed
    // out and takes ~1.5 s to adapt), read back as the frame starts, kept when its slot comes round again.
    static const uint64_t maxTexels = [] { const char* v = std::getenv("AC6_PIPELINE_RING_RESOLVED"); return v ? std::stoull(v) : 65536ull; }();
    if (maxTexels)
        RendererSnapshotResolved(maxTexels);
    // Every frame records the state its calls read from scratch.
    s_lastHash.clear();
    s_lastDevice.clear();
    s_frameShaders.clear();
}

// F12: the ring as a capture (on the worker after a present, the game waiting): the gamma ramp, the pages as the
// window started, then each frame's reads and calls, memory as it is now. A new frame takes the page copies made
// meanwhile, so the frames written don't change while they are (only this thread adds or drops frames).
void WriteRing()
{
    const auto started = std::chrono::steady_clock::now();
    size_t frames;
    {
        std::lock_guard lock(s_ringMutex);
        if (s_ring.empty())
            return;
        frames = s_ring.size();
        s_ring.back().counted = s_ring.back().Bytes();
        s_ringBytes += s_ring.back().counted;
        s_ring.emplace_back().frame = s_ring[frames - 1].frame + 1;
        s_ring.back().start = std::chrono::steady_clock::now();
        s_ring.back().shaderState = s_ringShaderState;
        s_out = &s_ring.back().events;
    }
    // The deque's first `frames` entries stay put: emplace_back keeps references, nothing pops until RingNextFrame.
    std::vector<uint8_t> head;
    {
        std::vector<uint8_t>* out = s_out;
        s_out = &head;
        GammaRampEvent();
        Event(kSeedFrame, s_ring[0].frame);
        s_out = out;
    }
    // The oldest copy of each page written in the window: what it held as the window started.
    std::set<uint32_t> pages;
    size_t undoPages = 0, reads = 0;
    for (size_t i = 0; i < frames; i++)
    {
        const RingFrame& f = s_ring[i];
        for (size_t at = 0; at + 4100 <= f.undo.size(); at += 4100)
        {
            uint32_t guest;
            std::memcpy(&guest, &f.undo[at], 4);
            if (pages.insert(guest).second)
            {
                PutMemory(head, guest, &f.undo[at + 4], 4096);
                undoPages++;
            }
        }
    }
    std::vector<std::pair<const uint8_t*, size_t>> parts{ { head.data(), head.size() } };
    for (size_t i = 0; i < frames; i++)
    {
        const RingFrame& f = s_ring[i];
        parts.push_back({ f.reads.data(), f.reads.size() });
        parts.push_back({ f.events.data(), f.events.size() });
        reads += f.reads.size();
    }
    const RingFrame& first = s_ring[0];
    const RingFrame& last = s_ring[frames - 1];
    WriteFile(std::filesystem::path(log::Directory()) / std::format("pipeline_{}.ac6cap", first.frame), uint32_t(frames), parts,
        *first.shaderState, first.resolved);
    log::Info("Pipeline ring written: frames {}-{} ({:.1f} s), {} pages put back, {:.1f} MB of reads, in {:.1f} s", first.frame,
        last.frame, std::chrono::duration<double>(last.start - first.start).count(), undoPages, reads / 1048576.0,
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count());
}

} // namespace

bool Recording() { return s_recording; }

bool RingEnabled() { return RingSeconds() > 0.0; }

void RequestRingCapture()
{
    if (!RingEnabled())
        return;
    s_ringDumpRequested = true;
    log::Info("Pipeline ring: writing the last {} s at the end of this frame", RingSeconds());
}

void OnRendererRead(uint32_t guest, uint32_t size)
{
    if (!RingEnabled() || !size || size > (64u << 20) || !g_memory.IsBacked(guest) || !g_memory.IsBacked(guest + size - 1))
        return;
    // Copied before the lock is taken (a read may commit a page, which takes the watch lock).
    std::vector<uint8_t> bytes(g_memory.As<uint8_t>(guest), g_memory.As<uint8_t>(guest) + size);
    std::lock_guard lock(s_ringMutex);
    if (s_ring.empty())
        return;
    RingFrame& f = s_ring.back();
    if (f.readRanges.insert({ guest, size }).second)
        PutMemory(f.reads, guest, bytes.data(), size);
}

std::atomic<uint64_t> s_requestedFrame{ UINT64_MAX };

bool Active()
{
    static const bool armed = [] { const char* v = std::getenv("AC6_CAPTURE"); return v && *v; }();
    return armed || s_recording || RingEnabled() || s_requestedFrame.load() != UINT64_MAX;
}

void RequestCapture(uint32_t frames)
{
    // With the ring, F12 writes it instead.
    if (RingEnabled())
    {
        RequestRingCapture();
        return;
    }
    std::lock_guard lock(s_mutex);
    if (s_recording)
        return;
    s_frameCount = frames;
    s_requestedFrame = RendererFrame() + 1;
    log::Info("Capture requested (F12): frames {}-{}", RendererFrame() + 1, RendererFrame() + frames);
}

// The scan-out gamma ramp the game set (at boot, before any capture starts).
void GammaRampEvent()
{
    uint32_t lut[1024];
    if (RendererGammaRamp(lut))
        Event(kGammaRamp, lut);
}

void OnFrameStart(uint64_t frame)
{
    if (RingEnabled())
    {
        std::lock_guard lock(s_mutex);
        if (!s_recording)
        {
            s_recording = true;
            g_memory.SetPageCopyHook(RingPageCopy);
            log::Info("Pipeline ring: the last {} s of frames kept for F12 (AC6_PIPELINE_RING)", RingSeconds());
        }
        RingNextFrame(frame);
        return;
    }
    if (frame == s_requestedFrame.load())
    {
        std::lock_guard lock(s_mutex);
        s_requestedFrame = UINT64_MAX;
        s_firstFrame = frame;
        s_framesDone = 0;
        s_recording = true;
        // From scratch: the first frame records every state it reads (a capture before this one left its last state).
        s_lastHash.clear();
        s_lastDevice.clear();
        GammaRampEvent();
        Event(kSeedFrame, frame);
        s_shaderState = ExportVertexShaderState();
        s_resolvedImages.clear();
        RendererSnapshotResolved();
        log::Info("Capture: recording frames {}-{}", s_firstFrame, s_firstFrame + s_frameCount - 1);
        return;
    }
    static const bool requested = [] {
        if (const char* v = std::getenv("AC6_CAPTURE"); v && *v)
        {
            std::string text(v);
            s_firstFrame = std::stoull(text);
            if (auto colon = text.find(':'); colon != std::string::npos)
                s_frameCount = std::max<uint64_t>(1, std::stoull(text.substr(colon + 1)));
            return true;
        }
        return false;
    }();
    if (!requested || frame != s_firstFrame)
        return;
    std::lock_guard lock(s_mutex);
    s_recording = true;
    // From scratch: the first frame records every state it reads (a capture before this one left its last state).
    s_lastHash.clear();
    s_lastDevice.clear();
    GammaRampEvent();
    Event(kSeedFrame, frame);
    s_shaderState = ExportVertexShaderState();
    s_resolvedImages.clear();
    RendererSnapshotResolved();
    log::Info("Capture: recording frames {}-{}", s_firstFrame, s_firstFrame + s_frameCount - 1);
}

void OnRingResolvedImages(uint64_t frame, std::vector<uint8_t> blob)
{
    std::lock_guard lock(s_ringMutex);
    for (auto it = s_ring.rbegin(); it != s_ring.rend(); ++it)
        if (it->frame == frame)
        {
            it->resolved = std::move(blob);
            if (it->counted)   // a frame already counted: its images now too
            {
                s_ringBytes += it->resolved.capacity();
                it->counted += it->resolved.capacity();
            }
            break;
        }
}

void OnResolvedImages(std::vector<uint8_t> blob)
{
    std::lock_guard lock(s_mutex);
    s_resolvedImages = std::move(blob);
}

void OnPresent()
{
    if (!s_recording)
        return;
    std::lock_guard lock(s_mutex);
    Event(kPresent);
    if (RingEnabled())
    {
        if (s_ringDumpRequested.exchange(false))
            WriteRing();
        return;
    }
    if (++s_framesDone < s_frameCount)
        return;
    s_recording = false;
    Write();
    s_events.clear();
    s_events.shrink_to_fit();
}

void OnClear(uint32_t device, uint32_t flags, const float color[4], float z, uint32_t stencil)
{
    if (!s_recording)
        return;
    std::lock_guard lock(s_mutex);
    Device(device);
    Event(kClear, device, flags, color[0], color[1], color[2], color[3], z, stencil);
}

void OnDraw(uint32_t device, uint32_t primType, uint32_t startVertex, uint32_t vertexCount)
{
    if (!s_recording)
        return;
    std::lock_guard lock(s_mutex);
    DrawState(device);
    Event(kDraw, device, primType, startVertex, vertexCount);
}

void OnDrawIndexed(uint32_t device, uint32_t primType, int32_t baseVertex, uint32_t startIndex, uint32_t indexCount)
{
    if (!s_recording)
        return;
    std::lock_guard lock(s_mutex);
    DrawState(device);
    if (const uint32_t ib = U32(device + kDevIndexBuffer); ib && g_memory.IsBacked(ib))
    {
        MemoryIfChanged(ib, 0x20);
        const bool wide = (U32(ib) & 0x80000000u) != 0;
        const uint32_t size = indexCount * (wide ? 4 : 2);
        const uint32_t start = (PhysicalToGuest(U32(ib + kIndexBufferAddress)) + startIndex * (wide ? 4 : 2)) & ~3u;
        Memory(start, (size + 7) & ~3u);
    }
    Event(kDrawIndexed, device, primType, baseVertex, startIndex, indexCount);
}

void OnResolve(uint32_t device, uint32_t flags, const ResolveRect* srcRect, uint32_t destTexture, const ResolvePoint* destPoint,
    uint32_t destLevel, const float* clearColor, float clearZ, uint32_t clearStencil)
{
    if (!s_recording)
        return;
    std::lock_guard lock(s_mutex);
    Device(device);
    if (destTexture)
        MemoryIfChanged(destTexture, kTextureObjectBytes);
    const ResolveRect rect = srcRect ? *srcRect : ResolveRect{};
    const ResolvePoint point = destPoint ? *destPoint : ResolvePoint{};
    float color[4] = {};
    if (clearColor)
        std::memcpy(color, clearColor, sizeof(color));
    const uint32_t has = (srcRect ? 1u : 0u) | (destPoint ? 2u : 0u) | (clearColor ? 4u : 0u);
    Event(kResolve, device, flags, has, rect, destTexture, point, destLevel, color, clearZ, clearStencil);
}

void OnBeginTiling(const ResolveRect* rects, uint32_t count)
{
    if (!s_recording)
        return;
    std::lock_guard lock(s_mutex);
    ResolveRect r[16]{};
    count = rects ? std::min<uint32_t>(count, 16) : 0;
    for (uint32_t i = 0; i < count; i++)
        r[i] = rects[i];
    Event(kBeginTiling, count, r);
}

void OnEndTiling()
{
    if (!s_recording)
        return;
    std::lock_guard lock(s_mutex);
    Event(kEndTiling);
}

void OnVertexShaderVariant(uint32_t object, uint32_t variant)
{
    if (!s_recording)
        return;
    std::lock_guard lock(s_mutex);
    Event(kVsVariant, object, variant);
}

void OnVertexShaderLoad(uint32_t object, uint32_t offset, const std::vector<uint8_t>& bytes)
{
    if (!s_recording)
        return;
    std::lock_guard lock(s_mutex);
    const uint32_t type = kVsLoad, size = uint32_t(bytes.size()), payload = 12 + size;
    Put(&type, 4);
    Put(&payload, 4);
    Put(&object, 4);
    Put(&offset, 4);
    Put(&size, 4);
    Put(bytes.data(), size);
}

void OnVertexShaderPatchedInPlace(uint32_t object, uint32_t variant)
{
    if (!s_recording)
        return;
    std::lock_guard lock(s_mutex);
    Event(kVsInPlace, object, variant);
}

int Replay(const char* path, uint32_t repeat)
{
    std::ifstream f(path, std::ios::binary);
    std::vector<uint8_t> file((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (file.size() < 36 || std::memcmp(file.data(), "AC6CAP02", 8) != 0)
    {
        log::Error("Replay: {} is not a capture", path);
        return 1;
    }
    // Every size in the file is checked against what's left of it (a capture cut short by a full disk, or damaged).
    size_t at = 8;
    bool bad = false;
    auto fits = [&](uint64_t n) {
        if (at > file.size() || n > file.size() - at)
            bad = true;
        return !bad;
    };
    auto read = [&](void* out, size_t n) {
        if (fits(n))
            std::memcpy(out, file.data() + at, n);
        else
            std::memset(out, 0, n);
        at += n;
    };
    auto skip = [&](uint64_t n) {
        if (fits(n))
            at += size_t(n);
    };
    uint32_t frames;
    uint64_t eventBytes, stateBytes;
    read(&frames, 4);
    read(&eventBytes, 8);
    const size_t events = at;
    skip(eventBytes);
    read(&stateBytes, 8);
    const size_t state = at;
    skip(stateBytes);
    uint64_t resolvedBytes;
    read(&resolvedBytes, 8);
    const size_t resolved = at;
    skip(resolvedBytes);
    uint32_t regions;
    read(&regions, 4);
    uint64_t total = 0;
    for (uint32_t i = 0; i < regions && !bad; i++)
    {
        uint32_t guest, size;
        read(&guest, 4);
        read(&size, 4);
        if (!fits(size) || uint64_t(guest) + size > 0x100000000ull)
        {
            bad = true;
            break;
        }
        g_memory.AddBackedRange(guest, uint64_t(guest) + size);
        g_memory.Commit(guest, size);
        std::memcpy(g_memory.Translate(guest), file.data() + at, size);
        at += size;
        total += size;
    }
    if (bad)
    {
        log::Error("Replay: {} is cut short or damaged (a section runs past its {} bytes)", path, file.size());
        return 1;
    }
    log::Info("Replay: {} ({} frames, {:.0f} MB of memory)", path, frames, total / 1048576.0);

    // AC6_REPLAY_LIST=1: log every call (not memory events) of the first pass, in order.
    static const bool list = [] { const char* v = std::getenv("AC6_REPLAY_LIST"); return v && *v == '1'; }();
    for (uint32_t pass = 0; pass < std::max(1u, repeat); pass++)
    {
        uint32_t index = 0;
        ImportVertexShaderState(file.data() + state, stateBytes);
        RendererRestoreResolved(file.data() + resolved, resolvedBytes);
        for (size_t e = events; e < events + eventBytes;)
        {
            uint32_t type, payload;
            if (events + eventBytes - e < 8)
            {
                log::Error("Replay: event {} is cut short", index);
                return 1;
            }
            std::memcpy(&type, file.data() + e, 4);
            std::memcpy(&payload, file.data() + e + 4, 4);
            if (payload > events + eventBytes - e - 8)
            {
                log::Error("Replay: event {} ({} bytes) runs past the events", index, payload);
                return 1;
            }
            const uint8_t* p = file.data() + e + 8;
            e += 8 + payload;
            auto w = [p](uint32_t i) { uint32_t v; std::memcpy(&v, p + i * 4, 4); return v; };
            auto fl = [p](uint32_t i) { float v; std::memcpy(&v, p + i * 4, 4); return v; };
            // The bytes each kind reads (its words, and the data some carry).
            const uint64_t need = type == kMemory ? (payload >= 8 ? 8ull + w(1) : 8ull)
                : type == kClear ? 32 : type == kDraw ? 16 : type == kDrawIndexed ? 20 : type == kResolve ? 68
                : type == kBeginTiling ? 4 + 16 * sizeof(ResolveRect) : type == kVsVariant || type == kVsInPlace ? 8
                : type == kVsLoad ? (payload >= 12 ? 12ull + w(2) : 12ull) : type == kGammaRamp ? 4096 : type == kSeedFrame ? 8 : 0;
            if (payload < need || (type == kMemory && uint64_t(w(0)) + w(1) > 0x100000000ull))
            {
                log::Error("Replay: event {} (kind {}) has {} bytes, needs {}: the capture is damaged", index, type, payload, need);
                return 1;
            }
            if (list && pass == 0 && type != kMemory)
            {
                static constexpr const char* kNames[] = { "", "memory", "clear", "draw", "draw indexed", "resolve", "begin tiling",
                    "end tiling", "present", "vs variant", "vs load", "vs in place", "gamma ramp", "seed frame" };
                std::string args;
                for (uint32_t i = 0; i < std::min<uint32_t>(payload / 4, 8); i++)
                    args += std::format(" {:08X}", w(i));
                log::Info("Replay event {}: {}{}", index, type < std::size(kNames) ? kNames[type] : "?", args);
            }
            // AC6_REPLAY_DEPTH_PROBE=<event>,...: read back the depth surface before those events (first pass).
            static const std::vector<uint32_t> probes = [] {
                std::vector<uint32_t> list;
                if (const char* v = std::getenv("AC6_REPLAY_DEPTH_PROBE"))
                    for (std::string item; const char c : std::string(v) + ",")
                    {
                        if (c == ',') { if (!item.empty()) list.push_back(uint32_t(std::stoul(item))); item.clear(); }
                        else item += c;
                    }
                return list;
            }();
            if (pass == 0 && type != kMemory && std::find(probes.begin(), probes.end(), index) != probes.end())
                RendererProbeDepth(0x40001A00, std::format("before event {}", index).c_str());
            index++;
            switch (type)
            {
            case kMemory:
                if (!g_memory.IsBacked(w(0)) || !g_memory.IsBacked(w(0) + w(1) - 1))
                {
                    // A page the snapshot left out as all zero (the pipeline ring's page copies).
                    g_memory.AddBackedRange(w(0), uint64_t(w(0)) + w(1));
                    g_memory.Commit(w(0), w(1));
                }
                std::memcpy(g_memory.Translate(w(0)), p + 8, w(1));
                break;
            case kClear:
            {
                const float color[4] = { fl(2), fl(3), fl(4), fl(5) };
                RendererClear(w(0), w(1), color, fl(6), w(7));
                break;
            }
            case kDraw:
                RendererDraw(w(0), w(1), w(2), w(3));
                break;
            case kDrawIndexed:
                RendererDrawIndexed(w(0), w(1), int32_t(w(2)), w(3), w(4));
                break;
            case kResolve:
            {
                const uint32_t has = w(2);
                const ResolveRect rect{ int32_t(w(3)), int32_t(w(4)), int32_t(w(5)), int32_t(w(6)) };
                const ResolvePoint point{ int32_t(w(8)), int32_t(w(9)) };
                const float color[4] = { fl(11), fl(12), fl(13), fl(14) };
                RendererResolve(w(0), w(1), (has & 1) ? &rect : nullptr, w(7), (has & 2) ? &point : nullptr, w(10),
                    (has & 4) ? color : nullptr, fl(15), w(16));
                break;
            }
            case kBeginTiling:
            {
                ResolveRect rects[16];
                std::memcpy(rects, p + 4, sizeof(rects));
                RendererBeginTiling(w(0) ? rects : nullptr, w(0));
                break;
            }
            case kEndTiling:
                RendererEndTiling();
                break;
            case kPresent:
                RendererPresent();
                break;
            case kVsVariant:
                NoteVertexShaderVariant(w(0), w(1));
                break;
            case kVsLoad:
                ReplayVertexShaderLoad(w(0), w(1), std::vector<uint8_t>(p + 12, p + 12 + w(2)));
                break;
            case kVsInPlace:
                NoteVertexShaderPatchedInPlace(w(0), w(1));
                break;
            case kGammaRamp:
                RendererRestoreGammaRamp(reinterpret_cast<const uint32_t*>(p));
                break;
            case kSeedFrame:
            {
                uint64_t frame;
                std::memcpy(&frame, p, 8);
                RendererSetSeedFrame(frame);
                break;
            }
            default:
                log::Error("Replay: unknown event {} at {}", type, e);
                return 1;
            }
        }
    }
    RendererWaitIdle();   // the last frame's screenshot is written by the renderer's worker
    log::Info("Replay: done ({} frames rendered)", RendererFrame());
    return 0;
}

} // namespace ac6::gpu::capture
