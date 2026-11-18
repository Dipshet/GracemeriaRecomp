// XAudio render-driver client (XAudioRegisterRenderDriverClient): the console's audio hardware asks the
// game's mixer for one 256-sample frame (48 kHz, 6 channels, planar big-endian float) every 5.33 ms.
// The client protocol follows UnleashedRecomp (UnleashedRecomp/apu, GPL-3.0): the callback gets a
// pointer to a guest copy of its parameter.
//
// The submitted frames go to an SDL3 audio stream: 5.1 at 48 kHz, which SDL downmixes
// to the device's layout. The device's consumption paces the mixer (AudioClock). Without a device a timer
// does, so the game's audio engine, and anything paced by it (movies), keeps moving either way.
#include "apu/audio_driver.h"
#include "apu/xma.h"
#include "cpu/guest_thread.h"
#include "heap.h"
#include "kernel/dispatcher.h"
#include "kernel/function.h"
#include "log.h"
#include "memory.h"
#include "settings.h"

#include <SDL3/SDL.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace ac6::apu {

namespace {

constexpr uint32_t kDriverHandle = 0x44415544;   // 'DAUD'
constexpr uint32_t kSampleRate = 48000;
constexpr uint32_t kFrameSamples = 256;

std::atomic<uint32_t> s_callback{ 0 };
uint32_t s_callbackParam = 0;   // guest address of the parameter copy
std::atomic<uint64_t> s_frames{ 0 };
SDL_AudioStream* s_stream = nullptr;
uint32_t s_targetFrames = 4;   // frames kept queued for the device (device-driven pacing)
uint64_t s_dropped = 0;
constexpr int kFrameBytes = int(kFrameSamples) * 6 * 4;

// AC6_AUDIO_DUMP=<file.wav>: the mixed output as a 6-channel float WAV (header rewritten every second).
struct WavDump
{
    std::FILE* file = nullptr;
    uint32_t dataBytes = 0;

    void Open(const char* path)
    {
        file = std::fopen(path, "wb");
        if (file)
            WriteHeader();
    }

    void WriteHeader()
    {
        auto put32 = [this](uint32_t v) { std::fwrite(&v, 4, 1, file); };
        auto put16 = [this](uint16_t v) { std::fwrite(&v, 2, 1, file); };
        std::fseek(file, 0, SEEK_SET);
        std::fwrite("RIFF", 1, 4, file);
        put32(36 + dataBytes);
        std::fwrite("WAVEfmt ", 1, 8, file);
        put32(16);
        put16(3);   // IEEE float
        put16(6);
        put32(kSampleRate);
        put32(kSampleRate * 6 * 4);
        put16(6 * 4);
        put16(32);
        std::fwrite("data", 1, 4, file);
        put32(dataBytes);
        std::fseek(file, 0, SEEK_END);
    }

    void Write(const float* samples, uint32_t bytes)
    {
        if (!file)
            return;
        std::fwrite(samples, 1, bytes, file);
        dataBytes += bytes;
        if (dataBytes % (kSampleRate * 6 * 4) < bytes)
        {
            WriteHeader();
            std::fflush(file);
        }
    }
} s_dump;

std::atomic<bool> s_muted{ false };   // audio switched off while running: silence to the device

// Opens the default playback device for 6-channel float frames (SDL's 5.1 order matches XAudio's: FL, FR,
// FC, LFE, BL, BR). AC6_AUDIO=0 starts without output, and AudioClock opens the device if audio is switched on later.
void OpenOutput(bool start = true)
{
    if (const char* v = std::getenv("AC6_AUDIO_DUMP"); start && v && *v)
    {
        s_dump.Open(v);
        log::Info("Audio: writing the mix to {}", v);
    }
    if (const char* v = std::getenv("AC6_AUDIO"); start && v && *v == '0')
    {
        log::Info("Audio output disabled (AC6_AUDIO=0)");
        return;
    }
    if (!SDL_InitSubSystem(SDL_INIT_AUDIO))
    {
        log::Error("SDL audio init failed: {}", SDL_GetError());
        return;
    }
    const SDL_AudioSpec spec{ SDL_AUDIO_F32, 6, int(kSampleRate) };
    s_stream = SDL_OpenAudioDeviceStream(SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK, &spec, nullptr, nullptr);
    if (!s_stream)
    {
        log::Error("SDL audio device failed: {}", SDL_GetError());
        return;
    }
    SDL_ResumeAudioStreamDevice(s_stream);
    // Keep the device's own buffer plus two frames queued.
    SDL_AudioSpec deviceSpec{};
    int deviceFrames = 0;
    if (SDL_GetAudioDeviceFormat(SDL_GetAudioStreamDevice(s_stream), &deviceSpec, &deviceFrames) && deviceFrames > 0)
        s_targetFrames = uint32_t((deviceFrames + kFrameSamples - 1) / kFrameSamples) + 2;
    log::Info("Audio output: {} (6 channels, {} Hz; device buffer {} samples, {} frames queued)", SDL_GetCurrentAudioDriver(),
        kSampleRate, deviceFrames, s_targetFrames);
}

// Runs the game's mixer, one frame per call of its callback. With a device, its consumption sets the pace:
// a frame is mixed whenever fewer than s_targetFrames are queued, so the game's audio clock follows the
// device's and the queue neither grows nor runs dry. Without one (AC6_AUDIO=0, or a device that stops
// consuming for a quarter second), a 48 kHz timer stands in.
void AudioClock(void*)
{
    using Clock = std::chrono::steady_clock;
    constexpr auto kPeriod = std::chrono::nanoseconds(1'000'000'000ull * kFrameSamples / kSampleRate);
    constexpr auto kStall = std::chrono::milliseconds(250);
    auto next = Clock::now();
    auto lastConsumed = Clock::now();
    int lastQueued = 0;
    uint64_t seen = UINT64_MAX;
    for (;;)
    {
        // Audio switched on or off by the settings overlay's Apply: off mutes the output. On, when the game started
        // without output, opens the device, whose consumption then sets the pace as at start.
        if (const uint64_t generation = settings::Generation(); generation != seen)
        {
            seen = generation;
            const bool off = settings::Get("AC6_AUDIO") == "0";
            if (off != s_muted.exchange(off))
                log::Info("Audio {}", off ? "off (muted)" : "on");
            if (!off && !s_stream)
                OpenOutput(false);
        }
        const uint32_t callback = s_callback.load(std::memory_order_acquire);
        const auto now = Clock::now();
        if (s_stream)
        {
            const int queued = SDL_GetAudioStreamQueued(s_stream);
            if (queued < lastQueued || queued == 0)
                lastConsumed = now;
            lastQueued = queued;
            if (now - lastConsumed < kStall)
            {
                if (queued < int(s_targetFrames) * kFrameBytes && callback)
                {
                    GuestCall(callback, s_callbackParam);
                    lastQueued = SDL_GetAudioStreamQueued(s_stream);
                }
                else
                    kernel::DelayExecution({ false, -10'000 });   // 1 ms
                next = Clock::now();
                continue;
            }
        }
        next += kPeriod;
        auto remaining = std::chrono::duration_cast<std::chrono::nanoseconds>(next - Clock::now()).count();
        if (remaining > 0)
            kernel::DelayExecution({ false, -(remaining / 100) });
        else if (remaining < -50'000'000)
            next = Clock::now();   // more than 50 ms behind (debugger, stall): resynchronise
        if (callback)
            GuestCall(callback, s_callbackParam);
    }
}

} // namespace

uint64_t RenderedSamples() { return s_frames.load(std::memory_order_relaxed) * kFrameSamples; }

} // namespace ac6::apu

namespace ac6::kernel {

uint32_t XAudioRegisterRenderDriverClient(be<uint32_t>* callback, be<uint32_t>* driver)
{
    using namespace apu;
    auto* param = static_cast<be<uint32_t>*>(g_heap.Alloc(sizeof(be<uint32_t>)));
    *param = callback[1];
    s_callbackParam = g_memory.MapVirtual(param);
    s_callback.store(callback[0], std::memory_order_release);
    *driver = kDriverHandle;
    log::Info("Audio render client {:08X}({:08X})",
        uint32_t(callback[0]), uint32_t(callback[1]));

    static std::atomic<bool> started{ false };
    if (!started.exchange(true))
    {
        OpenOutput();
        GuestThreadParams params;
        params.hostEntry = AudioClock;
        params.flags = 1u << 24;
        if (GuestThread* thread = GuestThread::Start(params))
        {
            thread->SetPriority(15);
            thread->Resume();
        }
    }
    return 0;
}

uint32_t XAudioUnregisterRenderDriverClient(uint32_t driver)
{
    apu::s_callback = 0;
    return 0;
}

// One frame: 256 samples of each of the 6 channels, channel after channel (planar), big-endian float.
uint32_t XAudioSubmitRenderDriverFrame(uint32_t driver, const be<float>* samples)
{
    using namespace apu;
    s_frames.fetch_add(1, std::memory_order_relaxed);
    if ((!s_stream && !s_dump.file) || !samples)
        return 0;
    float interleaved[kFrameSamples * 6];
    for (uint32_t c = 0; c < 6; c++)
        for (uint32_t i = 0; i < kFrameSamples; i++)
        {
            const float v = samples[c * kFrameSamples + i];
            interleaved[i * 6 + c] = v == v ? std::clamp(v, -1.0f, 1.0f) : 0.0f;
        }
    s_dump.Write(interleaved, kFrameBytes);
    if (!s_stream)
        return 0;
    if (s_muted.load(std::memory_order_relaxed))
        std::memset(interleaved, 0, sizeof(interleaved));
    // Latency bound for the timer fallback (device-driven pacing never queues this much): past 100 ms
    // queued, frames are dropped instead of letting the delay grow.
    if (SDL_GetAudioStreamQueued(s_stream) > int(kSampleRate / 10) * 6 * 4)
    {
        if (s_dropped++ % 256 == 0)
            log::Info("Audio: output queue full, {} frames dropped so far", s_dropped);
        return 0;
    }
    SDL_PutAudioStreamData(s_stream, interleaved, kFrameBytes);
    return 0;
}

uint32_t XAudioGetVoiceCategoryVolumeChangeMask(uint32_t driver, be<uint32_t>* mask)
{
    if (mask)
        *mask = 0;
    return 0;
}

uint32_t XAudioGetVoiceCategoryVolume(uint32_t category, be<float>* volume)
{
    if (volume)
        *volume = 1.0f;
    return 0;
}

// XMA decoder contexts: 64-byte blocks of the hardware's context array (apu/xma.cpp).
uint32_t XMACreateContext(be<uint32_t>* context)
{
    const uint32_t p = apu::XmaAllocateContext();
    if (!p)
        return X_STATUS_NO_MEMORY;
    *context = p;
    return X_STATUS_SUCCESS;
}

uint32_t XMAReleaseContext(uint32_t context)
{
    apu::XmaReleaseContext(context);
    return X_STATUS_SUCCESS;
}

} // namespace ac6::kernel

AC6_IMPORT(XMACreateContext)
AC6_IMPORT(XMAReleaseContext)
AC6_IMPORT(XAudioRegisterRenderDriverClient)
AC6_IMPORT(XAudioUnregisterRenderDriverClient)
AC6_IMPORT(XAudioSubmitRenderDriverFrame)
AC6_IMPORT(XAudioGetVoiceCategoryVolumeChangeMask)
AC6_IMPORT(XAudioGetVoiceCategoryVolume)
