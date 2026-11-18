#pragma once
// Guest address space: one 4 GB reservation. Pages inside declared "backed" ranges (heaps, image,
// function table, GPU register space) are committed lazily on first touch (Windows) or made accessible
// when the range is declared (Linux). Everything else, including the low guard range, faults and is
// reported by the crash handler.

#include "ppc.h"

#include <cstdint>
#include <functional>

namespace ac6 {

constexpr uint64_t kGuestSize = 0x100000000ull;

// Xbox 360 guest layout. User heap = runtime-owned objects (thread stacks, pools). The game's own
// virtual allocations get [kVirtualBegin, kVirtualEnd).
constexpr uint32_t kGuardEnd        = 0x00020000; // [0, 0x20000) never backed: null-pointer guard
constexpr uint32_t kUserHeapBegin   = 0x00020000;
constexpr uint32_t kUserHeapEnd     = 0x40000000;
constexpr uint32_t kVirtualBegin    = 0x40000000; // NtAllocateVirtualMemory range (kernel/vm.h), page-managed
constexpr uint32_t kVirtualEnd      = 0x7FC00000;
constexpr uint32_t kGpuRegsBegin    = 0x7FC00000; // GPU registers (read by the D3D library via PPC_MM_*)
constexpr uint32_t kGpuRegsEnd      = 0x80000000;
// Physical memory: 512 MB at 0xA0000000, so a GPU (physical) address maps back uniquely:
// guest = kPhysHeapBegin + (physical & 0x1FFFFFFF). As on the console, the same memory is also visible
// at 0xC0000000 (the D3D library reads shader microcode through it). Both are views of one host
// section. The console's third view at 0xE0000000 isn't mapped (its page offset is unverified).
constexpr uint32_t kPhysHeapBegin   = 0xA0000000;
constexpr uint64_t kPhysHeapEnd     = 0xC0000000ull;
constexpr uint32_t kPhysSize        = 0x20000000;
constexpr uint32_t kPhysAliasBegin  = 0xC0000000;
constexpr uint64_t kPhysAliasEnd    = 0xE0000000ull;
// Within it: runtime kernel objects (o1heap) first, then the game's physical allocations
// (MmAllocatePhysicalMemoryEx, page-granular: o1heap's power-of-two rounding would waste half of a
// 320 MB request).
constexpr uint32_t kPhysObjectsEnd  = 0xA2000000;
constexpr uint32_t PhysicalToGuest(uint32_t physical) { return kPhysHeapBegin + (physical & 0x1FFFFFFF); }

struct GuestMemory
{
    uint8_t* base = nullptr;

    bool Init();

    void* Translate(uint32_t guest) const { return base + guest; }
    template<typename T> T* As(uint32_t guest) const { return reinterpret_cast<T*>(base + guest); }
    uint32_t MapVirtual(const void* host) const
    {
        return host ? static_cast<uint32_t>(static_cast<const uint8_t*>(host) - base) : 0;
    }
    bool IsGuest(const void* host) const
    {
        auto p = static_cast<const uint8_t*>(host);
        return p >= base && p < base + kGuestSize;
    }

    // Declare [begin, end) as backed: first access commits pages there.
    void AddBackedRange(uint32_t begin, uint64_t end);
    bool IsBacked(uint32_t guest) const;
    // Zero a range. Large ranges inside backed memory are returned to the host instead of written.
    void Zero(uint32_t guest, uint64_t size);
    // Commit eagerly (e.g. the image before copying it in).
    bool Commit(uint32_t guest, uint64_t size);
    // Return pages to the host: they fault until committed again, then read as zeros.
    void Decommit(uint32_t guest, uint64_t size);
    // The committed ranges of [0, end) in address order (Linux: the accessible 64 KB chunks with memory behind them).
    void ForEachCommitted(uint64_t end, const std::function<void(uint64_t begin, uint64_t end)>& visit) const;

    PPCFunc*& Function(uint32_t guest) { return PPC_LOOKUP_FUNC(base, guest); }

    // Write tracking for physical memory, so the renderer can tell whether guest data
    // it has a copy of changed without reading it again. Watch() makes a range's pages read-only in both
    // views and returns a generation. The first write to a page afterwards faults, records a newer
    // generation for the page and makes it writable again. Unchanged() is true while every page of the
    // range is still watched and unwritten since that generation. Host system calls that write guest
    // memory (ReadFile) can't fault on the way: they call PrepareHostWrite first.
    uint64_t Watch(uint32_t guest, uint32_t size);
    bool Unchanged(uint32_t guest, uint32_t size, uint64_t since) const;
    void PrepareHostWrite(uint32_t guest, uint32_t size);
    // Host system calls that read or write guest memory (ReadFile, WriteFile) do it from the kernel, where a backed
    // page not committed yet (Windows commits them on first touch) fails the call instead of faulting into the lazy
    // commit: PrepareHostIo commits the range's backed pages first. (Linux maps backed memory up front: nothing to do.)
    void PrepareHostIo(uint32_t guest, uint32_t size);
    uint64_t WatchFaults() const;   // writes caught so far (diagnostics)
    // The pipeline ring (gpu/capture.cpp): `hook` gets each 4 KB physical page (its guest address in the physical heap)
    // as it starts being watched (`written` false: what the renderer is about to read) and as a watched page is first
    // written (true: what it held until then). Called with the watch lock held.
    using PageCopyHook = void (*)(uint32_t guest, const uint8_t* page, bool written);
    void SetPageCopyHook(PageCopyHook hook);

    // Pages the GPU reads in a frame: MarkGpuRead watches them and records the frame (the
    // renderer's GPU frame counter, gpu/video.h). A write to one before that frame is done counts as early.
    // SetGpuFrameQuery gives the counter.
    void SetGpuFrameQuery(uint64_t (*completed)());
    void MarkGpuRead(uint32_t guest, uint32_t size, uint64_t frame);
    struct GpuWriteStats { uint64_t early = 0, late = 0; };
    GpuWriteStats TakeGpuWriteStats();
};

extern GuestMemory g_memory;

} // namespace ac6
