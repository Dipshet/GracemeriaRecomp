#include "memory.h"

#include "log.h"

#include <algorithm>
#include <atomic>
#include <memory>
#include <cstring>
#include <mutex>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#else
#include "platform/platform.h"

#include <cerrno>
#include <cstdio>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace ac6 {

GuestMemory g_memory;

namespace {

struct Range { uint32_t begin; uint64_t end; };
std::vector<Range> s_backed;    // appended during init and by replays (gpu/capture.cpp), read by the fault handler
#if defined(_WIN32)
std::mutex s_commitMutex;                    // the lazy-commit handler's (Windows commits guest pages on demand)
std::atomic<uint64_t> s_lazyCommits{ 0 };
#endif

constexpr uintptr_t kCommitChunk = 0x10000;
bool s_physicalIsSection = false;   // physical memory is a section mapped at 0xA0000000 and 0xC0000000

bool InPhysicalView(uint64_t guest)
{
    return (guest >= kPhysHeapBegin && guest < kPhysHeapEnd) || (guest >= kPhysAliasBegin && guest < kPhysAliasEnd);
}

#if defined(_WIN32)
// Placeholder APIs (Windows 10 1803+), looked up at run time: the MinGW import libraries don't have them.
using VirtualAlloc2Fn = PVOID(WINAPI*)(HANDLE, PVOID, SIZE_T, ULONG, ULONG, void*, ULONG);
using MapViewOfFile3Fn = PVOID(WINAPI*)(HANDLE, HANDLE, PVOID, ULONG64, SIZE_T, ULONG, ULONG, void*, ULONG);

// Reserves the 4 GB guest space as placeholders, then turns [0, 0xA0000000) into an ordinary
// reservation and maps one 512 MB SEC_RESERVE section at 0xA0000000 and at 0xC0000000, so physical
// memory has the console's aliasing. Pages commit lazily through either view (commit is per section
// page, so both views see it). [0xE0000000, 4 GB) stays an inaccessible placeholder.
uint8_t* ReserveWithPhysicalAliases(uintptr_t preferred)
{
    HMODULE kernelBase = GetModuleHandleW(L"kernelbase.dll");
    auto virtualAlloc2 = kernelBase ? reinterpret_cast<VirtualAlloc2Fn>(GetProcAddress(kernelBase, "VirtualAlloc2")) : nullptr;
    auto mapViewOfFile3 = kernelBase ? reinterpret_cast<MapViewOfFile3Fn>(GetProcAddress(kernelBase, "MapViewOfFile3")) : nullptr;
    if (!virtualAlloc2 || !mapViewOfFile3)
        return nullptr;

    auto* p = static_cast<uint8_t*>(virtualAlloc2(nullptr, reinterpret_cast<void*>(preferred), kGuestSize,
        MEM_RESERVE | MEM_RESERVE_PLACEHOLDER, PAGE_NOACCESS, nullptr, 0));
    if (!p && preferred)
        p = static_cast<uint8_t*>(virtualAlloc2(nullptr, nullptr, kGuestSize, MEM_RESERVE | MEM_RESERVE_PLACEHOLDER, PAGE_NOACCESS, nullptr, 0));
    if (!p)
        return nullptr;

    // Split into [0, A) [A, C) [C, E) [E, 4G).
    bool ok = VirtualFree(p + kPhysHeapBegin, kGuestSize - kPhysHeapBegin, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER) &&
              VirtualFree(p + kPhysAliasBegin, kGuestSize - kPhysAliasBegin, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER) &&
              VirtualFree(p + kPhysAliasEnd, kGuestSize - kPhysAliasEnd, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER);
    ok = ok && virtualAlloc2(nullptr, p, kPhysHeapBegin, MEM_RESERVE | MEM_REPLACE_PLACEHOLDER, PAGE_READWRITE, nullptr, 0) == p;

    HANDLE section = ok ? CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE | SEC_RESERVE, 0, kPhysSize, nullptr) : nullptr;
    ok = section &&
         mapViewOfFile3(section, GetCurrentProcess(), p + kPhysHeapBegin, 0, kPhysSize, MEM_REPLACE_PLACEHOLDER, PAGE_READWRITE, nullptr, 0) &&
         mapViewOfFile3(section, GetCurrentProcess(), p + kPhysAliasBegin, 0, kPhysSize, MEM_REPLACE_PLACEHOLDER, PAGE_READWRITE, nullptr, 0);
    if (section)
        CloseHandle(section);   // the views keep the section alive
    if (!ok)
    {
        log::Warn("Physical memory aliasing unavailable (error {}); falling back to a plain reservation", GetLastError());
        UnmapViewOfFile(p + kPhysHeapBegin);
        UnmapViewOfFile(p + kPhysAliasBegin);
        for (uint64_t at : { uint64_t(0), uint64_t(kPhysHeapBegin), kPhysAliasBegin + uint64_t(0), kPhysAliasEnd })
            VirtualFree(p + at, 0, MEM_RELEASE);
        return nullptr;
    }
    return p;
}
#else
int s_physicalFd = -1;   // the memfd holding physical memory (both views map it)

// Linux: the 4 GB reservation is PROT_NONE. Backed ranges are made accessible as they are declared (the kernel then
// gives zero pages on first touch, so nothing commits lazily here), and physical memory is a 512 MB memfd mapped
// MAP_SHARED at 0xA0000000 and at 0xC0000000 for the console's aliasing. [0xE0000000, 4 GB) stays inaccessible.
uint8_t* ReserveWithPhysicalAliases(uintptr_t preferred)
{
    void* p = mmap(reinterpret_cast<void*>(preferred), kGuestSize, PROT_NONE,
        MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE | MAP_FIXED_NOREPLACE, -1, 0);
    if (p == MAP_FAILED)
        p = mmap(nullptr, kGuestSize, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (p == MAP_FAILED)
        return nullptr;
    auto* base = static_cast<uint8_t*>(p);
    const int fd = memfd_create("ac6-physical", MFD_CLOEXEC);
    bool ok = fd >= 0 && ftruncate(fd, kPhysSize) == 0;
    for (uint32_t view : { kPhysHeapBegin, kPhysAliasBegin })
        ok = ok && mmap(base + view, kPhysSize, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, fd, 0) != MAP_FAILED;
    if (!ok)
    {
        log::Warn("Physical memory aliasing unavailable (error {}); falling back to a plain reservation", errno);
        if (fd >= 0)
            close(fd);
        for (uint32_t view : { kPhysHeapBegin, kPhysAliasBegin })
            mmap(base + view, kPhysSize, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE | MAP_FIXED, -1, 0);
        return base;
    }
    s_physicalFd = fd;
    return base;
}
#endif

// --- Physical memory write tracking (GuestMemory::Watch) -------------------------------------------
constexpr uint32_t kWatchPageBits = 12;
constexpr uint32_t kWatchPages = kPhysSize >> kWatchPageBits;
std::mutex s_watchMutex;                                   // protection changes
std::atomic<uint64_t> s_watchGeneration{ 1 };
std::atomic<uint64_t> s_watchFaults{ 0 };
std::unique_ptr<std::atomic<uint64_t>[]> s_pageGeneration;   // generation of the last detected write
std::unique_ptr<std::atomic<uint8_t>[]> s_pageWatched;       // read-only in both views
std::unique_ptr<std::atomic<uint64_t>[]> s_pageGpuFrame;     // GPU frame reading the page (MarkGpuRead), 0: none
uint64_t (*s_gpuCompleted)() = nullptr;
std::atomic<GuestMemory::PageCopyHook> s_pageCopyHook{ nullptr };

// The pipeline ring's copy of a page (GuestMemory::SetPageCopyHook).
void CopyPage(uint32_t page, bool written)
{
    if (const auto hook = s_pageCopyHook.load(std::memory_order_acquire))
    {
        const uint32_t guest = kPhysHeapBegin + (page << kWatchPageBits);
        hook(guest, g_memory.As<uint8_t>(guest), written);
    }
}
std::atomic<uint64_t> s_gpuEarlyWrites{ 0 }, s_gpuLateWrites{ 0 };

// A write opening watched pages: those the GPU reads in a frame it hasn't finished count as early.
void NoteGpuWrite(uint32_t first, uint32_t count)
{
    if (!s_pageGpuFrame)
        return;
    uint64_t frame = 0;
    for (uint32_t p = first; p < first + count; p++)
        frame = std::max(frame, s_pageGpuFrame[p].exchange(0, std::memory_order_acq_rel));
    if (!frame)
        return;
    if (s_gpuCompleted && s_gpuCompleted() < frame)
        s_gpuEarlyWrites.fetch_add(1, std::memory_order_relaxed);
    else
        s_gpuLateWrites.fetch_add(1, std::memory_order_relaxed);
}

// Physical pages in both views: read-only (watched) or writable.
bool ProtectPages(uint32_t first, uint32_t count, bool writable)
{
    bool ok = true;
    for (uint32_t view : { kPhysHeapBegin, kPhysAliasBegin })
    {
        uint8_t* p = g_memory.base + view + (uint64_t(first) << kWatchPageBits);
        const uint64_t size = uint64_t(count) << kWatchPageBits;
#if defined(_WIN32)
        DWORD old = 0;
        ok &= VirtualProtect(p, size, writable ? PAGE_READWRITE : PAGE_READONLY, &old) != 0;
#else
        ok &= mprotect(p, size, writable ? PROT_READ | PROT_WRITE : PROT_READ) == 0;
#endif
    }
    return ok;
}

// A write to a watched page: record it and let the write through. Returns false for any other fault.
bool HandleWatchedWrite(bool write, uint32_t guest)
{
    if (!s_pageWatched || !write || !InPhysicalView(guest))
        return false;
    const uint32_t page = (guest & (kPhysSize - 1)) >> kWatchPageBits;
    std::lock_guard lock(s_watchMutex);
    if (s_pageWatched[page].load(std::memory_order_relaxed))
    {
        // The whole 64 KB block is opened: loads write sequentially, and a fault per 4 KB page would cost a
        // mission load about 2500 faults. Its other watched pages count as written (their textures are checked again).
        // (A block not wholly committed can't change protection at once: then just the page.)
        uint32_t first = page & ~15u, count = 16;
        if (!ProtectPages(first, count, true))
        {
            first = page;
            count = 1;
            ProtectPages(first, count, true);
        }
        NoteGpuWrite(first, count);
        if (s_pageCopyHook.load(std::memory_order_relaxed))
            for (uint32_t p = first; p < first + count; p++)
                if (s_pageWatched[p].load(std::memory_order_relaxed))
                    CopyPage(p, true);   // unchanged since watched, readable in either protection
        const uint64_t generation = s_watchGeneration.fetch_add(1) + 1;
        for (uint32_t p = first; p < first + count; p++)
            if (s_pageWatched[p].load(std::memory_order_relaxed))
            {
                s_pageGeneration[p].store(generation, std::memory_order_release);
                s_pageWatched[p].store(0, std::memory_order_release);
            }
        s_watchFaults.fetch_add(1, std::memory_order_relaxed);
        return true;
    }
    // Unwatched meanwhile by another thread: retry the write if the page is writable now.
#if defined(_WIN32)
    MEMORY_BASIC_INFORMATION mbi{};
    return VirtualQuery(g_memory.base + guest, &mbi, sizeof(mbi)) && mbi.State == MEM_COMMIT && mbi.Protect == PAGE_READWRITE;
#else
    return true;   // (physical pages are always mapped: an unwatched one is writable)
#endif
}

// Committing pages (VirtualAlloc MEM_COMMIT) sets their protection even where they were committed already, so
// a commit over watched pages would silently make them writable again: their writes would go unseen. Every
// commit of guest memory goes through here, which puts the watched pages back to read-only.
bool CommitGuest(uint32_t guest, uint64_t size)
{
    std::lock_guard lock(s_watchMutex);
#if defined(_WIN32)
    if (!VirtualAlloc(g_memory.base + guest, size, MEM_COMMIT, PAGE_READWRITE))
        return false;
#else
    // Physical memory is always mapped (and its watched pages keep their protection). Elsewhere the pages become
    // accessible.
    if (size && s_physicalFd >= 0 && InPhysicalView(guest) && InPhysicalView(guest + size - 1))
        return true;
    const uintptr_t begin = uintptr_t(g_memory.base + guest) & ~uintptr_t(0xFFF);
    const uintptr_t end = (uintptr_t(g_memory.base + guest) + size + 0xFFF) & ~uintptr_t(0xFFF);
    if (mprotect(reinterpret_cast<void*>(begin), end - begin, PROT_READ | PROT_WRITE) != 0)
        return false;
#endif
    if (!s_pageWatched || !size || !InPhysicalView(guest) || !InPhysicalView(guest + size - 1))
        return true;
    const uint32_t first = (guest & (kPhysSize - 1)) >> kWatchPageBits;
    const uint32_t last = ((guest + size - 1) & (kPhysSize - 1)) >> kWatchPageBits;
    // Writable for a moment, so they count as written.
    for (uint32_t page = first; page <= last; page++)
        if (s_pageWatched[page].load(std::memory_order_relaxed))
        {
            s_pageGeneration[page].store(s_watchGeneration.fetch_add(1) + 1, std::memory_order_release);
            if (!ProtectPages(page, 1, false))
            {
                s_pageWatched[page].store(0, std::memory_order_release);
                ProtectPages(page, 1, true);
            }
        }
    return true;
}

#if defined(_WIN32)
LONG CALLBACK LazyCommitHandler(EXCEPTION_POINTERS* info)
{
    if (info->ExceptionRecord->ExceptionCode != EXCEPTION_ACCESS_VIOLATION)
        return EXCEPTION_CONTINUE_SEARCH;

    auto addr = reinterpret_cast<uint8_t*>(info->ExceptionRecord->ExceptionInformation[1]);
    if (!g_memory.IsGuest(addr))
        return EXCEPTION_CONTINUE_SEARCH;

    uint32_t guest = g_memory.MapVirtual(addr);
    if (HandleWatchedWrite(info->ExceptionRecord->ExceptionInformation[0] == 1, guest))
        return EXCEPTION_CONTINUE_EXECUTION;
    if (!g_memory.IsBacked(guest))
        return EXCEPTION_CONTINUE_SEARCH; // a real fault: let the crash handler report it

    std::lock_guard lock(s_commitMutex);
    auto chunk = reinterpret_cast<uintptr_t>(addr) & ~(kCommitChunk - 1);
    MEMORY_BASIC_INFORMATION mbi{};
    if (VirtualQuery(addr, &mbi, sizeof(mbi)) && mbi.State == MEM_COMMIT)
        return EXCEPTION_CONTINUE_EXECUTION; // another thread committed it meanwhile
    // Committing already-committed pages is harmless (contents are kept), so commit the whole chunk.
    if (!CommitGuest(uint32_t(chunk - reinterpret_cast<uintptr_t>(g_memory.base)), kCommitChunk))
        return EXCEPTION_CONTINUE_SEARCH;
    s_lazyCommits++;
    return EXCEPTION_CONTINUE_EXECUTION;
}
#else
// Linux: only watched writes are resolved here (backed memory is accessible from the start).
bool GuestFaultHandler(const platform::Fault& fault)
{
    return g_memory.IsGuest(fault.address) && HandleWatchedWrite(fault.write, g_memory.MapVirtual(fault.address));
}
#endif

} // namespace

bool GuestMemory::Init()
{
    // Prefer 0x100000000 like UnleashedRecomp (keeps host pointers recognisable). Any address works.
    base = ReserveWithPhysicalAliases(0x100000000ull);
#if defined(_WIN32)
    s_physicalIsSection = base != nullptr;
#else
    s_physicalIsSection = s_physicalFd >= 0;
#endif
    if (s_physicalIsSection)
        AddBackedRange(kPhysAliasBegin, kPhysAliasEnd);
#if defined(_WIN32)
    if (!base)
        base = static_cast<uint8_t*>(VirtualAlloc(reinterpret_cast<void*>(0x100000000ull), kGuestSize, MEM_RESERVE, PAGE_READWRITE));
    if (!base)
        base = static_cast<uint8_t*>(VirtualAlloc(nullptr, kGuestSize, MEM_RESERVE, PAGE_READWRITE));
#endif
    if (!base)
    {
        log::Error("Failed to reserve 4 GB of guest address space");
        return false;
    }

    if (s_physicalIsSection)
    {
        s_pageGeneration = std::make_unique<std::atomic<uint64_t>[]>(kWatchPages);
        s_pageWatched = std::make_unique<std::atomic<uint8_t>[]>(kWatchPages);
    }
#if defined(_WIN32)
    AddVectoredExceptionHandler(1, LazyCommitHandler);
#else
    platform::AddFaultHandler(0, GuestFaultHandler);
#endif
    log::Info("Guest memory reserved at {} (commit on demand{})", static_cast<void*>(base),
        s_physicalIsSection ? ", physical memory aliased at C0000000" : "");
    return true;
}

void GuestMemory::AddBackedRange(uint32_t begin, uint64_t end)
{
    s_backed.push_back({ begin, end });
#if !defined(_WIN32)
    // Accessible now (zero pages on first touch), apart from the physical views, which are always mapped.
    auto open = [this](uint64_t from, uint64_t to) {
        if (to > from)
            mprotect(base + (from & ~uint64_t(0xFFF)), ((to + 0xFFF) & ~uint64_t(0xFFF)) - (from & ~uint64_t(0xFFF)), PROT_READ | PROT_WRITE);
    };
    if (s_physicalIsSection)
    {
        open(begin, std::min<uint64_t>(end, kPhysHeapBegin));
        open(std::max<uint64_t>(begin, kPhysAliasEnd), end);
    }
    else
        open(begin, end);
#endif
}

bool GuestMemory::IsBacked(uint32_t guest) const
{
    for (auto& r : s_backed)
        if (guest >= r.begin && guest < r.end)
            return true;
    return false;
}

void GuestMemory::Zero(uint32_t guest, uint64_t size)
{
    // Whole 64 KB chunks inside a backed range are decommitted instead of cleared: they come back as
    // zero pages on first touch, so a large zeroed allocation costs no host memory until it's used.
    uint64_t begin = guest, end = uint64_t(guest) + size;
    uint64_t chunkBegin = (begin + kCommitChunk - 1) & ~uint64_t(kCommitChunk - 1);
    uint64_t chunkEnd = end & ~uint64_t(kCommitChunk - 1);
    if (chunkEnd > chunkBegin && s_physicalIsSection && InPhysicalView(begin) && InPhysicalView(end - 1))
    {
#if !defined(_WIN32)
        // Whole chunks are punched out of the memfd (zero in both views, memory returned). That writes without
        // faulting, so watched pages are let go first (counted as written).
        PrepareHostWrite(guest, uint32_t(size));
        std::memset(base + begin, 0, chunkBegin - begin);
        std::memset(base + chunkEnd, 0, end - chunkEnd);
        if (fallocate(s_physicalFd, FALLOC_FL_PUNCH_HOLE | FALLOC_FL_KEEP_SIZE, off_t(chunkBegin & (kPhysSize - 1)), off_t(chunkEnd - chunkBegin)) != 0)
            std::memset(base + chunkBegin, 0, chunkEnd - chunkBegin);
        return;
#else
        // Section pages can't be decommitted through a view. Chunks never committed are still zero,
        // so only committed ones need clearing.
        std::memset(base + begin, 0, chunkBegin - begin);
        std::memset(base + chunkEnd, 0, end - chunkEnd);
        for (uint64_t c = chunkBegin; c < chunkEnd;)
        {
            MEMORY_BASIC_INFORMATION mbi{};
            if (!VirtualQuery(base + c, &mbi, sizeof(mbi)))
                break;
            uint64_t regionEnd = std::min<uint64_t>(chunkEnd, reinterpret_cast<uint8_t*>(mbi.BaseAddress) - base + mbi.RegionSize);
            if (mbi.State == MEM_COMMIT)
                std::memset(base + c, 0, regionEnd - c);
            c = regionEnd;
        }
        return;
#endif
    }
    if (chunkEnd > chunkBegin && IsBacked(guest) && IsBacked(static_cast<uint32_t>(end - 1)))
    {
        std::memset(base + begin, 0, chunkBegin - begin);
        std::memset(base + chunkEnd, 0, end - chunkEnd);
        // Not under s_commitMutex: the memsets above can fault into the lazy-commit handler, and a
        // concurrent commit of these chunks by another thread is harmless.
#if defined(_WIN32)
        VirtualFree(base + chunkBegin, chunkEnd - chunkBegin, MEM_DECOMMIT);
#else
        madvise(base + chunkBegin, chunkEnd - chunkBegin, MADV_DONTNEED);   // zero pages again on the next touch
#endif
        return;
    }
    std::memset(base + begin, 0, size);
}

bool GuestMemory::Commit(uint32_t guest, uint64_t size)
{
    return CommitGuest(guest, size);
}

void GuestMemory::ForEachCommitted(uint64_t end, const std::function<void(uint64_t, uint64_t)>& visit) const
{
#if defined(_WIN32)
    for (uint64_t at = 0; at < end;)
    {
        MEMORY_BASIC_INFORMATION info{};
        if (!VirtualQuery(base + at, &info, sizeof(info)))
            break;
        const uint64_t begin = reinterpret_cast<uint8_t*>(info.BaseAddress) - base;
        const uint64_t regionEnd = std::min<uint64_t>(begin + info.RegionSize, end);
        if (info.State == MEM_COMMIT && !(info.Protect & (PAGE_NOACCESS | PAGE_GUARD)))
            visit(begin, regionEnd);
        at = std::max(regionEnd, at + 0x1000);
    }
#else
    // Accessible mappings from /proc/self/maps, then the 64 KB chunks with a resident page (mincore): untouched memory
    // isn't listed, as uncommitted memory isn't on Windows.
    FILE* maps = std::fopen("/proc/self/maps", "r");
    if (!maps)
        return;
    const uintptr_t lo = uintptr_t(base), hi = uintptr_t(base) + end;
    char line[512];
    while (std::fgets(line, sizeof(line), maps))
    {
        unsigned long from = 0, to = 0;
        char perms[8] = {};
        if (std::sscanf(line, "%lx-%lx %7s", &from, &to, perms) != 3 || perms[0] != 'r')
            continue;
        const uintptr_t b = std::max<uintptr_t>(from, lo), e = std::min<uintptr_t>(to, hi);
        uint64_t runBegin = 0, runEnd = 0;
        for (uintptr_t chunk = b & ~uintptr_t(0xFFFF); chunk < e; chunk += 0x10000)
        {
            const uintptr_t cb = std::max(chunk, b), ce = std::min(chunk + 0x10000, e);
            unsigned char resident[16] = {};
            bool any = false;
            if (mincore(reinterpret_cast<void*>(cb), ce - cb, resident) == 0)
                for (size_t i = 0; i < (ce - cb + 0xFFF) / 0x1000 && !any; i++)
                    any = resident[i] & 1;
            if (!any)
                continue;
            if (runEnd == cb - lo)
                runEnd = ce - lo;
            else
            {
                if (runEnd > runBegin)
                    visit(runBegin, runEnd);
                runBegin = cb - lo;
                runEnd = ce - lo;
            }
        }
        if (runEnd > runBegin)
            visit(runBegin, runEnd);
    }
    std::fclose(maps);
#endif
}

void GuestMemory::Decommit(uint32_t guest, uint64_t size)
{
#if defined(_WIN32)
    VirtualFree(base + guest, size, MEM_DECOMMIT);
#else
    madvise(base + guest, size, MADV_DONTNEED);
    mprotect(base + guest, size, PROT_NONE);
#endif
}

uint64_t GuestMemory::Watch(uint32_t guest, uint32_t size)
{
    if (!s_pageWatched || !size || !InPhysicalView(guest) || !InPhysicalView(uint64_t(guest) + size - 1))
        return s_watchGeneration.fetch_add(1) + 1;
    const uint32_t first = (guest & (kPhysSize - 1)) >> kWatchPageBits;
    const uint32_t last = ((guest + size - 1) & (kPhysSize - 1)) >> kWatchPageBits;
    std::lock_guard lock(s_watchMutex);
    const uint64_t generation = s_watchGeneration.fetch_add(1) + 1;
    for (uint32_t page = first; page <= last;)
    {
        if (s_pageWatched[page].load(std::memory_order_relaxed))
        {
            page++;
            continue;
        }
        uint32_t end = page;
        while (end <= last && !s_pageWatched[end].load(std::memory_order_relaxed))
            end++;
        // Only committed pages can be protected (the caller has just read them, so they are).
        const bool ok = ProtectPages(page, end - page, false);
        // A page starting to be watched may have been written while it wasn't, unseen: it counts as written
        // now. (Keeping the generation of its last caught write would let a check against an earlier generation
        // take it as unchanged, missing the writes made while it was unwatched.)
        if (ok)
            for (uint32_t p = page; p < end; p++)
            {
                s_pageGeneration[p].store(generation, std::memory_order_release);
                s_pageWatched[p].store(1, std::memory_order_release);
                CopyPage(p, false);
            }
        else
            ProtectPages(page, end - page, true);
        page = end;
    }
    return generation;
}

uint64_t GuestMemory::WatchFaults() const { return s_watchFaults.load(std::memory_order_relaxed); }

void GuestMemory::SetGpuFrameQuery(uint64_t (*completed)())
{
    s_gpuCompleted = completed;
}

void GuestMemory::MarkGpuRead(uint32_t guest, uint32_t size, uint64_t frame)
{
    if (!s_pageWatched || !size || !InPhysicalView(guest) || !InPhysicalView(uint64_t(guest) + size - 1))
        return;
    if (!s_pageGpuFrame)
    {
        std::lock_guard lock(s_watchMutex);
        if (!s_pageGpuFrame)
            s_pageGpuFrame = std::make_unique<std::atomic<uint64_t>[]>(kWatchPages);
    }
    const uint32_t first = (guest & (kPhysSize - 1)) >> kWatchPageBits;
    const uint32_t last = ((guest + size - 1) & (kPhysSize - 1)) >> kWatchPageBits;
    for (uint32_t page = first; page <= last; page++)
        s_pageGpuFrame[page].store(frame, std::memory_order_release);
    Watch(guest, size);
}

GuestMemory::GpuWriteStats GuestMemory::TakeGpuWriteStats()
{
    return { s_gpuEarlyWrites.exchange(0, std::memory_order_relaxed), s_gpuLateWrites.exchange(0, std::memory_order_relaxed) };
}

bool GuestMemory::Unchanged(uint32_t guest, uint32_t size, uint64_t since) const
{
    if (!s_pageWatched || !size || !InPhysicalView(guest) || !InPhysicalView(uint64_t(guest) + size - 1))
        return false;
    const uint32_t first = (guest & (kPhysSize - 1)) >> kWatchPageBits;
    const uint32_t last = ((guest + size - 1) & (kPhysSize - 1)) >> kWatchPageBits;
    for (uint32_t page = first; page <= last; page++)
        if (!s_pageWatched[page].load(std::memory_order_acquire) || s_pageGeneration[page].load(std::memory_order_acquire) > since)
            return false;
    return true;
}

void GuestMemory::PrepareHostWrite(uint32_t guest, uint32_t size)
{
    if (!s_pageWatched || !size || !InPhysicalView(guest) || !InPhysicalView(uint64_t(guest) + size - 1))
        return;
    const uint32_t first = (guest & (kPhysSize - 1)) >> kWatchPageBits;
    const uint32_t last = ((guest + size - 1) & (kPhysSize - 1)) >> kWatchPageBits;
    std::lock_guard lock(s_watchMutex);
    for (uint32_t page = first; page <= last; page++)
        if (s_pageWatched[page].load(std::memory_order_relaxed))
        {
            CopyPage(page, true);
            s_pageGeneration[page].store(s_watchGeneration.fetch_add(1) + 1, std::memory_order_release);
            s_pageWatched[page].store(0, std::memory_order_release);
            ProtectPages(page, 1, true);
        }
}

void GuestMemory::PrepareHostIo(uint32_t guest, uint32_t size)
{
#if defined(_WIN32)
    if (!size)
        return;
    const uint64_t end = uint64_t(guest) + size;
    std::lock_guard lock(s_commitMutex);
    for (uint64_t c = guest & ~uint64_t(kCommitChunk - 1); c < end; c += kCommitChunk)
    {
        if (!IsBacked(uint32_t(c)))
            continue;
        // Whole chunks, as the lazy commit does. A chunk committed already is left alone (a commit would make its
        // watched pages count as written).
        MEMORY_BASIC_INFORMATION mbi{};
        if (VirtualQuery(base + c, &mbi, sizeof(mbi)) && mbi.State == MEM_COMMIT && mbi.RegionSize >= kCommitChunk)
            continue;
        if (CommitGuest(uint32_t(c), kCommitChunk))
            s_lazyCommits++;
    }
#else
    (void)guest;
    (void)size;
#endif
}

void GuestMemory::SetPageCopyHook(PageCopyHook hook)
{
    s_pageCopyHook.store(hook, std::memory_order_release);
}

} // namespace ac6
