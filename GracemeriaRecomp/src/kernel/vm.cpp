#include "kernel/vm.h"

#include "loader.h"
#include "log.h"
#include "memory.h"

#include <algorithm>
#include <map>
#include <mutex>
#include <optional>
#include <vector>

namespace ac6::kernel {

namespace {

constexpr uint32_t kPage = 0x1000;
constexpr uint32_t kGranularity = 0x10000;
constexpr uint32_t k16MB = 0x1000000;
constexpr uint32_t kPageCount = (kVirtualEnd - kVirtualBegin) / kPage;
constexpr uint16_t kCommitted = 0x8000;

struct Reservation
{
    uint32_t size;
    uint32_t protect;
    uint32_t pageSize;   // commit granularity
};

std::mutex s_mutex;
std::map<uint32_t, Reservation> s_reservations;   // base -> reservation
std::vector<uint16_t> s_pages(kPageCount);        // kCommitted | protect
uint64_t s_reserved = 0, s_committed = 0, s_peakCommitted = 0;

constexpr uint64_t AlignUp(uint64_t v, uint64_t a) { return (v + a - 1) & ~(a - 1); }
constexpr uint32_t AlignDown(uint32_t v, uint32_t a) { return v & ~(a - 1); }
size_t PageIndex(uint32_t address) { return (address - kVirtualBegin) / kPage; }
bool InVirtualRange(uint64_t begin, uint64_t end) { return begin >= kVirtualBegin && end <= kVirtualEnd && begin < end; }

// The reservation containing address, if any.
std::map<uint32_t, Reservation>::iterator FindReservation(uint32_t address)
{
    auto it = s_reservations.upper_bound(address);
    if (it == s_reservations.begin())
        return s_reservations.end();
    --it;
    return address < uint64_t(it->first) + it->second.size ? it : s_reservations.end();
}

bool Overlaps(uint64_t begin, uint64_t end)
{
    auto it = s_reservations.lower_bound(static_cast<uint32_t>(begin));
    if (it != s_reservations.end() && it->first < end)
        return true;
    if (it != s_reservations.begin())
    {
        --it;
        if (uint64_t(it->first) + it->second.size > begin)
            return true;
    }
    return false;
}

std::optional<uint32_t> FindFree(uint64_t size, uint32_t alignment, bool topDown)
{
    // Gaps between reservations, scanned in address order (or reverse for MEM_TOP_DOWN).
    std::vector<std::pair<uint64_t, uint64_t>> gaps;
    uint64_t cursor = kVirtualBegin;
    for (auto& [base, r] : s_reservations)
    {
        if (base > cursor)
            gaps.emplace_back(cursor, base);
        cursor = std::max<uint64_t>(cursor, uint64_t(base) + r.size);
    }
    if (cursor < kVirtualEnd)
        gaps.emplace_back(cursor, kVirtualEnd);

    if (topDown)
    {
        for (auto it = gaps.rbegin(); it != gaps.rend(); ++it)
        {
            if (it->second < size)
                continue;
            uint64_t start = (it->second - size) & ~uint64_t(alignment - 1);
            if (start >= it->first)
                return static_cast<uint32_t>(start);
        }
    }
    else
    {
        for (auto& [begin, end] : gaps)
        {
            uint64_t start = AlignUp(begin, alignment);
            if (start + size <= end)
                return static_cast<uint32_t>(start);
        }
    }
    return std::nullopt;
}

bool CommitPages(uint32_t begin, uint32_t end, uint32_t protect)
{
    if (!g_memory.Commit(begin, end - begin))   // keeps write-watched pages watched
        return false;
    for (size_t i = PageIndex(begin); i < PageIndex(end); i++)
    {
        if (!(s_pages[i] & kCommitted))
            s_committed += kPage;
        s_pages[i] = kCommitted | static_cast<uint16_t>(protect & 0x7FFF);
    }
    s_peakCommitted = std::max(s_peakCommitted, s_committed);
    return true;
}

void DecommitPages(uint32_t begin, uint32_t end)
{
    // Host decommit: the next commit of these pages reads zeros, as on the console.
    g_memory.Decommit(begin, end - begin);
    for (size_t i = PageIndex(begin); i < PageIndex(end); i++)
    {
        if (s_pages[i] & kCommitted)
            s_committed -= kPage;
        s_pages[i] = 0;
    }
}

} // namespace

uint32_t VmAllocate(uint32_t& base, uint32_t& size, uint32_t type, uint32_t protect)
{
    if (!(type & (X_MEM_COMMIT | X_MEM_RESERVE | X_MEM_RESET)) || size == 0)
        return X_STATUS_INVALID_PARAMETER;
    if (type & X_MEM_RESET)
        return X_STATUS_SUCCESS; // contents may be discarded: keeping them is allowed

    std::lock_guard lock(s_mutex);
    const uint32_t pageSize = (type & (X_MEM_LARGE_PAGES | X_MEM_16MB_PAGES)) ? kGranularity : kPage;
    const uint32_t alignment = (type & X_MEM_16MB_PAGES) ? k16MB : kGranularity;

    if (base == 0 || (type & X_MEM_RESERVE))
    {
        // New reservation (MEM_COMMIT with no base reserves too, as on NT).
        uint64_t begin, length;
        if (base == 0)
        {
            length = AlignUp(size, std::max(pageSize, kGranularity));
            auto found = FindFree(length, alignment, (type & X_MEM_TOP_DOWN) != 0);
            if (!found)
            {
                log::Error("NtAllocateVirtualMemory: out of guest virtual space for {:#x} bytes", size);
                return X_STATUS_NO_MEMORY;
            }
            begin = *found;
        }
        else
        {
            begin = AlignDown(base, alignment);
            length = AlignUp(uint64_t(base) + size, pageSize) - begin;
            if (!InVirtualRange(begin, begin + length) || Overlaps(begin, begin + length))
                return X_STATUS_CONFLICTING_ADDRESSES;
        }
        s_reservations[static_cast<uint32_t>(begin)] = { static_cast<uint32_t>(length), protect, pageSize };
        s_reserved += length;
        if ((type & X_MEM_COMMIT) && !CommitPages(static_cast<uint32_t>(begin), static_cast<uint32_t>(begin + length), protect))
        {
            s_reservations.erase(static_cast<uint32_t>(begin));
            s_reserved -= length;
            return X_STATUS_NO_MEMORY;
        }
        base = static_cast<uint32_t>(begin);
        size = static_cast<uint32_t>(length);
        return X_STATUS_SUCCESS;
    }

    // Commit inside an existing reservation.
    auto it = FindReservation(base);
    if (it == s_reservations.end())
        return X_STATUS_CONFLICTING_ADDRESSES;
    const uint32_t commitPage = std::max(pageSize, it->second.pageSize);
    uint32_t begin = AlignDown(base, commitPage);
    uint64_t end = AlignUp(uint64_t(base) + size, commitPage);
    if (begin < it->first || end > uint64_t(it->first) + it->second.size)
        return X_STATUS_CONFLICTING_ADDRESSES;
    if (!CommitPages(begin, static_cast<uint32_t>(end), protect))
        return X_STATUS_NO_MEMORY;
    base = begin;
    size = static_cast<uint32_t>(end - begin);
    return X_STATUS_SUCCESS;
}

uint32_t VmFree(uint32_t& base, uint32_t& size, uint32_t freeType)
{
    std::lock_guard lock(s_mutex);
    auto it = FindReservation(base);
    if (it == s_reservations.end())
        return X_STATUS_MEMORY_NOT_ALLOCATED;
    const uint32_t resBase = it->first;
    const uint64_t resEnd = uint64_t(resBase) + it->second.size;

    if (freeType & X_MEM_RELEASE)
    {
        if (base != resBase)
            return X_STATUS_FREE_VM_NOT_AT_BASE;
        DecommitPages(resBase, static_cast<uint32_t>(resEnd));
        s_reserved -= it->second.size;
        size = it->second.size;
        s_reservations.erase(it);
        return X_STATUS_SUCCESS;
    }
    if (freeType & X_MEM_DECOMMIT)
    {
        uint32_t begin = AlignDown(base, kPage);
        uint64_t end = size == 0 ? resEnd : AlignUp(uint64_t(base) + size, kPage);
        if (end > resEnd)
            return X_STATUS_INVALID_PARAMETER;
        DecommitPages(begin, static_cast<uint32_t>(end));
        base = begin;
        size = static_cast<uint32_t>(end - begin);
        return X_STATUS_SUCCESS;
    }
    return X_STATUS_INVALID_PARAMETER;
}

uint32_t VmQuery(uint32_t address, XMemoryBasicInformation& info)
{
    std::lock_guard lock(s_mutex);
    const uint32_t page = AlignDown(address, kPage);

    if (page >= kVirtualBegin && page < kVirtualEnd)
    {
        auto it = FindReservation(page);
        if (it == s_reservations.end())
        {
            // Free region: up to the next reservation.
            auto next = s_reservations.upper_bound(page);
            uint32_t end = next == s_reservations.end() ? kVirtualEnd : next->first;
            info = { page, 0, 0, end - page, X_MEM_FREE, X_PAGE_NOACCESS, 0 };
            return X_STATUS_SUCCESS;
        }
        // Run of pages with the same state and protection.
        const uint16_t state = s_pages[PageIndex(page)];
        const uint32_t resEnd = it->first + it->second.size;
        uint32_t end = page + kPage;
        while (end < resEnd && s_pages[PageIndex(end)] == state)
            end += kPage;
        const bool committed = (state & kCommitted) != 0;
        info = { page, it->first, it->second.protect, end - page,
            committed ? X_MEM_COMMIT : X_MEM_RESERVE, committed ? uint32_t(state & 0x7FFF) : 0u, X_MEM_PRIVATE };
        return X_STATUS_SUCCESS;
    }

    // Outside the managed range: the image, the runtime heaps, or nothing.
    if (address >= g_module.imageBase && address < g_module.imageBase + g_module.imageSize)
    {
        info = { page, g_module.imageBase, X_PAGE_READWRITE, g_module.imageBase + g_module.imageSize - page,
            X_MEM_COMMIT, X_PAGE_READWRITE, X_MEM_IMAGE };
        return X_STATUS_SUCCESS;
    }
    if (g_memory.IsBacked(address))
    {
        info = { page, page, X_PAGE_READWRITE, kPage, X_MEM_COMMIT, X_PAGE_READWRITE, X_MEM_PRIVATE };
        return X_STATUS_SUCCESS;
    }
    info = { page, 0, 0, kPage, X_MEM_FREE, X_PAGE_NOACCESS, 0 };
    return X_STATUS_SUCCESS;
}

uint32_t VmQueryProtect(uint32_t address)
{
    if (address >= kVirtualBegin && address < kVirtualEnd)
    {
        std::lock_guard lock(s_mutex);
        uint16_t state = s_pages[PageIndex(address)];
        return (state & kCommitted) ? (state & 0x7FFF) : 0;
    }
    return g_memory.IsBacked(address) ? X_PAGE_READWRITE : 0;
}

bool VmSetProtect(uint32_t base, uint32_t size, uint32_t protect)
{
    // Recorded for queries only: host pages stay read/write.
    std::lock_guard lock(s_mutex);
    uint64_t end = AlignUp(uint64_t(base) + size, kPage);
    for (uint64_t p = AlignDown(base, kPage); p < end; p += kPage)
        if (p >= kVirtualBegin && p < kVirtualEnd && (s_pages[PageIndex(uint32_t(p))] & kCommitted))
            s_pages[PageIndex(uint32_t(p))] = kCommitted | static_cast<uint16_t>(protect & 0x7FFF);
    return true;
}

VmStatistics VmGetStatistics()
{
    std::lock_guard lock(s_mutex);
    return { s_reserved, s_committed, s_peakCommitted };
}

} // namespace ac6::kernel
