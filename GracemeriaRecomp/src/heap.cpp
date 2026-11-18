// Adapted from UnleashedRecomp (UnleashedRecomp/kernel/heap.cpp, GPL-3.0).
// Differences: ranges come from memory.h, pages are committed on first touch (memory.cpp),
// and AC6's own heap functions are not hooked.
#include "heap.h"

#include "log.h"
#include "memory.h"

#include <algorithm>

#include <o1heap.h>

namespace ac6 {

Heap g_heap;

bool Heap::Init()
{
    g_memory.AddBackedRange(kUserHeapBegin, kUserHeapEnd);
    g_memory.AddBackedRange(kPhysHeapBegin, kPhysHeapEnd);

    heap = o1heapInit(g_memory.Translate(kUserHeapBegin), kUserHeapEnd - kUserHeapBegin);
    physicalHeap = o1heapInit(g_memory.Translate(kPhysHeapBegin), kPhysObjectsEnd - kPhysHeapBegin);
    freePages[kPhysObjectsEnd] = static_cast<uint32_t>(kPhysHeapEnd - kPhysObjectsEnd);
    if (!heap || !physicalHeap)
    {
        log::Error("o1heap init failed");
        return false;
    }
    return true;
}

void* Heap::Alloc(size_t size)
{
    std::lock_guard lock(mutex);
    return o1heapAllocate(heap, std::max<size_t>(1, size));
}

void* Heap::AllocPhysical(size_t size, size_t alignment)
{
    size = std::max<size_t>(1, size);
    alignment = alignment == 0 ? 0x1000 : std::max<size_t>(16, alignment);

    std::lock_guard lock(physicalMutex);
    void* ptr = o1heapAllocate(physicalHeap, size + alignment);
    if (!ptr)
        return nullptr;
    size_t aligned = (reinterpret_cast<size_t>(ptr) + alignment) & ~(alignment - 1);

    // Stash the raw pointer and size just below the aligned block (as UnleashedRecomp does).
    *(reinterpret_cast<void**>(aligned) - 1) = ptr;
    *(reinterpret_cast<size_t*>(aligned) - 2) = size + O1HEAP_ALIGNMENT;
    return reinterpret_cast<void*>(aligned);
}

void Heap::Free(void* ptr)
{
    if (!ptr)
        return;
    if (ptr >= g_memory.Translate(kPhysObjectsEnd) && ptr < g_memory.Translate(static_cast<uint32_t>(kPhysHeapEnd - 1)))
    {
        FreePages(g_memory.MapVirtual(ptr));
    }
    else if (ptr >= g_memory.Translate(kPhysHeapBegin))
    {
        std::lock_guard lock(physicalMutex);
        o1heapFree(physicalHeap, *(reinterpret_cast<void**>(ptr) - 1));
    }
    else
    {
        std::lock_guard lock(mutex);
        o1heapFree(heap, ptr);
    }
}

size_t Heap::Size(void* ptr)
{
    // Relies on o1heap's fragment header layout (as UnleashedRecomp does).
    return ptr ? *(reinterpret_cast<size_t*>(ptr) - 2) - O1HEAP_ALIGNMENT : 0;
}

size_t Heap::UserAllocated()
{
    std::lock_guard lock(mutex);
    return o1heapGetDiagnostics(heap).allocated;
}

size_t Heap::PhysicalAllocated()
{
    size_t objects;
    {
        std::lock_guard lock(physicalMutex);
        objects = o1heapGetDiagnostics(physicalHeap).allocated;
    }
    std::lock_guard lock(pagesMutex);
    return objects + pagesUsed;
}

uint32_t Heap::AllocPages(uint32_t size, uint32_t alignment, bool topDown)
{
    constexpr uint32_t kPage = 0x1000;
    size = (std::max<uint32_t>(size, 1) + kPage - 1) & ~(kPage - 1);
    alignment = std::max(alignment, kPage);
    std::lock_guard lock(pagesMutex);

    auto take = [&](std::map<uint32_t, uint32_t>::iterator it, uint32_t start) {
        const uint32_t blockStart = it->first, blockSize = it->second;
        const uint64_t blockEnd = uint64_t(blockStart) + blockSize, end = uint64_t(start) + size;
        freePages.erase(it);
        if (start > blockStart)
            freePages[blockStart] = start - blockStart;
        if (end < blockEnd)
            freePages[static_cast<uint32_t>(end)] = static_cast<uint32_t>(blockEnd - end);
        usedPages[start] = size;
        pagesUsed += size;
        return start;
    };

    if (topDown)
    {
        for (auto it = freePages.end(); it != freePages.begin();)
        {
            --it;
            uint64_t end = uint64_t(it->first) + it->second;
            if (end < size)
                continue;
            uint64_t start = (end - size) & ~uint64_t(alignment - 1);
            if (start >= it->first)
                return take(it, static_cast<uint32_t>(start));
        }
    }
    else
    {
        for (auto it = freePages.begin(); it != freePages.end(); ++it)
        {
            uint64_t start = (uint64_t(it->first) + alignment - 1) & ~uint64_t(alignment - 1);
            if (start + size <= uint64_t(it->first) + it->second)
                return take(it, static_cast<uint32_t>(start));
        }
    }
    return 0;
}

bool Heap::FreePages(uint32_t address)
{
    std::lock_guard lock(pagesMutex);
    auto used = usedPages.find(address);
    if (used == usedPages.end())
        return false;
    uint32_t start = address, size = used->second;
    pagesUsed -= size;
    usedPages.erase(used);
    // Coalesce with the neighbours.
    auto next = freePages.lower_bound(start);
    if (next != freePages.end() && next->first == start + size)
    {
        size += next->second;
        next = freePages.erase(next);
    }
    if (next != freePages.begin())
    {
        auto prev = std::prev(next);
        if (uint64_t(prev->first) + prev->second == start)
        {
            start = prev->first;
            size += prev->second;
            freePages.erase(prev);
        }
    }
    freePages[start] = size;
    return true;
}

} // namespace ac6
