#pragma once
// Guest heaps: o1heap over the user range and the physical range.
// Adapted from UnleashedRecomp (UnleashedRecomp/kernel/heap.{h,cpp}, GPL-3.0).

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <new>
#include <map>
#include <utility>

struct O1HeapInstance;

namespace ac6 {

struct Heap
{
    std::mutex mutex;
    O1HeapInstance* heap = nullptr;
    std::mutex physicalMutex;
    O1HeapInstance* physicalHeap = nullptr;
    std::mutex pagesMutex;
    std::map<uint32_t, uint32_t> freePages;    // start -> size
    std::map<uint32_t, uint32_t> usedPages;    // start -> size
    uint64_t pagesUsed = 0;

    bool Init();

    void* Alloc(size_t size);
    void* AllocPhysical(size_t size, size_t alignment);
    void Free(void* ptr);
    size_t Size(void* ptr);

    size_t UserAllocated();       // bytes in use, including o1heap's per-fragment overhead
    size_t PhysicalAllocated();   // kernel objects + game physical allocations

    // The game's physical memory (MmAllocatePhysicalMemoryEx): page-granular first fit in
    // [kPhysObjectsEnd, kPhysHeapEnd). Returns a guest address or 0.
    uint32_t AllocPages(uint32_t size, uint32_t alignment, bool topDown);
    bool FreePages(uint32_t address);

    template<typename T, typename... Args>
    T* New(Args&&... args)
    {
        return new (AllocPhysical(sizeof(T), alignof(T))) T(std::forward<Args>(args)...);
    }
};

extern Heap g_heap;

} // namespace ac6
