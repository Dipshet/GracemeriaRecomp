// Memory imports: virtual memory (kernel/vm.h), physical memory and pools (heap.h).
#include "heap.h"
#include "kernel/function.h"
#include "kernel/vm.h"
#include "loader.h"
#include "log.h"
#include "memory.h"

#include <algorithm>
#include <cstring>

namespace ac6::kernel {

uint32_t NtAllocateVirtualMemory(be<uint32_t>* baseAddress, be<uint32_t>* regionSize, uint32_t allocationType,
    uint32_t protect, uint32_t debugMemory)
{
    uint32_t base = *baseAddress, size = *regionSize;
    uint32_t status = VmAllocate(base, size, allocationType, protect);
    log::Trace("NtAllocateVirtualMemory({:08X}, {:#x}, type {:08X}, protect {:X}) -> {:08X} [{:08X}, +{:#x}]",
        uint32_t(*baseAddress), uint32_t(*regionSize), allocationType, protect, status, base, size);
    if (status == X_STATUS_SUCCESS)
    {
        *baseAddress = base;
        *regionSize = size;
    }
    return status;
}

uint32_t NtFreeVirtualMemory(be<uint32_t>* baseAddress, be<uint32_t>* regionSize, uint32_t freeType, uint32_t debugMemory)
{
    uint32_t base = *baseAddress, size = regionSize ? uint32_t(*regionSize) : 0;
    uint32_t status = VmFree(base, size, freeType);
    log::Trace("NtFreeVirtualMemory({:08X}, {:#x}, type {:08X}) -> {:08X}", uint32_t(*baseAddress), size, freeType, status);
    if (status == X_STATUS_SUCCESS)
    {
        *baseAddress = base;
        if (regionSize)
            *regionSize = size;
    }
    return status;
}

uint32_t NtQueryVirtualMemory(uint32_t address, XMemoryBasicInformation* info, uint32_t regionType)
{
    return info ? VmQuery(address, *info) : X_STATUS_INVALID_PARAMETER;
}

uint32_t MmQueryAddressProtect(uint32_t address)
{
    return VmQueryProtect(address);
}

void MmSetAddressProtect(uint32_t address, uint32_t size, uint32_t protect)
{
    VmSetProtect(address, size, protect);
}

namespace {

struct XMmStatistics
{
    be<uint32_t> length;
    be<uint32_t> totalPhysicalPages;
    be<uint32_t> kernelPages;
    be<uint32_t> titleAvailablePages;
    be<uint32_t> titleTotalVirtualMemoryBytes;
    be<uint32_t> titleReservedVirtualMemoryBytes;
    be<uint32_t> titlePhysicalPages;
    be<uint32_t> titlePoolPages;
    be<uint32_t> titleStackPages;
    be<uint32_t> titleImagePages;
    be<uint32_t> titleHeapPages;
    be<uint32_t> titleVirtualPages;
    be<uint32_t> titlePageTablePages;
    be<uint32_t> titleCachePages;
    be<uint32_t> systemAvailablePages;
    be<uint32_t> systemTotalVirtualMemoryBytes;
    be<uint32_t> systemReservedVirtualMemoryBytes;
    be<uint32_t> systemPhysicalPages;
    be<uint32_t> systemPoolPages;
    be<uint32_t> systemStackPages;
    be<uint32_t> systemImagePages;
    be<uint32_t> systemHeapPages;
    be<uint32_t> systemVirtualPages;
    be<uint32_t> systemPageTablePages;
    be<uint32_t> systemCachePages;
    be<uint32_t> highestPhysicalPage;
};
static_assert(sizeof(XMmStatistics) == 0x68);

// The console's memory budget: 512 MB, of which the system keeps 32 MB. The game sizes its main heap
// from what's left ("available minus 8 MB"), so this must look like a console, not like the host.
constexpr uint64_t kTotalPhysical = 512ull << 20;
constexpr uint64_t kSystemReserved = 32ull << 20;
constexpr uint32_t kPage = 0x1000;

} // namespace

uint32_t MmQueryStatistics(XMmStatistics* stats)
{
    if (!stats || stats->length != sizeof(XMmStatistics))
        return X_STATUS_INVALID_PARAMETER;

    const VmStatistics vm = VmGetStatistics();
    const uint64_t image = g_module.imageSize;
    const uint64_t physical = g_heap.PhysicalAllocated();
    const uint64_t user = g_heap.UserAllocated();
    const uint64_t used = image + physical + user + vm.committedBytes;
    const uint64_t titleBudget = kTotalPhysical - kSystemReserved;
    const uint64_t available = used < titleBudget ? titleBudget - used : 0;

    std::memset(reinterpret_cast<uint8_t*>(stats) + 4, 0, sizeof(XMmStatistics) - 4);
    stats->totalPhysicalPages = static_cast<uint32_t>(kTotalPhysical / kPage);
    stats->kernelPages = static_cast<uint32_t>(kSystemReserved / kPage);
    stats->titleAvailablePages = static_cast<uint32_t>(available / kPage);
    stats->titleTotalVirtualMemoryBytes = kVirtualEnd - kVirtualBegin;
    stats->titleReservedVirtualMemoryBytes = static_cast<uint32_t>(vm.reservedBytes);
    stats->titlePhysicalPages = static_cast<uint32_t>(physical / kPage);
    stats->titleImagePages = static_cast<uint32_t>(image / kPage);
    stats->titleHeapPages = static_cast<uint32_t>(user / kPage);
    stats->titleVirtualPages = static_cast<uint32_t>(vm.committedBytes / kPage);
    stats->systemAvailablePages = 0;
    stats->highestPhysicalPage = static_cast<uint32_t>(kTotalPhysical / kPage - 1);
    log::Info("MmQueryStatistics: title available {} MB (image {} MB, physical {} MB, heap {} MB, virtual {} MB)",
        available >> 20, image >> 20, physical >> 20, user >> 20, vm.committedBytes >> 20);
    return X_STATUS_SUCCESS;
}

uint32_t MmAllocatePhysicalMemoryEx(uint32_t flags, uint32_t size, uint32_t protect, uint32_t minAddress,
    uint32_t maxAddress, uint32_t alignment)
{
    // MEMORY_ALLOCATE_TOP_DOWN-style placement is ignored. The max address limits nothing here (all of
    // the 512 MB is addressable by the GPU).
    uint32_t address = g_heap.AllocPages(size, alignment, false);
    if (!address)
    {
        log::Error("MmAllocatePhysicalMemoryEx: out of physical memory ({:#x} bytes)", size);
        return 0;
    }
    g_memory.Zero(address, size);
    log::Trace("MmAllocatePhysicalMemoryEx({:#x}, align {:#x}, protect {:X}) -> {:08X}", size, alignment, protect, address);
    return address;
}

void MmFreePhysicalMemory(uint32_t type, uint32_t address)
{
    if (address && !g_heap.FreePages(address))
        log::Warn("MmFreePhysicalMemory({:08X}): not a physical allocation", address);
}

uint32_t MmGetPhysicalAddress(uint32_t address)
{
    return address & 0x1FFFFFFF;
}

uint32_t ExAllocatePool(uint32_t size)
{
    return g_memory.MapVirtual(g_heap.Alloc(size));
}

void ExFreePool(uint32_t address)
{
    if (address)
        g_heap.Free(g_memory.Translate(address));
}

uint32_t XamAlloc(uint32_t flags, uint32_t size, be<uint32_t>* out)
{
    void* p = g_heap.Alloc(size);
    if (!p)
        return 0x8007000E; // E_OUTOFMEMORY
    std::memset(p, 0, size);
    *out = g_memory.MapVirtual(p);
    return 0;
}

void XamFree(uint32_t address)
{
    if (address)
        g_heap.Free(g_memory.Translate(address));
}

void RtlFillMemoryUlong(be<uint32_t>* destination, uint32_t length, uint32_t pattern)
{
    std::fill_n(destination, length / 4, be<uint32_t>(pattern));
}

uint32_t RtlCompareMemoryUlong(const be<uint32_t>* source, uint32_t length, uint32_t pattern)
{
    uint32_t i = 0;
    while (i < length / 4 && source[i] == pattern)
        i++;
    return i * 4;
}

} // namespace ac6::kernel

AC6_IMPORT(NtAllocateVirtualMemory)
AC6_IMPORT(NtFreeVirtualMemory)
AC6_IMPORT(NtQueryVirtualMemory)
AC6_IMPORT(MmQueryAddressProtect)
AC6_IMPORT(MmSetAddressProtect)
AC6_IMPORT(MmQueryStatistics)
AC6_IMPORT(MmAllocatePhysicalMemoryEx)
AC6_IMPORT(MmFreePhysicalMemory)
AC6_IMPORT(MmGetPhysicalAddress)
AC6_IMPORT(ExAllocatePool)
AC6_IMPORT(ExFreePool)
AC6_IMPORT(XamAlloc)
AC6_IMPORT(XamFree)
AC6_IMPORT(RtlFillMemoryUlong)
AC6_IMPORT(RtlCompareMemoryUlong)
