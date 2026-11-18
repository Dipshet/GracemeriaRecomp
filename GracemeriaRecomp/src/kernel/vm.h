#pragma once
// Guest virtual memory for NtAllocateVirtualMemory / NtFreeVirtualMemory / NtQueryVirtualMemory.
// The game's range [kVirtualBegin, kVirtualEnd) is managed page by page (4 KB): reservations are 64 KB aligned,
// commits map host pages, decommits
// return them to the host so a later commit reads zeros, as on the console. Reserved-but-uncommitted
// pages fault like they would on hardware (they are not a lazily backed range).

#include "kernel/xtypes.h"

#include <cstdint>

namespace ac6::kernel {

// Allocation types / free types / states (winnt.h values, as on the console).
constexpr uint32_t X_MEM_COMMIT      = 0x00001000;
constexpr uint32_t X_MEM_RESERVE     = 0x00002000;
constexpr uint32_t X_MEM_DECOMMIT    = 0x00004000;
constexpr uint32_t X_MEM_RELEASE     = 0x00008000;
constexpr uint32_t X_MEM_FREE        = 0x00010000;
constexpr uint32_t X_MEM_PRIVATE     = 0x00020000;
constexpr uint32_t X_MEM_RESET       = 0x00080000;
constexpr uint32_t X_MEM_TOP_DOWN    = 0x00100000;
constexpr uint32_t X_MEM_NOZERO      = 0x00800000;
constexpr uint32_t X_MEM_IMAGE       = 0x01000000;
constexpr uint32_t X_MEM_LARGE_PAGES = 0x20000000;   // 64 KB pages
constexpr uint32_t X_MEM_16MB_PAGES  = 0x80000000;

constexpr uint32_t X_PAGE_NOACCESS  = 0x01;
constexpr uint32_t X_PAGE_READONLY  = 0x02;
constexpr uint32_t X_PAGE_READWRITE = 0x04;

// In/out arguments follow the NT calls: base and size are rounded and written back.
uint32_t VmAllocate(uint32_t& base, uint32_t& size, uint32_t allocationType, uint32_t protect);
uint32_t VmFree(uint32_t& base, uint32_t& size, uint32_t freeType);
uint32_t VmQuery(uint32_t address, XMemoryBasicInformation& info);
uint32_t VmQueryProtect(uint32_t address);                 // 0 if not committed
bool VmSetProtect(uint32_t base, uint32_t size, uint32_t protect);

struct VmStatistics
{
    uint64_t reservedBytes;
    uint64_t committedBytes;
    uint64_t peakCommittedBytes;
};
VmStatistics VmGetStatistics();

} // namespace ac6::kernel
