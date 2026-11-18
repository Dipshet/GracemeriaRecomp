#pragma once
// The machine at the top of the log: OS, CPU, RAM, and on Windows the GPUs with their driver
// versions (DXGI, before Vulkan starts). The Vulkan device line (gpu/video.cpp) adds the driver version of the GPU
// the renderer uses, through DriverVersion().

#include <cstdint>
#include <string>

namespace ac6::system_info {

void Log();

// A Vulkan driverVersion as its vendor writes it (NVIDIA 566.36, Intel on Windows 101.5590, others major.minor.patch).
std::string DriverVersion(uint32_t vendorId, uint64_t driverVersion);

} // namespace ac6::system_info
