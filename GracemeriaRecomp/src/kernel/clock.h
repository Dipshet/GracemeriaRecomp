#pragma once
// Guest time sources.
//
// The recompiled code reads the PowerPC time base (mftb) as the host TSC (__rdtsc), so the guest's
// "performance frequency" must be the TSC frequency, measured at startup. KeTimeStampBundle (a
// variable import some code reads for tick counts) is refreshed by a host thread every millisecond.

#include <cstdint>

namespace ac6::kernel {

void InitClock();                    // calibrates the TSC and raises the host timer resolution
uint64_t TimebaseFrequency();        // ticks per second of mftb
uint64_t BootTime100ns();            // system time (FILETIME units) when the runtime started

// Fills the variable imports the kernel exports (version, command line, time stamp bundle, ...).
void InitVariableImports();

} // namespace ac6::kernel
