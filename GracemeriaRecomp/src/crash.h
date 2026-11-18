#pragma once
// Fault reporting: maps host crashes back to the recompiled guest function and dumps guest state.

#include <cstdint>
#include <string>

namespace ac6::crash {

void Install();
// The recompiled (or hooked) guest function whose host code contains `host`, 0 if none.
uint32_t GuestFunctionAt(uintptr_t host);
// Diagnostics: logs every write to [guest, guest + size) (value, writer, host stack), up to 2000 writes.
void ArmWriteWatch(uint32_t guest, uint32_t size);
// Diagnostics, Windows only: AC6_WATCH_HW=<hex guest address> logs every write to that word (debug registers, no
// slowdown). AC6_WATCH_HW=<address>:rw logs reads too.
void StartHardwareWatch();
// The hang report (AC6_HANG_SECONDS, see crash.cpp) and, for automated runs, AC6_RUN_SECONDS=N: exits (with the
// summary) after N seconds.
void StartWatchdog();
// What this thread is doing, for a fault's report: `task` is logged if the thread faults, and
// `onFault` (if any) is called there, on the faulting thread, with `context`. The `task` string must stay valid until
// the task is cleared, and a null `task` clears it.
void SetThreadTask(const char* task, void (*onFault)(void*) = nullptr, void* context = nullptr);
// Called once per presented frame, feeds the hang report.
void NoteFrame();
// Windowed builds: Shutdown with a non-zero code also tells the user in a message box.
void ShowFatalErrors(bool enable);
void ShowFatalError(const char* message);
// This thread's host stack as a chain of recompiled functions (sub_X) and other frames, innermost first (diagnostics).
std::string HostStack();
// Write the import summary and exit.
[[noreturn]] void Shutdown(int code, const char* reason);

} // namespace ac6::crash
