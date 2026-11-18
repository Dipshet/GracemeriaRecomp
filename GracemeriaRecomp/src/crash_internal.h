#pragma once
// Between crash.cpp (the platform-independent half of crash.h) and crash_win32.cpp / crash_linux.cpp.

#include <cstdint>

namespace ac6::crash::detail {

// The guest address of the recompiled or hooked function starting at host address `start`, 0 if none.
uint32_t GuestAtFunctionStart(uintptr_t start);
// The calling thread's guest registers, call chain and the memory its registers point at.
void DumpGuestContext();
// Every running guest thread (with host stacks where the platform can take them).
void DumpGuestThreads();
// std::terminate, abort() and the like: logged with this thread's state, then Shutdown(3).
[[noreturn]] void FatalNonException(const char* what);
// The platform's fault and interrupt handlers.
void InstallPlatform();

} // namespace ac6::crash::detail
