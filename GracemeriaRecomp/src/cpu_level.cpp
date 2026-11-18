// The CPU level check. The main executables are built for x86-64-v3 (AVX2, FMA, BMI1/2, F16C,
// LZCNT, MOVBE). On Windows a CPU without them gets the x86-64-v2 compatibility version when it is next to them
// (<name>_compat.exe, as build_compat.bat makes it), started with the same command line, its exit code passed on.
// Releases ship that version as a download of its own (the -windows-x64-compat archive, its executable named as the
// main one): without it the message says to download it. Linux builds need AVX2. This file is compiled for plain x86-64
// and runs as the first constructor, before any code that may use v3 instructions.
// AC6_CPU_LEVEL=v2 takes the fallback on any CPU (Windows, tests).
#if !defined(AC6_X86_V2)

#include <cpuid.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(_WIN32)
#include <windows.h>
#endif

namespace {

bool HasV3()
{
    unsigned a, b, c, d;
    if (!__get_cpuid(1, &a, &b, &c, &d))
        return false;
    const bool fma = c & (1u << 12), movbe = c & (1u << 22), osxsave = c & (1u << 27), avx = c & (1u << 28), f16c = c & (1u << 29);
    if (!(fma && movbe && osxsave && avx && f16c))
        return false;
    // The OS saves the AVX state (XCR0 bits 1 and 2).
    unsigned lo, hi;
    __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    if ((lo & 6) != 6)
        return false;
    if (!__get_cpuid_count(7, 0, &a, &b, &c, &d))
        return false;
    const bool bmi1 = b & (1u << 3), avx2 = b & (1u << 5), bmi2 = b & (1u << 8);
    if (!__get_cpuid(0x80000001, &a, &b, &c, &d))
        return false;
    const bool lzcnt = c & (1u << 5);
    return bmi1 && avx2 && bmi2 && lzcnt;
}

#if defined(_WIN32)
bool WantV2()
{
    const char* force = std::getenv("AC6_CPU_LEVEL");
    return !HasV3() || (force && std::strcmp(force, "v2") == 0);
}
#endif

#if !defined(_WIN32)
// Linux builds are AVX2-only, with no v2 version: a CPU without AVX2 gets a message instead of an
// illegal-instruction crash.
__attribute__((constructor(101))) void CheckCpuLevel()
{
    if (HasV3())
        return;
    std::fprintf(stderr, "Ace Combat 6 needs a CPU with AVX2 (x86-64-v3).\n");
    std::_Exit(1);
}
#else
__attribute__((constructor(101))) void CheckCpuLevel()
{
    if (!WantV2())
        return;
    // <dir>\<name>.exe -> <dir>\<name>_compat.exe
    wchar_t path[MAX_PATH + 8];
    const DWORD n = GetModuleFileNameW(nullptr, path, MAX_PATH);
    if (n == 0 || n >= MAX_PATH || n < 4 || _wcsicmp(path + n - 4, L".exe") != 0)
        std::_Exit(1);
    wcscpy(path + n - 4, L"_compat.exe");
    STARTUPINFOW si{};
    si.cb = sizeof(si);
    PROCESS_INFORMATION pi{};
    if (!CreateProcessW(path, GetCommandLineW(), nullptr, nullptr, TRUE, 0, nullptr, nullptr, &si, &pi))
    {
        std::fprintf(stderr, "This version of Ace Combat 6 needs a CPU with AVX2 (x86-64-v3). Download the compatibility "
            "version instead: the release whose name ends in -windows-x64-compat.zip.\n");
        // Background runs (AC6_BACKGROUND=1, used by tests) get only the console line, no message box.
        if (const char* v = std::getenv("AC6_BACKGROUND"); !(v && *v == '1'))
            MessageBoxW(nullptr,
                L"This version of Ace Combat 6 needs a CPU with AVX2 (x86-64-v3).\n\n"
                L"Download the compatibility version instead: on the releases page, the file whose name ends in "
                L"-windows-x64-compat.zip.",
                L"Ace Combat 6", MB_OK | MB_ICONERROR);
        std::_Exit(1);
    }
    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD code = 1;
    GetExitCodeProcess(pi.hProcess, &code);
    std::_Exit(int(code));
}
#endif

} // namespace

#endif
