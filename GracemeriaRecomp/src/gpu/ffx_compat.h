#pragma once
// Force-included into the FidelityFX SDK's sources (CMakeLists.txt ac6_ffx): what MSVC provides them
// implicitly, for llvm-mingw and GCC. The SDK's files stay as AMD ships them.
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cwchar>
#include <mutex>
#include <new>

#ifndef _countof
#define _countof(array) (sizeof(array) / sizeof((array)[0]))
#endif

#if !defined(_MSC_VER)
#define swprintf_s swprintf
#endif

#if !defined(_WIN32)
// What the SDK's non-Windows paths use without including: memset / memcpy, std::wstring_convert (ffx_vk.cpp's name
// conversions), FFX_UNUSED (ffx_message.cpp).
#include <cstring>
#include <locale>
#include <FidelityFX/host/ffx_util.h>
// The C11 Annex K / MSVC "secure" copies the SDK uses (llvm-mingw's CRT has them, glibc doesn't): truncating, always
// terminated.
#include <cstdlib>
inline int ac6_wcscpy_s(wchar_t* dst, size_t size, const wchar_t* src)
{
    if (!dst || !size)
        return 22;
    size_t i = 0;
    for (; i + 1 < size && src && src[i]; i++)
        dst[i] = src[i];
    dst[i] = 0;
    return 0;
}
inline int wcscpy_s(wchar_t* dst, size_t size, const wchar_t* src) { return ac6_wcscpy_s(dst, size, src); }
template <size_t N>
inline int wcscpy_s(wchar_t (&dst)[N], const wchar_t* src) { return ac6_wcscpy_s(dst, N, src); }
inline int strcpy_s(char* dst, size_t size, const char* src)
{
    if (!dst || !size)
        return 22;
    size_t i = 0;
    for (; i + 1 < size && src && src[i]; i++)
        dst[i] = src[i];
    dst[i] = 0;
    return 0;
}
inline int wcstombs_s(size_t* converted, char* dst, size_t size, const wchar_t* src, size_t count)
{
    size_t n = (dst && size) ? std::wcstombs(dst, src, std::min(size - 1, count)) : 0;
    if (n == size_t(-1))
        n = 0;
    if (dst && size)
        dst[n] = 0;
    if (converted)
        *converted = n + 1;
    return 0;
}
#define sprintf_s snprintf
#endif
