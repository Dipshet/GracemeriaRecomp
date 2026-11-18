#pragma once
// Replaces XenosRecomp's pch.h for the runtime build: its big-endian helper and an fmt shim over
// std::format (XenosRecomp uses {fmt}, the runtime uses the standard library).

#include <bit>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <format>
#include <iterator>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>

namespace fmt {
using std::format;
using std::make_format_args;
using std::vformat_to;
template<class... Args>
using format_string = std::format_string<Args...>;
} // namespace fmt

namespace xenos {

template<typename T>
static T byteSwap(T value)
{
    if constexpr (sizeof(T) == 1)
        return value;
    else if constexpr (sizeof(T) == 2)
        return static_cast<T>(__builtin_bswap16(static_cast<uint16_t>(value)));
    else if constexpr (sizeof(T) == 4)
        return static_cast<T>(__builtin_bswap32(static_cast<uint32_t>(value)));
    else
        return static_cast<T>(__builtin_bswap64(static_cast<uint64_t>(value)));
}

// Read-only big-endian view (as in XenosRecomp).
template<typename T>
struct be
{
    T value;

    T get() const
    {
        if constexpr (std::is_enum_v<T>)
            return T(byteSwap(std::underlying_type_t<T>(value)));
        else
            return byteSwap(value);
    }

    operator T() const { return get(); }
};

} // namespace xenos
