#pragma once
// Big-endian guest values. The runtime's own be<T>: XenonUtils' xbox.h version has operator& and
// operator| that modify the stored value ("x & mask" writes back into guest memory and corrupts guest
// objects). This one only converts.

#include <bit>
#include <cstdint>
#include <cstring>
#include <type_traits>

namespace ac6 {

template<typename T>
T ByteSwapValue(T value)
{
    if constexpr (sizeof(T) == 1)
    {
        return value;
    }
    else
    {
        using U = std::conditional_t<sizeof(T) == 2, uint16_t, std::conditional_t<sizeof(T) == 4, uint32_t, uint64_t>>;
        U u;
        std::memcpy(&u, &value, sizeof(T));
        u = std::byteswap(u);
        T out;
        std::memcpy(&out, &u, sizeof(T));
        return out;
    }
}

template<typename T>
struct be
{
    T value{};   // stored big-endian

    be() = default;
    be(T v) : value(ByteSwapValue(v)) {}

    T get() const { return ByteSwapValue(value); }
    void set(T v) { value = ByteSwapValue(v); }

    operator T() const { return get(); }
    be& operator=(T v)
    {
        set(v);
        return *this;
    }
};

static_assert(sizeof(be<uint32_t>) == 4 && sizeof(be<uint64_t>) == 8);

} // namespace ac6
