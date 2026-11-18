#pragma once
// Lookup tables on the per-draw path. libc++'s std::hash of an integer is the integer
// itself, so its tables keep prime bucket counts and divide on every lookup. Guest addresses and hashes
// mixed here can use power-of-two bucket counts instead, which libc++ keeps once it has one (a mask per
// lookup). FastMap starts with one.

#include <cstddef>
#include <cstdint>
#include <unordered_map>

namespace ac6 {

struct FastHash
{
    size_t operator()(uint64_t k) const noexcept
    {
        k ^= k >> 33;
        k *= 0xFF51AFD7ED558CCDull;
        k ^= k >> 33;
        return size_t(k);
    }
    template <class T>
    size_t operator()(T* p) const noexcept { return (*this)(uint64_t(reinterpret_cast<uintptr_t>(p))); }
};

template <class K, class V>
struct FastMap : std::unordered_map<K, V, FastHash>
{
    FastMap() { this->rehash(64); }
};

} // namespace ac6
