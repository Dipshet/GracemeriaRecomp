// Raw deflate decoding (RFC 1951), written for the port. Huffman codes are decoded with a 10-bit first-level table and,
// for longer codes, a canonical walk over the code lengths.
#include "game/inflate.h"

#include <cstring>

namespace ac6::game {

namespace {

constexpr int kMaxBits = 15;
constexpr int kFastBits = 10;

struct Huffman
{
    uint16_t count[kMaxBits + 1] = {};   // codes of each length
    uint16_t symbol[320] = {};           // symbols ordered by code
    uint16_t fast[1 << kFastBits] = {};  // (length << 9 | symbol) for codes up to kFastBits long, 0 = longer
};

struct Bits
{
    const uint8_t* in;
    size_t size, pos = 0;
    uint64_t buffer = 0;
    int count = 0;
    bool overrun = false;

    void Fill()
    {
        while (count <= 56)
        {
            uint64_t b = 0;
            if (pos < size)
                b = in[pos];
            else if (pos >= size + 8)   // reads past the end are zeros, past 8 of them the stream is damaged
                overrun = true;
            pos++;
            buffer |= b << count;
            count += 8;
        }
    }
    uint32_t Get(int n)
    {
        if (count < n)
            Fill();
        const uint32_t v = uint32_t(buffer & ((1ull << n) - 1));
        buffer >>= n;
        count -= n;
        return v;
    }
    uint32_t Peek(int n)
    {
        if (count < n)
            Fill();
        return uint32_t(buffer & ((1ull << n) - 1));
    }
    void Drop(int n)
    {
        buffer >>= n;
        count -= n;
    }
};

uint32_t Reverse(uint32_t code, int len)
{
    uint32_t r = 0;
    for (int i = 0; i < len; i++)
        r = r << 1 | ((code >> i) & 1);
    return r;
}

// Codes from lengths (0 = unused), false for an over-subscribed set.
bool Build(Huffman& h, const uint8_t* lengths, int n)
{
    std::memset(&h, 0, sizeof h);
    for (int i = 0; i < n; i++)
        h.count[lengths[i]]++;
    h.count[0] = 0;
    int left = 1;
    for (int len = 1; len <= kMaxBits; len++)
    {
        left <<= 1;
        left -= h.count[len];
        if (left < 0)
            return false;
    }
    uint16_t offs[kMaxBits + 2] = {};
    for (int len = 1; len <= kMaxBits; len++)
        offs[len + 1] = uint16_t(offs[len] + h.count[len]);
    for (int i = 0; i < n; i++)
        if (lengths[i])
            h.symbol[offs[lengths[i]]++] = uint16_t(i);
    // The fast table: canonical codes, bit-reversed (deflate sends them most significant bit first).
    uint32_t code = 0;
    int index = 0;
    for (int len = 1; len <= kMaxBits; len++)
    {
        for (int k = 0; k < h.count[len]; k++, index++, code++)
            if (len <= kFastBits)
            {
                const uint32_t rev = Reverse(code, len);
                for (uint32_t fill = rev; fill < (1u << kFastBits); fill += 1u << len)
                    h.fast[fill] = uint16_t(len << 9 | h.symbol[index]);
            }
        code <<= 1;
    }
    return true;
}

int Decode(Bits& b, const Huffman& h)
{
    const uint16_t f = h.fast[b.Peek(kFastBits)];
    if (f)
    {
        b.Drop(f >> 9);
        return f & 511;
    }
    // Longer codes: walk length by length (the canonical decode, as zlib's puff decoder does it).
    int code = 0, first = 0, index = 0;
    for (int len = 1; len <= kMaxBits; len++)
    {
        code |= int(b.Get(1));
        const int count = h.count[len];
        if (code - count < first)
            return h.symbol[index + (code - first)];
        index += count;
        first += count;
        first <<= 1;
        code <<= 1;
    }
    return -1;
}

constexpr uint16_t kLengthBase[29] = { 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115,
                                       131, 163, 195, 227, 258 };
constexpr uint8_t kLengthExtra[29] = { 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0 };
constexpr uint16_t kDistBase[30] = { 1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
                                     2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577 };
constexpr uint8_t kDistExtra[30] = { 0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13 };

} // namespace

size_t Inflate(const uint8_t* in, size_t inSize, uint8_t* out, size_t outSize)
{
    Bits b{ in, inSize };
    size_t made = 0;
    static Huffman* fixedLit = nullptr;
    static Huffman* fixedDist = nullptr;
    static const bool fixedBuilt = [] {
        static Huffman lit, dist;
        uint8_t l[288];
        for (int i = 0; i < 288; i++)
            l[i] = i < 144 ? 8 : i < 256 ? 9 : i < 280 ? 7 : 8;
        Build(lit, l, 288);
        uint8_t d[30];
        std::memset(d, 5, sizeof d);
        Build(dist, d, 30);
        fixedLit = &lit;
        fixedDist = &dist;
        return true;
    }();
    (void)fixedBuilt;
    Huffman lit, dist;
    for (bool last = false; !last;)
    {
        last = b.Get(1) != 0;
        const uint32_t type = b.Get(2);
        if (type == 0)
        {
            // Stored: to a byte boundary, then LEN, NLEN and the bytes.
            b.Drop(b.count & 7);
            const uint32_t len = b.Get(16), nlen = b.Get(16);
            if ((len ^ 0xFFFF) != nlen || made + len > outSize)
                return SIZE_MAX;
            for (uint32_t i = 0; i < len; i++)
                out[made++] = uint8_t(b.Get(8));
        }
        else if (type == 1 || type == 2)
        {
            const Huffman* l = fixedLit;
            const Huffman* d = fixedDist;
            if (type == 2)
            {
                const int nlen = int(b.Get(5)) + 257, ndist = int(b.Get(5)) + 1, ncode = int(b.Get(4)) + 4;
                if (nlen > 286 || ndist > 30)
                    return SIZE_MAX;
                static constexpr uint8_t kOrder[19] = { 16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15 };
                uint8_t lengths[320] = {};
                for (int i = 0; i < ncode; i++)
                    lengths[kOrder[i]] = uint8_t(b.Get(3));
                Huffman codeLen;
                if (!Build(codeLen, lengths, 19))
                    return SIZE_MAX;
                std::memset(lengths, 0, sizeof lengths);
                for (int i = 0; i < nlen + ndist;)
                {
                    const int sym = Decode(b, codeLen);
                    if (sym < 0)
                        return SIZE_MAX;
                    if (sym < 16)
                    {
                        lengths[i++] = uint8_t(sym);
                        continue;
                    }
                    uint8_t value = 0;
                    int repeat;
                    if (sym == 16)
                    {
                        if (i == 0)
                            return SIZE_MAX;
                        value = lengths[i - 1];
                        repeat = 3 + int(b.Get(2));
                    }
                    else if (sym == 17)
                        repeat = 3 + int(b.Get(3));
                    else
                        repeat = 11 + int(b.Get(7));
                    if (i + repeat > nlen + ndist)
                        return SIZE_MAX;
                    while (repeat--)
                        lengths[i++] = value;
                }
                if (!Build(lit, lengths, nlen) || !Build(dist, lengths + nlen, ndist))
                    return SIZE_MAX;
                l = &lit;
                d = &dist;
            }
            for (;;)
            {
                int sym = Decode(b, *l);
                if (sym < 0 || b.overrun)
                    return SIZE_MAX;
                if (sym < 256)
                {
                    if (made >= outSize)
                        return SIZE_MAX;
                    out[made++] = uint8_t(sym);
                    continue;
                }
                if (sym == 256)
                    break;
                sym -= 257;
                if (sym >= 29)
                    return SIZE_MAX;
                const size_t len = kLengthBase[sym] + b.Get(kLengthExtra[sym]);
                const int ds = Decode(b, *d);
                if (ds < 0 || ds >= 30)
                    return SIZE_MAX;
                const size_t distance = kDistBase[ds] + b.Get(kDistExtra[ds]);
                if (distance > made || made + len > outSize)
                    return SIZE_MAX;
                uint8_t* dst = out + made;
                const uint8_t* src = dst - distance;
                if (distance >= len)
                    std::memcpy(dst, src, len);
                else
                    for (size_t i = 0; i < len; i++)
                        dst[i] = src[i];
                made += len;
            }
        }
        else
            return SIZE_MAX;
        if (b.overrun)
            return SIZE_MAX;
    }
    return made;
}

} // namespace ac6::game
