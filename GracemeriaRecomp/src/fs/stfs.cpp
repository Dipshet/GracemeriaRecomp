// STFS content packages, written from the format's public description.
//
// Layout: a header (signature, licences, the header's SHA-1 at 0x32C, metadata from 0x344 with the
// volume descriptor at 0x379), then 4 KB blocks from the header size rounded up to 4 KB. Data blocks are
// interleaved with hash tables: one table of 170 entries (SHA-1, status, next block) before every 170
// data blocks, a level-1 table (hashing level-0 tables) once there are more than 170 blocks and a
// level-2 table past 170 * 170, each placed where it is first needed. Packages written with two copies
// of every table (block separation bit 0 clear) keep the active copy's index in the parent entry's
// status (0x40). The top table is hashed in the volume descriptor. The file table (64-byte entries:
// name, flags, block counts, first block, parent directory index, size) lives in data blocks too, and
// files follow their blocks' "next" links unless they are marked contiguous.
#include "fs/stfs.h"

#include "log.h"
#include "platform/platform.h"

#include <algorithm>
#include <cstring>
#include <format>

#if defined(_WIN32)
#include <windows.h>
#include <bcrypt.h>
#endif

#include <memory_mapped_file.h>

namespace ac6::fs {

namespace {

constexpr uint64_t kBlock = 0x1000;
constexpr uint64_t kPerTable = 170;
constexpr uint64_t kPerL1 = kPerTable * kPerTable;
constexpr uint32_t kMagicCon = 0x434F4E20, kMagicLive = 0x4C495645, kMagicPirs = 0x50495253;
constexpr uint32_t kNoBlock = 0xFFFFFF;

uint32_t Be32(const uint8_t* p) { return uint32_t(p[0]) << 24 | uint32_t(p[1]) << 16 | uint32_t(p[2]) << 8 | p[3]; }
uint32_t Be24(const uint8_t* p) { return uint32_t(p[0]) << 16 | uint32_t(p[1]) << 8 | p[2]; }
uint16_t Be16(const uint8_t* p) { return uint16_t(p[0] << 8 | p[1]); }
uint32_t Le24(const uint8_t* p) { return uint32_t(p[0]) | uint32_t(p[1]) << 8 | uint32_t(p[2]) << 16; }
uint16_t Le16(const uint8_t* p) { return uint16_t(p[0] | p[1] << 8); }

std::string Utf16BeToUtf8(const uint8_t* p, size_t maxChars)
{
    std::string out;
    for (size_t i = 0; i < maxChars; i++)
    {
        uint32_t c = Be16(p + i * 2);
        if (!c)
            break;
        if (c >= 0xD800 && c < 0xDC00 && i + 1 < maxChars)
        {
            c = 0x10000 + ((c - 0xD800) << 10) + (Be16(p + ++i * 2) - 0xDC00);
        }
        if (c < 0x80) out += char(c);
        else if (c < 0x800) { out += char(0xC0 | (c >> 6)); out += char(0x80 | (c & 0x3F)); }
        else if (c < 0x10000) { out += char(0xE0 | (c >> 12)); out += char(0x80 | ((c >> 6) & 0x3F)); out += char(0x80 | (c & 0x3F)); }
        else { out += char(0xF0 | (c >> 18)); out += char(0x80 | ((c >> 12) & 0x3F)); out += char(0x80 | ((c >> 6) & 0x3F)); out += char(0x80 | (c & 0x3F)); }
    }
    return out;
}

bool IsMagic(uint32_t m) { return m == kMagicCon || m == kMagicLive || m == kMagicPirs; }

uint64_t AlignBlock(uint64_t v) { return (v + kBlock - 1) & ~(kBlock - 1); }

bool ParseHeader(const uint8_t* d, uint64_t size, StfsHeader& h, std::string& error)
{
    if (size < 4 || !IsMagic(Be32(d)))
        return false;
    h.magic = Be32(d);
    if (size < 0x3AD)
    {
        error = std::format("truncated: {} bytes, not even a whole header", size);
        return false;
    }
    h.headerSize = Be32(d + 0x340);
    if (h.headerSize < 0x3AD || h.headerSize > 0x100000)
    {
        error = std::format("its header size ({:#x}) is invalid: the header is corrupt", h.headerSize);
        return false;
    }
    if (size < h.headerSize)
    {
        error = std::format("truncated: {} bytes, but its header alone is {}", size, h.headerSize);
        return false;
    }
    h.contentType = Be32(d + 0x344);
    h.titleId = Be32(d + 0x360);
    h.descriptorType = Be32(d + 0x3A9);
    std::memcpy(h.contentId.data(), d + 0x32C, 20);
    for (int i = 0; i < 16; i++)
    {
        const uint8_t* e = d + 0x22C + i * 0x10;
        const uint64_t licensee = uint64_t(Be32(e)) << 32 | Be32(e + 4);
        const uint32_t bits = Be32(e + 8), flags = Be32(e + 12);
        if (licensee || flags)
        {
            h.licenseEntries++;
            h.licenseBits |= bits;
        }
    }
    if (h.headerSize >= 0x411 + 0x80)
        h.displayName = Utf16BeToUtf8(d + 0x411, 0x40);
    return true;
}

// Where blocks are, for one of the two table layouts (t = blocks per table: 1, or 2 with two copies).
struct Layout
{
    uint64_t dataStart = 0;
    uint64_t t = 1;

    uint64_t Step0() const { return kPerTable + t; }                // a level-0 group: table + 170 blocks
    uint64_t Step1() const { return kPerTable * Step0() + t; }      // a level-1 group
    uint64_t DataBlock(uint64_t n) const
    {
        uint64_t p = n + t * (n / kPerTable + 1);
        if (n >= kPerTable)
            p += t * (n / kPerL1 + 1);
        if (n >= kPerL1)
            p += t;
        return p;
    }
    // The table of `level` covering data block n.
    uint64_t Table(int level, uint64_t n) const
    {
        if (level == 0)
        {
            if (n < kPerTable)
                return 0;
            uint64_t p = n / kPerTable * Step0() + t * (n / kPerL1 + 1);
            if (n >= kPerL1)
                p += t;
            return p;
        }
        if (level == 1)
            return n < kPerL1 ? Step0() : t + n / kPerL1 * Step1();
        return Step1();
    }
    uint64_t Offset(uint64_t physical) const { return dataStart + physical * kBlock; }
};

uint64_t Span(int level) { return level == 0 ? 1 : level == 1 ? kPerTable : kPerL1; }

struct StfsDevice : Device
{
    MemoryMappedFile file;
    std::vector<std::vector<uint64_t>> blocks;   // per file entry (Entry::offset): its blocks' file offsets

    const char* Kind() const override { return "STFS container"; }
    bool ReadOnly() const override { return true; }
    const char* FileSystemName() const override { return "STFS"; }

    uint64_t Read(const Entry& entry, uint64_t offset, void* destination, uint64_t length) override
    {
        if (entry.directory || offset >= entry.size)
            return 0;
        const auto& list = blocks[entry.offset];
        const uint64_t n = std::min(length, entry.size - offset);
        auto* out = static_cast<uint8_t*>(destination);
        uint64_t done = 0;
        while (done < n)
        {
            const uint64_t at = offset + done, index = at / kBlock, within = at % kBlock;
            if (index >= list.size())
                break;
            uint64_t chunk = std::min(n - done, kBlock - within);
            const uint64_t from = list[index] + within;
            if (from >= file.size())
                break;
            chunk = std::min<uint64_t>(chunk, file.size() - from);
            std::memcpy(out + done, file.data() + from, chunk);
            done += chunk;
        }
        return done;
    }
};

struct RawEntry
{
    std::string name;
    bool used = false, directory = false, contiguous = false;
    uint32_t validBlocks = 0, allocatedBlocks = 0, firstBlock = 0, size = 0;
    uint16_t parent = 0xFFFF;
};

} // namespace

#if !defined(_WIN32)
namespace {

// SHA-1 (FIPS 180-4) on Linux, a 64-byte block at a time. Windows has it in bcrypt.
uint32_t Rotl(uint32_t v, int n) { return (v << n) | (v >> (32 - n)); }

void Sha1Block(uint32_t h[5], const uint8_t* p)
{
    uint32_t w[80];
    for (int i = 0; i < 16; i++)
        w[i] = uint32_t(p[i * 4]) << 24 | uint32_t(p[i * 4 + 1]) << 16 | uint32_t(p[i * 4 + 2]) << 8 | p[i * 4 + 3];
    for (int i = 16; i < 80; i++)
        w[i] = Rotl(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1);
    uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4];
    for (int i = 0; i < 80; i++)
    {
        const uint32_t f = i < 20 ? (b & c) | (~b & d) : i < 40 ? b ^ c ^ d : i < 60 ? (b & c) | (b & d) | (c & d) : b ^ c ^ d;
        const uint32_t k = i < 20 ? 0x5A827999 : i < 40 ? 0x6ED9EBA1 : i < 60 ? 0x8F1BBCDC : 0xCA62C1D6;
        const uint32_t t = Rotl(a, 5) + f + e + k + w[i];
        e = d;
        d = c;
        c = Rotl(b, 30);
        b = a;
        a = t;
    }
    h[0] += a;
    h[1] += b;
    h[2] += c;
    h[3] += d;
    h[4] += e;
}

} // namespace
#endif

std::array<uint8_t, 20> Sha1(const void* data, size_t size)
{
#if !defined(_WIN32)
    uint32_t h[5] = { 0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0 };
    const auto* p = static_cast<const uint8_t*>(data);
    const size_t blocks = size / 64, rest = size % 64;
    for (size_t i = 0; i < blocks; i++)
        Sha1Block(h, p + i * 64);
    // The tail: the rest, 0x80, zeros and the length in bits (big-endian) to fill one or two blocks.
    uint8_t tail[128] = {};
    std::memcpy(tail, p + blocks * 64, rest);
    tail[rest] = 0x80;
    const size_t tailSize = rest < 56 ? 64 : 128;
    const uint64_t bits = uint64_t(size) * 8;
    for (int i = 0; i < 8; i++)
        tail[tailSize - 1 - i] = uint8_t(bits >> (i * 8));
    Sha1Block(h, tail);
    if (tailSize == 128)
        Sha1Block(h, tail + 64);
    std::array<uint8_t, 20> out{};
    for (int i = 0; i < 20; i++)
        out[i] = uint8_t(h[i / 4] >> (24 - (i % 4) * 8));
    return out;
#else
    static const BCRYPT_ALG_HANDLE algorithm = [] {
        BCRYPT_ALG_HANDLE h = nullptr;
        BCryptOpenAlgorithmProvider(&h, BCRYPT_SHA1_ALGORITHM, nullptr, 0);
        return h;
    }();
    std::array<uint8_t, 20> out{};
    BCRYPT_HASH_HANDLE hash = nullptr;
    if (algorithm && BCryptCreateHash(algorithm, &hash, nullptr, 0, nullptr, 0, 0) >= 0)
    {
        auto* p = static_cast<PUCHAR>(const_cast<void*>(data));
        for (size_t done = 0; done < size;)
        {
            const ULONG n = ULONG(std::min<size_t>(size - done, 1u << 30));
            BCryptHashData(hash, p + done, n, 0);
            done += n;
        }
        BCryptFinishHash(hash, out.data(), 20, 0);
        BCryptDestroyHash(hash);
    }
    return out;
#endif
}

std::string Hex(const uint8_t* data, size_t size)
{
    static const char digits[] = "0123456789ABCDEF";
    std::string s;
    for (size_t i = 0; i < size; i++)
    {
        s += digits[data[i] >> 4];
        s += digits[data[i] & 15];
    }
    return s;
}

bool ReadStfsHeader(const std::filesystem::path& path, StfsHeader& header, std::string& error)
{
    error.clear();
    platform::HostFile h;
    if (!h.Open(path, platform::HostFile::Mode::Read))
    {
        error = std::format("can't be opened ({})", platform::ErrorText(h.Error()));
        return false;
    }
    uint64_t size = 0;
    h.Size(size);
    std::vector<uint8_t> head(size_t(std::min<uint64_t>(size, 0x2000)));
    uint32_t got = 0;
    const bool ok = head.empty() || h.ReadAt(head.data(), uint32_t(head.size()), 0, got);
    const uint32_t readError = ok ? 0 : h.Error();
    h.Close();
    if (!ok)
    {
        error = std::format("can't be read ({})", platform::ErrorText(readError));
        return false;
    }
    if (got < 4 || !IsMagic(Be32(head.data())))
        return false;
    // Checked against the whole file's size. The fields read all sit in the first 8 KB.
    StfsHeader parsed;
    if (!ParseHeader(head.data(), size, parsed, error))
        return false;
    header = parsed;
    return true;
}

StfsOpenResult OpenStfs(const std::filesystem::path& path, const StfsHeader& header)
{
    StfsOpenResult result;
    auto device = std::make_shared<StfsDevice>();
    if (!device->file.open(path))
    {
        result.errors.push_back(std::format("can't be opened ({})", platform::ErrorText(platform::LastError())));
        return result;
    }
    const uint8_t* d = device->file.data();
    const uint64_t size = device->file.size();

    // The header's own hash (the content id): only the console's mount needs it, but a mismatch says
    // the header was edited (licences, typically) without rehashing.
    const uint64_t dataStart = AlignBlock(header.headerSize);
    if (size < dataStart)
    {
        result.errors.push_back(std::format("truncated: {} bytes, ends inside its header ({} bytes)", size, dataStart));
        return result;
    }
    if (Sha1(d + 0x344, dataStart - 0x344) != header.contentId && Sha1(d + 0x344, header.headerSize - 0x344) != header.contentId)
        result.warnings.push_back("its header hash doesn't match (edited without rehashing, e.g. a licence patch): "
                                  "a console would refuse it; the runtime doesn't need it");

    const uint8_t blockSeparation = d[0x37B];
    const uint32_t fileTableBlocks = Le16(d + 0x37C);
    const uint32_t fileTableBlock = Le24(d + 0x37E);
    const uint8_t* topHash = d + 0x381;
    const uint64_t allocated = Be32(d + 0x395), unallocated = Be32(d + 0x399);
    const uint64_t total = allocated + unallocated;
    if (total == 0 || total > kPerL1 * kPerTable)
    {
        result.errors.push_back(std::format("its block count ({}) is invalid: the volume descriptor is corrupt", total));
        return result;
    }

    // Find the layout whose top table matches the descriptor's hash: the table copies (from block
    // separation) and the tree's height (from the block count) are both tried, so a package written by
    // an unusual tool still opens when its tree is consistent.
    const int expectedLevel = total <= kPerTable ? 0 : total <= kPerL1 ? 1 : 2;
    const uint64_t preferredT = (blockSeparation & 1) ? 1 : 2;
    Layout layout{ dataStart, preferredT };
    int topLevel = expectedLevel;
    uint64_t topCopy = (preferredT == 2 && (blockSeparation & 2)) ? 1 : 0;
    bool topFound = false;
    for (uint64_t t : { preferredT, 3 - preferredT })
    {
        for (int level : { expectedLevel, 0, 1, 2 })
        {
            Layout l{ dataStart, t };
            const uint64_t base = level == 0 ? 0 : level == 1 ? l.Step0() : l.Step1();
            for (uint64_t copy = 0; copy < t && !topFound; copy++)
            {
                const uint64_t at = l.Offset(base + copy);
                if (at + kBlock <= size && std::memcmp(Sha1(d + at, kBlock).data(), topHash, 20) == 0)
                {
                    layout = l;
                    topLevel = level;
                    topCopy = copy;
                    topFound = true;
                }
            }
            if (topFound)
                break;
        }
        if (topFound)
            break;
    }
    if (!topFound)
        result.errors.push_back("its top hash table doesn't match the volume descriptor: the container is corrupt "
                                "(or truncated, or edited without rehashing)");
    else if (topLevel != expectedLevel)
        result.warnings.push_back(std::format("its hash tree is {} levels deep, the block count says {}", topLevel + 1, expectedLevel + 1));

    // Walk the tree: each data block's level-0 entry, and whether every table above it checked out.
    std::vector<const uint8_t*> entryOf(total, nullptr);
    std::vector<uint8_t> trusted(total, topFound ? 1 : 0);
    uint64_t badTables = 0, cutTables = 0, firstCut = 0;
    auto walk = [&](auto&& self, int level, uint64_t first, uint64_t tableAt) -> void {
        for (uint64_t i = 0; i < kPerTable; i++)
        {
            const uint64_t child = first + i * Span(level);
            if (child >= total)
                break;
            const uint8_t* entry = d + tableAt + i * 0x18;
            if (level == 0)
            {
                entryOf[child] = entry;
                continue;
            }
            const uint64_t copy = layout.t == 2 && (entry[20] & 0x40) ? 1 : 0;
            const uint64_t childAt = layout.Offset(layout.Table(level - 1, child) + copy);
            const uint64_t last = std::min(total, child + Span(level)) - 1;
            if (childAt + kBlock > size)
            {
                if (!cutTables++)
                    firstCut = child;
                continue;
            }
            if (std::memcmp(Sha1(d + childAt, kBlock).data(), entry, 20) != 0)
            {
                badTables++;
                if (badTables <= 4)
                    result.errors.push_back(std::format("the hash table for blocks {}-{} doesn't match its parent: corrupt", child, last));
                for (uint64_t b = child; b <= last; b++)
                    trusted[b] = 0;
            }
            self(self, level - 1, child, childAt);
        }
    };
    const uint64_t topAt = layout.Offset((topLevel == 0 ? 0 : topLevel == 1 ? layout.Step0() : layout.Step1()) + topCopy);
    if (topAt + kBlock > size)
    {
        result.errors.push_back("truncated: its top hash table is past the end of the file");
        return result;
    }
    walk(walk, topLevel, 0, topAt);
    if (badTables > 4)
        result.errors.push_back(std::format("... {} damaged hash tables in all", badTables));
    if (cutTables)
        result.errors.push_back(std::format("truncated: the file is {} bytes, it ends before block {} of {}", size, firstCut, total));

    // One data block: its file offset, whether it is inside the file, and its hash.
    enum class BlockState { Ok, Unverified, Bad, Missing };
    auto checkBlock = [&](uint64_t n, uint64_t bytesNeeded, uint64_t& at) {
        at = layout.Offset(layout.DataBlock(n));
        if (at + bytesNeeded > size)
            return BlockState::Missing;
        if (!entryOf[n] || !trusted[n])
            return BlockState::Unverified;
        if (at + kBlock > size || std::memcmp(Sha1(d + at, kBlock).data(), entryOf[n], 20) != 0)
            return BlockState::Bad;
        result.verifiedBlocks++;
        return BlockState::Ok;
    };
    auto nextOf = [&](uint64_t n) -> uint64_t { return entryOf[n] ? Be24(entryOf[n] + 21) : kNoBlock; };

    // The file table.
    std::vector<RawEntry> raw;
    {
        uint64_t b = fileTableBlock;
        std::vector<uint8_t> tableSeen(total, 0);   // a block met twice: the table's links loop
        for (uint32_t i = 0; i < fileTableBlocks; i++)
        {
            if (b >= total)
            {
                result.errors.push_back(std::format("its file table runs off the container (block {} of {}): corrupt", b, total));
                break;
            }
            if (tableSeen[b])
            {
                result.errors.push_back(std::format("its file table links back to block {}: corrupt", b));
                break;
            }
            tableSeen[b] = 1;
            uint64_t at = 0;
            const BlockState state = checkBlock(b, kBlock, at);
            if (state == BlockState::Missing)
            {
                result.errors.push_back("truncated: its file table is past the end of the file");
                break;
            }
            if (state == BlockState::Bad)
                result.errors.push_back(std::format("its file table (block {}) doesn't match its hash: corrupt", b));
            for (int k = 0; k < 64; k++)
            {
                const uint8_t* e = d + at + k * 0x40;
                RawEntry r;
                const uint8_t flags = e[0x28];
                const size_t nameLength = flags & 0x3F;
                if (nameLength)
                {
                    r.used = true;
                    r.name.assign(reinterpret_cast<const char*>(e), std::min<size_t>(nameLength, 0x28));
                    r.directory = flags & 0x80;
                    r.contiguous = flags & 0x40;
                    r.validBlocks = Le24(e + 0x29);
                    r.allocatedBlocks = Le24(e + 0x2C);
                    r.firstBlock = Le24(e + 0x2F);
                    r.parent = Be16(e + 0x32);
                    r.size = Be32(e + 0x34);
                }
                raw.push_back(std::move(r));
            }
            b = i + 1 < fileTableBlocks ? nextOf(b) : b;
        }
    }

    // The tree. Entries whose parent chain doesn't reach the root are reported and left out.
    std::vector<std::unique_ptr<Entry>> made(raw.size());
    device->root.directory = true;
    device->root.attributes = 0x10;
    for (size_t i = 0; i < raw.size(); i++)
    {
        const RawEntry& r = raw[i];
        if (!r.used)
            continue;
        bool reaches = false;
        size_t at = i;
        for (int depth = 0; depth < 64; depth++)
        {
            const uint16_t p = raw[at].parent;
            if (p == 0xFFFF)
            {
                reaches = true;
                break;
            }
            if (p >= raw.size() || !raw[p].used || !raw[p].directory)
                break;
            at = p;
        }
        if (!reaches)
        {
            result.errors.push_back(std::format("its file table entry '{}' has no valid parent folder: corrupt", r.name));
            continue;
        }
        auto e = std::make_unique<Entry>();
        e->name = r.name;
        e->key = ToLower(r.name);
        e->directory = r.directory;
        e->attributes = r.directory ? 0x10 : 0;
        e->size = r.directory ? 0 : r.size;
        made[i] = std::move(e);
    }

    // Files: their block chains, each block checked against its hash. chainSeen[b] is the file (index + 1) whose chain
    // last met block b. A block met twice in the same file's chain means the chain loops.
    std::vector<uint32_t> chainSeen(total, 0);
    for (size_t i = 0; i < raw.size(); i++)
    {
        if (!made[i] || raw[i].directory)
            continue;
        const RawEntry& r = raw[i];
        const uint64_t count = (uint64_t(r.size) + kBlock - 1) / kBlock;
        if (count > total)
        {
            result.errors.push_back(std::format("'{}' says it has {} bytes, more than the container holds: corrupt", r.name, r.size));
            made[i].reset();
            continue;
        }
        std::vector<uint64_t> list;
        list.reserve(count);
        uint64_t bad = 0, missing = 0;
        bool broken = false;
        if (count > r.validBlocks && r.validBlocks)
            result.warnings.push_back(std::format("'{}' needs {} blocks but lists {}", r.name, count, r.validBlocks));
        uint64_t b = r.firstBlock;
        for (uint64_t k = 0; k < count; k++)
        {
            if (b >= total || chainSeen[b] == uint32_t(i + 1))
            {
                broken = true;
                break;
            }
            chainSeen[b] = uint32_t(i + 1);
            const uint64_t bytes = k + 1 < count ? kBlock : r.size - k * kBlock;
            uint64_t at = 0;
            const BlockState state = checkBlock(b, bytes, at);
            if (state == BlockState::Missing)
            {
                missing = count - k;
                break;
            }
            bad += state == BlockState::Bad;
            list.push_back(at);
            if (k + 1 < count)
                b = r.contiguous ? b + 1 : nextOf(b);
        }
        if (broken)
            result.errors.push_back(std::format("'{}': its block chain leaves the container or loops after {} of {} blocks: corrupt", r.name, list.size(), count));
        if (missing)
            result.errors.push_back(std::format("'{}': truncated, its last {} of {} blocks are past the end of the file", r.name, missing, count));
        if (bad)
            result.damage.push_back(std::format("'{}': {} of its {} blocks don't match their hashes", r.name, bad, count));
        made[i]->offset = device->blocks.size();
        device->blocks.push_back(std::move(list));
    }

    // Attach children to parents (raw pointers stay valid as the unique_ptrs move).
    std::vector<Entry*> ptr(raw.size(), nullptr);
    for (size_t i = 0; i < raw.size(); i++)
        ptr[i] = made[i].get();
    for (size_t i = 0; i < raw.size(); i++)
    {
        if (!made[i])
            continue;
        Entry* parent = raw[i].parent == 0xFFFF ? &device->root : ptr[raw[i].parent];
        if (!parent)
            continue;   // its parent was rejected (reported above)
        made[i]->parent = parent;
        parent->children.push_back(std::move(made[i]));
    }
    device->Finish();
    if (device->root.children.empty())
        result.errors.push_back("it contains no files");

    if (const platform::PathInfo info = platform::StatPath(path); info.exists)
        device->timestamp = info.write;
    device->sectorSize = uint32_t(kBlock);
    device->totalBytes = size;
    result.device = std::move(device);
    return result;
}

} // namespace ac6::fs
