// Cutscene subtitles re-timed for Force Japanese Audio: see game/captions.h.
#include "game/captions.h"
#include "game/language.h"
#include "game/packages.h"
#include "endian.h"
#include "log.h"
#include "memory.h"

#include <algorithm>
#include <bit>
#include <cstring>
#include <map>
#include <mutex>
#include <set>
#include <vector>

namespace ac6::game::captions {

namespace {

uint32_t Be32(const uint8_t* p) { return uint32_t(p[0]) << 24 | uint32_t(p[1]) << 16 | uint32_t(p[2]) << 8 | p[3]; }

void Put32(uint8_t* p, uint32_t v)
{
    p[0] = uint8_t(v >> 24);
    p[1] = uint8_t(v >> 16);
    p[2] = uint8_t(v >> 8);
    p[3] = uint8_t(v);
}

// An FHM's item k, or nothing.
struct Span
{
    uint8_t* p = nullptr;
    size_t size = 0;
};

bool IsFhm(Span s)
{
    return s.p && s.size >= 0x14 && std::memcmp(s.p, "FHM ", 4) == 0 && 0x14 + 16ull * Be32(s.p + 0x10) <= s.size;
}

uint32_t Count(Span s)
{
    return IsFhm(s) ? Be32(s.p + 0x10) : 0;
}

Span Item(Span s, uint32_t k)
{
    const uint32_t n = Count(s);
    if (k >= n)
        return {};
    const uint32_t offset = Be32(s.p + 0x14 + 4 * k), size = Be32(s.p + 0x14 + 4 * (n + k));
    if (!size || uint64_t(offset) + size > s.size)
        return {};
    return { s.p + offset, size };
}

// A caption slot's times (its item 2/3), or nothing when `s` isn't a slot.
Span Times(Span s)
{
    const Span capt = Item(s, 0);
    if (capt.size < 4 || std::memcmp(capt.p, "CAPT", 4) != 0)
        return {};
    const Span times = Item(Item(s, 2), 3);
    return times.size >= 64 && times.size % 32 == 0 ? times : Span{};
}

struct Line
{
    float start, end;
};

std::vector<Line> ReadTimes(Span t)
{
    std::vector<Line> lines;
    for (size_t at = 32; at + 32 <= t.size; at += 32)
        lines.push_back({ std::bit_cast<float>(Be32(t.p + at + 4)), std::bit_cast<float>(Be32(t.p + at + 8)) });
    return lines;
}

// The offset of the Japanese slot's second copy (timed to the English voices): the game's own constant.
float SecondCopyOffset()
{
    static const float offset = [] {
        const float v = std::bit_cast<float>(g_memory.As<be<uint32_t>>(0x82069DA4)->get());
        const bool ok = v > 100.0f && v < 5000.0f;
        log::Info("Force Japanese Audio: the Japanese subtitles' English-voice copy starts {} s on{}", ok ? v : 600.0f,
                  ok ? "" : " (the game's constant unreadable: assumed)");
        return ok ? v : 600.0f;
    }();
    return offset;
}

float Overlap(Line a, Line b)
{
    return std::max(0.0f, std::min(a.end, b.end) - std::max(a.start, b.start));
}

// `lines` (one slot) re-timed from the Japanese voices' lines: each Japanese line goes to the line it overlaps most (with
// every other line it covers a third of, a Japanese line spoken over several of them shared by their lengths); a
// line's new span is that of its Japanese lines; a line no Japanese line overlaps keeps its times. Returns how many moved.
uint32_t Retime(const std::vector<Line>& japanese, std::vector<Line>& lines)
{
    std::map<std::vector<size_t>, std::vector<Line>> owners;
    for (const Line& j : japanese)
    {
        if (j.end <= j.start)
            continue;
        size_t best = SIZE_MAX;
        float bestOverlap = 0.0f;
        std::vector<size_t> sharers;
        for (size_t i = 0; i < lines.size(); i++)
        {
            const float o = Overlap(lines[i], j);
            if (o > bestOverlap)
            {
                bestOverlap = o;
                best = i;
            }
            if (o > 0.0f && o >= (lines[i].end - lines[i].start) / 3.0f)
                sharers.push_back(i);
        }
        if (best == SIZE_MAX)
            continue;
        if (std::find(sharers.begin(), sharers.end(), best) == sharers.end())
            sharers = { best };
        owners[sharers].push_back(j);
    }
    std::map<size_t, Line> spans;
    auto extend = [&](size_t i, Line s) {
        auto [it, added] = spans.try_emplace(i, s);
        if (!added)
            it->second = { std::min(it->second.start, s.start), std::max(it->second.end, s.end) };
    };
    for (const auto& [sharers, js] : owners)
    {
        Line all{ js[0].start, js[0].end };
        for (const Line& j : js)
            all = { std::min(all.start, j.start), std::max(all.end, j.end) };
        if (sharers.size() == 1)
        {
            extend(sharers[0], all);
            continue;
        }
        std::vector<size_t> order = sharers;
        std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return lines[a].start < lines[b].start; });
        float total = 0.0f;
        for (size_t i : order)
            total += lines[i].end - lines[i].start;
        float t = all.start;
        for (size_t i : order)
        {
            const float d = total > 0.0f ? (all.end - all.start) * (lines[i].end - lines[i].start) / total : 0.0f;
            extend(i, { t, t + d });
            t += d;
        }
    }
    const std::vector<Line> original = lines;
    for (const auto& [i, s] : spans)
        lines[i] = s;
    // In order and apart: a line that followed another still starts once that one has ended.
    std::vector<size_t> order(lines.size());
    for (size_t i = 0; i < order.size(); i++)
        order[i] = i;
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return original[a].start < original[b].start; });
    for (size_t k = 1; k < order.size(); k++)
    {
        const size_t prev = order[k - 1], i = order[k];
        if (original[i].start >= original[prev].end && lines[i].start < lines[prev].end)
        {
            lines[i].start = lines[prev].end;
            lines[i].end = std::max(lines[i].end, lines[i].start + 0.5f);
        }
    }
    return uint32_t(spans.size());
}

// The Japanese voices' lines of a Japanese slot's times (the first copy).
std::vector<Line> JapaneseLines(Span times)
{
    std::vector<Line> japanese = ReadTimes(times);
    const float offset = SecondCopyOffset();
    std::erase_if(japanese, [&](const Line& l) { return l.start >= offset; });
    return japanese;
}

// Slot `s` re-timed in place from the Japanese lines. Returns the lines moved.
uint32_t RetimeSlot(Span s, const std::vector<Line>& japanese)
{
    const Span times = Times(s);
    if (!times.p)
        return 0;
    std::vector<Line> lines = ReadTimes(times);
    const uint32_t moved = Retime(japanese, lines);
    for (size_t i = 0; i < lines.size(); i++)
    {
        uint8_t* r = times.p + 32 * (i + 1);
        Put32(r + 4, std::bit_cast<uint32_t>(lines[i].start));
        Put32(r + 8, std::bit_cast<uint32_t>(lines[i].end));
    }
    return moved;
}

// Caption groups found under `s` (an FHM whose items 0 and 1 are slots), each re-timed. Returns the lines moved.
uint32_t Walk(Span s, int depth, uint32_t& groups)
{
    const uint32_t n = Count(s);
    if (n >= 2 && Times(Item(s, 0)).p && Times(Item(s, 1)).p)
    {
        const std::vector<Line> japanese = JapaneseLines(Times(Item(s, 0)));
        uint32_t moved = 0;
        for (uint32_t k = 1; k < n; k++)
            moved += RetimeSlot(Item(s, k), japanese);
        groups++;
        return moved;
    }
    uint32_t moved = 0;
    if (depth < 4)
        for (uint32_t k = 0; k < n; k++)
            if (const Span item = Item(s, k); IsFhm(item))
                moved += Walk(item, depth + 1, groups);
    return moved;
}

// Package 466's Japanese lines (a slot of its own: the other languages' are packages 467 to 472), unpacked once.
const std::vector<Line>& Package466Lines()
{
    static const std::vector<Line> lines = [] {
        std::vector<uint8_t> bytes;
        if (!UnpackPackage(466, bytes))
            return std::vector<Line>{};
        const Span root{ bytes.data(), bytes.size() };
        return JapaneseLines(Times(Item(root, 0)));
    }();
    return lines;
}

} // namespace

void OnUnpacked(uint32_t index, uint8_t* data, size_t size)
{
    if (!ForceJapaneseAudio())
        return;
    const Span root{ data, size };
    if (!IsFhm(root))
        return;
    uint32_t groups = 0, moved = 0;
    if (index >= 467 && index <= 472 && Times(Item(root, 0)).p)
    {
        if (!Package466Lines().empty())
        {
            moved = RetimeSlot(Item(root, 0), Package466Lines());
            groups = 1;
        }
    }
    else
        moved = Walk(root, 0, groups);
    if (!groups)
        return;
    static std::mutex mutex;
    static std::set<uint32_t> logged;
    std::lock_guard lock(mutex);
    if (logged.insert(index).second)
        log::Info("Force Japanese Audio: package {}'s subtitles re-timed to the Japanese voices ({} group(s), {} line(s))", index,
                  groups, moved);
}

} // namespace ac6::game::captions
