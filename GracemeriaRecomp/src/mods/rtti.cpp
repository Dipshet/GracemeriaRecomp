// The game's C++ classes from its MSVC RTTI (mods/rtti.h).
#include "mods/rtti.h"

#include "log.h"
#include "memory.h"
#include "ppc.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <mutex>
#include <unordered_map>

namespace ac6::mods::rtti {

namespace {

struct Index
{
    std::vector<Class> classes;
    std::unordered_map<std::string, size_t> byName;     // both spellings
    std::unordered_map<uint32_t, std::pair<size_t, uint32_t>> byVtable;   // vtable -> class, subobject offset
};

std::mutex s_mutex;
Index* s_index = nullptr;
const Index s_empty;   // before the image is loaded (early.lua): nothing, and nothing kept

uint32_t Word(uint32_t address)
{
    uint32_t v;
    std::memcpy(&v, g_memory.base + address, 4);
    return __builtin_bswap32(v);
}

constexpr uint32_t kImageBegin = uint32_t(PPC_IMAGE_BASE);
constexpr uint32_t kImageEnd = uint32_t(PPC_IMAGE_BASE + PPC_IMAGE_SIZE);

bool InImage(uint32_t a) { return a >= kImageBegin && a + 4 <= kImageEnd; }

bool IsFunction(uint32_t a)
{
    return a >= PPC_CODE_BASE && a < PPC_CODE_BASE + PPC_CODE_SIZE && (a & 3) == 0 && g_memory.Function(a) != nullptr;
}

void Build()
{
    const auto start = std::chrono::steady_clock::now();
    auto* index = new Index();
    const uint8_t* image = g_memory.base + kImageBegin;
    const uint32_t size = kImageEnd - kImageBegin;

    // Type descriptors: their names.
    std::unordered_map<uint32_t, size_t> byDescriptor;   // type descriptor -> class
    for (uint32_t i = 8; i + 5 < size; i++)
    {
        if (image[i] != '.' || image[i + 1] != '?' || image[i + 2] != 'A' || (image[i + 3] != 'V' && image[i + 3] != 'U'))
            continue;
        uint32_t end = i;
        while (end < size && image[end] != 0 && end - i < 512)
            end++;
        if (end >= size || image[end] != 0)
            continue;
        Class c;
        c.mangled.assign(reinterpret_cast<const char*>(image + i), end - i);
        c.name = Demangle(c.mangled);
        c.typeDescriptor = kImageBegin + i - 8;
        byDescriptor.emplace(c.typeDescriptor, index->classes.size());
        index->classes.push_back(std::move(c));
        i = end;
    }

    // Complete object locators (signature 0, the descriptor at +12), then the vtables after a word pointing at one.
    std::unordered_map<uint32_t, std::pair<size_t, uint32_t>> locators;   // locator -> class, offset
    for (uint32_t i = 12; i + 4 <= size; i += 4)
    {
        const uint32_t w = Word(kImageBegin + i);
        const auto d = byDescriptor.find(w);
        if (d == byDescriptor.end())
            continue;
        const uint32_t locator = kImageBegin + i - 12;
        if (Word(locator) != 0)
            continue;
        locators.emplace(locator, std::make_pair(d->second, Word(locator + 4)));
    }
    for (uint32_t i = 0; i + 8 <= size; i += 4)
    {
        const auto l = locators.find(Word(kImageBegin + i));
        if (l == locators.end())
            continue;
        const uint32_t vtable = kImageBegin + i + 4;
        if (!IsFunction(Word(vtable)))
            continue;
        index->byVtable.emplace(vtable, l->second);
        Vtable v{ vtable, l->second.second, 0 };
        index->classes[l->second.first].vtables.push_back(v);
    }
    // The slots: words pointing at functions, up to the next vtable (its locator word before it isn't one).
    for (Class& c : index->classes)
    {
        for (Vtable& v : c.vtables)
            for (uint32_t a = v.address; InImage(a) && IsFunction(Word(a)) && (a == v.address || !index->byVtable.contains(a)); a += 4)
                v.slots++;
        std::sort(c.vtables.begin(), c.vtables.end(), [](const Vtable& a, const Vtable& b) { return a.offset < b.offset; });
    }
    size_t vtables = 0;
    for (size_t i = 0; i < index->classes.size(); i++)
    {
        index->byName.emplace(index->classes[i].name, i);
        index->byName.emplace(index->classes[i].mangled, i);
        vtables += index->classes[i].vtables.size();
    }
    s_index = index;
    log::Info("RTTI: {} classes, {} vtables ({:.0f} ms)", index->classes.size(), vtables,
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
}

const Index& Get()
{
    std::lock_guard lock(s_mutex);
    if (!s_index)
    {
        if (Word(kImageBegin) >> 16 != 0x4D5A)   // "MZ": the image isn't loaded yet
            return s_empty;
        Build();
    }
    return *s_index;
}

} // namespace

std::string Demangle(std::string_view mangled)
{
    // .?AV / .?AU, then names innermost first, each ending in '@', the list ending in "@@". Templates (?$) are left
    // as MSVC spells them.
    std::string_view s = mangled;
    if (s.size() < 4 || !s.starts_with(".?A"))
        return std::string(mangled);
    s.remove_prefix(4);
    if (s.find("?$") != std::string_view::npos)
        return std::string(s.ends_with("@@") ? s.substr(0, s.size() - 2) : s);
    if (s.ends_with("@@"))
        s.remove_suffix(2);
    std::vector<std::string_view> parts;
    while (!s.empty())
    {
        const size_t at = s.find('@');
        parts.push_back(s.substr(0, at));
        if (at == std::string_view::npos)
            break;
        s.remove_prefix(at + 1);
    }
    std::string out;
    for (size_t i = parts.size(); i-- > 0;)
    {
        out += parts[i];
        if (i)
            out += "::";
    }
    return out;
}

const std::vector<Class>& Classes() { return Get().classes; }

const Class* Find(std::string_view name)
{
    const Index& index = Get();
    const auto it = index.byName.find(std::string(name));
    return it != index.byName.end() ? &index.classes[it->second] : nullptr;
}

const Class* Of(uint32_t object, uint32_t* offset)
{
    if (!g_memory.IsBacked(object) || !g_memory.IsBacked(object + 3))
        return nullptr;
    const uint32_t vtable = Word(object);
    if (!InImage(vtable))
        return nullptr;
    const Index& index = Get();
    const auto it = index.byVtable.find(vtable);
    if (it == index.byVtable.end())
        return nullptr;
    if (offset)
        *offset = it->second.second;
    return &index.classes[it->second.first];
}

} // namespace ac6::mods::rtti
