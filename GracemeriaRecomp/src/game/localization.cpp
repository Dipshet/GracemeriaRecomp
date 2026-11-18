// Language packs (game/localization.h).
#include "game/localization.h"

#include "log.h"
#include "platform/platform.h"
#include "settings.h"

#include <algorithm>
#include <cctype>
#include <system_error>

#define TOML_EXCEPTIONS 0
#include <toml++/toml.hpp>

namespace ac6::localization {

namespace {

constexpr const char* kGameLanguages[] = { "english", "japanese", "german", "french", "spanish", "italian" };

std::string Lower(std::string s)
{
    for (char& c : s)
        c = char(std::tolower(uint8_t(c)));
    return s;
}

bool IsGameLanguage(const std::string& id)
{
    return std::find(std::begin(kGameLanguages), std::end(kGameLanguages), id) != std::end(kGameLanguages);
}

std::vector<Pack> Scan()
{
    std::vector<Pack> packs;
    std::error_code ec;
    const std::filesystem::path root = Directory();
    if (!std::filesystem::is_directory(root, ec))
        return packs;
    std::vector<std::filesystem::path> folders;
    for (const auto& e : std::filesystem::directory_iterator(root, ec))
        if (e.is_directory(ec))
            folders.push_back(e.path());
    std::sort(folders.begin(), folders.end());
    for (const std::filesystem::path& dir : folders)
    {
        const std::string folder = platform::PathUtf8(dir.filename());
        Pack pack;
        pack.id = Lower(folder);
        pack.name = folder;
        pack.base = "english";
        pack.dir = dir;
        const std::filesystem::path config = dir / "language.toml";
        const bool hasConfig = std::filesystem::is_regular_file(config, ec);
        if (!hasConfig && !std::filesystem::is_directory(dir / "packages", ec) && !std::filesystem::is_directory(dir / "textures", ec))
            continue;   // (not a pack: nothing in it the game would use)
        if (IsGameLanguage(pack.id) || pack.id.find_first_of("|=") != std::string::npos)
        {
            log::Warn("Localization: '{}' isn't used: a pack's folder can't be named like one of the game's languages, or "
                      "contain | or =", folder);
            continue;
        }
        if (hasConfig)
        {
            toml::parse_result parsed = toml::parse_file(platform::PathUtf8(config));
            if (!parsed)
            {
                log::Warn("Localization: {}: {} (line {}); the pack isn't used", platform::PathUtf8(config),
                    parsed.error().description(), parsed.error().source().begin.line);
                continue;
            }
            if (auto name = parsed.table()["name"].value<std::string>(); name && !name->empty() && name->find('|') == std::string::npos)
                pack.name = *name;
            if (auto base = parsed.table()["base"].value<std::string>())
            {
                if (IsGameLanguage(Lower(*base)))
                    pack.base = Lower(*base);
                else
                    log::Warn("Localization: {}: base = \"{}\" isn't one of the game's languages (english, japanese, german, "
                              "french, spanish, italian): english", folder, *base);
            }
        }
        packs.push_back(std::move(pack));
    }
    return packs;
}

} // namespace

std::filesystem::path Directory()
{
    return platform::ExecutableDirectory() / "localization";
}

const std::vector<Pack>& Packs()
{
    static const std::vector<Pack> packs = [] {
        std::vector<Pack> found = Scan();
        for (const Pack& p : found)
            log::Info("Localization: pack '{}' ({}, from the {} text) in {}", p.name, p.id, p.base, platform::PathUtf8(p.dir));
        return found;
    }();
    return packs;
}

const Pack* Active()
{
    const std::vector<Pack>& packs = Packs();
    const std::string value = Lower(settings::Get("AC6_LANGUAGE"));
    if (value.empty())
        return packs.empty() ? nullptr : &packs.front();
    for (const Pack& p : packs)
        if (p.id == value)
            return &p;
    return nullptr;
}

std::string GameLanguage()
{
    if (const Pack* pack = Active())
        return pack->base;
    return settings::Get("AC6_LANGUAGE");
}

} // namespace ac6::localization
