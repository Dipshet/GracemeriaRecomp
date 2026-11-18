#pragma once
// Language packs: localization/<pack>/ next to the game, each a translation of the game's text. A pack holds
// language.toml (its name in the Language setting, and the game language it is translated from, which also gives its
// voices and cutscenes) and the replacements, laid out as a mod's: packages/ (the text banks, the missions' subtitle
// banks) and textures/ (pictures with words in them). Every pack found is offered in the Language setting after the
// game's own languages, and the first one is the default. The chosen pack's packages and textures are used whether
// or not mods are on.

#include <filesystem>
#include <string>
#include <vector>

namespace ac6::localization {

struct Pack
{
    std::string id;          // the folder's name, lower case: the Language setting's value
    std::string name;        // shown in the Language setting
    std::string base;        // the game language it is translated from: english, japanese, german, french, spanish, italian
    std::filesystem::path dir;
};

// The packs in the localization folder, in name order (read once).
const std::vector<Pack>& Packs();
// The pack the Language setting names, or the first pack when the setting is unset, null for one of the game's languages.
const Pack* Active();
// The game language to run: the setting's value, or the active pack's base ("" = the default, English).
std::string GameLanguage();
// The folder the packs are read from.
std::filesystem::path Directory();

} // namespace ac6::localization
