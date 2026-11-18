#pragma once
// The settings file: ac6_settings.toml next to the executable. Each setting stands for one of the
// AC6_* environment variables the runtime reads. A variable set in the environment overrides the file.
#include <filesystem>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ac6::settings {

// Reads the file (writes a new one holding only a header comment when there is none) and sets the variables it gives.
// First thing in main: before anything reads them, and before the log is open (Log() reports afterwards).
void Load(const std::filesystem::path& directory);
// What Load did, into the log.
void Log();

// The settings overlay (src/ui/overlay.cpp): every setting with its tab, label, offered values and default.
enum class ValueKind { Bool, Number, Text };
struct Entry
{
    const char* section;
    const char* key;
    const char* env;
    ValueKind kind;
    const char* help;
    const char* tab;       // "Game", "Display", "Graphics", "Controls", "Advanced controls", "Debug", or "" (not shown)
    const char* label;
    const char* choices;   // "value=Label|value=Label", or "" (free text, or a range for numbers)
    double min, max, step;
    const char* def;       // the default, as the variable's text ("1" / "0" for booleans)
    const char* group;     // the heading it is listed under in its tab
};
std::vector<Entry> Entries();
std::filesystem::path Path();
// The setting's value in the file, as the variable's text ("" when it isn't set there: the default applies).
std::string FileValue(const Entry& entry);
// What's wrong with the settings file when it doesn't parse (its line and the error), "" when it parses or is missing.
// While it doesn't parse, none of its settings apply and nothing is written into it.
std::string FileError();
// Writes the values into the file in place, keeping its comments and everything else. "" or the default takes the
// setting's line out, so the default applies.
bool Save(const std::vector<std::pair<Entry, std::string>>& values, std::string& error);
// The same for any line by section and key: `value` as TOML text, or "" to take the line out (an older file's
// commented template line is commented out again, with `def`). The keyboard and mouse bindings use it (input/kbm.cpp).
struct TomlLine
{
    std::string section, key, value, def;
};
bool SaveLines(const std::vector<TomlLine>& lines, std::string& error);
// The input binding sections, which hold no settings: [menu_keys] and [flight_keys] (input/kbm.cpp) and
// [advanced_bindings] (input/advanced.cpp). [mouse] isn't one: it holds settings and input/kbm.cpp's own keys.
bool IsInputSection(std::string_view section);
// Unsets the variables Load set from the file (before starting the game again: the new copy reads the file itself, and
// variables in its environment would override it).
void ForgetApplied();
// Its variable was set in the environment (not by Load): the file's value doesn't apply.
bool FromEnvironment(const Entry& entry);
// The variable was set by the settings file (its setting or its [env] section), not by the environment.
bool SetByFile(const char* env);

// Live changes (the settings overlay): the settings that take effect at once are read through Get,
// from any thread: the overlay's value when it set one ("" = unset: the default), else the environment's. Generation
// goes up with each change: readers that keep a value read it again then.
std::string Get(const char* env);
void SetLive(const char* env, const std::string& value);
uint64_t Generation();
// The variable's text for a setting's value as the overlay holds it (its aliases applied: "exclusive" -> "1").
std::string VariableText(const Entry& entry, const std::string& value);

// The profile name as the game can take it: only an Xbox gamertag's characters (A-Z, a-z, digits and spaces) are
// kept, the spaces at the ends are trimmed, and at most kProfileNameMax (15) characters remain. "" when nothing is
// left: the default name, Player, applies then.
constexpr size_t kProfileNameMax = 15;
bool ProfileNameChar(unsigned int c);
std::string CleanProfileName(std::string_view name);

} // namespace ac6::settings
