#pragma once
// AC6 runtime plugins (plugin/ac6_plugin.h): the runtime's side.
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ac6::plugin {

// Loads each mods/<mod>/*.dll (.so on Linux) and calls its ac6_mod_init (the native mod API, mods/ac6_mod_api.h)
// and/or ac6_plugin_init (ac6_plugin.h), before the game starts (from main). AC6_PLUGINS=0 loads none.
void Load();
// An event to the plugins' handlers (ac6_plugin.h Event).
void Event(uint32_t event, int64_t a);
// A DLL a Lua mod loaded (ac6.native.load) exporting ac6_plugin_init: initialised as a plugin named `name`, at any time
// (from the mod's scripts). False: its init declined (its hooks and handlers removed: the DLL itself stays loaded).
bool InitNative(const std::string& name, void* init);
// Such a plugin's Lua mod failed: its hooks and event handlers removed (the DLL stays loaded: its code may be running).
void RemoveNative(const std::string& name);

// The plugins the loader found ("<mod>/<file>"), whether each started, and why not.
struct PluginStatus
{
    std::string name;
    bool loaded = false;
    std::string error;
};
std::vector<PluginStatus> Status();
// The hooks a plugin has in place through this API (ac6_plugin.h).
size_t HookCountOf(const std::string& name);

} // namespace ac6::plugin
