#pragma once
// AC6 runtime plugins (plugin/ac6_plugin.h): the runtime's side.
#include <cstdint>

namespace ac6::plugin {

// Loads each mods/<mod>/*.dll (.so on Linux) and calls its ac6_plugin_init, before the game starts (from main).
// AC6_PLUGINS=0 loads none.
void Load();
// An event to the plugins' handlers (ac6_plugin.h Event).
void Event(uint32_t event, int64_t a);

} // namespace ac6::plugin
