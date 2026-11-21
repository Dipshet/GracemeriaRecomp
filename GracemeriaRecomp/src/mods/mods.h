#pragma once
// Lua mods: mods/<mod>/ next to the executable, with a mod.toml and Lua scripts, run by LuaJIT (the JIT and the FFI on).
// docs/modding/lua.md is the modders' guide, src/mods/lua/ac6.lua the API scripts see.
//
//   mods/<mod>/mod.toml    the manifest: name, version, author, description, api (the Lua API version the mod is
//                          written for: newer than the runtime's and it isn't loaded), load_order (lower first, then the
//                          folders' names), enabled
//   mods/<mod>/early.lua   run before the window and the Vulkan device are made (a mod that changes how they're made)
//   mods/<mod>/init.lua    run before the game's code starts
//
// Each mod has a Lua state of its own and a lock: its hooks and events run one at a time, whatever the thread. A script
// that fails (a load error, an error in early.lua or init.lua) leaves the mod off: its hooks are removed, those of the
// DLLs it loaded too (their event handlers with them; the DLLs themselves stay loaded). Errors in a hook or an event
// handler are logged; a hook failing again and again is removed.
//
// Native code: scripts can load DLLs (ac6.native.load, LuaJIT's ffi.load) and the FFI reaches all of the process: a Lua
// mod can do anything a program can. Run only mods you trust. AC6_LUA=0 loads no Lua mods, AC6_LUA_NATIVE=0 lets them
// load no DLLs, AC6_LUA_DISABLE=<mod>[,<mod>...] leaves those out.
//
// A DLL is started (its ac6_mod_init / ac6_plugin_init run) once, whoever loads it: the plugin loader starts the DLLs at
// the top of each mods/<mod>/ folder, so a Lua mod keeps the ones its scripts load elsewhere (bin/).
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

struct ac6_mod_api;

namespace ac6::mods {

constexpr uint32_t kLuaApiVersion = 2;   // 2: input (ac6.input, the "pad" event) and drawing (ac6.ui, the "draw" event)

// The mods found and their early.lua run: on the main thread, before the window and the Vulkan device are made.
void LoadEarly();
// Their init.lua: before the game's code starts (after the native plugins load).
void LoadInit();

// Events to the mods (ac6.events). OnFrame: on the game's main thread at each presented frame, with its guest context
// (handlers may call the game's functions). OnMode: a mode task entered (game/game_modes.h).
void OnFrame();
void OnMode(const char* mode);
// The mods' drawing (ui/overlay.h ac6::ui::draw): whether any has a draw handler, and the handlers, from the overlay's
// frame (the window's thread: the game's main thread, at each present, before the pad filter and OnFrame).
bool WantsDraw();
void OnDraw();

// The native mod API (mods/ac6_mod_api.h) for a plugin: a DLL in mods/<mod>/ the plugin loader found exporting
// ac6_mod_init (plugin/plugins.cpp). `name` is "<mod>/<file>" (its log lines' and hooks' owner), `dir` its folder, its
// `lua` null. Kept for the process's life. DropPluginNativeApi: its init declined, what it registered goes.
const ac6_mod_api* PluginNativeApi(const std::string& name, const std::string& file, const std::filesystem::path& dir);
void DropPluginNativeApi(const ac6_mod_api* api);

// The DLLs started, by file (the same file however its path is spelled). ClaimNative: `who` starts `file`; false when
// it already was started, `by` then who did. ReleaseNative: its init declined, it can be started again.
bool ClaimNative(const std::filesystem::path& file, const std::string& who, std::string* by = nullptr);
void ReleaseNative(const std::filesystem::path& file);

// The mods as the settings overlay lists them: the Lua mods, then the plugins the plugin loader started.
struct ModStatus
{
    std::string id, name, version, author, description;
    bool plugin = false;        // a plugin (a DLL in mods/<mod>/, no Lua)
    bool loaded = false;        // its scripts ran (a plugin: its init didn't decline)
    bool native = false;        // it loaded a DLL
    std::string error;          // why it isn't loaded, or its last error
    uint32_t hooks = 0;         // hooks in place
};
std::vector<ModStatus> Status();

} // namespace ac6::mods
