// AC6 runtime plugins (plugin/ac6_plugin.h): the loader, the Api, the hooks and the events.
//
// Hooks: with hook gates (the default build, mods/hooks.h) a plugin hooks any of the game's functions through the hook
// registry, its hook an entry in the function's chain (several plugins and mods can hook one function). Without them,
// only the six Hookable functions, one plugin each, through their replacements at the end of this file.
#include "plugin/plugins.h"
#include "plugin/ac6_plugin.h"
#include "log.h"
#include "memory.h"
#include "mods/ac6_mod_api.h"
#include "mods/hooks.h"
#include "mods/mods.h"
#include "platform/platform.h"
#include "ppc.h"
#include "settings.h"

#include <SDL3/SDL.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

namespace ac6::plugin {

namespace {

// A hooked call as plugins see it: the registry's call (with hook gates), or the replacement's state (without).
#if defined(PPC_HOOK_GATES)
PPCContext& Ctx(Call* call) { return reinterpret_cast<mods::HookCall*>(call)->ctx; }
#else
struct CallState
{
    PPCContext* ctx;
    uint8_t* base;
    PPCFunc* original;
};
CallState* State(Call* call) { return reinterpret_cast<CallState*>(call); }
PPCContext& Ctx(Call* call) { return *State(call)->ctx; }
#endif

PPCRegister* R(PPCContext& c, int n)
{
    switch (n)
    {
    case 3: return &c.r3; case 4: return &c.r4; case 5: return &c.r5; case 6: return &c.r6;
    case 7: return &c.r7; case 8: return &c.r8; case 9: return &c.r9; case 10: return &c.r10;
    default: return nullptr;
    }
}

PPCRegister* F(PPCContext& c, int n)
{
    switch (n)
    {
    case 1: return &c.f1; case 2: return &c.f2; case 3: return &c.f3; case 4: return &c.f4; case 5: return &c.f5;
    case 6: return &c.f6; case 7: return &c.f7; case 8: return &c.f8; case 9: return &c.f9; case 10: return &c.f10;
    case 11: return &c.f11; case 12: return &c.f12; case 13: return &c.f13;
    default: return nullptr;
    }
}

uint64_t ApiR(Call* call, int n)
{
    const PPCRegister* r = R(Ctx(call), n);
    return r ? r->u64 : 0;
}
void ApiSetR(Call* call, int n, uint64_t v)
{
    if (PPCRegister* r = R(Ctx(call), n))
        r->u64 = v;
}
double ApiF(Call* call, int n)
{
    const PPCRegister* r = F(Ctx(call), n);
    return r ? r->f64 : 0.0;
}
void ApiSetF(Call* call, int n, double v)
{
    if (PPCRegister* r = F(Ctx(call), n))
        r->f64 = v;
}
void ApiOriginal(Call* call)
{
#if defined(PPC_HOOK_GATES)
    mods::CallNext(*reinterpret_cast<mods::HookCall*>(call));   // the rest of the chain: other hooks, then the game's
#else
    CallState* s = State(call);
    s->original(*s->ctx, s->base);
#endif
}

// The event handlers, and which plugin's init registered each (to remove a plugin's).
struct Handler
{
    EventHandler fn;
    std::string owner;
};
std::vector<Handler> s_events;
std::string s_loading;   // the plugin whose init is running (its hooks' owner, its name in the loader's log lines)
std::vector<PluginStatus> s_status;   // the plugins the loader found

void RemoveEventsOf(const std::string& plugin)
{
    std::erase_if(s_events, [&](const Handler& h) { return h.owner == plugin; });
}

#if defined(PPC_HOOK_GATES)
std::string Owner(const std::string& plugin) { return "plugin " + plugin; }

void RunPluginHook(mods::HookCall& call, void* hook)
{
    reinterpret_cast<Hook>(hook)(reinterpret_cast<Call*>(&call));
}

// Any gated function. False: not a function start (the registry logs it).
bool ApiHook(uint32_t function, Hook hook)
{
    if (!hook)
        return false;
    return mods::AddHook(function, RunPluginHook, reinterpret_cast<void*>(hook), 0, Owner(s_loading)) != 0;
}
#else
// The hookable functions (ac6_plugin.h Hookable), in this order.
constexpr std::array<uint32_t, 6> kHookable = { kDriftStartCheck, kDriftEndCheck, kDriftUpdate, kDriftPitchAuthority, kDriftDecel,
    kCameraFollowAttitude };
std::array<Hook, kHookable.size()> s_hooks{};
std::array<std::string, kHookable.size()> s_hookOwners;

bool ApiHook(uint32_t function, Hook hook)
{
    for (size_t i = 0; i < kHookable.size(); i++)
        if (kHookable[i] == function)
        {
            if (s_hooks[i])
            {
                log::Warn("Plugins: {} asked for {:08X}, which {} has", s_loading, function, s_hookOwners[i]);
                return false;
            }
            s_hooks[i] = hook;
            s_hookOwners[i] = s_loading;
            return true;
        }
    log::Warn("Plugins: {} asked for {:08X}, which isn't hookable", s_loading, function);
    return false;
}
#endif

void ApiOnEvent(EventHandler handler)
{
    if (handler)
        s_events.push_back({ handler, s_loading });
}

size_t ApiSetting(const char* name, char* buf, size_t size)
{
    const std::string v = name ? settings::Get(name) : std::string();
    if (buf && size)
    {
        const size_t n = std::min(v.size(), size - 1);
        std::memcpy(buf, v.data(), n);
        buf[n] = 0;
    }
    return v.size();
}

uint64_t ApiGeneration() { return settings::Generation(); }

void ApiLog(int level, const char* text)
{
    // Called from a plugin's code at any time, when the runtime doesn't know which plugin is calling: plugins name
    // themselves in their text.
    const std::string t = text ? text : "";
    if (level >= kError)
        log::Error("[plugin] {}", t);
    else if (level == kWarn)
        log::Warn("[plugin] {}", t);
    else
        log::Info("[plugin] {}", t);
}

uint64_t ApiTicks() { return SDL_GetTicks(); }

Api s_api;

const Api* GetApi()
{
    if (s_api.version == 0)
        s_api = { kApiVersion, g_memory.base, ApiR, ApiSetR, ApiF, ApiSetF, ApiOriginal, ApiHook, ApiOnEvent, ApiSetting, ApiGeneration,
            ApiLog, ApiTicks };
    return &s_api;
}

// A declined plugin's hooks go with it.
void RemoveHooks(const std::string& plugin)
{
#if defined(PPC_HOOK_GATES)
    mods::RemoveHooksOf(Owner(plugin));
#else
    for (size_t i = 0; i < s_hooks.size(); i++)
        if (s_hooks[i] && s_hookOwners[i] == plugin)
        {
            s_hooks[i] = nullptr;
            s_hookOwners[i].clear();
        }
#endif
}

} // namespace

void Load()
{
    const std::string off = settings::Get("AC6_PLUGINS");
    if (off == "0" || off == "false")
    {
        log::Info("Plugins: off (AC6_PLUGINS=0)");
        return;
    }
    GetApi();
    const std::filesystem::path root = platform::ExecutableDirectory() / "mods";
    std::error_code ec;
    if (!std::filesystem::is_directory(root, ec))
        return;
#if defined(_WIN32)
    const char* kExtension = ".dll";
#else
    const char* kExtension = ".so";
#endif
    std::vector<std::filesystem::path> files;
    for (const auto& mod : std::filesystem::directory_iterator(root, ec))
        if (mod.is_directory(ec))
            for (const auto& file : std::filesystem::directory_iterator(mod.path(), ec))
                if (file.is_regular_file(ec) && file.path().extension() == kExtension)
                    files.push_back(file.path());
    std::sort(files.begin(), files.end());
    for (const std::filesystem::path& file : files)
    {
        s_loading = platform::PathUtf8(file.parent_path().filename()) + "/" + platform::PathUtf8(file.filename());
        // (A Lua mod's early.lua may have started it already: ac6.native.load.)
        if (std::string by; !mods::ClaimNative(file, "the plugin loader", &by))
        {
            log::Warn("Plugins: {} left alone: {} started it already", s_loading, by);
            continue;
        }
        PluginStatus& status = s_status.emplace_back(PluginStatus{ s_loading, false, {} });
        SDL_SharedObject* object = SDL_LoadObject(reinterpret_cast<const char*>(file.u8string().c_str()));
        if (!object)
        {
            status.error = SDL_GetError();
            log::Error("Plugins: {} not loaded: {}", s_loading, status.error);
            mods::ReleaseNative(file);
            continue;
        }
        // ac6_mod_init: the native mod API (mods/ac6_mod_api.h); ac6_plugin_init: this file's. A plugin exports one or
        // both, each called in that order. One declining unloads it.
        const auto modInit = reinterpret_cast<ac6_mod_init_fn>(SDL_LoadFunction(object, "ac6_mod_init"));
        const auto init = reinterpret_cast<Ac6PluginInit>(SDL_LoadFunction(object, "ac6_plugin_init"));
        if (!init && !modInit)
        {
            status.error = "no ac6_mod_init or ac6_plugin_init";
            log::Error("Plugins: {} has no ac6_mod_init or ac6_plugin_init", s_loading);
            SDL_UnloadObject(object);
            mods::ReleaseNative(file);
            continue;
        }
        const ac6_mod_api* modApi = nullptr;
        bool ok = true;
        if (modInit)
        {
            modApi = mods::PluginNativeApi(s_loading, platform::PathUtf8(file.filename()), file.parent_path());
            ok = modInit(modApi) != 0;
        }
        if (ok && init)
            ok = init(&s_api);
        log::Info("Plugins: {} {}", s_loading, ok ? "loaded" : "declined (its init returned false)");
        status.loaded = ok;
        if (!ok)
        {
            // A declined plugin leaves nothing behind: its hooks and event handlers go, and so does its code.
            status.error = "its init declined";
            if (modApi)
                mods::DropPluginNativeApi(modApi);
            RemoveHooks(s_loading);
            RemoveEventsOf(s_loading);
            SDL_UnloadObject(object);
            mods::ReleaseNative(file);
        }
    }
    s_loading = "?";
}

void Event(uint32_t event, int64_t a)
{
    for (const Handler& h : s_events)
        h.fn(event, a);
}

bool InitNative(const std::string& name, void* init)
{
    const std::string previous = s_loading;
    s_loading = name;
    const bool ok = reinterpret_cast<Ac6PluginInit>(init)(GetApi());
    log::Info("Plugins: {} {} (loaded by its Lua mod)", name, ok ? "initialised" : "declined (its init returned false)");
    if (!ok)
    {
        RemoveHooks(name);
        RemoveEventsOf(name);
    }
    s_loading = previous;
    return ok;
}

void RemoveNative(const std::string& name)
{
    RemoveHooks(name);
    RemoveEventsOf(name);
}

std::vector<PluginStatus> Status() { return s_status; }

size_t HookCountOf(const std::string& name)
{
#if defined(PPC_HOOK_GATES)
    return mods::HookCountOf(Owner(name));
#else
    size_t n = 0;
    for (size_t i = 0; i < s_hooks.size(); i++)
        n += s_hooks[i] && s_hookOwners[i] == name;
    return n;
#endif
}

#if !defined(PPC_HOOK_GATES)
namespace {
// The replacements for the hookable functions: each runs the plugin's hook when one is registered, else the game's
// own function.
template <size_t I>
void Run(PPCContext& ctx, uint8_t* base, PPCFunc* original)
{
    if (const Hook h = s_hooks[I])
    {
        CallState state{ &ctx, base, original };
        h(reinterpret_cast<Call*>(&state));
    }
    else
        original(ctx, base);
}
} // namespace
#endif

} // namespace ac6::plugin

#if !defined(PPC_HOOK_GATES)
#define AC6_HOOKABLE(addr, index)                                                                                      \
    extern "C" PPC_FUNC(__imp__sub_##addr);                                                                             \
    PPC_FUNC(sub_##addr) { ac6::plugin::Run<index>(ctx, base, __imp__sub_##addr); }

AC6_HOOKABLE(822A5C90, 0)
AC6_HOOKABLE(822A5D78, 1)
AC6_HOOKABLE(822A5F08, 2)
AC6_HOOKABLE(822A5FB8, 3)
AC6_HOOKABLE(822A5E90, 4)
AC6_HOOKABLE(8225E368, 5)
#endif
