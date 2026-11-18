// AC6 runtime plugins (plugin/ac6_plugin.h): the loader, the Api, the hookable functions and the events.
#include "plugin/plugins.h"
#include "plugin/ac6_plugin.h"
#include "log.h"
#include "memory.h"
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

// A hooked call as plugins see it: the guest's registers, and the game's own function.
struct CallState
{
    PPCContext* ctx;
    uint8_t* base;
    PPCFunc* original;
};

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

CallState* State(Call* call) { return reinterpret_cast<CallState*>(call); }

uint64_t ApiR(Call* call, int n)
{
    const PPCRegister* r = R(*State(call)->ctx, n);
    return r ? r->u64 : 0;
}
void ApiSetR(Call* call, int n, uint64_t v)
{
    if (PPCRegister* r = R(*State(call)->ctx, n))
        r->u64 = v;
}
double ApiF(Call* call, int n)
{
    const PPCRegister* r = F(*State(call)->ctx, n);
    return r ? r->f64 : 0.0;
}
void ApiSetF(Call* call, int n, double v)
{
    if (PPCRegister* r = F(*State(call)->ctx, n))
        r->f64 = v;
}
void ApiOriginal(Call* call)
{
    CallState* s = State(call);
    s->original(*s->ctx, s->base);
}

// The hookable functions (ac6_plugin.h Hookable), in this order.
constexpr std::array<uint32_t, 6> kHookable = { kDriftStartCheck, kDriftEndCheck, kDriftUpdate, kDriftPitchAuthority, kDriftDecel,
    kCameraFollowAttitude };
std::array<Hook, kHookable.size()> s_hooks{};
std::array<std::string, kHookable.size()> s_hookOwners;
std::vector<EventHandler> s_events;
std::string s_loading;   // the plugin whose init is running (its hooks' owner, its name in the loader's log lines)

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

void ApiOnEvent(EventHandler handler)
{
    if (handler)
        s_events.push_back(handler);
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

} // namespace

void Load()
{
    const std::string off = settings::Get("AC6_PLUGINS");
    if (off == "0" || off == "false")
    {
        log::Info("Plugins: off (AC6_PLUGINS=0)");
        return;
    }
    s_api = { kApiVersion, g_memory.base, ApiR, ApiSetR, ApiF, ApiSetF, ApiOriginal, ApiHook, ApiOnEvent, ApiSetting, ApiGeneration, ApiLog,
        ApiTicks };
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
        SDL_SharedObject* object = SDL_LoadObject(reinterpret_cast<const char*>(file.u8string().c_str()));
        if (!object)
        {
            log::Error("Plugins: {} not loaded: {}", s_loading, SDL_GetError());
            continue;
        }
        const auto init = reinterpret_cast<Ac6PluginInit>(SDL_LoadFunction(object, "ac6_plugin_init"));
        if (!init)
        {
            log::Error("Plugins: {} has no ac6_plugin_init", s_loading);
            SDL_UnloadObject(object);
            continue;
        }
        const size_t eventsBefore = s_events.size();
        const bool ok = init(&s_api);
        log::Info("Plugins: {} {}", s_loading, ok ? "loaded" : "declined (its init returned false)");
        if (!ok)
        {
            // A declined plugin leaves nothing behind: its hooks and event handlers go, and so does its code.
            for (size_t i = 0; i < s_hooks.size(); i++)
                if (s_hooks[i] && s_hookOwners[i] == s_loading)
                {
                    s_hooks[i] = nullptr;
                    s_hookOwners[i].clear();
                }
            s_events.resize(eventsBefore);
            SDL_UnloadObject(object);
        }
    }
    s_loading = "?";
}

void Event(uint32_t event, int64_t a)
{
    for (EventHandler h : s_events)
        h(event, a);
}

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

} // namespace ac6::plugin

#define AC6_HOOKABLE(addr, index)                                                                                      \
    extern "C" PPC_FUNC(__imp__sub_##addr);                                                                             \
    PPC_FUNC(sub_##addr) { ac6::plugin::Run<index>(ctx, base, __imp__sub_##addr); }

AC6_HOOKABLE(822A5C90, 0)
AC6_HOOKABLE(822A5D78, 1)
AC6_HOOKABLE(822A5F08, 2)
AC6_HOOKABLE(822A5FB8, 3)
AC6_HOOKABLE(822A5E90, 4)
AC6_HOOKABLE(8225E368, 5)
