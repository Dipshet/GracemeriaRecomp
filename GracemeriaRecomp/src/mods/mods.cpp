// Lua mods (mods/mods.h): finding them, their Lua states, the ac6.core functions the API script (mods/lua/ac6.lua)
// builds the mods' API on, hooks and events into Lua, and native modules.
//
// ac6.core is the API script's only way into the runtime: plain C functions on Lua values. The script makes what mods
// use of them (ac6.mem through the FFI on guest memory, the hooks' call objects, the event lists). Each mod's core
// functions carry the mod as their upvalue.
#define TOML_EXCEPTIONS 0
#include "mods/mods.h"

#include "cpu/guest_thread.h"
#include "endian.h"
#include "game/game_modes.h"
#include "gpu/renderer.h"
#include "heap.h"
#include "input/input.h"
#include "log.h"
#include "memory.h"
#include "mods/ac6_mod_api.h"
#include "mods/hooks.h"
#include "mods/rtti.h"
#include "platform/platform.h"
#include "plugin/plugins.h"
#include "ppc.h"
#include "settings.h"
#include "ui/overlay.h"

#include <lua.hpp>
#include <toml++/toml.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <cwctype>
#include <exception>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace ac6::gpu {
extern const char* const kModsLuaApi;   // mods/lua/ac6.lua (embedded, tools/cmake/embed_text.cmake)
extern const char* const kModsSymbols;  // mods/symbols.toml (embedded)
}

namespace ac6::mods {

namespace {

struct Mod
{
    std::string id;                 // the folder's name
    std::string name, version, author, description;
    std::filesystem::path dir;
    int64_t loadOrder = 0;
    int64_t api = kLuaApiVersion;
    lua_State* L = nullptr;
    std::recursive_mutex mutex;     // the Lua state: one hook or event at a time (recursive: a hook's call into the game
                                    // can reach another of the mod's hooks on the same thread)
    bool loaded = false;            // init.lua ran: events go to it
    bool failed = false;
    bool native = false;
    std::string error;
    int tracebackRef = LUA_NOREF;
    int eventSinkRef = LUA_NOREF;   // the API script's event dispatcher: (name, ...)
    int hookSinkRef = LUA_NOREF;    // its hook dispatcher: (hook function, call, context, function address)
    std::atomic<uint32_t> hooks{ 0 };
    // It has handlers for the draw and pad events (the API script says, core.want): only then is it called for them.
    std::atomic<bool> wantsDraw{ false }, wantsPad{ false };
    // Its native modules (ac6.native.load): what each gave its scripts (a registry reference), by file (NativeKey), and
    // the plugins they started (plugin::InitNative's names). Under the mod's lock.
    std::unordered_map<std::string, int> modules;
    std::vector<std::string> plugins;
};

std::vector<std::unique_ptr<Mod>> s_mods;

// A Lua hook: its mod and its function (a registry reference in the mod's state). A call on its way in when the hook is
// removed finds `alive` false and passes the call on; the record itself is freed by the hook registry (ReleaseLuaHook)
// once no call can reach it.
struct LuaHook
{
    Mod* mod = nullptr;
    int fnRef = LUA_NOREF;
    uint64_t id = 0;
    uint32_t function = 0;
    std::atomic<bool> alive{ true };
    uint32_t errors = 0;
};
// The Lua hooks in place, by id (a removed one leaves at once).
std::mutex s_luaHooksMutex;
std::unordered_map<uint64_t, LuaHook*> s_luaHooks;

void ReleaseLuaHook(void* user) { delete static_cast<LuaHook*>(user); }

// Takes a Lua hook out, under its mod's lock: off, unlisted, its function's reference dropped. False: it already was.
// The caller removes it from the registry after (RemoveHook: the record may be freed from then on).
bool TakeOut(LuaHook& h)
{
    if (!h.alive.exchange(false))
        return false;
    {
        std::lock_guard lock(s_luaHooksMutex);
        s_luaHooks.erase(h.id);
    }
    if (h.mod->L && h.fnRef != LUA_NOREF)
        luaL_unref(h.mod->L, LUA_REGISTRYINDEX, h.fnRef);
    h.fnRef = LUA_NOREF;
    h.mod->hooks--;
    return true;
}

// A hooked call as the mod's Lua sees it (a light userdata): whether its hook passed it on yet.
struct LuaCall
{
    HookCall* call;
    bool nexted;
};

// A guest thread ending (ExTerminateThread throws GuestThreadExit) inside the game's code a script called: LuaJIT's
// pcall would catch the C++ exception as a Lua error. Kept here instead, the script gets an error, and the runtime
// throws it again past the script.
thread_local std::exception_ptr t_guestExit;
void RethrowGuestExit()
{
    if (t_guestExit)
        std::rethrow_exception(std::exchange(t_guestExit, nullptr));
}

constexpr uint32_t kMaxHookErrors = 10;   // a hook failing this many times is removed

Mod& ModOf(lua_State* L) { return *static_cast<Mod*>(lua_touserdata(L, lua_upvalueindex(1))); }

std::string Owner(const Mod& m) { return "mod " + m.id; }

int Traceback(lua_State* L)
{
    const char* message = lua_tostring(L, 1);
    luaL_traceback(L, L, message ? message : "(an error that isn't a string)", 1);
    return 1;
}

// Calls the function under `nargs` arguments on the stack, with a traceback on error. Returns the error ("" none).
std::string ProtectedCall(Mod& m, int nargs, int nresults)
{
    lua_State* L = m.L;
    const int function = lua_gettop(L) - nargs;
    lua_rawgeti(L, LUA_REGISTRYINDEX, m.tracebackRef);
    lua_insert(L, function);
    const int status = lua_pcall(L, nargs, nresults, function);
    lua_remove(L, function);
    if (status == 0)
        return {};
    std::string error = lua_tostring(L, -1) ? lua_tostring(L, -1) : "(error)";
    lua_pop(L, 1);
    return error;
}

std::string ReadFile(const std::filesystem::path& path)
{
    std::ifstream file(path, std::ios::binary);
    std::stringstream s;
    s << file.rdbuf();
    return s.str();
}

uint32_t CheckAddress(lua_State* L, int index)
{
    const lua_Number n = luaL_checknumber(L, index);
    if (n < 0 || n > 4294967295.0)
        luaL_argerror(L, index, "not a 32-bit guest address");
    return uint32_t(n);
}

// --- ac6.core ---------------------------------------------------------------------------------------------------------

// core.log(level, text): 0 info, 1 warn, 2 error.
int CoreLog(lua_State* L)
{
    Mod& m = ModOf(L);
    const int level = int(luaL_checkinteger(L, 1));
    const char* text = luaL_checkstring(L, 2);
    if (level >= 2)
        log::Error("[mod {}] {}", m.id, text);
    else if (level == 1)
        log::Warn("[mod {}] {}", m.id, text);
    else
        log::Info("[mod {}] {}", m.id, text);
    return 0;
}

// core.setting(name) -> string ("" when unset): a setting or AC6_* variable as the runtime reads it.
int CoreSetting(lua_State* L)
{
    const std::string v = settings::Get(luaL_checkstring(L, 1));
    lua_pushlstring(L, v.data(), v.size());
    return 1;
}

int CoreInfo(lua_State* L)
{
    Mod& m = ModOf(L);
    lua_createtable(L, 0, 8);
    auto field = [&](const char* key, const std::string& value) {
        lua_pushlstring(L, value.data(), value.size());
        lua_setfield(L, -2, key);
    };
    field("id", m.id);
    field("name", m.name);
    field("version", m.version);
    field("author", m.author);
    field("description", m.description);
    field("dir", platform::PathUtf8(m.dir));
    field("exe_dir", platform::PathUtf8(platform::ExecutableDirectory()));
    lua_pushinteger(L, kLuaApiVersion);
    lua_setfield(L, -2, "api");
    return 1;
}

int CoreGuestBase(lua_State* L)
{
    lua_pushlightuserdata(L, g_memory.base);
    return 1;
}

// core.registers() -> { r = { [n] = offset }, f = { [n] = offset } }: where the guest registers that carry a call's
// arguments and results (r3-r10, f1-f13), its stack pointer r1 and r13 (the thread's PCR) are in a PPCContext.
int CoreRegisters(lua_State* L)
{
    lua_createtable(L, 0, 2);
    lua_createtable(L, 16, 0);
    auto r = [&](int n, size_t offset) {
        lua_pushinteger(L, lua_Integer(offset));
        lua_rawseti(L, -2, n);
    };
    r(1, offsetof(PPCContext, r1));
    r(3, offsetof(PPCContext, r3));
    r(4, offsetof(PPCContext, r4));
    r(5, offsetof(PPCContext, r5));
    r(6, offsetof(PPCContext, r6));
    r(7, offsetof(PPCContext, r7));
    r(8, offsetof(PPCContext, r8));
    r(9, offsetof(PPCContext, r9));
    r(10, offsetof(PPCContext, r10));
    r(13, offsetof(PPCContext, r13));
    lua_setfield(L, -2, "r");
    lua_createtable(L, 16, 0);
    r(1, offsetof(PPCContext, f1));
    r(2, offsetof(PPCContext, f2));
    r(3, offsetof(PPCContext, f3));
    r(4, offsetof(PPCContext, f4));
    r(5, offsetof(PPCContext, f5));
    r(6, offsetof(PPCContext, f6));
    r(7, offsetof(PPCContext, f7));
    r(8, offsetof(PPCContext, f8));
    r(9, offsetof(PPCContext, f9));
    r(10, offsetof(PPCContext, f10));
    r(11, offsetof(PPCContext, f11));
    r(12, offsetof(PPCContext, f12));
    r(13, offsetof(PPCContext, f13));
    lua_setfield(L, -2, "f");
    return 1;
}

// core.set_sinks(event dispatcher, hook dispatcher): the API script's.
int CoreSetSinks(lua_State* L)
{
    Mod& m = ModOf(L);
    luaL_checktype(L, 1, LUA_TFUNCTION);
    luaL_checktype(L, 2, LUA_TFUNCTION);
    lua_pushvalue(L, 1);
    m.eventSinkRef = luaL_ref(L, LUA_REGISTRYINDEX);
    lua_pushvalue(L, 2);
    m.hookSinkRef = luaL_ref(L, LUA_REGISTRYINDEX);
    return 0;
}

// A Lua hook's entry in a function's chain: the mod's hook dispatcher runs it, under the mod's lock.
void RunLuaHook(HookCall& call, void* user)
{
    LuaHook& h = *static_cast<LuaHook*>(user);
    Mod& m = *h.mod;
    LuaCall luaCall{ &call, false };
    {
        std::lock_guard lock(m.mutex);
        if (h.alive.load(std::memory_order_acquire) && m.L && !m.failed)
        {
            lua_State* L = m.L;
            const int top = lua_gettop(L);
            lua_rawgeti(L, LUA_REGISTRYINDEX, m.hookSinkRef);
            lua_rawgeti(L, LUA_REGISTRYINDEX, h.fnRef);
            lua_pushlightuserdata(L, &luaCall);
            lua_pushlightuserdata(L, &call.ctx);
            lua_pushnumber(L, call.function);
            // (The dispatcher returns the hook's error with its traceback, taken where it failed.)
            std::string error = ProtectedCall(m, 4, 1);
            if (error.empty() && lua_isstring(L, -1))
                error = lua_tostring(L, -1);
            lua_settop(L, top);
            if (!error.empty() && !t_guestExit)
            {
                log::Error("[mod {}] hook on {:08X}: {}", m.id, call.function, error);
                // (This call holds the chain that has the hook: its record stays until the call ends.)
                if (++h.errors >= kMaxHookErrors && TakeOut(h))
                {
                    log::Error("[mod {}] hook on {:08X} removed after {} errors", m.id, call.function, h.errors);
                    RemoveHook(h.id);
                }
            }
            RethrowGuestExit();
            if (error.empty() || luaCall.nexted)
                return;
        }
    }
    // Not run (removed, or the mod off), or it failed before passing the call on: the game's function runs as usual.
    if (!luaCall.nexted)
        CallNext(call);
}

// core.hook(address, function, priority) -> id, or nil and why.
int CoreHook(lua_State* L)
{
    Mod& m = ModOf(L);
    const uint32_t function = CheckAddress(L, 1);
    luaL_checktype(L, 2, LUA_TFUNCTION);
    const int32_t priority = int32_t(luaL_optinteger(L, 3, 0));
    if (!Hookable(function))
    {
        lua_pushnil(L);
        lua_pushfstring(L, "%08X isn't the start of one of the game's functions", function);
        return 2;
    }
    auto* h = new LuaHook();
    h->mod = &m;
    h->function = function;
    lua_pushvalue(L, 2);
    h->fnRef = luaL_ref(L, LUA_REGISTRYINDEX);
    // (A call on another thread may reach it at once: it waits for the mod's lock, which this script holds, so the hook's
    // id and listing are in place by the time it runs.)
    h->id = AddHook(function, RunLuaHook, h, priority, Owner(m), ReleaseLuaHook);
    if (!h->id)
    {
        luaL_unref(L, LUA_REGISTRYINDEX, h->fnRef);
        delete h;
        lua_pushnil(L);
        lua_pushstring(L, "the hook registry refused it (see the log)");
        return 2;
    }
    {
        std::lock_guard lock(s_luaHooksMutex);
        s_luaHooks.emplace(h->id, h);
    }
    m.hooks++;
    lua_pushnumber(L, lua_Number(h->id));
    return 1;
}

// core.unhook(id) -> whether it was there.
int CoreUnhook(lua_State* L)
{
    Mod& m = ModOf(L);
    const uint64_t id = uint64_t(luaL_checknumber(L, 1));
    LuaHook* found = nullptr;
    {
        std::lock_guard lock(s_luaHooksMutex);
        if (const auto it = s_luaHooks.find(id); it != s_luaHooks.end() && it->second->mod == &m)
            found = it->second;
    }
    if (!found || !TakeOut(*found))
    {
        lua_pushboolean(L, 0);
        return 1;
    }
    RemoveHook(id);   // (`found` may be freed from here on)
    lua_pushboolean(L, 1);
    return 1;
}

// core.call_next(call): from a hook, the rest of the chain (the next hook, or the game's function).
int CoreCallNext(lua_State* L)
{
    auto* luaCall = static_cast<LuaCall*>(lua_touserdata(L, 1));
    if (!luaCall)
        return luaL_argerror(L, 1, "not a hooked call");
    luaCall->nexted = true;
    bool exited = false;
    try
    {
        CallNext(*luaCall->call);
    }
    catch (const GuestThreadExit&)
    {
        t_guestExit = std::current_exception();
        exited = true;
    }
    if (exited)
        return luaL_error(L, "the game's thread is ending");
    return 0;
}

int CoreHookable(lua_State* L)
{
    lua_pushboolean(L, Hookable(CheckAddress(L, 1)));
    return 1;
}

constexpr int kMaxIntArgs = 16;     // r3-r10, then 8 on the stack
constexpr int kMaxFloatArgs = 13;   // f1-f13

// Calls the game's function at `address` on the calling game thread, in a copy of its registers (the same stack, r13
// and FPSCR): integer arguments in r3-r10 and past those in a stack frame of their own (the Xbox ABI's parameter
// area: argument 9 + k at r1 + 0x54 + 8k, as kernel/function.h reads them), floats in f1-f13. The results are in
// `out`. Null on success, else why it couldn't call. A guest thread ending inside (GuestThreadExit) goes through.
const char* CallGame(uint32_t address, const uint64_t* ints, int intCount, const double* floats, int floatCount, PPCContext& out)
{
    PPCContext* current = GetPPCContext();
    if (!current)
        return "not on a game thread (call the game from a hook or the frame event)";
    PPCFunc* fn = address >= PPC_CODE_BASE && address < PPC_CODE_BASE + PPC_CODE_SIZE && (address & 3) == 0 ? g_memory.Function(address)
                                                                                                            : nullptr;
    if (!fn)
        return "not one of the game's functions";
    if (intCount < 0 || intCount > kMaxIntArgs || floatCount < 0 || floatCount > kMaxFloatArgs)
        return "too many arguments (16 integers and 13 floats at most)";
    out = *current;
    PPCRegister* intRegs[] = { &out.r3, &out.r4, &out.r5, &out.r6, &out.r7, &out.r8, &out.r9, &out.r10 };
    PPCRegister* floatRegs[] = { &out.f1, &out.f2, &out.f3, &out.f4, &out.f5, &out.f6, &out.f7, &out.f8, &out.f9, &out.f10,
        &out.f11, &out.f12, &out.f13 };
    for (int i = 0; i < intCount && i < 8; i++)
        intRegs[i]->u64 = ints[i];
    for (int i = 0; i < floatCount; i++)
        floatRegs[i]->f64 = floats[i];
    if (intCount > 8)
    {
        // A frame below the caller's: its back chain, then the parameter area's 8-byte slots (the value in the low word).
        const uint32_t sp = out.r1.u32;
        const uint32_t frame = (0x50 + 8 * uint32_t(intCount - 8) + 0x10 + 15) & ~15u;
        const uint32_t newSp = (sp - frame) & ~15u;
        *g_memory.As<be<uint32_t>>(newSp) = sp;
        for (int k = 8; k < intCount; k++)
            *g_memory.As<be<uint64_t>>(newSp + 0x50 + 8 * uint32_t(k - 8)) = ints[k];
        out.r1.u64 = newSp;
    }
    struct Restore
    {
        PPCContext* current;
        PPCContext& call;
        ~Restore()
        {
            current->fpscr = call.fpscr;
            SetPPCContext(*current);
        }
    } restore{ current, out };
    SetPPCContext(out);
    fn(out, g_memory.base);
    return nullptr;
}

// core.call(address, integers, floats [, highs]) -> r3 (u32), f1, r3's high 32 bits: the game's function at `address`
// with integers (an array: r3, r4, ... and past 8 on the stack, 16 at most) and floats (f1 ... f13), on the calling game
// thread. highs[i], where set, is integer i's high 32 bits (the integer then its low 32): 64-bit values a Lua number
// can't hold exactly.
int CoreCall(lua_State* L)
{
    const uint32_t address = CheckAddress(L, 1);
    uint64_t ints[kMaxIntArgs] = {};
    double floats[kMaxFloatArgs] = {};
    int intCount = 0, floatCount = 0;
    if (lua_istable(L, 2))
    {
        intCount = int(lua_objlen(L, 2));
        if (intCount > kMaxIntArgs)
            return luaL_error(L, "ac6.game.call: %d integer arguments (16 at most)", intCount);
        const bool highs = lua_istable(L, 4);
        for (int i = 0; i < intCount; i++)
        {
            lua_rawgeti(L, 2, i + 1);
            const lua_Number v = lua_tonumber(L, -1);
            lua_pop(L, 1);
            ints[i] = v < 0 ? uint64_t(int64_t(v)) : uint64_t(v);
            if (highs)
            {
                lua_rawgeti(L, 4, i + 1);
                if (lua_isnumber(L, -1))
                    ints[i] = uint64_t(uint32_t(lua_tonumber(L, -1))) << 32 | uint32_t(ints[i]);
                lua_pop(L, 1);
            }
        }
    }
    if (lua_istable(L, 3))
    {
        floatCount = int(lua_objlen(L, 3));
        if (floatCount > kMaxFloatArgs)
            return luaL_error(L, "ac6.game.call: %d float arguments (13 at most)", floatCount);
        for (int i = 0; i < floatCount; i++)
        {
            lua_rawgeti(L, 3, i + 1);
            floats[i] = lua_tonumber(L, -1);
            lua_pop(L, 1);
        }
    }
    PPCContext call;
    const char* error = nullptr;
    bool exited = false;
    try
    {
        error = CallGame(address, ints, intCount, floats, floatCount, call);
    }
    catch (const GuestThreadExit&)
    {
        t_guestExit = std::current_exception();
        exited = true;
    }
    if (exited)
        return luaL_error(L, "the game's thread is ending");
    if (error)
        return luaL_error(L, "ac6.game.call(%08X): %s", address, error);
    lua_pushnumber(L, call.r3.u32);
    lua_pushnumber(L, call.f1.f64);
    lua_pushnumber(L, uint32_t(call.r3.u64 >> 32));
    return 3;
}

int CoreIsGameThread(lua_State* L)
{
    lua_pushboolean(L, GetPPCContext() != nullptr);
    return 1;
}

// core.alloc(size) -> guest address of `size` zeroed bytes in the guest heap (16-byte aligned), or nil.
int CoreAlloc(lua_State* L)
{
    const lua_Number size = luaL_checknumber(L, 1);
    if (size <= 0 || size > 256.0 * 1024 * 1024)
        return luaL_argerror(L, 1, "size out of range");
    void* p = g_heap.Alloc(size_t(size));
    if (!p)
    {
        lua_pushnil(L);
        return 1;
    }
    std::memset(p, 0, size_t(size));
    lua_pushnumber(L, g_memory.MapVirtual(p));
    return 1;
}

int CoreFree(lua_State* L)
{
    const uint32_t address = CheckAddress(L, 1);
    if (address)
        g_heap.Free(g_memory.Translate(address));
    return 0;
}

int CoreBacked(lua_State* L)
{
    lua_pushboolean(L, g_memory.IsBacked(CheckAddress(L, 1)));
    return 1;
}

int CoreFrame(lua_State* L)
{
    lua_pushnumber(L, lua_Number(gpu::RendererFrame()));
    return 1;
}

int CoreMode(lua_State* L)
{
    lua_pushstring(L, game::CurrentMode());
    return 1;
}

int CoreTime(lua_State* L)
{
    lua_pushnumber(L, log::Seconds());
    return 1;
}

// --- Input and drawing (ac6.input, ac6.ui) -----------------------------------------------------------------------------

void FilterPad(uint32_t user, input::PadState& pad);   // (below: the mods' pad filter)

// The pad filter in place, from the first pad handler on (Lua or native).
void EnsurePadFilter()
{
    static std::once_flag once;
    std::call_once(once, [] { input::SetPadFilter(FilterPad); });
}

// core.want(event, on): the mod has handlers for "draw" or "pad" now (or none any more).
int CoreWant(lua_State* L)
{
    Mod& m = ModOf(L);
    const std::string_view event = luaL_checkstring(L, 1);
    const bool on = lua_toboolean(L, 2);
    if (event == "draw")
        m.wantsDraw = on;
    else if (event == "pad")
    {
        m.wantsPad = on;
        if (on)
            EnsurePadFilter();
    }
    return 0;
}

// A pad as the mods' Lua sees it: { connected, buttons, lt, rt, lx, ly, rx, ry [, pressed] }.
void PushPad(lua_State* L, const input::PadState& p, const uint16_t* pressed)
{
    lua_createtable(L, 0, 9);
    lua_pushboolean(L, p.connected);
    lua_setfield(L, -2, "connected");
    auto number = [&](const char* key, double v) {
        lua_pushnumber(L, v);
        lua_setfield(L, -2, key);
    };
    number("buttons", p.buttons);
    number("lt", p.leftTrigger);
    number("rt", p.rightTrigger);
    number("lx", p.thumbLX);
    number("ly", p.thumbLY);
    number("rx", p.thumbRX);
    number("ry", p.thumbRY);
    if (pressed)
        number("pressed", *pressed);
}

// The table at `t` back into the pad (a handler's changes), each field kept to its range.
void ReadPad(lua_State* L, int t, input::PadState& p)
{
    auto number = [&](const char* key, double low, double high, double current) {
        lua_getfield(L, t, key);
        const double v = lua_isnumber(L, -1) ? lua_tonumber(L, -1) : current;
        lua_pop(L, 1);
        return std::clamp(v, low, high);
    };
    lua_getfield(L, t, "connected");
    p.connected = lua_toboolean(L, -1);
    lua_pop(L, 1);
    lua_getfield(L, t, "buttons");
    if (lua_isnumber(L, -1))
        p.buttons = uint16_t(int64_t(lua_tonumber(L, -1)));   // (the low 16 bits: bit.* results may be negative)
    lua_pop(L, 1);
    p.leftTrigger = uint8_t(number("lt", 0, 255, p.leftTrigger));
    p.rightTrigger = uint8_t(number("rt", 0, 255, p.rightTrigger));
    p.thumbLX = int16_t(number("lx", -32768, 32767, p.thumbLX));
    p.thumbLY = int16_t(number("ly", -32768, 32767, p.thumbLY));
    p.thumbRX = int16_t(number("rx", -32768, 32767, p.thumbRX));
    p.thumbRY = int16_t(number("ry", -32768, 32767, p.thumbRY));
}

// core.pad(user, game) -> the player's pad as they left it (before the mods' filters, with `pressed`: its buttons gone
// down since the frame before), or the game's (game true: after them).
int CorePad(lua_State* L)
{
    const lua_Integer user = luaL_optinteger(L, 1, 0);
    if (user < 0 || user > 3)
        return luaL_argerror(L, 1, "players are 0-3");
    if (lua_toboolean(L, 2))
    {
        PushPad(L, input::GetState(uint32_t(user)), nullptr);
        return 1;
    }
    uint16_t pressed = 0;
    const input::PadState p = input::PlayerState(uint32_t(user), &pressed);
    PushPad(L, p, &pressed);
    return 1;
}

// core.key(name) -> held, pressed since the frame before; nothing for a name that isn't a key.
int CoreKey(lua_State* L)
{
    bool pressed = false;
    const int held = input::KeyState(luaL_checkstring(L, 1), &pressed);
    if (held < 0)
        return 0;
    lua_pushboolean(L, held);
    lua_pushboolean(L, pressed);
    return 2;
}

// core.mouse() -> x, y (window units), buttons held, buttons gone down (SDL_BUTTON_MASK bits), the wheel's notches.
int CoreMouse(lua_State* L)
{
    const input::MouseState s = input::Mouse();
    lua_pushnumber(L, s.x);
    lua_pushnumber(L, s.y);
    lua_pushnumber(L, s.buttons);
    lua_pushnumber(L, s.pressed);
    lua_pushnumber(L, s.wheel);
    return 5;
}

void CheckDrawing(lua_State* L)
{
    if (!ui::draw::Active())
        luaL_error(L, "ac6.ui draws only in a draw handler (ac6.events.on(\"draw\", ...))");
}

float Arg(lua_State* L, int index) { return float(luaL_checknumber(L, index)); }
float OptArg(lua_State* L, int index, float otherwise) { return float(luaL_optnumber(L, index, otherwise)); }

// A colour, 0xAARRGGBB (white when left out): a number, the bit library's signed results too.
uint32_t ColorArg(lua_State* L, int index)
{
    const lua_Number n = luaL_optnumber(L, index, 4294967295.0);
    if (n < 0)
        return uint32_t(int32_t(std::max(n, -2147483648.0)));
    return uint32_t(std::min(n, 4294967295.0));
}

int CoreDrawSize(lua_State* L)
{
    CheckDrawing(L);
    float w = 0.0f, h = 0.0f;
    ui::draw::Size(w, h);
    lua_pushnumber(L, w);
    lua_pushnumber(L, h);
    return 2;
}

template<void (*Get)(float&, float&, float&, float&)>
int CoreDrawRectOf(lua_State* L)
{
    CheckDrawing(L);
    float x = 0.0f, y = 0.0f, w = 0.0f, h = 0.0f;
    Get(x, y, w, h);
    lua_pushnumber(L, x);
    lua_pushnumber(L, y);
    lua_pushnumber(L, w);
    lua_pushnumber(L, h);
    return 4;
}

// core.draw_line(x0, y0, x1, y1, color, thickness)
int CoreDrawLine(lua_State* L)
{
    CheckDrawing(L);
    ui::draw::Line(Arg(L, 1), Arg(L, 2), Arg(L, 3), Arg(L, 4), ColorArg(L, 5), OptArg(L, 6, 1.0f));
    return 0;
}

// core.draw_rect(x0, y0, x1, y1, color, thickness (0: filled), rounding)
int CoreDrawRect(lua_State* L)
{
    CheckDrawing(L);
    ui::draw::Rect(Arg(L, 1), Arg(L, 2), Arg(L, 3), Arg(L, 4), ColorArg(L, 5), OptArg(L, 7, 0.0f), OptArg(L, 6, 1.0f));
    return 0;
}

// core.draw_circle(x, y, radius, color, thickness (0: filled))
int CoreDrawCircle(lua_State* L)
{
    CheckDrawing(L);
    ui::draw::Circle(Arg(L, 1), Arg(L, 2), Arg(L, 3), ColorArg(L, 4), OptArg(L, 5, 1.0f));
    return 0;
}

// core.draw_triangle(x0, y0, x1, y1, x2, y2, color, thickness (0: filled))
int CoreDrawTriangle(lua_State* L)
{
    CheckDrawing(L);
    ui::draw::Triangle(Arg(L, 1), Arg(L, 2), Arg(L, 3), Arg(L, 4), Arg(L, 5), Arg(L, 6), ColorArg(L, 7), OptArg(L, 8, 1.0f));
    return 0;
}

// core.draw_poly({ x0, y0, x1, y1, ... }, color, closed, thickness (0 and closed: filled, convex))
int CoreDrawPoly(lua_State* L)
{
    CheckDrawing(L);
    luaL_checktype(L, 1, LUA_TTABLE);
    const int n = int(lua_objlen(L, 1));
    std::vector<float> xy(size_t(n - n % 2));
    for (int i = 0; i < int(xy.size()); i++)
    {
        lua_rawgeti(L, 1, i + 1);
        xy[size_t(i)] = float(lua_tonumber(L, -1));
        lua_pop(L, 1);
    }
    ui::draw::Polyline(xy.data(), int(xy.size() / 2), ColorArg(L, 2), lua_toboolean(L, 3), OptArg(L, 4, 1.0f));
    return 0;
}

// core.draw_text(x, y, text, color, size, mono)
int CoreDrawText(lua_State* L)
{
    CheckDrawing(L);
    ui::draw::Text(Arg(L, 1), Arg(L, 2), luaL_checkstring(L, 3), ColorArg(L, 4), OptArg(L, 5, 0.0f), lua_toboolean(L, 6));
    return 0;
}

// core.text_size(text, size, mono) -> width, height
int CoreTextSize(lua_State* L)
{
    CheckDrawing(L);
    float w = 0.0f, h = 0.0f;
    ui::draw::TextSize(luaL_checkstring(L, 1), OptArg(L, 2, 0.0f), lua_toboolean(L, 3), w, h);
    lua_pushnumber(L, w);
    lua_pushnumber(L, h);
    return 2;
}

int CorePushClip(lua_State* L)
{
    CheckDrawing(L);
    ui::draw::PushClip(Arg(L, 1), Arg(L, 2), Arg(L, 3), Arg(L, 4));
    return 0;
}

int CorePopClip(lua_State* L)
{
    CheckDrawing(L);
    ui::draw::PopClip();
    return 0;
}

// --- The native mod API (mods/ac6_mod_api.h) --------------------------------------------------------------------------

// One per DLL: the api it was given (first, so the api's address is the record's), its Lua mod (null for a plugin the
// plugin loader loaded), and its hooks' owner.
struct NativeApi
{
    ac6_mod_api api{};
    Mod* mod = nullptr;
    std::string id;      // its mod's id or the plugin's name: the log's "[mod <id>]"
    std::string owner;   // "native <mod>/<file>" (a plugin: "native <plugin>")
    std::string dir;
    std::atomic<bool> declined{ false };   // its init declined, or its Lua mod failed: what it registered is gone
    // Its handlers get events: its Lua mod is loaded (or, a plugin, its init didn't decline).
    bool Active() const { return !declined && (!mod || mod->loaded); }
};
NativeApi& Native(const ac6_mod_api* api) { return *reinterpret_cast<NativeApi*>(const_cast<ac6_mod_api*>(api)); }

// A native hook's record: freed by the hook registry (ReleaseNativeHook) once no call can reach it.
struct NativeHook
{
    ac6_hook_fn fn;
    void* user;
};
void ReleaseNativeHook(void* user) { delete static_cast<NativeHook*>(user); }
struct NativeFrameHandler
{
    NativeApi* api;
    ac6_frame_fn fn;
    void* user;
};
struct NativeModeHandler
{
    NativeApi* api;
    ac6_mode_fn fn;
    void* user;
};
struct NativePadHandler
{
    NativeApi* api;
    ac6_pad_fn fn;
    void* user;
};
struct NativeDrawHandler
{
    NativeApi* api;
    ac6_draw_fn fn;
    void* user;
};
std::mutex s_nativeMutex;   // the lists below (the handlers are called without it, from a copy)
std::vector<std::unique_ptr<NativeApi>> s_nativeApis;
std::vector<NativeFrameHandler> s_nativeFrame;
std::vector<NativeModeHandler> s_nativeMode;
std::vector<NativePadHandler> s_nativePad;
std::vector<NativeDrawHandler> s_nativeDraw;
std::unordered_map<uint64_t, NativeApi*> s_nativeHookOwner;

void RunNativeHook(HookCall& call, void* user)
{
    const NativeHook& h = *static_cast<NativeHook*>(user);
    h.fn(reinterpret_cast<ac6_call*>(&call), h.user);
}

uint64_t NativeHookFn(const ac6_mod_api* api, uint32_t function, ac6_hook_fn fn, void* user, int32_t priority)
{
    NativeApi& n = Native(api);
    if (!fn || n.declined)
        return 0;   // (declined, or its Lua mod failed: it hooks nothing more)
    auto* h = new NativeHook{ fn, user };
    const uint64_t id = AddHook(function, RunNativeHook, h, priority, n.owner, ReleaseNativeHook);
    if (!id)
    {
        delete h;
        return 0;
    }
    bool dropped = false;
    {
        // (DropNative marks it declined before it removes its hooks: a hook added meanwhile is either removed there, or
        // seen dropped here.)
        std::lock_guard lock(s_nativeMutex);
        dropped = n.declined;
        if (!dropped)
        {
            s_nativeHookOwner.emplace(id, &n);
            if (n.mod)
                n.mod->hooks++;
        }
    }
    if (dropped)
    {
        RemoveHook(id);
        return 0;
    }
    return id;
}

int NativeUnhook(const ac6_mod_api* api, uint64_t id)
{
    {
        std::lock_guard lock(s_nativeMutex);
        const auto it = s_nativeHookOwner.find(id);
        if (it == s_nativeHookOwner.end() || it->second != &Native(api))
            return 0;
        s_nativeHookOwner.erase(it);
    }
    if (Mod* m = Native(api).mod)
        m->hooks--;
    return RemoveHook(id) ? 1 : 0;
}

int NativeHookable(uint32_t function) { return Hookable(function) ? 1 : 0; }

uint64_t* NativeReg(ac6_call* c, int n)
{
    PPCContext& ctx = reinterpret_cast<HookCall*>(c)->ctx;
    switch (n)
    {
    case 1: return &ctx.r1.u64; case 3: return &ctx.r3.u64; case 4: return &ctx.r4.u64; case 5: return &ctx.r5.u64;
    case 6: return &ctx.r6.u64; case 7: return &ctx.r7.u64; case 8: return &ctx.r8.u64; case 9: return &ctx.r9.u64;
    case 10: return &ctx.r10.u64; case 13: return &ctx.r13.u64;
    default: return nullptr;
    }
}

double* NativeFreg(ac6_call* c, int n)
{
    PPCContext& ctx = reinterpret_cast<HookCall*>(c)->ctx;
    switch (n)
    {
    case 1: return &ctx.f1.f64; case 2: return &ctx.f2.f64; case 3: return &ctx.f3.f64; case 4: return &ctx.f4.f64;
    case 5: return &ctx.f5.f64; case 6: return &ctx.f6.f64; case 7: return &ctx.f7.f64; case 8: return &ctx.f8.f64;
    case 9: return &ctx.f9.f64; case 10: return &ctx.f10.f64; case 11: return &ctx.f11.f64; case 12: return &ctx.f12.f64;
    case 13: return &ctx.f13.f64;
    default: return nullptr;
    }
}

uint32_t NativeFunction(ac6_call* c) { return reinterpret_cast<HookCall*>(c)->function; }
void NativeCallNext(ac6_call* c) { CallNext(*reinterpret_cast<HookCall*>(c)); }

int NativeGuestCall(uint32_t function, const uint64_t* ints, int intCount, const double* floats, int floatCount, uint64_t* r3, double* f1)
{
    PPCContext call;
    if (CallGame(function, ints, intCount, floats, floatCount, call))
        return 0;
    if (r3)
        *r3 = call.r3.u64;
    if (f1)
        *f1 = call.f1.f64;
    return 1;
}

int NativeIsGameThread() { return GetPPCContext() != nullptr; }

uint32_t NativeAlloc(uint32_t size)
{
    void* p = size ? g_heap.Alloc(size) : nullptr;
    if (!p)
        return 0;
    std::memset(p, 0, size);
    return g_memory.MapVirtual(p);
}

void NativeFree(uint32_t address)
{
    if (address)
        g_heap.Free(g_memory.Translate(address));
}

void NativeOnFrame(const ac6_mod_api* api, ac6_frame_fn fn, void* user)
{
    if (!fn)
        return;
    std::lock_guard lock(s_nativeMutex);
    s_nativeFrame.push_back({ &Native(api), fn, user });
}

void NativeOnMode(const ac6_mod_api* api, ac6_mode_fn fn, void* user)
{
    if (!fn)
        return;
    std::lock_guard lock(s_nativeMutex);
    s_nativeMode.push_back({ &Native(api), fn, user });
}

// --- Version 2: input and drawing ---

ac6_pad ToNative(const input::PadState& p)
{
    ac6_pad c{};
    c.buttons = p.buttons;
    c.left_trigger = p.leftTrigger;
    c.right_trigger = p.rightTrigger;
    c.lx = p.thumbLX;
    c.ly = p.thumbLY;
    c.rx = p.thumbRX;
    c.ry = p.thumbRY;
    c.connected = p.connected ? 1 : 0;
    return c;
}

void FromNative(const ac6_pad& c, input::PadState& p)
{
    p.buttons = c.buttons;
    p.leftTrigger = c.left_trigger;
    p.rightTrigger = c.right_trigger;
    p.thumbLX = c.lx;
    p.thumbLY = c.ly;
    p.thumbRX = c.rx;
    p.thumbRY = c.ry;
    p.connected = c.connected != 0;
}

int NativePad(uint32_t user, int game, ac6_pad* out, uint16_t* pressed)
{
    if (user > 3 || !out)
        return 0;
    uint16_t went = 0;
    *out = ToNative(game ? input::GetState(user) : input::PlayerState(user, &went));
    if (pressed)
        *pressed = went;
    return 1;
}

int NativeKey(const char* name, int* pressed)
{
    bool went = false;
    const int held = name ? input::KeyState(name, &went) : -1;
    if (pressed)
        *pressed = held >= 0 && went ? 1 : 0;
    return held;
}

void NativeMouse(ac6_mouse* out)
{
    if (!out)
        return;
    const input::MouseState s = input::Mouse();
    *out = { s.x, s.y, s.buttons, s.pressed, s.wheel };
}

void NativeOnPad(const ac6_mod_api* api, ac6_pad_fn fn, void* user)
{
    if (!fn || Native(api).declined)
        return;
    {
        std::lock_guard lock(s_nativeMutex);
        s_nativePad.push_back({ &Native(api), fn, user });
    }
    EnsurePadFilter();
}

void NativeOnDraw(const ac6_mod_api* api, ac6_draw_fn fn, void* user)
{
    if (!fn || Native(api).declined)
        return;
    std::lock_guard lock(s_nativeMutex);
    s_nativeDraw.push_back({ &Native(api), fn, user });
}

// (Outside a draw handler: zeros, and drawing does nothing.)
void NativeDrawSize(float* width, float* height)
{
    float w = 0.0f, h = 0.0f;
    if (ui::draw::Active())
        ui::draw::Size(w, h);
    if (width)
        *width = w;
    if (height)
        *height = h;
}

template<void (*Get)(float&, float&, float&, float&)>
void NativeDrawRectOf(float* x, float* y, float* width, float* height)
{
    float a = 0.0f, b = 0.0f, c = 0.0f, d = 0.0f;
    if (ui::draw::Active())
        Get(a, b, c, d);
    if (x)
        *x = a;
    if (y)
        *y = b;
    if (width)
        *width = c;
    if (height)
        *height = d;
}

void NativeDrawPolyline(const float* xy, int points, uint32_t color, int closed, float thickness)
{
    ui::draw::Polyline(xy, points, color, closed != 0, thickness);
}

void NativeDrawText(float x, float y, const char* text, uint32_t color, float size, int mono)
{
    ui::draw::Text(x, y, text, color, size, mono != 0);
}

void NativeTextSize(const char* text, float size, int mono, float* width, float* height)
{
    float w = 0.0f, h = 0.0f;
    ui::draw::TextSize(text, size, mono != 0, w, h);
    if (width)
        *width = w;
    if (height)
        *height = h;
}

const char* NativeRttiOf(uint32_t object)
{
    const rtti::Class* c = rtti::Of(object);
    return c ? c->name.c_str() : nullptr;
}

uint32_t NativeVfunc(const char* className, uint32_t slot)
{
    const rtti::Class* c = className ? rtti::Find(className) : nullptr;
    if (!c || c->vtables.empty() || slot >= c->vtables[0].slots)
        return 0;
    return g_memory.As<be<uint32_t>>(c->vtables[0].address + slot * 4)->get();
}

void NativeLog(const ac6_mod_api* api, int level, const char* text)
{
    const std::string& id = Native(api).id;
    const std::string t = text ? text : "";
    if (level >= AC6_LOG_ERROR)
        log::Error("[mod {}] {}", id, t);
    else if (level == AC6_LOG_WARN)
        log::Warn("[mod {}] {}", id, t);
    else
        log::Info("[mod {}] {}", id, t);
}

size_t NativeSetting(const char* name, char* buf, size_t size)
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

// A DLL's api: for a Lua mod's module (`mod`), or for a plugin (mod null: `id` its name, `dir` its folder).
NativeApi* MakeNativeApi(Mod* mod, const std::string& id, const std::string& file, const std::filesystem::path& dir)
{
    auto n = std::make_unique<NativeApi>();
    n->mod = mod;
    n->id = id;
    n->owner = mod ? "native " + id + "/" + file : "native " + id;   // (a plugin's id already names its file)
    n->dir = platform::PathUtf8(dir);
    std::replace(n->dir.begin(), n->dir.end(), '\\', '/');
    ac6_mod_api& a = n->api;
    a.version = AC6_MOD_API_VERSION;
    a.size = sizeof(ac6_mod_api);
    a.mod_id = n->id.c_str();
    a.mod_dir = n->dir.c_str();
    a.lua = mod ? mod->L : nullptr;
    a.guest_base = g_memory.base;
    a.hook = NativeHookFn;
    a.unhook = NativeUnhook;
    a.hookable = NativeHookable;
    a.reg = NativeReg;
    a.freg = NativeFreg;
    a.function = NativeFunction;
    a.call_next = NativeCallNext;
    a.guest_call = NativeGuestCall;
    a.is_game_thread = NativeIsGameThread;
    a.guest_alloc = NativeAlloc;
    a.guest_free = NativeFree;
    a.on_frame = NativeOnFrame;
    a.on_mode = NativeOnMode;
    a.rtti_of = NativeRttiOf;
    a.vfunc = NativeVfunc;
    a.log = NativeLog;
    a.setting = NativeSetting;
    // Version 2.
    a.pad = NativePad;
    a.key = NativeKey;
    a.mouse = NativeMouse;
    a.on_pad = NativeOnPad;
    a.on_draw = NativeOnDraw;
    a.draw_size = NativeDrawSize;
    a.draw_view = NativeDrawRectOf<ui::draw::View>;
    a.draw_canvas = NativeDrawRectOf<ui::draw::Canvas>;
    a.draw_line = ui::draw::Line;
    a.draw_rect = ui::draw::Rect;
    a.draw_circle = ui::draw::Circle;
    a.draw_triangle = ui::draw::Triangle;
    a.draw_polyline = NativeDrawPolyline;
    a.draw_text = NativeDrawText;
    a.text_size = NativeTextSize;
    a.draw_push_clip = ui::draw::PushClip;
    a.draw_pop_clip = ui::draw::PopClip;
    std::lock_guard lock(s_nativeMutex);
    return s_nativeApis.emplace_back(std::move(n)).get();
}

// A DLL's init declined, or its Lua mod failed: what it registered goes, and it registers nothing more.
void DropNative(NativeApi& n)
{
    {
        std::lock_guard lock(s_nativeMutex);
        if (n.declined.exchange(true))
            return;
    }
    RemoveHooksOf(n.owner);
    std::lock_guard lock(s_nativeMutex);
    for (auto it = s_nativeHookOwner.begin(); it != s_nativeHookOwner.end();)
        if (it->second == &n)
        {
            if (n.mod)
                n.mod->hooks--;
            it = s_nativeHookOwner.erase(it);
        }
        else
            ++it;
    std::erase_if(s_nativeFrame, [&](const NativeFrameHandler& h) { return h.api == &n; });
    std::erase_if(s_nativeMode, [&](const NativeModeHandler& h) { return h.api == &n; });
    std::erase_if(s_nativePad, [&](const NativePadHandler& h) { return h.api == &n; });
    std::erase_if(s_nativeDraw, [&](const NativeDrawHandler& h) { return h.api == &n; });
}

// The DLLs started (mods.h ClaimNative): file -> who started it.
std::mutex s_startedMutex;
std::unordered_map<std::string, std::string> s_started;

// A file's key: its absolute path, symbolic links and . / .. resolved (on Windows in lower case, as its file system
// compares names).
std::string NativeKey(const std::filesystem::path& file)
{
    std::error_code ec;
    std::filesystem::path p = std::filesystem::weakly_canonical(file, ec);
    if (ec)
        p = std::filesystem::absolute(file, ec);
#if defined(_WIN32)
    std::wstring s = p.make_preferred().native();
    for (wchar_t& c : s)
        c = wchar_t(std::towlower(c));
    return platform::PathUtf8(std::filesystem::path(s));
#else
    return p.native();
#endif
}

// core.load_native(path) -> what its luaopen_<name> returned, or true: a DLL (.so) loaded with the folder it's in
// searched for the DLLs it needs. One exporting ac6_mod_init is started as a native mod (mods/ac6_mod_api.h), one
// exporting ac6_plugin_init as a plugin (plugin/ac6_plugin.h), one exporting luaopen_<its file name> (dots as
// underscores) is opened as a Lua module in the mod's state. It stays loaded. A DLL is started once (ClaimNative):
// one the plugin loader or another mod started is only opened as a module here. The mod loading the same file again
// gets what it got the first time, as require does.
int CoreLoadNative(lua_State* L)
{
    Mod& m = ModOf(L);
    const std::string off = settings::Get("AC6_LUA_NATIVE");
    if (off == "0" || off == "false")
        return luaL_error(L, "native modules are off (AC6_LUA_NATIVE=0)");
    const char* utf8 = luaL_checkstring(L, 1);
    std::filesystem::path path(std::u8string(reinterpret_cast<const char8_t*>(utf8)));   // (Lua's strings: UTF-8)
    if (path.is_relative())
        path = m.dir / path;
    std::error_code ec;
    if (!std::filesystem::is_regular_file(path, ec))
        return luaL_error(L, "no file %s", platform::PathUtf8(path).c_str());
    const std::string key = NativeKey(path);
    if (const auto it = m.modules.find(key); it != m.modules.end())
    {
        lua_rawgeti(L, LUA_REGISTRYINDEX, it->second);
        return 1;
    }
    const std::string self = Owner(m);
    std::string by;
    const bool start = ClaimNative(path, self, &by);
    void* object = nullptr;
    auto symbol = [&](const char* name) -> void* {
#if defined(_WIN32)
        return reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(object), name));
#else
        return dlsym(object, name);
#endif
    };
#if defined(_WIN32)
    object = LoadLibraryExW(path.c_str(), nullptr, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    if (!object)
    {
        const int error = int(GetLastError());
        if (start)
            ReleaseNative(path);
        return luaL_error(L, "%s not loaded (error %d)", platform::PathUtf8(path).c_str(), error);
    }
#else
    object = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!object)
    {
        const std::string error = dlerror();
        if (start)
            ReleaseNative(path);
        return luaL_error(L, "%s not loaded: %s", platform::PathUtf8(path).c_str(), error.c_str());
    }
#endif
    m.native = true;
    const std::string file = platform::PathUtf8(path.filename());
    log::Info("[mod {}] loaded native module {}", m.id, file);
    if (!start && by != self)
        log::Warn("[mod {}] {}: {} started it already, its ac6_mod_init / ac6_plugin_init aren't run again (keep a Lua "
                  "mod's DLLs out of the top of its folder, where the plugin loader starts them: in bin/)",
            m.id, file, by);
    if (start)
    {
        NativeApi* n = nullptr;
        if (void* init = symbol("ac6_mod_init"))
        {
            n = MakeNativeApi(&m, m.id, file, m.dir);
            if (!reinterpret_cast<ac6_mod_init_fn>(init)(&n->api))
            {
                DropNative(*n);
                ReleaseNative(path);
                return luaL_error(L, "%s declined (its ac6_mod_init returned 0)", file.c_str());
            }
        }
        if (void* init = symbol("ac6_plugin_init"))
        {
            const std::string name = m.id + "/" + file;
            if (!plugin::InitNative(name, init))
            {
                // (Neither half stays: its native mod API's registrations go too.)
                if (n)
                    DropNative(*n);
                ReleaseNative(path);
                return luaL_error(L, "%s declined (its ac6_plugin_init returned false)", file.c_str());
            }
            m.plugins.push_back(name);
        }
    }
    std::string stem = platform::PathUtf8(path.stem());
    std::replace(stem.begin(), stem.end(), '.', '_');
    if (void* open = symbol(("luaopen_" + stem).c_str()))
    {
        lua_pushcfunction(L, reinterpret_cast<lua_CFunction>(open));
        lua_pushstring(L, stem.c_str());
        lua_call(L, 1, 1);
    }
    else
        lua_pushnil(L);
    if (lua_isnil(L, -1))
    {
        lua_pop(L, 1);
        lua_pushboolean(L, 1);
    }
    lua_pushvalue(L, -1);
    m.modules[key] = luaL_ref(L, LUA_REGISTRYINDEX);   // (for the next load of the same file)
    return 1;
}

// core.symbols() -> { functions = { name = address }, data = { name = address }, signatures = { name = signature },
// sources = { file, ... } }: the built-in names (mods/symbols.toml), then symbols/*.toml next to the executable, then
// the mod's own symbols.toml, a later file's name winning.
int CoreSymbols(lua_State* L)
{
    Mod& m = ModOf(L);
    std::map<std::string, uint32_t> functions, data;
    std::map<std::string, std::string> signatures;
    std::vector<std::string> sources;
    auto addresses = [](const toml::table& t, const char* section, std::map<std::string, uint32_t>& out) {
        if (const toml::table* s = t[section].as_table())
            for (const auto& [key, node] : *s)
                if (const auto address = node.value<int64_t>(); address && *address > 0 && *address <= 0xFFFFFFFFll)
                    out[std::string(key.str())] = uint32_t(*address);
    };
    auto load = [&](std::string_view text, const std::string& source) {
        toml::parse_result parsed = toml::parse(text, source);
        if (!parsed)
        {
            log::Warn("[mod {}] {}: {} (line {}), its symbols left out", m.id, source, parsed.error().description(),
                parsed.error().source().begin.line);
            return;
        }
        const toml::table& t = parsed.table();
        addresses(t, "functions", functions);
        addresses(t, "data", data);
        if (const toml::table* s = t["signatures"].as_table())
            for (const auto& [key, node] : *s)
                if (const auto signature = node.value<std::string>())
                    signatures[std::string(key.str())] = *signature;
        sources.push_back(source);
    };
    load(gpu::kModsSymbols, "symbols.toml (built in)");
    std::error_code ec;
    std::vector<std::filesystem::path> files;
    for (const auto& entry : std::filesystem::directory_iterator(platform::ExecutableDirectory() / "symbols", ec))
        if (entry.is_regular_file(ec) && entry.path().extension() == ".toml")
            files.push_back(entry.path());
    std::sort(files.begin(), files.end());
    if (std::filesystem::is_regular_file(m.dir / "symbols.toml", ec))
        files.push_back(m.dir / "symbols.toml");
    for (const auto& file : files)
        load(ReadFile(file), platform::PathUtf8(file));

    lua_createtable(L, 0, 4);
    const std::pair<const char*, const std::map<std::string, uint32_t>*> tables[] = { { "functions", &functions }, { "data", &data } };
    for (const auto& [name, map] : tables)
    {
        lua_createtable(L, 0, int(map->size()));
        for (const auto& [key, address] : *map)
        {
            lua_pushnumber(L, address);
            lua_setfield(L, -2, key.c_str());
        }
        lua_setfield(L, -2, name);
    }
    lua_createtable(L, 0, int(signatures.size()));
    for (const auto& [key, signature] : signatures)
    {
        lua_pushlstring(L, signature.data(), signature.size());
        lua_setfield(L, -2, key.c_str());
    }
    lua_setfield(L, -2, "signatures");
    lua_createtable(L, int(sources.size()), 0);
    for (size_t i = 0; i < sources.size(); i++)
    {
        lua_pushlstring(L, sources[i].data(), sources[i].size());
        lua_rawseti(L, -2, int(i + 1));
    }
    lua_setfield(L, -2, "sources");
    return 1;
}

// core.functions() -> every hookable function's address, in order.
int CoreFunctions(lua_State* L)
{
    const std::vector<uint32_t>& functions = HookableFunctions();
    lua_createtable(L, int(functions.size()), 0);
    for (size_t i = 0; i < functions.size(); i++)
    {
        lua_pushnumber(L, functions[i]);
        lua_rawseti(L, -2, int(i + 1));
    }
    return 1;
}

// core.rtti_of(object) -> class name, the subobject's offset; or nil.
int CoreRttiOf(lua_State* L)
{
    uint32_t offset = 0;
    const rtti::Class* c = rtti::Of(CheckAddress(L, 1), &offset);
    if (!c)
        return 0;
    lua_pushlstring(L, c->name.data(), c->name.size());
    lua_pushnumber(L, offset);
    return 2;
}

void PushClass(lua_State* L, const rtti::Class& c)
{
    lua_createtable(L, 0, 4);
    lua_pushlstring(L, c.name.data(), c.name.size());
    lua_setfield(L, -2, "name");
    lua_pushlstring(L, c.mangled.data(), c.mangled.size());
    lua_setfield(L, -2, "mangled");
    lua_pushnumber(L, c.typeDescriptor);
    lua_setfield(L, -2, "type_descriptor");
    lua_createtable(L, int(c.vtables.size()), 0);
    for (size_t i = 0; i < c.vtables.size(); i++)
    {
        lua_createtable(L, 0, 3);
        lua_pushnumber(L, c.vtables[i].address);
        lua_setfield(L, -2, "address");
        lua_pushnumber(L, c.vtables[i].offset);
        lua_setfield(L, -2, "offset");
        lua_pushnumber(L, c.vtables[i].slots);
        lua_setfield(L, -2, "slots");
        lua_rawseti(L, -2, int(i + 1));
    }
    lua_setfield(L, -2, "vtables");
}

// core.rtti_class(name) -> { name, mangled, type_descriptor, vtables = { { address, offset, slots }... } }, or nil.
int CoreRttiClass(lua_State* L)
{
    const rtti::Class* c = rtti::Find(luaL_checkstring(L, 1));
    if (!c)
        return 0;
    PushClass(L, *c);
    return 1;
}

// core.rtti_classes() -> every class's name.
int CoreRttiClasses(lua_State* L)
{
    const auto& classes = rtti::Classes();
    lua_createtable(L, int(classes.size()), 0);
    for (size_t i = 0; i < classes.size(); i++)
    {
        lua_pushlstring(L, classes[i].name.data(), classes[i].name.size());
        lua_rawseti(L, -2, int(i + 1));
    }
    return 1;
}

const luaL_Reg kCore[] = {
    { "log", CoreLog },
    { "setting", CoreSetting },
    { "info", CoreInfo },
    { "guest_base", CoreGuestBase },
    { "registers", CoreRegisters },
    { "set_sinks", CoreSetSinks },
    { "hook", CoreHook },
    { "unhook", CoreUnhook },
    { "call_next", CoreCallNext },
    { "hookable", CoreHookable },
    { "call", CoreCall },
    { "is_game_thread", CoreIsGameThread },
    { "alloc", CoreAlloc },
    { "free", CoreFree },
    { "backed", CoreBacked },
    { "frame", CoreFrame },
    { "mode", CoreMode },
    { "time", CoreTime },
    { "load_native", CoreLoadNative },
    { "symbols", CoreSymbols },
    { "functions", CoreFunctions },
    { "rtti_of", CoreRttiOf },
    { "rtti_class", CoreRttiClass },
    { "rtti_classes", CoreRttiClasses },
    { "want", CoreWant },
    { "pad", CorePad },
    { "key", CoreKey },
    { "mouse", CoreMouse },
    { "draw_size", CoreDrawSize },
    { "draw_view", CoreDrawRectOf<ui::draw::View> },
    { "draw_canvas", CoreDrawRectOf<ui::draw::Canvas> },
    { "draw_line", CoreDrawLine },
    { "draw_rect", CoreDrawRect },
    { "draw_circle", CoreDrawCircle },
    { "draw_triangle", CoreDrawTriangle },
    { "draw_poly", CoreDrawPoly },
    { "draw_text", CoreDrawText },
    { "text_size", CoreTextSize },
    { "push_clip", CorePushClip },
    { "pop_clip", CorePopClip },
    { nullptr, nullptr },
};

// --- Loading ----------------------------------------------------------------------------------------------------------

// The mod off (under its lock): its Lua hooks, and what its native modules registered (their hooks and handlers, the
// plugins they started). The DLLs stay loaded: their code may be running on another thread.
void Fail(Mod& m, const std::string& error)
{
    m.failed = true;
    m.loaded = false;
    m.error = error;
    log::Error("[mod {}] {}; the mod is off", m.id, error);
    const size_t before = HookCount();
    {
        std::lock_guard lock(s_luaHooksMutex);
        for (auto it = s_luaHooks.begin(); it != s_luaHooks.end();)
        {
            LuaHook* h = it->second;
            if (h->mod != &m)
            {
                ++it;
                continue;
            }
            h->alive = false;
            if (m.L && h->fnRef != LUA_NOREF)
                luaL_unref(m.L, LUA_REGISTRYINDEX, h->fnRef);
            h->fnRef = LUA_NOREF;
            it = s_luaHooks.erase(it);
        }
    }
    RemoveHooksOf(Owner(m));
    std::vector<NativeApi*> natives;
    {
        std::lock_guard lock(s_nativeMutex);
        for (auto& n : s_nativeApis)
            if (n->mod == &m)
                natives.push_back(n.get());
    }
    for (NativeApi* n : natives)
        DropNative(*n);
    for (const std::string& name : m.plugins)
        plugin::RemoveNative(name);
    m.plugins.clear();
    m.hooks = 0;
    const size_t after = HookCount();
    if (before > after)
        log::Info("[mod {}] {} hook(s) removed{}", m.id, before - after, natives.empty() ? "" : " (its native modules' too)");
}

// The mod's manifest. False (and the mod's error set): it can't be loaded.
bool ReadManifest(Mod& m)
{
    m.name = m.id;
    const std::filesystem::path path = m.dir / "mod.toml";
    std::error_code ec;
    if (!std::filesystem::is_regular_file(path, ec))
        return true;   // (scripts without a manifest: the defaults)
    toml::parse_result parsed = toml::parse_file(platform::PathUtf8(path));
    if (!parsed)
    {
        m.error = std::format("mod.toml: {} (line {})", parsed.error().description(), parsed.error().source().begin.line);
        return false;
    }
    const toml::table& t = parsed.table();
    m.name = t["name"].value_or(m.id);
    m.version = t["version"].value_or(std::string());
    m.author = t["author"].value_or(std::string());
    m.description = t["description"].value_or(std::string());
    m.loadOrder = t["load_order"].value_or(int64_t(0));
    m.api = t["api"].value_or(int64_t(kLuaApiVersion));
    if (!t["enabled"].value_or(true))
    {
        m.error = "disabled in its mod.toml";
        return false;
    }
    if (m.api > int64_t(kLuaApiVersion))
    {
        m.error = std::format("written for Lua API {}, this runtime has {} (a newer version of the game needed)", m.api, kLuaApiVersion);
        return false;
    }
    return true;
}

// The mod's Lua state: the standard libraries (with the FFI and the JIT), its folder on the module paths, and the API
// script's `ac6` global.
bool CreateState(Mod& m)
{
    lua_State* L = luaL_newstate();
    if (!L)
    {
        m.error = "no memory for a Lua state";
        return false;
    }
    m.L = L;
    luaL_openlibs(L);
    lua_pushcfunction(L, Traceback);
    m.tracebackRef = luaL_ref(L, LUA_REGISTRYINDEX);

    // require: the mod's folder, its lib/, then lua/ next to the executable (shared libraries, LuaJIT's jit.*).
    // Native Lua modules from its bin/.
    auto slash = [](std::string s) {
        std::replace(s.begin(), s.end(), '\\', '/');
        return s;
    };
    const std::string dir = slash(platform::PathUtf8(m.dir));
    const std::string exe = slash(platform::PathUtf8(platform::ExecutableDirectory()));
    const std::string path = dir + "/?.lua;" + dir + "/?/init.lua;" + dir + "/lib/?.lua;" + dir + "/lib/?/init.lua;" + exe + "/lua/?.lua;" +
        exe + "/lua/?/init.lua";
#if defined(_WIN32)
    const std::string cpath = dir + "/bin/?.dll";
#else
    const std::string cpath = dir + "/bin/?.so";
#endif
    lua_getglobal(L, "package");
    lua_pushstring(L, path.c_str());
    lua_setfield(L, -2, "path");
    lua_pushstring(L, cpath.c_str());
    lua_setfield(L, -2, "cpath");
    lua_pop(L, 1);

    // The API script, given ac6.core: it returns the ac6 table.
    const std::string_view api = gpu::kModsLuaApi;
    if (luaL_loadbuffer(L, api.data(), api.size(), "=ac6.lua") != 0)
    {
        m.error = std::string("the API script: ") + lua_tostring(L, -1);
        lua_pop(L, 1);
        return false;
    }
    lua_newtable(L);
    lua_pushlightuserdata(L, &m);
    luaL_openlib(L, nullptr, kCore, 1);
    if (std::string error = ProtectedCall(m, 1, 1); !error.empty())
    {
        m.error = "the API script: " + error;
        return false;
    }
    lua_setglobal(L, "ac6");
    return true;
}

// Runs the mod's `file` if it has one. False (the mod failed): it didn't run through.
bool RunScript(Mod& m, const char* file)
{
    const std::filesystem::path path = m.dir / file;
    std::error_code ec;
    if (!std::filesystem::is_regular_file(path, ec))
        return true;
    const std::string code = ReadFile(path);
    std::lock_guard lock(m.mutex);
    lua_State* L = m.L;
    const std::string chunk = "@mods/" + m.id + "/" + file;
    if (luaL_loadbuffer(L, code.data(), code.size(), chunk.c_str()) != 0)
    {
        Fail(m, lua_tostring(L, -1));
        lua_pop(L, 1);
        return false;
    }
    if (std::string error = ProtectedCall(m, 0, 0); !error.empty())
    {
        Fail(m, error);
        return false;
    }
    return true;
}

bool Off(const char* name)
{
    const std::string v = settings::Get(name);
    return v == "0" || v == "false";
}

std::vector<std::string> DisabledIds()
{
    std::vector<std::string> ids;
    std::stringstream list(settings::Get("AC6_LUA_DISABLE"));
    for (std::string id; std::getline(list, id, ',');)
        if (!id.empty())
            ids.push_back(id);
    return ids;
}

} // namespace

void LoadEarly()
{
    if (Off("AC6_LUA"))
    {
        log::Info("Lua mods: off (AC6_LUA=0)");
        return;
    }
    const std::filesystem::path root = platform::ExecutableDirectory() / "mods";
    std::error_code ec;
    if (!std::filesystem::is_directory(root, ec))
        return;
    const std::vector<std::string> disabled = DisabledIds();
    for (const auto& entry : std::filesystem::directory_iterator(root, ec))
    {
        if (!entry.is_directory(ec))
            continue;
        const std::filesystem::path& dir = entry.path();
        if (!std::filesystem::exists(dir / "mod.toml", ec) && !std::filesystem::exists(dir / "init.lua", ec) &&
            !std::filesystem::exists(dir / "early.lua", ec))
            continue;   // (a texture, package or plugin mod)
        auto m = std::make_unique<Mod>();
        m->id = platform::PathUtf8(dir.filename());
        m->dir = dir;
        if (std::find(disabled.begin(), disabled.end(), m->id) != disabled.end())
        {
            m->name = m->id;
            m->error = "left out (AC6_LUA_DISABLE)";
            m->failed = true;
        }
        else if (!ReadManifest(*m))
            m->failed = true;
        if (m->failed)
            log::Info("[mod {}] not loaded: {}", m->id, m->error);
        s_mods.push_back(std::move(m));
    }
    std::sort(s_mods.begin(), s_mods.end(), [](const auto& a, const auto& b) {
        return a->loadOrder != b->loadOrder ? a->loadOrder < b->loadOrder : a->id < b->id;
    });
    log::Info("Lua mods: {} found ({}, Lua API {})", s_mods.size(), LUAJIT_VERSION, kLuaApiVersion);
    for (auto& m : s_mods)
    {
        if (m->failed)
            continue;
        log::Info("[mod {}] {}{}{}", m->id, m->name, m->version.empty() ? std::string() : " " + m->version,
            m->author.empty() ? std::string() : " by " + m->author);
        if (!CreateState(*m))
        {
            Fail(*m, m->error);
            continue;
        }
        RunScript(*m, "early.lua");
    }
}

void LoadInit()
{
    for (auto& m : s_mods)
    {
        if (m->failed)
            continue;
        if (RunScript(*m, "init.lua"))
        {
            m->loaded = true;
            log::Info("[mod {}] loaded ({} hook(s){})", m->id, m->hooks.load(), m->native ? ", native code" : "");
        }
    }
}

namespace {

// An event to a loaded mod: `push` pushes its arguments and returns how many; `after` (if given) runs with the stack as
// the dispatcher left it plus what push put below the call (its index: the stack's top before, + 1), on success.
template<typename Push>
void Send(Mod& m, const char* name, Push&& push, void (*after)(lua_State*, int, void*) = nullptr, void* context = nullptr)
{
    if (!m.loaded)
        return;
    std::lock_guard lock(m.mutex);
    if (!m.loaded || m.eventSinkRef == LUA_NOREF)
        return;
    lua_State* L = m.L;
    const int top = lua_gettop(L);
    lua_rawgeti(L, LUA_REGISTRYINDEX, m.eventSinkRef);
    lua_pushstring(L, name);
    const int n = push(L);
    const std::string error = ProtectedCall(m, 1 + n, 0);
    if (error.empty() && after)
        after(L, top + 1, context);
    lua_settop(L, top);
    if (!error.empty() && !t_guestExit)
        log::Error("[mod {}] {} event: {}", m.id, name, error);
    RethrowGuestExit();
}

// An event to every loaded mod.
template<typename Push>
void Broadcast(const char* name, Push&& push)
{
    for (auto& m : s_mods)
        Send(*m, name, push);
}

// The pad filter (input::SetPadFilter): the native ones, then the Lua mods' "pad" handlers, in their order, each on the
// one before's result.
void FilterPad(uint32_t user, input::PadState& pad)
{
    std::vector<NativePadHandler> natives;
    {
        std::lock_guard lock(s_nativeMutex);
        natives = s_nativePad;
    }
    for (const NativePadHandler& h : natives)
        if (h.api->Active())
        {
            ac6_pad c = ToNative(pad);
            h.fn(user, &c, h.user);
            FromNative(c, pad);
        }
    for (auto& m : s_mods)
    {
        if (!m->wantsPad)
            continue;
        // The pad's table under the call (its handlers change it), read back after.
        Send(
            *m, "pad",
            [&](lua_State* L) {
                // (Stack: dispatcher, name.) A copy of the table under the dispatcher, then: user, table.
                PushPad(L, pad, nullptr);
                lua_pushvalue(L, -1);
                lua_insert(L, -4);
                lua_pushinteger(L, user);
                lua_insert(L, -2);
                return 2;
            },
            [](lua_State* L, int t, void* p) { ReadPad(L, t, *static_cast<input::PadState*>(p)); }, &pad);
    }
}

} // namespace

bool WantsDraw()
{
    for (auto& m : s_mods)
        if (m->loaded && m->wantsDraw)
            return true;
    std::lock_guard lock(s_nativeMutex);
    for (const NativeDrawHandler& h : s_nativeDraw)
        if (h.api->Active())
            return true;
    return false;
}

void OnDraw()
{
    std::vector<NativeDrawHandler> natives;
    {
        std::lock_guard lock(s_nativeMutex);
        natives = s_nativeDraw;
    }
    for (const NativeDrawHandler& h : natives)
        if (h.api->Active())
        {
            h.fn(h.user);
            ui::draw::PopClips();
        }
    for (auto& m : s_mods)
        if (m->wantsDraw)
        {
            Send(*m, "draw", [](lua_State*) { return 0; });
            ui::draw::PopClips();
        }
}

void OnFrame()
{
    static auto last = std::chrono::steady_clock::now();
    const auto now = std::chrono::steady_clock::now();
    const double dt = std::chrono::duration<double>(now - last).count();
    last = now;
    std::vector<NativeFrameHandler> natives;
    {
        std::lock_guard lock(s_nativeMutex);
        if (s_mods.empty() && s_nativeFrame.empty())
            return;
        natives = s_nativeFrame;
    }
    const uint64_t frameNumber = gpu::RendererFrame();
    const double frame = double(frameNumber);
    for (const NativeFrameHandler& h : natives)
        if (h.api->Active())
            h.fn(dt, frameNumber, h.user);
    if (s_mods.empty())
        return;
    Broadcast("frame", [&](lua_State* L) {
        lua_pushnumber(L, dt);
        lua_pushnumber(L, frame);
        return 2;
    });
}

void OnMode(const char* mode)
{
    std::vector<NativeModeHandler> natives;
    {
        std::lock_guard lock(s_nativeMutex);
        if (s_mods.empty() && s_nativeMode.empty())
            return;
        natives = s_nativeMode;
    }
    for (const NativeModeHandler& h : natives)
        if (h.api->Active())
            h.fn(mode, h.user);
    if (s_mods.empty())
        return;
    Broadcast("mode", [&](lua_State* L) {
        lua_pushstring(L, mode);
        return 1;
    });
}

const ac6_mod_api* PluginNativeApi(const std::string& name, const std::string& file, const std::filesystem::path& dir)
{
    return &MakeNativeApi(nullptr, name, file, dir)->api;
}

void DropPluginNativeApi(const ac6_mod_api* api)
{
    DropNative(Native(api));
}

bool ClaimNative(const std::filesystem::path& file, const std::string& who, std::string* by)
{
    const std::string key = NativeKey(file);
    std::lock_guard lock(s_startedMutex);
    const auto [it, claimed] = s_started.emplace(key, who);
    if (!claimed && by)
        *by = it->second;
    return claimed;
}

void ReleaseNative(const std::filesystem::path& file)
{
    const std::string key = NativeKey(file);
    std::lock_guard lock(s_startedMutex);
    s_started.erase(key);
}

std::vector<ModStatus> Status()
{
    std::vector<ModStatus> out;
    for (auto& m : s_mods)
    {
        ModStatus s;
        s.id = m->id;
        s.name = m->name;
        s.version = m->version;
        s.author = m->author;
        s.description = m->description;
        s.loaded = m->loaded;
        s.native = m->native;
        s.error = m->error;
        s.hooks = m->hooks.load();
        out.push_back(std::move(s));
    }
    for (const plugin::PluginStatus& p : plugin::Status())
    {
        ModStatus s;
        s.id = p.name;
        s.name = p.name;
        s.plugin = true;
        s.loaded = p.loaded;
        s.native = true;
        s.error = p.error;
        // (Its hooks through either API: ac6_plugin.h's, and the native mod API's.)
        s.hooks = uint32_t(plugin::HookCountOf(p.name) + HookCountOf("native " + p.name));
        out.push_back(std::move(s));
    }
    return out;
}

} // namespace ac6::mods
