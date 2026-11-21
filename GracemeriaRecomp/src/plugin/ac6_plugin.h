#pragma once
// AC6 runtime plugins: native code mods, mods/<mod>/<name>.dll next to the game (.so on Linux), loaded
// before the game starts. A plugin exports
//
//     extern "C" AC6_PLUGIN_EXPORT bool ac6_plugin_init(const ac6::plugin::Api* api);
//
// and, from it, registers hooks on the game's functions and handlers for its events.
// Returning false declines: the plugin is unloaded, with the hooks and handlers it registered.
// A hook runs in place of the game's function: it reads and sets the call's registers and runs the game's own through
// `original` (or not). Any of the game's functions can be hooked (by its start address: the runtime's hook gates,
// GracemeriaRecomp/src/mods/hooks.h), Hookable below names a few, and several plugins and mods can hook the same one:
// `original` then runs the next hook in its chain, the game's function after the last. (A build without hook gates
// offers only the Hookable functions, one plugin each.)
// Guest memory is big-endian, at api->guest_base + address (the helpers below).
//
// Native code: a plugin can crash or harm the game and the machine like any program. Run only plugins you trust.
// AC6_PLUGINS=0 loads none. This header is the whole interface: plugins don't link against the runtime.
#include <cstddef>
#include <cstdint>
#include <cstring>

#if defined(_WIN32)
#define AC6_PLUGIN_EXPORT __declspec(dllexport)
#else
#define AC6_PLUGIN_EXPORT __attribute__((visibility("default")))
#endif

namespace ac6::plugin {

constexpr uint32_t kApiVersion = 1;

// The game's functions a hook can take (guest addresses).
enum Hookable : uint32_t
{
    kDriftStartCheck = 0x822A5C90,      // flight_drift_start_check (r3 drift, r5 inputs, r6 stall bit, f1 dt) -> r3
    kDriftEndCheck = 0x822A5D78,        // flight_drift_end_check (r3 drift, r5 inputs, f1 dt) -> r3 still active
    kDriftUpdate = 0x822A5F08,          // flight_drift_update (r3 drift, r5 inputs, f1 dt) -> r3 active
    kDriftPitchAuthority = 0x822A5FB8,  // flight_drift_pitch_authority (r3 drift) -> f1
    kDriftDecel = 0x822A5E90,           // flight_drift_decel (r3 drift, f1 speed) -> f1
    kCameraFollowAttitude = 0x8225E368, // camera_follow_attitude (r3 camera, f1 dt)
};

// A hooked call: opaque, its registers through the Api.
struct Call;
using Hook = void (*)(Call* call);

enum Event : uint32_t
{
    kPackageUnpacked = 1,   // a: the package's index (DATA.TBL, DLC packages have their own numbers)
};
using EventHandler = void (*)(uint32_t event, int64_t a);

enum LogLevel : int { kInfo = 0, kWarn = 1, kError = 2 };

struct Api
{
    uint32_t version;       // kApiVersion of the runtime
    uint8_t* guest_base;    // guest memory
    // The call's argument registers: r3-r10 (64-bit) and f1-f13 (double). Others read 0 and ignore writes.
    uint64_t (*r)(Call* call, int n);
    void (*set_r)(Call* call, int n, uint64_t value);
    double (*f)(Call* call, int n);
    void (*set_f)(Call* call, int n, double value);
    void (*original)(Call* call);   // the game's function, on the call's registers as they are now
    // false: not the start of one of the game's functions (without hook gates: not a Hookable address, or another
    // plugin has it).
    bool (*hook)(uint32_t function, Hook hook);
    void (*on_event)(EventHandler handler);
    // A setting or AC6_* variable as the runtime reads it (the settings overlay's value, ac6_settings.toml's [env], the
    // environment), copied into buf and always NUL-terminated. Returns the value's full length (0 when unset), which
    // is more than size - 1 when buf was too small. settings_generation goes up on each settings change.
    size_t (*setting)(const char* name, char* buf, size_t size);
    uint64_t (*settings_generation)();
    void (*log)(int level, const char* text);   // into the runtime's log, "[plugin] " in front (name yourself in the text)
    uint64_t (*ticks_ms)();                     // milliseconds since the start
};

// Guest memory helpers.
inline uint32_t Load32(const Api* api, uint32_t address)
{
    uint32_t v;
    std::memcpy(&v, api->guest_base + address, 4);
    return __builtin_bswap32(v);
}
inline float LoadFloat(const Api* api, uint32_t address)
{
    const uint32_t v = Load32(api, address);
    float f;
    std::memcpy(&f, &v, 4);
    return f;
}
inline uint8_t Load8(const Api* api, uint32_t address) { return api->guest_base[address]; }
inline void Store32(const Api* api, uint32_t address, uint32_t value)
{
    const uint32_t v = __builtin_bswap32(value);
    std::memcpy(api->guest_base + address, &v, 4);
}
inline void StoreFloat(const Api* api, uint32_t address, float value)
{
    uint32_t v;
    std::memcpy(&v, &value, 4);
    Store32(api, address, v);
}
inline void Store8(const Api* api, uint32_t address, uint8_t value) { api->guest_base[address] = value; }

} // namespace ac6::plugin

using Ac6PluginInit = bool (*)(const ac6::plugin::Api* api);
