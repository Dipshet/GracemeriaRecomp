/* The native mod API: what a DLL (.so) gets from the runtime. A C header: mods are written in C or C++ and don't link
 * against the runtime. The DLL exports
 *
 *     AC6_MOD_EXPORT int ac6_mod_init(const ac6_mod_api* api);
 *
 * and is loaded one of two ways:
 *   - as a plugin: mods/<mod>/<name>.dll, at the top of the mod's folder, found by the plugin loader before the game's
 *     code starts (no Lua needed; api->lua is NULL, api->mod_dir the DLL's folder, api->mod_id "<mod>/<name>.dll");
 *   - by a Lua mod's script (ac6.native.load), from anywhere in the mod's folder but its top (bin/ for instance: the
 *     plugin loader would start a DLL there first, as a plugin). ac6_mod_init runs before luaopen_<name>, if it exports
 *     that too.
 * Either way a DLL is started once: ac6_mod_init isn't called again when another load finds it started. Returning 0
 * declines: the plugin is unloaded (or the script's load fails), and the hooks and handlers it registered are removed.
 * A Lua mod whose scripts fail is turned off with what its DLLs registered (their code stays loaded).
 *
 * Hooks run in place of the game's function on whatever thread calls it, with no lock taken: a native hook pays a
 * call through the chain, nothing else. A hook passes the call on with call_next (the next hook, or the game's
 * function), or not. Guest memory is big-endian, at guest_base + address. Calls into the game (guest_call) are made
 * from a game thread only: a hook, or the frame handler.
 *
 * Native code can crash or harm the game and the machine like any program. Run only mods you trust.
 * GracemeriaRecomp/src/mods/mods.h has the rest of the picture; plugin/ac6_plugin.h is the older plugin API (both can
 * be exported by one DLL). */
#ifndef AC6_MOD_API_H
#define AC6_MOD_API_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#define AC6_MOD_EXPORT __declspec(dllexport)
#else
#define AC6_MOD_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* 1: hooks, game calls, events, RTTI. 2: input and drawing (the fields after `setting`: check api->size, an older
 * runtime's api ends before them). */
#define AC6_MOD_API_VERSION 2

struct lua_State;
typedef struct ac6_call ac6_call;   /* a hooked call */
typedef void (*ac6_hook_fn)(ac6_call* call, void* user);
typedef void (*ac6_frame_fn)(double dt, uint64_t frame, void* user);
typedef void (*ac6_mode_fn)(const char* mode, void* user);

enum { AC6_LOG_INFO = 0, AC6_LOG_WARN = 1, AC6_LOG_ERROR = 2 };

/* A pad as the game reads it (XINPUT_GAMEPAD): its buttons (AC6_PAD_*), triggers 0-255, sticks -32768-32767. */
typedef struct ac6_pad
{
    uint16_t buttons;
    uint8_t left_trigger, right_trigger;
    int16_t lx, ly, rx, ry;
    uint8_t connected;
} ac6_pad;
enum
{
    AC6_PAD_UP = 0x0001, AC6_PAD_DOWN = 0x0002, AC6_PAD_LEFT = 0x0004, AC6_PAD_RIGHT = 0x0008,
    AC6_PAD_START = 0x0010, AC6_PAD_BACK = 0x0020, AC6_PAD_LS = 0x0040, AC6_PAD_RS = 0x0080,
    AC6_PAD_LB = 0x0100, AC6_PAD_RB = 0x0200, AC6_PAD_A = 0x1000, AC6_PAD_B = 0x2000, AC6_PAD_X = 0x4000, AC6_PAD_Y = 0x8000
};
/* The mouse: the cursor in window units, its buttons held and gone down since the frame before (AC6_MOUSE_*), the
 * wheel's notches in that frame (up positive). */
typedef struct ac6_mouse
{
    float x, y;
    uint32_t buttons, pressed;
    float wheel;
} ac6_mouse;
enum { AC6_MOUSE_LEFT = 1, AC6_MOUSE_MIDDLE = 2, AC6_MOUSE_RIGHT = 4, AC6_MOUSE_X1 = 8, AC6_MOUSE_X2 = 16 };
typedef void (*ac6_pad_fn)(uint32_t user, ac6_pad* pad, void* data);
typedef void (*ac6_draw_fn)(void* user);

typedef struct ac6_mod_api
{
    uint32_t version;   /* AC6_MOD_API_VERSION of the runtime */
    uint32_t size;      /* sizeof(ac6_mod_api) of the runtime: newer fields are past older runtimes' size */

    /* The mod that loaded the DLL. */
    const char* mod_id;
    const char* mod_dir;            /* UTF-8, '/' separated */
    struct lua_State* lua;          /* its Lua state: only from the mod's own Lua calls (luaopen_*, functions it gave Lua) */
    uint8_t* guest_base;            /* guest memory: guest address a is at guest_base + a */

    /* Hooks. hook(): 0 when `function` isn't the start of one of the game's functions. Higher priorities run first.
     * (The functions that act for the mod take its api first.) */
    uint64_t (*hook)(const struct ac6_mod_api* api, uint32_t function, ac6_hook_fn fn, void* user, int32_t priority);
    int (*unhook)(const struct ac6_mod_api* api, uint64_t id);
    int (*hookable)(uint32_t function);

    /* A hooked call: its registers (64-bit integer registers r1, r3-r10, r13; floating-point f1-f13; NULL for others),
     * the hooked function's address, and the rest of its chain. */
    uint64_t* (*reg)(ac6_call* call, int n);
    double* (*freg)(ac6_call* call, int n);
    uint32_t (*function)(ac6_call* call);
    void (*call_next)(ac6_call* call);

    /* A call into the game, from a game thread: integer arguments in r3... (up to 16: past 8 on the stack), floats in
     * f1... (up to 13). Results in *r3 / *f1 (either may be NULL). Returns 0 when it couldn't call (not a game thread,
     * not a function). */
    int (*guest_call)(uint32_t function, const uint64_t* ints, int int_count, const double* floats, int float_count,
        uint64_t* r3, double* f1);
    int (*is_game_thread)(void);
    /* Zeroed guest heap memory, 16-byte aligned: its guest address (0: none). */
    uint32_t (*guest_alloc)(uint32_t size);
    void (*guest_free)(uint32_t address);

    /* Events. frame: on the game's main thread at each presented frame (guest calls allowed). mode: a mode task entered
     * ("CModeTaskGame", ...). */
    void (*on_frame)(const struct ac6_mod_api* api, ac6_frame_fn fn, void* user);
    void (*on_mode)(const struct ac6_mod_api* api, ac6_mode_fn fn, void* user);

    /* The game's classes (RTTI): an object's class name (NULL: none; the string lives as long as the process), and a
     * class's virtual function (slot from 0 in its primary vtable; 0: none). */
    const char* (*rtti_of)(uint32_t object);
    uint32_t (*vfunc)(const char* class_name, uint32_t slot);

    void (*log)(const struct ac6_mod_api* api, int level, const char* text);   /* ac6_runtime.log, "[mod <id>] " in front */
    /* A setting or AC6_* variable as the runtime reads it, copied into buf (always NUL-terminated). Returns its full length
     * (0: unset). */
    size_t (*setting)(const char* name, char* buf, size_t size);

    /* --- Version 2 ------------------------------------------------------------------------------------------------ */

    /* Input, from any thread, as of the last frame (nothing held while the settings overlay is open). pad: player
     * `user`'s (0-3) as the player left it (game 0: before the mods' pad filters, `pressed` its buttons gone down since
     * the frame before; may be NULL) or as the game reads it (game 1). Returns 0 for no such player. key: a key or mouse
     * button by the settings file's names ("F5", "LeftControl", "Numpad1", "Mouse1"): -1 no such key, else 1 held or 0;
     * *pressed (may be NULL): gone down since the frame before. */
    int (*pad)(uint32_t user, int game, ac6_pad* out, uint16_t* pressed);
    int (*key)(const char* name, int* pressed);
    void (*mouse)(ac6_mouse* out);
    /* The pad filter: each player's pad at each frame, before the game reads it (main thread, the overlay closed):
     * change *pad. Filters run in the mods' order, each on the one before's result. */
    void (*on_pad)(const struct ac6_mod_api* api, ac6_pad_fn fn, void* user);

    /* Drawing: a layer over the game (under the settings overlay). on_draw's handlers run at each frame on the main
     * thread (game calls allowed). The draw_* functions draw only from them, and do nothing elsewhere. Window units
     * ((0, 0) the top left; draw_size the window, draw_view the game's picture in it, draw_canvas the game's 1280x720
     * 2D canvas, 16:9 in the picture). Colours 0xAARRGGBB (0xFF in the top byte: opaque). thickness <= 0:
     * filled (a filled polyline: closed and convex). Text: at its top left, `size` its height (0: 18), mono the
     * overlay's numbers' font. Clips are popped after each handler. */
    void (*on_draw)(const struct ac6_mod_api* api, ac6_draw_fn fn, void* user);
    void (*draw_size)(float* width, float* height);
    void (*draw_view)(float* x, float* y, float* width, float* height);
    void (*draw_canvas)(float* x, float* y, float* width, float* height);
    void (*draw_line)(float x0, float y0, float x1, float y1, uint32_t color, float thickness);
    void (*draw_rect)(float x0, float y0, float x1, float y1, uint32_t color, float rounding, float thickness);
    void (*draw_circle)(float x, float y, float radius, uint32_t color, float thickness);
    void (*draw_triangle)(float x0, float y0, float x1, float y1, float x2, float y2, uint32_t color, float thickness);
    void (*draw_polyline)(const float* xy, int points, uint32_t color, int closed, float thickness);
    void (*draw_text)(float x, float y, const char* text, uint32_t color, float size, int mono);
    void (*text_size)(const char* text, float size, int mono, float* width, float* height);
    void (*draw_push_clip)(float x0, float y0, float x1, float y1);
    void (*draw_pop_clip)(void);
} ac6_mod_api;

/* Whether the runtime's api has a field (an older runtime's ends before the newer ones):
 * if (AC6_MOD_API_HAS(api, on_draw)) ... */
#define AC6_MOD_API_HAS(api, field) ((api)->size >= offsetof(ac6_mod_api, field) + sizeof((api)->field))

typedef int (*ac6_mod_init_fn)(const ac6_mod_api* api);

/* Guest memory helpers. */
static inline uint32_t ac6_load32(const ac6_mod_api* api, uint32_t address)
{
    uint32_t v;
    __builtin_memcpy(&v, api->guest_base + address, 4);
    return __builtin_bswap32(v);
}
static inline float ac6_load_float(const ac6_mod_api* api, uint32_t address)
{
    const uint32_t v = ac6_load32(api, address);
    float f;
    __builtin_memcpy(&f, &v, 4);
    return f;
}
static inline void ac6_store32(const ac6_mod_api* api, uint32_t address, uint32_t value)
{
    const uint32_t v = __builtin_bswap32(value);
    __builtin_memcpy(api->guest_base + address, &v, 4);
}
static inline void ac6_store_float(const ac6_mod_api* api, uint32_t address, float value)
{
    uint32_t v;
    __builtin_memcpy(&v, &value, 4);
    ac6_store32(api, address, v);
}

#ifdef __cplusplus
}
#endif

#endif
