# Lua mods

A Lua mod is a folder in `mods/` next to the game with a `mod.toml` and Lua scripts. The game runs the scripts with
LuaJIT (JIT and FFI on). A mod can read and write the game's memory, hook any of the game's functions, call them,
read the controllers, keyboard and mouse, change what the game reads from the controllers, draw over the game, and
load native code.

**Mods run with the game's full access.** A mod can do anything a program on your computer can. Use only mods you
trust.

The API is `GracemeriaRecomp/src/mods/lua/ac6.lua` (the `ac6` global), the runtime's side
`GracemeriaRecomp/src/mods/mods.h`, and the native API `GracemeriaRecomp/src/mods/ac6_mod_api.h`. The examples are in
`examples/mods/`; the build puts them, ready to copy, in `build/example_mods/`.

## A mod's folder

```
mods/my_mod/
  mod.toml        the manifest
  early.lua       run before the window and the graphics device are made (optional)
  init.lua        run before the game's code starts (optional)
  lib/            more Lua, for require() (the mod's folder itself is searched too)
  bin/            its DLLs (.so on Linux): native Lua modules, ac6.native.load
  symbols.toml    names for the game's functions and data (optional)
```

`mod.toml`:

```toml
name = "My mod"
version = "1.0.0"
author = "me"
description = "What it does (the settings overlay shows it)."
api = 2            # the Lua API version it's written for: a newer one than the game's and it isn't loaded
                   # (1: memory, hooks, calls, events; 2: ac6.input, ac6.ui, the draw and pad events)
load_order = 0     # lower first, then by folder name
enabled = true     # false: not loaded
```

Every field is optional, and a folder with only `init.lua` loads with the defaults.

Settings, in `ac6_settings.toml`'s `[env]` or the environment:

| Setting | Effect |
|-|-|
| `AC6_LUA=0` | Loads no Lua mods. |
| `AC6_LUA_DISABLE=a,b` | Leaves those mods (folder names) out. |
| `AC6_LUA_NATIVE=0` | Lua mods can load no DLLs (`ac6.native.load` fails). |
| `AC6_PLUGINS=0` | The plugin loader starts no plugins. |

The settings overlay (F4) lists the mods and plugins on the Game tab, under MODS: each one's state, its hooks, and why
it's off when it is.

## When things run

1. `early.lua` of each mod, in load order: before the window and the Vulkan device. The game isn't loaded yet:
   its memory is empty, and `ac6.game.class` and `Class::vfN` names find nothing. Hooks can be set already.
2. The plugin loader starts the DLLs at the top of each `mods/<mod>/` folder.
3. `init.lua` of each mod: the game is loaded, its code hasn't started.
4. Events, as the game runs. At each presented frame, on the game's main thread, in this order:
   1. `draw`: the layer over the game is drawn;
   2. `pad`, once for each player: the controllers are read, and the handlers change what the game will read;
   3. `frame`: `ac6.input` has this frame's input.

   `mode` comes when the game enters a mode (`CModeTaskTitleMovie`, `CModeTaskGame`, ...).

A script that fails turns its mod off. A load error, or an error in `early.lua` or `init.lua`, is logged with its
traceback, and the mod's hooks are removed, along with what its DLLs registered. An error in a hook or an event handler
is logged and the game goes on. A hook that fails 10 times is removed.

Everything a mod logs goes into `ac6_runtime.log` with `[mod <folder>]` in front.

## The API

### ac6.mod, ac6.log, ac6.settings

```lua
ac6.mod.id, ac6.mod.name, ac6.mod.version, ac6.mod.author, ac6.mod.description, ac6.mod.dir
ac6.mod.path("data/x.bin")    -- a path in the mod's folder
ac6.exe_dir                   -- the game's folder
ac6.api                       -- the Lua API version (1)

ac6.log.info(...), ac6.log.warn(...), ac6.log.error(...)   -- print() is ac6.log.info
ac6.settings.get("AC6_SLOTS")  -- a setting or AC6_* variable as the game reads it ("" when unset)
```

Lua's standard libraries are all there (`io`, `os`, `debug`, `ffi`, `bit`, `jit`); `ac6.ffi` and `ac6.bit` are
LuaJIT's.

### ac6.mem: the game's memory

Addresses are guest addresses, numbers like `0x82000000`. The game's memory is big-endian; these functions convert.

```lua
mem.u8(a)  mem.s8(a)  mem.u16(a)  mem.s16(a)  mem.u32(a)  mem.s32(a)  mem.f32(a)  mem.f64(a)
mem.u64(a)                    -- a uint64_t cdata
mem.ptr32(a)                  -- a guest pointer (as u32)
mem.set_u8(a, v) ... mem.set_f64(a, v), mem.set_ptr32(a, v)

mem.vec3(a) -> x, y, z        mem.set_vec3(a, x, y, z)
mem.vec4(a) -> x, y, z, w     mem.set_vec4(a, x, y, z, w)
mem.mat4(a) -> { 16 floats }  mem.set_mat4(a, m)

mem.string(a [, max])         -- NUL-terminated, 256 bytes at most by default
mem.wstring(a [, max])        -- NUL-terminated UTF-16 (big-endian), as UTF-8
mem.bytes(a, n)  mem.write(a, s)  mem.fill(a, n [, byte])  mem.copy(to, from, n)

mem.valid(a)                  -- whether the address is memory the game can use
mem.alloc(size) -> a          -- zeroed bytes in the game's heap, 16-byte aligned (nil: none)
mem.free(a)
mem.ptr(a)                    -- the host pointer (uint8_t*), for the FFI
mem.base                      -- guest address 0, as a uint8_t*
```

Nothing is checked: an address outside the game's memory crashes the game. Check pointers you follow with `mem.valid`.

Struct views read and write fields in place:

```lua
local Camera = mem.struct{ fov = { "f32", 212 }, aspect = { "f32", 216 }, pos = { "vec3", 0x40 } }
local cam = Camera(address)
cam.fov = cam.fov * 1.2
local other = cam:at(0x300)   -- the same view, 0x300 further
```

Field types are `u8 s8 u16 s16 u32 s32 u64 f32 f64 ptr vec3 vec4 mat4`, with vectors as `{ x, y, z }` tables.

### ac6.sym, ac6.data, ac6.signatures: names

`ac6.sym.name` is a function's address and `ac6.data.name` a global's. `ac6.signatures.name` is a function's
signature, for `ac6.game.fn`. The names come from these sources, the later ones winning:

1. the built-in list, `GracemeriaRecomp/src/mods/symbols.toml`: the functions the runtime itself patches;
2. every `symbols/*.toml` next to the game: `tools/ida/export_symbols.py` writes one from an IDA database;
3. the mod's own `symbols.toml`.

```toml
[functions]
my_function = 0x82123456
[data]
my_global = 0x826E0000
[signatures]
my_function = "f32(ptr, f32)"
```

The game's classes come from its RTTI on top of these. `ac6.sym["ACE6::CAce6CameraManager::vf0"]` is slot 0 of the
class's vtable.

### ac6.game: hooks

```lua
local h = ac6.game.hook(target, hook [, { priority = n }])
h:remove()
```

`target` is an address or an `ac6.sym` name. Any of the game's functions can be hooked: `ac6.game.hookable(a)` says
whether `a` is the start of one, and `ac6.game.functions()` lists them all. Several hooks (from several mods, plugins
and native code) can be on one function. They run highest priority first, equal priorities in the order they were
added, then the game's function.

A hook is a function that runs **in place of** the game's:

```lua
-- (r3 drift, f1 speed) -> f1: the post-stall drift's deceleration.
ac6.game.hook("flight_drift_decel", function(c)
    c.f[1] = c.f[1] * 0.5    -- an argument: the speed it's given
    c:next()                 -- the rest of the chain, then the game's function
    c.f[1] = c.f[1] * 0.9    -- the result
end)
```

Or it can be a table, `{ pre = function(c), post = function(c), priority = n }`. `pre` runs before the game's function,
and returning `false` from it skips both the function and `post`. `post` runs after the function, with the results in
`c.r[3]` and `c.f[1]`.

The call object `c`:

| | |
|-|-|
| `c.fn` | the hooked function's address |
| `c.r[n]` | register rn's low 32 bits: r3-r10 the integer and pointer arguments, r3 the result, r1 the stack pointer, r13 the thread's block |
| `c.r64[n]` | all 64 bits, as a uint64_t |
| `c.f[n]` | float register fn, f1-f13: the float arguments, f1 the float result |
| `c:next()` | the rest of the chain. Call it once, more than once, or not at all (the call skipped: set the results yourself) |

`c` is good only while its hook runs, and is reused for later calls. Using it after its hook has returned is an error:
copy out the values you need.

**Threads.** A hook runs on whatever thread calls the function: the game's main thread, its render thread, its
workers. Each mod's Lua state is used by one thread at a time: a mod's hooks and events run one at a time, under the
mod's lock. The lock is held for the whole hook, including `c:next()` and the game's function under it. So:

- A Lua hook on a function that many threads call makes them take turns. Hook hot functions from native code instead.
- If the function a hook passes on to waits for another game thread, and that thread then runs into a hook of the
  same mod, the game deadlocks. Don't put a `c:next()` hook (or a `pre` / `post` one) on a function that waits for
  other threads when your mod also hooks what those threads run.

### ac6.game: calling the game

```lua
local decel = ac6.game.fn("flight_drift_decel", "f32(ptr, f32)")   -- or the signature from ac6.signatures
local speed = decel(drift, 300)

local r3, f1, r3_high = ac6.game.call(address, { 1, 2, 0x82000000 }, { 1.5 })   -- raw: integers, floats
```

The signature types are `void bool s8 u8 s16 u16 s32 u32 s64 u64 ptr f32 f64`. Integers and pointers go in r3-r10,
and past 8 on the stack (16 at most). Floats go in f1-f13, and `f32` arguments are rounded to single precision.
Integers are numbers, or 64-bit cdata (`5ULL`, `-2LL`) for values a number can't hold; `s64` / `u64` results are
cdata.

The game's functions are called from a game thread only: in a hook, or the `frame` event. `ac6.game.is_game_thread()`
says whether the script is on one.

### ac6.game: classes and the rest

```lua
ac6.game.rtti(object) -> "ACE6::CAce6CameraManager", subobject offset   -- or nil
ac6.game.class(name)  -> { name, mangled, type_descriptor, vtables = { { address, offset, slots }, ... } }
ac6.game.classes()    -> every class's name
ac6.game.vfunc(object_or_class, slot) -> the virtual function's address
ac6.game.vcall(object, slot, signature, ...)   -- a virtual call, the object in r3
ac6.game.name_of(address)   -- a name ac6.sym has for it
ac6.game.mode()             -- the mode the game is in
ac6.game.frame()            -- frames presented so far
ac6.game.time()             -- seconds since the start
```

### ac6.events

```lua
local e = ac6.events.on("frame", function(dt, frame) ... end)   -- the main thread, game calls allowed
ac6.events.on("mode", function(mode) ... end)                    -- "CModeTaskGame", ...
ac6.events.on("draw", function() ... end)                        -- draw with ac6.ui (below)
ac6.events.on("pad", function(user, pad) ... end)                -- change what the game reads (below)
e:remove()
```

`frame`, `draw` and `pad` run on the game's main thread, where the game's functions can be called. `mode` runs on the
thread that changed the mode.

### ac6.input: controllers, keyboard, mouse

What the player is doing, as of this frame, from any thread. While the settings overlay is open, nothing is held or
pressed.

```lua
local pad = ac6.input.pad()        -- player 1 (ac6.input.pad(1) is player 2, ... 3)
-- { connected, buttons, lt, rt, lx, ly, rx, ry, pressed }: triggers 0-255, sticks -32768-32767,
-- `buttons` held and `pressed` gone down since the frame before (bits: ac6.input.BUTTONS)
ac6.input.button("a")              -- held (player 1; a second argument picks another)
ac6.input.button_pressed("start")  -- went down this frame
ac6.input.game_pad()               -- what the game reads (after the mods' pad handlers)

ac6.input.key("LeftControl")       -- held
ac6.input.key_pressed("F6")        -- went down this frame (a tap between two frames counts)

local m = ac6.input.mouse()
-- { x, y, wheel, left, right, middle, x1, x2, pressed = { left, ... } }: the cursor in window units (ac6.ui's),
-- the buttons held and gone down this frame, the wheel's notches this frame (up positive)
```

Button names are `a b x y lb rb ls rs start back up down left right`. Keys use the names of the keyboard bindings in
`ac6_settings.toml`: `A`-`Z`, `0`-`9`, `F1`-`F24`, `Space`, `Return`, `Escape`, `Tab`, `Backspace`, `Up` / `Down` /
`Left` / `Right`, `LeftShift`, `RightControl`, `LeftAlt`, `Shift` / `Control` / `Alt` (either side), `Numpad0`-`Numpad9`,
`NumpadEnter`, `Home`, `End`, `PageUp`, `Insert`, `Delete`, `Backtick`, `Minus`, `Plus`, `Comma`, `Period`, `Slash`,
`Mouse1` (left) to `Mouse5`, and so on. An unknown name is an error.

A key a mod uses as a hotkey still reaches the game. With keyboard controls on (`AC6_KBM=1`), pick keys the game's
bindings don't use.

### The pad event: changing what the game reads

At each frame, before the game reads the controllers, each `pad` handler gets each player's pad (`user` 0-3) as a
table. Whatever the handler leaves in the table is what the game reads. Mods run in load order, each on the one
before's result.

```lua
-- A and B swapped for player 1.
ac6.events.on("pad", function(user, pad)
    if user ~= 0 then return end
    local a, b = ac6.input.has(pad, "a"), ac6.input.has(pad, "b")
    ac6.input.set(pad, "a", b)
    ac6.input.set(pad, "b", a)
end)
```

The fields are those of `ac6.input.pad()` (`connected`, `buttons`, `lt`, `rt`, `lx`, `ly`, `rx`, `ry`).
`ac6.input.has(pad, name)` and `ac6.input.set(pad, name, down)` read and change one button. `ac6.input.pad()` keeps
showing what the player did; `ac6.input.game_pad()` shows what the game got. With the settings overlay open, the game
gets idle pads and the handlers aren't called.

### ac6.ui: drawing over the game

`draw` handlers draw a layer over the game's picture: under the settings overlay, over everything else. `ac6.ui`
works only inside them; anywhere else it's an error. The layer is drawn again from nothing at each frame.

Coordinates are window units, with (0, 0) at the window's top left: the same units as the mouse. Colours are
`0xAARRGGBB` numbers. `0xFF` in the top byte is opaque and `0x00` is transparent: `0x00FF0000` is invisible.
`ui.rgb(0xFF0000)` is opaque red, `ui.rgb(0xFF0000, 0.5)` half transparent, and `ui.color(255, 0, 0 [, alpha 0-255])`
also works. A colour left out is white.

```lua
ac6.events.on("draw", function()
    local ui = ac6.ui
    local s = ui.scale()                         -- the window against 1920x1080: multiply sizes by it
    local x, y = ui.from_canvas(40, 40)          -- the game's 1280x720 canvas, in window units
    ui.fill_rect(x, y, x + 300 * s, y + 60 * s, ui.rgb(0x04051A, 0.7), 6 * s)
    ui.text(x + 12 * s, y + 18 * s, "Hello from a mod", ui.rgb(0xE4E9FA), 24 * s)
end)
```

| | |
|-|-|
| `ui.size()` | the window: width, height |
| `ui.view()` | the game's picture in it (letterboxed): x, y, width, height |
| `ui.canvas()` | the game's 2D canvas in the picture, 16:9, where it draws its HUD and menus: x, y, width, height |
| `ui.from_canvas(cx, cy)` | a point of the 1280x720 canvas (the game's HUD coordinates) in window units |
| `ui.scale()` | min(width / 1920, height / 1080) |
| `ui.line(x0, y0, x1, y1, color [, thickness])` | |
| `ui.rect(x0, y0, x1, y1, color [, thickness [, rounding]])` / `ui.fill_rect(x0, y0, x1, y1, color [, rounding])` | |
| `ui.circle(x, y, r, color [, thickness])` / `ui.fill_circle(x, y, r, color)` | |
| `ui.triangle(x0, y0, x1, y1, x2, y2, color [, thickness])` / `ui.fill_triangle(...)` | |
| `ui.polyline({ x0, y0, x1, y1, ... }, color [, thickness [, closed]])` / `ui.fill_polygon(points, color)` | a filled polygon is convex |
| `ui.text(x, y, text, color [, size [, mono]])` | at its top left; `size` its height (18 by default); `mono` the font with equal-width digits |
| `ui.text_size(text [, size [, mono]])` | width, height |
| `ui.clip(x0, y0, x1, y1, fn)` | `fn()` draws clipped to the rectangle |

The game's own world-to-screen projection (for markers over aircraft, say) gives 1280x720 canvas coordinates:
`ui.from_canvas` turns them into window units.

Drawing is immediate and has no widgets: there are no buttons or text fields to click. Take input from
`ac6.input`.

## Native code

`ac6.native.load("bin/my_module.dll")` loads a DLL (`.so` on Linux) from the mod's folder, and returns what its
`luaopen_my_module` returns (or `true`). The DLL can be any of these, or all of them at once:

- a **native mod**: it exports `ac6_mod_init(const ac6_mod_api*)` (`GracemeriaRecomp/src/mods/ac6_mod_api.h`, a C
  header). From there it hooks functions in C, with nothing but a call through the chain added to a hooked call. It
  also gets frame and mode events, calls the game, and allocates in the game's heap. Version 2 of the header adds the
  same input (`pad`, `key`, `mouse`, the `on_pad` filter) and drawing (`on_draw`, `draw_*`) as Lua has. Check
  `AC6_MOD_API_HAS(api, on_draw)` before using them, so the DLL still loads in an older game;
- a **plugin** of the older API, `ac6_plugin_init` (`GracemeriaRecomp/src/plugin/ac6_plugin.h`);
- a **Lua module**: `luaopen_<file name>`. It links against the game's LuaJIT, `lua51.dll` next to the executable.

A DLL is started (its `ac6_mod_init` / `ac6_plugin_init` run) once. Loading the same file again returns what the first
load did.

**Keep a Lua mod's DLLs out of the top of its folder**, in `bin/` for instance. The plugin loader starts every DLL at
the top of a `mods/<mod>/` folder as a plugin before `init.lua` runs. A Lua mod's load of such a DLL then only opens it
as a Lua module: its init isn't run again, and its hooks belong to the plugin, not to the mod. The log warns when that
happens.

A DLL meant to run without Lua is a plugin: put it at the top of its folder, with no scripts
(`examples/mods/aircraft_slots`).

To build one on Windows with the toolchain the game's build downloaded, from the repository's top folder (after the
game's build, which makes `lua51.dll` and its import library):

```bat
toolchain\llvm-mingw\bin\x86_64-w64-mingw32-clang.exe -shared -O2 -I GracemeriaRecomp\src\mods ^
    -I thirdparty\LuaJIT\src -I build\luajit my_module.c build\liblua51.dll.a -o mods\my_mod\bin\my_module.dll
```

On Linux, build a `.so` against `build/libluajit-5.1.so`.

## Examples

| | |
|-|-|
| `examples/mods/aircraft_list` | Reads the game's tables and logs the aircraft. Hooks the hangar's list. |
| `examples/mods/input_display` | Draws player 1's controller over the game (F6 hides it), and swaps A and B through the pad event (F7). |

## When something goes wrong

Look in `ac6_runtime.log`:

- `Lua mods: N found` lists the mods; `[mod x] not loaded: ...` says why one isn't loaded.
- `[mod x] ...; the mod is off` is a script that failed, with its traceback.
- `[mod x] hook on 8224C788: ...` is an error in a hook.
- `Hooks: ... asked for X, which isn't a hookable function` means the address isn't a function's start.
- `Plugins: ...` lines are for the DLLs the plugin loader started.
