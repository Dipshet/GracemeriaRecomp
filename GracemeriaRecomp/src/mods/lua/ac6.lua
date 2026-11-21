-- The Lua mods' API: the `ac6` global of every mod's scripts (GracemeriaRecomp/src/mods/mods.h).
-- Built on ac6.core (src/mods/mods.cpp), this chunk's argument. Embedded in the executable.
--
--   ac6.mod        the mod: id, name, version, author, description, dir, path(relative)
--   ac6.log        info / warn / error (print goes to info): into ac6_runtime.log, "[mod <id>]" in front
--   ac6.settings   get(name): a setting or AC6_* variable as the runtime reads it
--   ac6.mem        guest memory (big-endian): u8 ... f64 reads, set_* writes, vec3 / vec4 / mat4, strings, struct views,
--                  alloc / free in the guest heap. mem.base is guest address 0 as an FFI uint8_t* (host memory)
--   ac6.game       hook any of the game's functions, call them (fn: typed, call: raw registers), the mode and frame
--   ac6.sym        names for some of the game's functions (their guest addresses)
--   ac6.events     on(name, handler): "frame" (dt, frame number: the game's main thread, game calls allowed),
--                  "mode" (the mode task entered, e.g. "CModeTaskGame"), "draw" (ac6.ui), "pad" (user, pad: change
--                  what the game reads)
--   ac6.input      the pads, the keyboard and the mouse as the player uses them
--   ac6.ui         drawing over the game, from "draw" handlers
--   ac6.native     load(path): a DLL from the mod's folder (a Lua module, a plugin, or both)
--   ac6.ffi, ac6.bit  LuaJIT's
local core = ...
local ffi = require("ffi")
local bit = require("bit")

local ac6 = { core = core, ffi = ffi, bit = bit }
local info = core.info()
ac6.api = info.api

local format, concat, select, tostring, type = string.format, table.concat, select, tostring, type

------------------------------------------------------------------------------------------------------------------------
-- ac6.mod
------------------------------------------------------------------------------------------------------------------------
ac6.mod = {
    id = info.id, name = info.name, version = info.version, author = info.author, description = info.description,
    dir = info.dir,
}
-- A path inside the mod's folder.
function ac6.mod.path(relative)
    return info.dir .. "/" .. relative
end
ac6.exe_dir = info.exe_dir

------------------------------------------------------------------------------------------------------------------------
-- ac6.log
------------------------------------------------------------------------------------------------------------------------
local function text(...)
    local n = select("#", ...)
    if n == 1 then return tostring((...)) end
    local parts = {}
    for i = 1, n do parts[i] = tostring((select(i, ...))) end
    return concat(parts, "\t")
end

ac6.log = {
    info = function(...) core.log(0, text(...)) end,
    warn = function(...) core.log(1, text(...)) end,
    error = function(...) core.log(2, text(...)) end,
}
print = ac6.log.info

------------------------------------------------------------------------------------------------------------------------
-- ac6.settings
------------------------------------------------------------------------------------------------------------------------
ac6.settings = {
    get = function(name) return core.setting(name) end,
}

------------------------------------------------------------------------------------------------------------------------
-- ac6.mem: guest memory. Addresses are guest addresses (numbers); values are big-endian in memory.
------------------------------------------------------------------------------------------------------------------------
local base = ffi.cast("uint8_t*", core.guest_base())
local u8p, u16p, u32p, u64p = ffi.typeof("uint8_t*"), ffi.typeof("uint16_t*"), ffi.typeof("uint32_t*"), ffi.typeof("uint64_t*")
local i64p, f64p = ffi.typeof("int64_t*"), ffi.typeof("double*")
local bswap, band, bor, lshift, rshift = bit.bswap, bit.band, bit.bor, bit.lshift, bit.rshift
local conv = ffi.new("union { uint32_t u; int32_t i; float f; uint64_t u64; double d; }")
local uint64 = ffi.typeof("uint64_t")

local mem = { base = base }

function mem.ptr(address) return base + address end   -- the host pointer (uint8_t*) to a guest address

function mem.u8(a) return base[a] end
function mem.s8(a) local v = base[a]; return v >= 128 and v - 256 or v end
function mem.u16(a) local v = ffi.cast(u16p, base + a)[0]; return bor(rshift(v, 8), band(lshift(v, 8), 0xFF00)) end
function mem.s16(a) local v = mem.u16(a); return v >= 32768 and v - 65536 or v end
function mem.u32(a) local v = bswap(ffi.cast(u32p, base + a)[0]); return v < 0 and v + 4294967296 or v end
function mem.s32(a) return bswap(ffi.cast(u32p, base + a)[0]) end
function mem.u64(a) return bswap(ffi.cast(u64p, base + a)[0]) end   -- (a uint64_t cdata)
function mem.f32(a) conv.i = bswap(ffi.cast(u32p, base + a)[0]); return conv.f end
function mem.f64(a) conv.u64 = bswap(ffi.cast(u64p, base + a)[0]); return conv.d end
mem.ptr32 = mem.u32   -- a guest pointer

function mem.set_u8(a, v) base[a] = v end
function mem.set_u16(a, v) v = band(v, 0xFFFF); ffi.cast(u16p, base + a)[0] = bor(rshift(v, 8), lshift(band(v, 0xFF), 8)) end
function mem.set_u32(a, v) ffi.cast(u32p, base + a)[0] = bswap(v) end
function mem.set_u64(a, v) ffi.cast(u64p, base + a)[0] = bswap(ffi.cast(uint64, v)) end
function mem.set_f32(a, v) conv.f = v; ffi.cast(u32p, base + a)[0] = bswap(conv.i) end
function mem.set_f64(a, v) conv.d = v; ffi.cast(u64p, base + a)[0] = bswap(conv.u64) end
mem.set_s8, mem.set_s16, mem.set_s32, mem.set_ptr32 = mem.set_u8, mem.set_u16, mem.set_u32, mem.set_u32

-- Vectors and matrices of floats: vec3 / vec4 return their components, mat4 a table of 16 (as stored: row by row).
function mem.vec3(a) return mem.f32(a), mem.f32(a + 4), mem.f32(a + 8) end
function mem.vec4(a) return mem.f32(a), mem.f32(a + 4), mem.f32(a + 8), mem.f32(a + 12) end
function mem.set_vec3(a, x, y, z) mem.set_f32(a, x); mem.set_f32(a + 4, y); mem.set_f32(a + 8, z) end
function mem.set_vec4(a, x, y, z, w) mem.set_f32(a, x); mem.set_f32(a + 4, y); mem.set_f32(a + 8, z); mem.set_f32(a + 12, w) end
function mem.mat4(a)
    local m = {}
    for i = 0, 15 do m[i + 1] = mem.f32(a + i * 4) end
    return m
end
function mem.set_mat4(a, m)
    for i = 0, 15 do mem.set_f32(a + i * 4, m[i + 1]) end
end

-- Bytes and strings.
function mem.bytes(a, n) return ffi.string(base + a, n) end
function mem.write(a, s) ffi.copy(base + a, s, #s) end
function mem.fill(a, n, byte) ffi.fill(base + a, n, byte or 0) end
function mem.copy(to, from, n) ffi.copy(base + to, base + from, n) end
-- A NUL-terminated string (at most `max` bytes, 256 by default).
function mem.string(a, max)
    max = max or 256
    local n = 0
    while n < max and base[a + n] ~= 0 do n = n + 1 end
    return ffi.string(base + a, n)
end
-- A NUL-terminated UTF-16 (big-endian) string, as UTF-8.
function mem.wstring(a, max)
    max = max or 256
    local out, i = {}, 0
    while i < max do
        local c = mem.u16(a + i * 2)
        if c == 0 then break end
        if c >= 0xD800 and c < 0xDC00 then   -- a surrogate pair
            local low = mem.u16(a + i * 2 + 2)
            c = 0x10000 + (c - 0xD800) * 0x400 + (low - 0xDC00)
            i = i + 1
        end
        if c < 0x80 then out[#out + 1] = string.char(c)
        elseif c < 0x800 then out[#out + 1] = string.char(0xC0 + rshift(c, 6), 0x80 + band(c, 0x3F))
        elseif c < 0x10000 then out[#out + 1] = string.char(0xE0 + rshift(c, 12), 0x80 + band(rshift(c, 6), 0x3F), 0x80 + band(c, 0x3F))
        else out[#out + 1] = string.char(0xF0 + rshift(c, 18), 0x80 + band(rshift(c, 12), 0x3F), 0x80 + band(rshift(c, 6), 0x3F), 0x80 + band(c, 0x3F)) end
        i = i + 1
    end
    return concat(out)
end

-- Whether a guest address is in memory the game can use (a page of it may still be uncommitted).
function mem.valid(a) return core.backed(a) end
-- Zeroed bytes in the guest heap (16-byte aligned): their guest address. Freed with mem.free.
function mem.alloc(size) return core.alloc(size) end
function mem.free(a) core.free(a) end

-- Struct views: mem.struct{ name = { type, offset }, ... } makes a constructor; view = Ctor(address) reads and writes
-- the fields in place (view.fov = 60). Types: u8 s8 u16 s16 u32 s32 u64 f32 f64 ptr vec3 vec4 mat4. view._address is the
-- address; view:at(offset) another view of the same type there.
local readers = { u8 = mem.u8, s8 = mem.s8, u16 = mem.u16, s16 = mem.s16, u32 = mem.u32, s32 = mem.s32, u64 = mem.u64, f32 = mem.f32,
    f64 = mem.f64, ptr = mem.u32, vec3 = function(a) return { mem.vec3(a) } end, vec4 = function(a) return { mem.vec4(a) } end,
    mat4 = mem.mat4 }
local writers = { u8 = mem.set_u8, s8 = mem.set_u8, u16 = mem.set_u16, s16 = mem.set_u16, u32 = mem.set_u32, s32 = mem.set_u32,
    u64 = mem.set_u64, f32 = mem.set_f32, f64 = mem.set_f64, ptr = mem.set_u32,
    vec3 = function(a, v) mem.set_vec3(a, v[1], v[2], v[3]) end, vec4 = function(a, v) mem.set_vec4(a, v[1], v[2], v[3], v[4]) end,
    mat4 = mem.set_mat4 }
function mem.struct(fields)
    for name, f in pairs(fields) do
        if not readers[f[1]] then error(format("ac6.mem.struct: field %s has an unknown type %s", name, tostring(f[1])), 2) end
    end
    local mt = {}
    mt.__index = function(view, key)
        local f = fields[key]
        if f then return readers[f[1]](rawget(view, "_address") + f[2]) end
        if key == "at" then return function(self, offset) return setmetatable({ _address = rawget(self, "_address") + offset }, mt) end end
        return nil
    end
    mt.__newindex = function(view, key, value)
        local f = fields[key]
        if not f then error(format("ac6.mem.struct: no field %s", tostring(key)), 2) end
        writers[f[1]](rawget(view, "_address") + f[2], value)
    end
    mt.__tostring = function(view) return format("struct view at %08X", rawget(view, "_address")) end
    return function(address) return setmetatable({ _address = address }, mt) end
end

ac6.mem = mem

------------------------------------------------------------------------------------------------------------------------
-- ac6.sym: names for the game's functions -> their addresses. The built-in list (src/mods/symbols.toml), symbols/*.toml
-- next to the executable, the mod's own symbols.toml; and "Class::vfN" for any class in the game's RTTI (slot N of its
-- primary vtable, e.g. "ACE6::CAce6CameraManager::vf0"). ac6.signatures: their signatures, where known.
------------------------------------------------------------------------------------------------------------------------
local symbols = core.symbols()
ac6.signatures = symbols.signatures
ac6.symbol_sources = symbols.sources
ac6.data = symbols.data   -- names for the game's data (globals) -> their addresses

local function vfunc_of_class(class, slot)
    local c = core.rtti_class(class)
    local vt = c and c.vtables[1]
    if not vt or slot < 0 or slot >= vt.slots then return nil end
    return mem.u32(vt.address + slot * 4)
end

ac6.sym = setmetatable(symbols.functions, {
    __index = function(t, name)
        if type(name) ~= "string" then return nil end
        local class, slot = name:match("^(.-)::vf(%d+)$")
        local address = class and vfunc_of_class(class, tonumber(slot))
        if address then rawset(t, name, address) end
        return address
    end,
})

local function resolve(target, level)
    if type(target) == "number" then return target end
    local address = ac6.sym[target]
    if not address then error(format("no function named %s in ac6.sym", tostring(target)), level + 1) end
    return address
end

------------------------------------------------------------------------------------------------------------------------
-- ac6.game
------------------------------------------------------------------------------------------------------------------------
local game = {}
local regs = core.registers()
local roff, foff = regs.r, regs.f

-- A hooked call, as a hook gets it:
--   c.fn          the hooked function's address
--   c.r[n]        guest register rn's low 32 bits (r1, r3-r10, r13): arguments in r3.., the result in r3
--   c.r64[n]      all 64 bits (a uint64_t)
--   c.f[n]        floating-point register fn (f1-f13): float arguments, the float result in f1
--   c:next()      the rest of the chain: the next hook, or the game's function (more than once, or not at all)
-- The object is good only while its hook runs (the registers and the call are gone after it returns: using it then is
-- an error), and is reused for later calls: copy out what's needed later.
local function new_call()
    local c = { _ctx = nil, _call = nil, fn = 0 }
    local function at(table, n, kind)
        local ctx = c._ctx
        if not ctx then error("this hooked call has returned: use its call object only inside its hook", 3) end
        local o = table[n]
        if not o then error(format("%s%s isn't available to hooks", kind, tostring(n)), 3) end
        return ctx + o
    end
    c.r = setmetatable({}, {
        __index = function(_, n) return ffi.cast(u32p, at(roff, n, "r"))[0] end,
        __newindex = function(_, n, v) ffi.cast(i64p, at(roff, n, "r"))[0] = v end,
    })
    c.r64 = setmetatable({}, {
        __index = function(_, n) return ffi.cast(u64p, at(roff, n, "r"))[0] end,
        __newindex = function(_, n, v) ffi.cast(u64p, at(roff, n, "r"))[0] = v end,
    })
    c.f = setmetatable({}, {
        __index = function(_, n) return ffi.cast(f64p, at(foff, n, "f"))[0] end,
        __newindex = function(_, n, v) ffi.cast(f64p, at(foff, n, "f"))[0] = v end,
    })
    function c:next()
        if not self._call then error("this hooked call has returned: call c:next() only inside its hook", 2) end
        core.call_next(self._call)
    end
    return c
end

local pool, depth = {}, 0
local traceback = debug.traceback

-- core's hook dispatcher: runs the mod's hook function with the call. Returns nil, or the error with its traceback.
-- (One call object per nesting level, emptied when its hook returns: the registers and the call it points at live on
-- the runtime's stack only until then.)
local function hook_sink(fn, luacall, ctx, address)
    depth = depth + 1
    local c = pool[depth]
    if not c then c = new_call(); pool[depth] = c end
    c._ctx, c._call, c.fn = ffi.cast(u8p, ctx), luacall, address
    local ok, err = xpcall(fn, traceback, c)
    c._ctx, c._call = nil, nil
    depth = depth - 1
    if not ok then return tostring(err) end
end

-- ac6.game.hook(function, hook [, options]) -> handle (handle:remove()).
--   function   a guest address, or a name in ac6.sym
--   hook       a function(c) that runs in place of the game's (c:next() runs it: not calling it skips the call), or
--              { pre = function(c), post = function(c), priority = n }: pre before the game's function (returning
--              false skips it and post), post after it (the results in c.r[3] / c.f[1])
--   options    { priority = n }: higher first in the function's chain (default 0)
function game.hook(target, hook, options)
    local address = resolve(target, 2)
    local fn, priority
    if type(hook) == "function" then
        fn, priority = hook, options and options.priority or 0
    elseif type(hook) == "table" then
        local pre, post = hook.pre, hook.post
        priority = hook.priority or (options and options.priority) or 0
        fn = function(c)
            if pre and pre(c) == false then return end
            c:next()
            if post then post(c) end
        end
    else
        error("ac6.game.hook: the hook is a function or a { pre, post } table", 2)
    end
    local id, why = core.hook(address, fn, priority)
    if not id then error(format("ac6.game.hook(%s): %s", tostring(target), why), 2) end
    local handle = { id = id, address = address }
    function handle:remove() return core.unhook(self.id) end
    return handle
end

-- Whether a guest address is the start of one of the game's functions (hookable).
function game.hookable(address) return core.hookable(address) end

-- Integer arguments as core.call takes them: numbers and booleans as they are, 64-bit cdata (uint64_t / int64_t, which a
-- number can't always hold) split into their low 32 bits and, in `highs`, their high 32 bits.
local int64 = ffi.typeof("int64_t")
local function int_arg(ints, highs, i, v)
    if v == true then ints[i] = 1
    elseif v == false or v == nil then ints[i] = 0
    elseif type(v) == "cdata" then
        local u = ffi.cast(uint64, v)
        ints[i] = tonumber(band(u, 0xFFFFFFFFULL))
        highs = highs or {}
        highs[i] = tonumber(rshift(u, 32))
    else ints[i] = tonumber(v) or error(format("an integer argument is a %s", type(v)), 3) end
    return highs
end

-- ac6.game.call(function, integers, floats) -> r3 (low 32 bits), f1, r3's high 32 bits: the raw call, `integers` an
-- array for r3-r10 (and past 8 on the stack: 16 at most; numbers, or 64-bit cdata), `floats` for f1-f13. From a game
-- thread only (a hook, the frame event).
function game.call(target, integers, floats)
    local address = resolve(target, 2)
    local ints, highs = {}, nil
    for i = 1, integers and #integers or 0 do highs = int_arg(ints, highs, i, integers[i]) end
    return core.call(address, ints, floats, highs)
end

-- ac6.game.fn(function, signature) -> a Lua function calling it: signature "ret(arg, arg, ...)", the types void bool s8
-- u8 s16 u16 s32 u32 s64 u64 ptr f32 f64 (integers and pointers in r3.., floats in f1..), e.g. "f32(ptr, f32)". A named
-- function's signature may come from ac6.signatures instead. f32 arguments are rounded to single precision, as the
-- game's own callers pass them; s64 / u64 take and return 64-bit cdata (numbers too).
local int_types = { bool = true, s8 = true, u8 = true, s16 = true, u16 = true, s32 = true, u32 = true, s64 = true, u64 = true, ptr = true }
local float_types = { f32 = true, f64 = true }
local single = ffi.new("float[1]")
function game.fn(target, signature)
    local address = resolve(target, 2)
    signature = signature or (type(target) == "string" and ac6.signatures[target])
    if not signature then error(format("ac6.game.fn(%s): no signature given or known", tostring(target)), 2) end
    local ret, args = signature:match("^%s*(%w+)%s*%((.*)%)%s*$")
    if not ret then error(format("ac6.game.fn: bad signature %q", signature), 2) end
    local kinds = {}
    for a in args:gmatch("[^,]+") do
        a = a:match("^%s*(%w+)")
        if not (int_types[a] or float_types[a]) then error(format("ac6.game.fn: unknown argument type %q", tostring(a)), 2) end
        kinds[#kinds + 1] = a
    end
    if not (ret == "void" or int_types[ret] or float_types[ret]) then error(format("ac6.game.fn: unknown return type %q", ret), 2) end
    return function(...)
        local ints, floats, highs = {}, {}, nil
        for i = 1, #kinds do
            local v, kind = select(i, ...), kinds[i]
            if kind == "f32" then single[0] = v or 0; floats[#floats + 1] = single[0]
            elseif kind == "f64" then floats[#floats + 1] = v or 0
            else highs = int_arg(ints, highs, #ints + 1, v) end
        end
        local r3, f1, high = core.call(address, ints, floats, highs)
        if ret == "void" then return
        elseif float_types[ret] then return f1
        elseif ret == "bool" then return band(r3, 0xFF) ~= 0
        elseif ret == "s32" then return r3 >= 2147483648 and r3 - 4294967296 or r3
        elseif ret == "s16" then r3 = band(r3, 0xFFFF); return r3 >= 32768 and r3 - 65536 or r3
        elseif ret == "u16" then return band(r3, 0xFFFF)
        elseif ret == "s8" then r3 = band(r3, 0xFF); return r3 >= 128 and r3 - 256 or r3
        elseif ret == "u8" then return band(r3, 0xFF)
        elseif ret == "u64" then return ffi.cast(uint64, high) * 4294967296ULL + r3
        elseif ret == "s64" then return ffi.cast(int64, ffi.cast(uint64, high) * 4294967296ULL + r3)
        else return r3 end
    end
end

-- The game's classes (its RTTI). rtti(object) -> the class name of the object at a guest address (its first word a
-- vtable) and which subobject's vtable it was (0: the primary), or nil. class(name) -> { name, mangled, vtables = {
-- { address, offset, slots }, ... } } or nil. classes() -> every class's name.
function game.rtti(object) return core.rtti_of(object) end
function game.class(name) return core.rtti_class(name) end
function game.classes() return core.rtti_classes() end

-- The address of a virtual function: vfunc(object, slot) through the object's own vtable, vfunc("Class", slot) through
-- the class's primary vtable. Slots count from 0.
function game.vfunc(object_or_class, slot)
    if type(object_or_class) == "number" then
        return mem.u32(mem.u32(object_or_class) + slot * 4)
    end
    local address = vfunc_of_class(object_or_class, slot)
    if not address then error(format("ac6.game.vfunc: %s has no slot %d", tostring(object_or_class), slot), 2) end
    return address
end

-- A virtual call: vcall(object, slot, signature, ...) calls the object's slot with the object in r3 (the signature's
-- first argument, a ptr) and the rest after it.
function game.vcall(object, slot, signature, ...)
    return game.fn(game.vfunc(object, slot), signature)(object, ...)
end

-- Every hookable function's address, in order (all of the game's functions but its register save/restore helpers).
function game.functions() return core.functions() end

-- The name ac6.sym has for an address, or nil.
local names
function game.name_of(address)
    if not names then
        names = {}
        for name, a in pairs(ac6.sym) do names[a] = names[a] or name end
    end
    return names[address]
end

function game.mode() return core.mode() end            -- the mode task the game is in ("CModeTaskGame", ...)
function game.frame() return core.frame() end          -- frames presented so far
function game.is_game_thread() return core.is_game_thread() end
function game.time() return core.time() end            -- seconds since the start

ac6.game = game

------------------------------------------------------------------------------------------------------------------------
-- ac6.events
------------------------------------------------------------------------------------------------------------------------
local known_events = { frame = true, mode = true, draw = true, pad = true }
local wanted = { draw = true, pad = true }   -- the runtime calls the mod for these only while it has handlers
local handlers = {}   -- name -> array of handler records (replaced, not changed, when one is added or removed)

local events = {}
-- ac6.events.on(name, handler) -> handle (handle:remove()).
--   "frame"  (dt, frame): at each presented frame, on the game's main thread (game calls allowed)
--   "mode"   (mode): the game entered a mode ("CModeTaskGame", ...)
--   "draw"   (): at each frame, the time to draw with ac6.ui (the main thread, game calls allowed)
--   "pad"    (user, pad): each player's pad (0-3) at each frame, before the game reads it: what the handler changes in
--            `pad` ({ connected, buttons, lt, rt, lx, ly, rx, ry }) is what the game gets
function events.on(name, fn)
    if not known_events[name] then error(format("ac6.events.on: no event %q (frame, mode, draw, pad)", tostring(name)), 2) end
    if type(fn) ~= "function" then error("ac6.events.on: the handler is a function", 2) end
    local record = { fn = fn }
    local list = {}
    for i, r in ipairs(handlers[name] or {}) do list[i] = r end
    list[#list + 1] = record
    handlers[name] = list
    if wanted[name] and #list == 1 then core.want(name, true) end
    local handle = {}
    function handle:remove()
        local kept = {}
        for _, r in ipairs(handlers[name] or {}) do if r ~= record then kept[#kept + 1] = r end end
        handlers[name] = kept
        if wanted[name] and #kept == 0 then core.want(name, false) end
    end
    return handle
end

-- core's event dispatcher: every handler of the event, one failing not stopping the others.
local function event_sink(name, ...)
    local list = handlers[name]
    if not list then return end
    for i = 1, #list do
        local ok, err = xpcall(list[i].fn, traceback, ...)
        if not ok then core.log(2, format("%s handler: %s", name, tostring(err))) end
    end
end

ac6.events = events

------------------------------------------------------------------------------------------------------------------------
-- ac6.input: the pads, the keyboard and the mouse as the player uses them, as of this frame (none of it while the
-- settings overlay is open). Any thread. To change what the game reads from a pad: the "pad" event.
------------------------------------------------------------------------------------------------------------------------
local input = {}

-- The pads' buttons (XINPUT_GAMEPAD_*).
input.BUTTONS = { up = 0x0001, down = 0x0002, left = 0x0004, right = 0x0008, start = 0x0010, back = 0x0020, ls = 0x0040,
    rs = 0x0080, lb = 0x0100, rb = 0x0200, a = 0x1000, b = 0x2000, x = 0x4000, y = 0x8000 }
local BUTTONS = input.BUTTONS

local function button_bit(name, level)
    local b = BUTTONS[name]
    if not b then error(format("ac6.input: no button %q (a b x y lb rb ls rs start back up down left right)", tostring(name)), level + 1) end
    return b
end

-- pad([user]) -> { connected, buttons, lt, rt, lx, ly, rx, ry, pressed }: player `user`'s pad (0-3, 0 by default) as the
-- player left it (before the mods' "pad" handlers); `pressed` its buttons gone down since the frame before. Triggers
-- 0-255, sticks -32768-32767.
function input.pad(user) return core.pad(user or 0, false) end
-- game_pad([user]): the pad as the game reads it (after the mods' "pad" handlers).
function input.game_pad(user) return core.pad(user or 0, true) end

-- Whether the player holds a button ("a", "lb", "start", ...), and whether it went down this frame.
function input.button(name, user) return band(core.pad(user or 0, false).buttons, button_bit(name, 2)) ~= 0 end
function input.button_pressed(name, user) return band(core.pad(user or 0, false).pressed, button_bit(name, 2)) ~= 0 end

-- On a pad table (the "pad" event's, or one of the above): whether a button is down, and setting it down or up.
function input.has(pad, name) return band(pad.buttons, button_bit(name, 2)) ~= 0 end
function input.set(pad, name, down)
    local b = button_bit(name, 2)
    pad.buttons = down == false and band(pad.buttons, bit.bnot(b)) or bor(pad.buttons, b)
end

-- Keys and mouse buttons by the settings file's names: "F5", "A", "1", "Space", "LeftControl", "Shift" (either),
-- "Numpad1", "Up", "Escape", "Mouse1" (left) ... "Mouse5". key: held; key_pressed: went down this frame (a tap between
-- two frames too). An unknown name is an error.
local function key_state(name, level)
    local held, pressed = core.key(name)
    if held == nil then error(format("ac6.input: no key %q", tostring(name)), level + 1) end
    return held, pressed
end
function input.key(name) return (key_state(name, 2)) end
function input.key_pressed(name) return select(2, key_state(name, 2)) end

-- mouse() -> { x, y, wheel, left, right, middle, x1, x2, pressed = { left, ... } }: the cursor in window units (as
-- ac6.ui draws), the buttons held and gone down this frame, the wheel's notches this frame (up positive).
local mouse_bits = { left = 1, middle = 2, right = 4, x1 = 8, x2 = 16 }
function input.mouse()
    local x, y, held, pressed, wheel = core.mouse()
    local m = { x = x, y = y, wheel = wheel, pressed = {} }
    for name, b in pairs(mouse_bits) do
        m[name] = band(held, b) ~= 0
        m.pressed[name] = band(pressed, b) ~= 0
    end
    return m
end

ac6.input = input

------------------------------------------------------------------------------------------------------------------------
-- ac6.ui: drawing over the game, in "draw" handlers (anywhere else it's an error). Window units, (0, 0) the top left
-- (the mouse's units). Colours 0xAARRGGBB (0xFF000000 and up opaque, alpha 0 transparent), white when left out:
-- ui.color(r, g, b [, a]) makes one from 0-255 components, ui.rgb(0xRRGGBB [, alpha]) from a hex colour and an alpha
-- 0-1 (1 by default).
------------------------------------------------------------------------------------------------------------------------
local ui = {}

function ui.color(r, g, b, a)
    return bor(lshift(band(a or 255, 255), 24), lshift(band(r, 255), 16), lshift(band(g, 255), 8), band(b, 255)) % 4294967296
end
function ui.rgb(hex, alpha)
    local a = math.floor(math.max(0, math.min(1, alpha or 1)) * 255 + 0.5)
    return bor(lshift(a, 24), band(hex, 0xFFFFFF)) % 4294967296
end

-- The window's size; the game's picture in it (x, y, width, height: letterboxed); the game's 2D canvas, its 1280x720 in
-- the picture (where it draws its HUD and menus, 16:9). from_canvas: a canvas point (0-1280, 0-720) in window units.
function ui.size() return core.draw_size() end
function ui.view() return core.draw_view() end
function ui.canvas() return core.draw_canvas() end
function ui.from_canvas(cx, cy)
    local x, y, w, h = core.draw_canvas()
    return x + cx * w / 1280, y + cy * h / 720
end
-- A scale for sizes: the window against 1920x1080 (the smaller of the two ratios).
function ui.scale()
    local w, h = core.draw_size()
    return math.min(w / 1920, h / 1080)
end

-- Shapes: outlines `thickness` wide (1 by default), and filled.
function ui.line(x0, y0, x1, y1, color, thickness) core.draw_line(x0, y0, x1, y1, color, thickness or 1) end
function ui.rect(x0, y0, x1, y1, color, thickness, rounding) core.draw_rect(x0, y0, x1, y1, color, thickness or 1, rounding or 0) end
function ui.fill_rect(x0, y0, x1, y1, color, rounding) core.draw_rect(x0, y0, x1, y1, color, 0, rounding or 0) end
function ui.circle(x, y, radius, color, thickness) core.draw_circle(x, y, radius, color, thickness or 1) end
function ui.fill_circle(x, y, radius, color) core.draw_circle(x, y, radius, color, 0) end
function ui.triangle(x0, y0, x1, y1, x2, y2, color, thickness) core.draw_triangle(x0, y0, x1, y1, x2, y2, color, thickness or 1) end
function ui.fill_triangle(x0, y0, x1, y1, x2, y2, color) core.draw_triangle(x0, y0, x1, y1, x2, y2, color, 0) end
-- points: { x0, y0, x1, y1, ... }. A filled polygon is convex.
function ui.polyline(points, color, thickness, closed) core.draw_poly(points, color, closed and true or false, thickness or 1) end
function ui.fill_polygon(points, color) core.draw_poly(points, color, true, 0) end

-- Text at its top left, `size` its height in window units (18 by default), mono: the overlay's numbers' font (its
-- digits all one width). text_size -> width, height.
function ui.text(x, y, text, color, size, mono) core.draw_text(x, y, tostring(text), color, size or 0, mono and true or false) end
function ui.text_size(text, size, mono) return core.text_size(tostring(text), size or 0, mono and true or false) end

-- Drawing clipped to a rectangle: fn() draws inside it.
function ui.clip(x0, y0, x1, y1, fn)
    core.push_clip(x0, y0, x1, y1)
    local ok, err = pcall(fn)
    core.pop_clip()
    if not ok then error(err, 0) end
end

ac6.ui = ui

------------------------------------------------------------------------------------------------------------------------
-- ac6.native
------------------------------------------------------------------------------------------------------------------------
ac6.native = {
    -- A DLL (.so on Linux), its path relative to the mod's folder: what its luaopen_<name> returns, or true. Keep it out
    -- of the top of the folder (bin/, say): the plugin loader starts the DLLs there first. Loading the same file again
    -- returns what the first load did.
    load = function(path) return core.load_native(path) end,
}

ac6.util = {
    hex = function(n) return format("%08X", n) end,
}

core.set_sinks(event_sink, hook_sink)
return ac6
