-- What the game knows about its aircraft, for mods (require("aircraft")). Found in the game's code (see README.md):
--
--   names       100-byte records at 0x823F9418, one per aircraft id: { u32 id, char code[32], char name[64] }
--   data        the aircraft table, item 19 of the resource manager at [0x8293B930] + 1419896 (data the game loads):
--               1840 bytes per aircraft, { u32 weapon_sets, u32 set_cost[3], u8 set_weapon[4], paint[35] } with
--               52-byte paint records (u32 price first, the ammunition of each weapon in words 8-12)
--   save        each save slot's aircraft state, at root + 44452 + 42976 * slot: credits at +4, the owned paints of
--               aircraft id at +665 + 35 * id (one byte each), its owned weapon sets at +1188 + 4 * id + 2
--   select      the aircraft list the hangar and the aircraft select build (sub_8214C6A0, CSelectAircraftManager):
--               52-byte nodes, the first at manager + 120364, their count at manager + 120368, the id at node + 48
local mem = ac6.mem

local M = {}

M.COUNT = 15                       -- the game's aircraft ids 0-14 (its select loop's bound)
M.NAMES = 0x823F9418
M.NAME_RECORD = 100
M.RESOURCE_BASE = 0x8293B930       -- a pointer: the resource manager holding the aircraft table is [it] + 1419896
M.RESOURCE_OFFSET = 1419896
M.AIRCRAFT_ITEM = 19
M.RECORD = 1840
M.PAINTS = 35
M.PAINT_RECORD = 52
M.ROOT = 0x826E4E54                -- the game's root object pointer (ac6.data.game_root)
M.BUILD_LIST = 0x8214C6A0          -- CSelectAircraftManager's list build

function M.name(id) return mem.string(M.NAMES + id * M.NAME_RECORD + 0x24, 64) end
function M.code(id) return mem.string(M.NAMES + id * M.NAME_RECORD + 4, 32) end

-- The aircraft table's first record (nil until the game has loaded it).
local function guest(a) return a > 0 and a <= 0xFFFFFFFF and mem.valid(a) end

function M.table()
    local holder = mem.u32(M.RESOURCE_BASE)
    if holder == 0 then return nil end
    local r = holder + M.RESOURCE_OFFSET
    if not guest(r) then return nil end
    local count, base, offsets = mem.u32(r), mem.u32(r + 4), mem.u32(r + 12)
    if count <= M.AIRCRAFT_ITEM or base == 0 or not guest(offsets) then return nil end
    local offset = mem.u32(offsets + M.AIRCRAFT_ITEM * 4)
    if offset == 0 or not guest(base + offset) then return nil end
    local item = base + offset
    local data = item + mem.u32(item + 8)
    if not guest(data) or not guest(data + M.RECORD * M.COUNT - 1) then return nil end
    return data, item
end

-- An aircraft's data from the table: its weapon sets (the special weapons, set 0 the standard one) and its paints.
function M.data(id)
    local t = M.table()
    if not t then return nil end
    local r = t + id * M.RECORD
    local d = { weapon_sets = mem.u32(r), set_cost = {}, set_weapon = {}, paints = {} }
    for i = 1, 3 do d.set_cost[i] = mem.s32(r + 4 * i) end
    for i = 0, 3 do d.set_weapon[i] = mem.u8(r + 16 + i) end
    for p = 0, M.PAINTS - 1 do
        local a = r + 20 + p * M.PAINT_RECORD
        local price = mem.s32(a)
        local ammo = {}
        for w = 8, 12 do ammo[#ammo + 1] = mem.s32(a + 4 * w) end
        d.paints[p] = { price = price, ammo = ammo, empty = price == 0 and ammo[1] == 0 and ammo[2] == 0 }
    end
    return d
end

-- The save slot in use: its aircraft state's address, or nil before a save is loaded.
function M.save()
    local root = mem.u32(M.ROOT)
    if root == 0 or not mem.valid(root) then return nil end
    local slot = mem.u32(root + 148584)
    if slot > 2 then slot = 0 end
    return root + 44452 + 42976 * slot
end

function M.credits(save) return mem.s32(save + 4) end

-- The paints of an aircraft the save owns (their indices), and its owned special weapon sets.
function M.owned(save, id)
    local paints, sets = {}, {}
    for p = 0, M.PAINTS - 1 do
        if mem.u8(save + 665 + 35 * id + p) == 1 then paints[#paints + 1] = p end
    end
    for s = 0, 3 do
        if mem.u8(save + 1188 + 4 * id + 2 + s) ~= 0 then sets[#sets + 1] = s end
    end
    return paints, sets
end

-- The ids in a CSelectAircraftManager's list, in its order (a circular list of nodes).
function M.list(manager)
    local ids = {}
    local node, count = mem.u32(manager + 120364), mem.u32(manager + 120368)
    for _ = 1, math.min(count, 64) do
        if node == 0 then break end
        ids[#ids + 1] = mem.u32(node + 48)
        node = mem.u32(node)
    end
    return ids
end

return M
