-- Prints the game's playable aircraft (mod.toml). Three parts, each when the game has what it needs:
--   at the start       the 15 aircraft from the game's own name table
--   once loaded        each one's data from the game's aircraft table (its price, its special weapon sets)
--   in the hangar or the aircraft select   the ones the save has unlocked, as the game lists them
local aircraft = require("aircraft")
local fmt = string.format

print(fmt("Playable aircraft (%d):", aircraft.COUNT))
for id = 0, aircraft.COUNT - 1 do
    print(fmt("  %2d  %-14s %s", id, aircraft.name(id), aircraft.code(id)))
end

-- The aircraft table: the game loads it with its boot data. Looked for once a second until it's there.
local printed = false
ac6.events.on("frame", function(dt, frame)
    if printed or frame % 60 ~= 0 or not aircraft.table() then return end
    printed = true
    local _, item = aircraft.table()
    print(fmt("The aircraft table (item %d at %08X):", aircraft.AIRCRAFT_ITEM, item))
    for id = 0, aircraft.COUNT - 1 do
        local d = aircraft.data(id)
        -- Special weapon sets: the weapon of each (255: none), and what the second and later cost (the first is free).
        local sets, costs = {}, {}
        for s = 0, d.weapon_sets - 1 do sets[#sets + 1] = tostring(d.set_weapon[s]) end
        for s = 1, d.weapon_sets - 1 do costs[#costs + 1] = tostring(d.set_cost[s]) end
        print(fmt("  %2d  %-14s price %6d; special weapon sets %s (costs %s)", id, aircraft.name(id), d.paints[0].price,
            table.concat(sets, "/"), table.concat(costs, "/")))
    end
end)

-- The list the hangar and the aircraft select build: the aircraft the save has unlocked. Priority 1000: the outermost
-- hook, so its post runs after other mods' (the aircraft_slots mod appends its aircraft there).
ac6.game.hook(aircraft.BUILD_LIST, {
    priority = 1000,
    post = function(c)
        local manager = c.r[3]
        local ids = aircraft.list(manager)
        local save = aircraft.save()
        print(fmt("The game lists %d aircraft (%s)%s:", #ids, ac6.game.mode(),
            save and fmt(", %d credits", aircraft.credits(save)) or ""))
        for _, id in ipairs(ids) do
            local paints, sets = {}, {}
            if save then paints, sets = aircraft.owned(save, id) end
            if id < aircraft.COUNT then
                print(fmt("  %2d  %-14s paints %s, weapon sets %s", id, aircraft.name(id), table.concat(paints, ","),
                    table.concat(sets, ",")))
            else
                print(fmt("  %2d  (a new aircraft slot)", id))
            end
        end
    end,
})
