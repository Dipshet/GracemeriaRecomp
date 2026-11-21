-- Player 1's controller over the game (mod.toml): ac6.input to read it, ac6.ui to draw it, and the "pad" event to change
-- what the game reads.
--   F6   shows or hides the display
--   F7   swaps A and B for the game (off at the start)
local input, ui = ac6.input, ac6.ui

local shown, swapped = true, false

-- The hotkeys: checked once a frame (key_pressed is "went down since the frame before").
ac6.events.on("frame", function()
    if input.key_pressed("F6") then shown = not shown end
    if input.key_pressed("F7") then
        swapped = not swapped
        print("A and B " .. (swapped and "swapped" or "back"))
    end
end)

-- The pad event: what this handler leaves in `pad` is what the game reads. The display below shows the player's own
-- presses (input.pad), so it doesn't change with the swap.
ac6.events.on("pad", function(user, pad)
    if not swapped or user ~= 0 then return end
    local a, b = input.has(pad, "a"), input.has(pad, "b")
    input.set(pad, "a", b)
    input.set(pad, "b", a)
end)

local BACK = ui.color(4, 5, 26, 170)
local EDGE = ui.color(120, 140, 220)
local LIT = ui.color(242, 210, 122)
local DIM = ui.color(90, 107, 200, 140)
local TEXT = ui.color(228, 233, 250)

-- A stick: its ring, and a dot where it points.
local function stick(x, y, r, sx, sy, s)
    ui.circle(x, y, r, EDGE, 2 * s)
    ui.fill_circle(x + sx / 32768 * r, y - sy / 32768 * r, 6 * s, LIT)
end

-- A trigger: a bar filled as far as it's pulled.
local function trigger(x0, y0, x1, y1, value, s)
    ui.rect(x0, y0, x1, y1, EDGE, 2 * s)
    local fill = y1 - (y1 - y0) * value / 255
    if value > 0 then ui.fill_rect(x0 + 2 * s, fill, x1 - 2 * s, y1 - 2 * s, LIT) end
end

-- A button: a disc with its name, lit while held.
local function button(x, y, r, name, held, s)
    ui.fill_circle(x, y, r, held and LIT or DIM)
    local label = name:upper()
    local tw, th = ui.text_size(label, 14 * s, true)
    ui.text(x - tw / 2, y - th / 2, label, held and ui.color(4, 5, 26) or TEXT, 14 * s, true)
end

ac6.events.on("draw", function()
    if not shown then return end
    local pad = input.pad(0)
    -- In the bottom left of the game's 2D canvas, sized with the window.
    local s = ui.scale()
    local x0, y0 = ui.from_canvas(24, 720 - 24)
    local w, h = 360 * s, 170 * s
    y0 = y0 - h
    ui.fill_rect(x0, y0, x0 + w, y0 + h, BACK, 8 * s)
    ui.rect(x0, y0, x0 + w, y0 + h, EDGE, 1.5 * s, 8 * s)

    trigger(x0 + 14 * s, y0 + 14 * s, x0 + 30 * s, y0 + 90 * s, pad.lt, s)
    trigger(x0 + w - 30 * s, y0 + 14 * s, x0 + w - 14 * s, y0 + 90 * s, pad.rt, s)
    stick(x0 + 80 * s, y0 + 60 * s, 34 * s, pad.lx, pad.ly, s)
    stick(x0 + 230 * s, y0 + 112 * s, 34 * s, pad.rx, pad.ry, s)

    local function held(name) return input.has(pad, name) end
    -- The face buttons in a diamond, the shoulders and the D-pad as a row.
    local fx, fy, r = x0 + 290 * s, y0 + 52 * s, 13 * s
    button(fx, fy - 26 * s, r, "y", held("y"), s)
    button(fx - 26 * s, fy, r, "x", held("x"), s)
    button(fx + 26 * s, fy, r, "b", held("b"), s)
    button(fx, fy + 26 * s, r, "a", held("a"), s)
    local row = { "lb", "rb", "back", "start", "up", "down", "left", "right" }
    local bx = x0 + 44 * s
    for _, name in ipairs(row) do
        local label = name == "left" and "<" or name == "right" and ">" or name == "up" and "^" or name == "down" and "v" or name
        local tw = ui.text_size(label:upper(), 13 * s, true)
        local cw = math.max(tw + 10 * s, 22 * s)
        ui.fill_rect(bx, y0 + h - 30 * s, bx + cw, y0 + h - 10 * s, held(name) and LIT or DIM, 4 * s)
        ui.text(bx + (cw - tw) / 2, y0 + h - 28 * s, label:upper(), held(name) and ui.color(4, 5, 26) or TEXT, 13 * s, true)
        bx = bx + cw + 4 * s
    end
    ui.text(x0 + 120 * s, y0 + 10 * s, swapped and "F7: A/B SWAPPED" or "F6 HIDES", TEXT, 13 * s)
end)
