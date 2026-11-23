// The settings file, see settings.h. Every setting is one AC6_* environment variable under a readable name, so the
// runtime reads its settings through those variables. The file only sets those the environment doesn't (test scripts
// and launchers still override it). A new file holds only a header comment, and the overlay writes only the settings
// changed from their defaults, so a default changed in a later release reaches every existing file.
#include "settings.h"
#include "game/localization.h"

#include "input/kbm.h"
#include "log.h"
#include "platform/platform.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <map>
#include <mutex>
#include <optional>
#include <format>
#include <fstream>
#include <iterator>
#include <string>
#include <string_view>
#include <vector>

#define TOML_EXCEPTIONS 0
#include <toml++/toml.hpp>

namespace ac6::settings {

namespace {

enum Kind { kBool, kNumber, kText };

struct Setting
{
    const char* section;
    const char* key;
    const char* env;
    Kind kind;
    const char* example;   // an example value (TOML), for the documentation: new files hold only changed settings
    const char* help;      // the overlay's help text, ending with its "Default:" line
    const char* aliases = nullptr;   // text values given to the variable as another: "from=to,from=to"
};

// The settings players change. Any other AC6_* variable (the diagnostics too) goes under [env].
const Setting kSettings[] = {
    { "display", "monitor", "AC6_MONITOR", kNumber, "2",
      "The monitor to display the game on\n"
      "Default: Primary" },
    { "display", "resolution", "AC6_RESOLUTION", kText, "\"native\"",
      "The game's resolution\n"
      "Native uses the desktop's resolution\n"
      "Default: Native" },
    { "display", "fullscreen", "AC6_FULLSCREEN", kText, "\"exclusive\"",
      "Sets the display mode\n"
      "Borderless fills the screen without a border\n"
      "Exclusive takes over the display and lets you choose the refresh rate\n"
      "Default: Borderless",
      "exclusive=1,off=0" },
    { "display", "refresh", "AC6_REFRESH", kNumber, "144",
      "Sets the display's refresh rate\n"
      "Only in Exclusive fullscreen\n"
      "Windowed and Borderless follow the desktop's\n"
      "Default: Highest" },
    { "display", "render_resolution", "AC6_RENDER_SCALE", kNumber, "1.5",
      "Multiplies the 3D scene's resolution, from 0.5x to 4x\n"
      "Minimum 3D resolution is 640x360\n"
      "Default: 1" },
    { "display", "widescreen", "AC6_WIDESCREEN", kBool, "false",
      "The game's aspect ratio\n"
      "Auto fills the screen\n"
      "16:9 keeps the original aspect ratio, with bars at the sides on wider screens and at the top and bottom on narrower ones\n"
      "Default: Auto" },
    { "display", "cutscene_widescreen", "AC6_WIDESCREEN_CUTSCENES", kBool, "true",
      "The aspect ratio of in-engine cutscenes\n"
      "They were originally framed for 16:9: wider or narrower, they can show parts of the scene that weren't meant to be seen\n"
      "Requires aspect ratio set to Auto\n"
      "Default: 16:9" },
    { "display", "hud_edges", "AC6_HUD_EDGES", kNumber, "1.0",
      "Moves the HUD's side panels toward the screen edges on screens wider or narrower than 16:9\n"
      "1 puts them at the edges, 0 keeps the original 16:9 positions\n"
      "Requires aspect ratio set to Auto\n"
      "Default: 0" },
    { "display", "hud_custom", "AC6_HUD_CUSTOM", kText, "\"custom\"",
      "Allow changing the HUD's colours to custom or to colourblind presets\n"
      "Default: Off" },
    { "display", "hud_colour", "AC6_HUD_COLOUR", kText, "\"FFB000\"",
      "The flight HUD's colour\n"
      "Default: Green" },
    { "display", "hud_friendly", "AC6_HUD_FRIENDLY", kText, "\"00E0C0\"",
      "The colour of friendly units on the HUD\n"
      "Auto uses the game's blue or a colour contrast to the HUD colour\n"
      "Default: Auto" },
    { "display", "hud_enemy", "AC6_HUD_ENEMY", kText, "\"FF00C0\"",
      "The colour of enemy units and HUD alerts\n"
      "Auto uses the game's red, or a colour contrast to the HUD colour\n"
      "Default: Auto" },
    { "display", "fov_offset", "AC6_FOV_OFFSET", kNumber, "10",
      "Widens the gameplay field of view by up to 40 degrees\n"
      "Default: 0" },
    { "display", "vsync", "AC6_VSYNC", kBool, "true", "Syncs frames to the display's refresh rate, to prevent tearing\n"
                                                      "Default: On" },
    { "display", "frame_rate", "AC6_FRAME_RATE", kNumber, "120",
      "Caps the frame rate\n"
      "Match display caps it at the display's refresh rate\n"
      "Console uses the original 60 FPS in menus and 30 FPS in missions\n"
      "Uncapped removes the cap\n"
      "With VSync on, the frame rate stays at or below the display's refresh rate\n"
      "Default: Match display" },
    { "display", "cutscene_unlock", "AC6_CUTSCENE_UNLOCK", kBool, "false", "Unlocks the frame rate in in-engine cutscenes\n"
                                                                           "Off keeps them at 30 FPS\n"
                                                                           "Default: On" },
    { "display", "menu_unlock", "AC6_MENU_UNLOCK", kBool, "false",
      "Unlocks the frame rate in menus\n"
      "Off keeps them at 60 FPS\n"
      "Default: On" },

    { "graphics", "aa", "AC6_AA", kText, "\"smaa\"",
      "Anti-aliasing to smooth jagged edges\n"
      "Enabling FSR 3.1 allows FSR upscaling\n"
      "Default: Off" },
    { "graphics", "fsr_quality", "AC6_FSR_QUALITY", kText, "\"quality\"",
      "Adjust the 3D scene resolution before being upscaled with FSR\n"
      "Native: 1x\n"
      "Quality: 0.67x, Balanced: 0.59x, Performance: 0.5x, Ultra performance: 0.33x\n"
      "Only with FSR 3.1 anti-aliasing\n"
      "Default: Native" },
    { "graphics", "msaa", "AC6_MSAA", kText, "\"4\"", "Multisample anti-aliasing of the 3D scene\n"
                                                      "Console is the original 2x\n"
                                                      "Default: Console (2x)" },
    { "graphics", "max_anisotropy", "AC6_MAX_ANISO", kNumber, "16", "Caps the anisotropic filtering of every draw\n"
                                                                    "Settings file only" },
    { "graphics", "anisotropic_override", "AC6_ANISO_OVERRIDE", kNumber, "16",
      "Forces anisotropic filtering on every texture, beyond what the game asks for\n"
      "Sharper textures at steep angles\n"
      "Default: Console" },
    { "graphics", "volumetrics", "AC6_VOLUMETRICS", kBool, "true", "Draws smoke, fire and missile trails as lit volumes, and the clouds as a 3D cloudscape\n"
                                                                   "EXPERIMENTAL AND UNFINISHED\n"
                                                                   "Default: Off" },
    { "graphics", "volumetric_kinds", "AC6_VOL_KINDS", kText, "\"smoke,fire,cloud,trail\"",
      "Set which effects become volumes\n"
      "Only with volumetric effects on\n"
      "Default: smoke,fire,cloud,trail" },
    { "graphics", "lod_scale", "AC6_LOD_SCALE", kNumber, "2",
      "Keeps models detailed, and draws the map's props, up to this many times farther away\n"
      "From 1 to 8\n"
      "Default: 1" },
    // (The Graphics tab's Effects in this order: depth of field, bloom, then the rest.)
    { "graphics", "dof_fix", "AC6_DOF_FIX", kBool, "true", "Enhanced removes the depth of field striping on resolutions above 1280 x 720\n"
                                                           "Default: Console" },
    { "graphics", "bloom", "AC6_BLOOM", kText, "\"enhanced\"",
      "The glow around bright objects\n"
      "Console renders the glow at a fixed 160 x 90 resolution\n"
      "Enhanced renders the glow at half the screen's resolution\n"
      "Default: Console" },
    { "graphics", "ambient_occlusion", "AC6_AO", kBool, "true",
      "Add soft shading in creases and where objects meet\n"
      "Default: Off" },
    { "graphics", "shadow_detail", "AC6_SHADOW_DETAIL", kBool, "true",
      "Sharper shadows on characters in cutscenes\n"
      "Off keeps the console's shadow resolution, with visible steps\n"
      "Default: Off" },
    { "graphics", "planar_reflection", "AC6_PLANAR_REFLECTION", kBool, "true",
      "Enables planar reflection on bodies of water\n"
      "Default: Off" },
    { "graphics", "motion_blur", "AC6_MOTION_BLUR", kNumber, "0.5",
      "Add blurs to moving objects in the 3D scene\n"
      "Only with TAA, SMAA + TAA or FSR 3.1 anti-aliasing\n"
      "Default: 0" },
    { "graphics", "film_grain", "AC6_FILM_GRAIN", kNumber, "0.5",
      "Add perceptual film grain effects\n"
      "Default: 0" },
    { "graphics", "vignette", "AC6_VIGNETTE", kNumber, "0.5",
      "Darkens the corners of the 3D scene\n"
      "Default: 0" },
    { "graphics", "chromatic_aberration", "AC6_CHROMATIC_ABERRATION", kNumber, "0.5",
      "Colour fringes toward the corners of the scene\n"
      "Default: 0" },
    { "graphics", "centroid", "AC6_CENTROID", kBool, "true", "Removes the black dots along aircraft edges\n"
                                                             "Off looks as on the console\n"
                                                             "Default: On" },
    { "graphics", "deband", "AC6_DEBAND", kNumber, "1", "Smooths colour banding in gradients such as the sky\n"
                                                        "Strength in 8-bit steps\n"
                                                        "Default: 0 (off)" },
    { "graphics", "wide_blurs", "AC6_WIDE_BLURS", kBool, "true", "Keeps the bloom round on non 16:9 resolution\n"
                                                                 "Default: On" },
    { "graphics", "sdr_10bit", "AC6_SDR_10BIT", kBool, "true",
      "Sends 10 bits per colour to the display instead of 8\n"
      "Produce smoother gradients, less banding\n"
      "8-bit when the display or driver doesn't offer 10-bit\n"
      "SDR only\n"
      "Default: Off" },

    { "output", "mode", "AC6_OUTPUT", kText, "\"hdr\"",
      "Enables HDR output\n"
      "Needs HDR on for this display in the system's display settings\n"
      "Default: Off" },
    { "output", "hdr_tonemap", "AC6_HDR_TONEMAP", kText, "\"psychov30\"",
      "How HDR brightness is fitted to the display\n"
      "PsychoV30 and RenoDRT are RenoDX's tone mappers\n"
      "Default: PsychoV30" },
    { "output", "hdr_scene", "AC6_HDR_SCENE", kBool, "true",
      "Keeps the highlights the game renders above white (the sun, glints, fire) for HDR, instead of estimating them from the SDR picture\n"
      "Default: On" },
    // The HDR grading: RenoDX's sliders, for RenoDRT and PsychoV-30.
    { "output", "hdr_exposure", "AC6_HDR_EXPOSURE", kNumber, "1", "Overall HDR brightness, from 0.5 to 2\n"
                                                                  "Default: 1" },
    { "output", "hdr_highlights", "AC6_HDR_HIGHLIGHTS", kNumber, "50",
      "Brighter highlights above 50, dimmer below, from 0 to 100\n"
      "Default: 50" },
    { "output", "hdr_shadows", "AC6_HDR_SHADOWS", kNumber, "50", "Lifted shadows above 50, deeper below, from 0 to 100\n"
                                                                 "Default: 50" },
    { "output", "hdr_contrast", "AC6_HDR_CONTRAST", kNumber, "50", "HDR contrast, from 0 to 100\n"
                                                                   "Default: 50" },
    { "output", "hdr_saturation", "AC6_HDR_SATURATION", kNumber, "50", "HDR colour saturation, from 0 to 100\n"
                                                                       "Default: 50" },
    { "output", "hdr_hue_correction", "AC6_HDR_HUE_CORRECTION", kNumber, "50",
      "How closely the highlights' colour follows the SDR picture, from 0 to 100\n"
      "Default: 50" },
    { "output", "hdr_gamut_compression", "AC6_HDR_GAMUT_COMPRESSION", kNumber, "100",
      "PsychoV30: brings colours the display can't show back into its range, from 0 to 100\n"
      "Default: 100" },
    { "output", "hdr_compression", "AC6_HDR_COMPRESSION", kNumber, "1",
      "PsychoV30: how the highlights roll off toward the peak, from 0 to 2\n"
      "Higher keeps them darker for longer, 0 sets it from the peak\n"
      "Default: 1" },
    { "output", "hdr_clouds", "AC6_HDR_CLOUDS", kNumber, "2",
      "Brightens the clouds' sunlit parts after they are drawn, from 1 to 6\n"
      "Replaced by Cloud light\n"
      "Default: 1" },
    { "output", "hdr_cloud_light", "AC6_RELIGHT_CLOUD_SUN", kNumber, "1",
      "The sunlight the game lights the clouds with, from 0.5 to 4\n"
      "Default: 1" },
    { "output", "hdr_cloud_edge", "AC6_RELIGHT_CLOUD_RIM", kNumber, "1.5",
      "The bright rim on the clouds' thin edges when facing the sun, from 0 to 8\n"
      "Cloud light scales it too\n"
      "Default: 1.5 (1 is the console's)" },
    { "output", "hdr_aircraft_light", "AC6_RELIGHT_OBJECT_SUN", kNumber, "1.5",
      "The sunlight the game lights aircraft and objects with, from 0.5 to 4\n"
      "Default: 1.5 (1 is the console's)" },
    { "output", "hdr_afterburner", "AC6_HDR_AFTERBURNER", kNumber, "3",
      "Extra afterburner glow on top of the effects' brightness, from 0 to 6\n"
      "0 keeps them as the game draws them\n"
      "Default: 3" },
    { "output", "hdr_effects", "AC6_HDR_EFFECTS", kNumber, "1",
      "How bright fire, explosions, flares, glows and afterburners are, from 0 to 4\n"
      "0 keeps them as the game draws them, no brighter than white\n"
      "Default: 1" },
    { "output", "hdr_lit_highlights", "AC6_HDR_LIT_HIGHLIGHTS", kNumber, "100",
      "How much of lit surfaces' brightness above white is kept (sunlit clouds and roofs, glints), from 0 to 100\n"
      "Default: 100 (all of it)" },
    { "output", "hdr_stripes", "AC6_HDR_STRIPES", kText, "\"off\"",
      "Overlays a stripe which measures HDR brightness\n"
      "Display brightness: blue, green and yellow are the thirds from paper white to 90% of the peak, orange up to 98%, red at the peak\n"
      "Game's values: blue 1-2x white, green 2-4x, yellow 4-16x, orange 16-256x, red above\n"
      "Default: Off" },
    { "output", "hdr_paper_white", "AC6_HDR_PAPER_WHITE", kNumber, "200", "How bright white is in HDR, in nits\n"
                                                                          "Default: 203" },
    { "output", "hdr_peak", "AC6_HDR_PEAK", kNumber, "1000", "Display's peak brightness, in nits\n"
                                                             "Default: your display's peak brightness" },
    { "output", "hdr_ui_brightness", "AC6_HDR_UI_BRIGHTNESS", kNumber, "250", "How bright the HUD and menus are, in nits\n"
                                                                              "Default: 250" },

    { "controls", "keyboard_mouse", "AC6_KBM", kBool, "true",
      "Enables keyboard and mouse controls\n"
      "Default keybinds follow Ace Combat 7 scheme\n"
      "Default: Off" },
    { "controls", "cursor_hide_seconds", "AC6_CURSOR_HIDE_SECONDS", kNumber, "3", "Hides the mouse cursor once it hasn't moved for this many seconds\n"
                                                                                  "0 never hides it\n"
                                                                                  "Default: 3" },

    // The mouse in flight with keyboard and mouse on (input/kbm.cpp, the overlay's Controls tab).
    // The rest of [mouse] (mode, smoothing, curve, deadzones, recentring) stays the file's only.
    { "mouse", "mouse_controls", "AC6_MOUSE_CONTROLS", kText, "\"camera\"",
      "What the mouse controls in flight\n"
      "Aircraft: the mouse steers, hold the camera key (Left Alt) to look around\n"
      "Camera: the mouse looks around, hold the camera key (Left Alt) to steer\n"
      "Default: Aircraft" },
    { "mouse", "steer_model", "AC6_MOUSE_STEER_MODEL", kText, "\"position\"",
      "How the mouse steers the aircraft\n"
      "Velocity: how fast you move the mouse sets the stick, similar to Ace Combat 7\n"
      "Position: moving the mouse pushes the stick, which springs back to the centre when you stop\n"
      "Default: Velocity" },
    // Sensitivity: 1 stands for an internal scale of 1.5 (input/kbm.cpp kSensitivityUnit).
    { "mouse", "sensitivity_x", "AC6_MOUSE_SENSITIVITY_X", kNumber, "1.5", "How much sideways mouse movement rolls the aircraft\n"
                                                                           "Default: 1" },
    { "mouse", "sensitivity_y", "AC6_MOUSE_SENSITIVITY_Y", kNumber, "1.5", "How much up and down mouse movement pitches the aircraft\n"
                                                                           "Default: 1" },
    { "mouse", "invert_x", "AC6_MOUSE_INVERT_X", kBool, "true", "Reverses the mouse's sideways movement\n"
                                                                "Default: Off" },
    { "mouse", "invert_y", "AC6_MOUSE_INVERT_Y", kBool, "false", "On: moving the mouse down points the nose down\n"
                                                                 "Off: moving the mouse down points the nose up\n"
                                                                 "Default: On" },

    // The controller buttons the settings overlay shows, listed above Touchpad as Back.
    { "controller", "button_style", "AC6_BUTTON_STYLE", kText, "\"playstation\"",
      "The controller buttons the settings overlay and the game's prompts show\n"
      "The PlayStation buttons are partial, some in-game glyphs still show Xbox buttons\n"
      "Default: Xbox" },
    // A DualShock 4 / DualSense's touchpad click as Back (input/input.cpp).
    { "controller", "touchpad_back", "AC6_TOUCHPAD_BACK", kBool, "true",
      "Makes the touchpad click act as Back on DualShock 4 and DualSense controllers\n"
      "Default: Off" },
    { "controller", "adaptive_triggers", "AC6_ADAPTIVE_TRIGGERS", kBool, "false",
      "DualSense adaptive trigger effects, similar to Ace Combat 8\n"
      "Default: On" },
    { "controller", "analog_yaw", "AC6_ANALOG_YAW", kBool, "true",
      "The rudder follows how far a trigger is pulled\n"
      "Only with THROTTLE/YAW on REVERSE in the game's Controller Settings\n"
      "Default: Off" },
    // The gate for custom controls and a HOTAS (game/flight_input.cpp), and Advanced controls'
    // switch (input/advanced.cpp).
    { "controller", "custom_flight", "AC6_CUSTOM_FLIGHT", kBool, "true",
      "Enables custom axes and key bindings to use with controller and HOTAS\n"
      "Actions without bindings keep the game's controls\n"
      "Does not affect keyboard and mouse flight\n"
      "Default: Off" },

    // Advanced controls' axes: percentages.
    { "advanced_controls", "pitch_deadzone", "AC6_ADV_PITCH_DEADZONE", kNumber, "5", "The dead zone around the centre of the pitch axis, in %\n"
                                                                                     "Default: 5" },
    { "advanced_controls", "pitch_full", "AC6_ADV_PITCH_FULL", kNumber, "100", "Where pitch reaches full deflection, in % of the axis\n"
                                                                               "Default: 100" },
    { "advanced_controls", "pitch_curve", "AC6_ADV_PITCH_CURVE", kNumber, "0", "The pitch response: 0 is straight, 100 a cube, finer near the centre\n"
                                                                               "Default: 0" },
    { "advanced_controls", "pitch_invert", "AC6_ADV_PITCH_INVERT", kBool, "true", "Reverses pitch\n"
                                                                                  "Default: Off" },
    { "advanced_controls", "roll_deadzone", "AC6_ADV_ROLL_DEADZONE", kNumber, "5", "The dead zone around the centre of the roll axis, in %\n"
                                                                                   "Default: 5" },
    { "advanced_controls", "roll_full", "AC6_ADV_ROLL_FULL", kNumber, "100", "Where roll reaches full deflection, in % of the axis\n"
                                                                             "Default: 100" },
    { "advanced_controls", "roll_curve", "AC6_ADV_ROLL_CURVE", kNumber, "0", "The roll response: 0 is straight, 100 a cube, finer near the centre\n"
                                                                             "Default: 0" },
    { "advanced_controls", "roll_invert", "AC6_ADV_ROLL_INVERT", kBool, "true", "Reverses roll\n"
                                                                                "Default: Off" },
    { "advanced_controls", "yaw_deadzone", "AC6_ADV_YAW_DEADZONE", kNumber, "5", "The dead zone around the centre of the yaw axis, in %\n"
                                                                                 "Default: 5" },
    { "advanced_controls", "yaw_full", "AC6_ADV_YAW_FULL", kNumber, "100", "Where yaw reaches full deflection, in % of the axis\n"
                                                                           "Default: 100" },
    { "advanced_controls", "yaw_curve", "AC6_ADV_YAW_CURVE", kNumber, "0", "The yaw response: 0 is straight, 100 a cube, finer near the centre\n"
                                                                           "Default: 0" },
    { "advanced_controls", "yaw_invert", "AC6_ADV_YAW_INVERT", kBool, "true", "Reverses yaw\n"
                                                                              "Default: Off" },
    { "advanced_controls", "throttle_deadzone", "AC6_ADV_THROTTLE_DEADZONE", kNumber, "2", "The dead zone at rest for thrust and brake, in %\n"
                                                                                           "Default: 2" },
    { "advanced_controls", "throttle_full", "AC6_ADV_THROTTLE_FULL", kNumber, "100", "Where thrust and brake reach full, in % of the axis\n"
                                                                                     "Default: 100" },
    { "advanced_controls", "throttle_curve", "AC6_ADV_THROTTLE_CURVE", kNumber, "0", "The thrust and brake response: 0 is straight, 100 a cube, finer near rest\n"
                                                                                     "Default: 0" },
    { "advanced_controls", "lever_detent", "AC6_ADV_LEVER_DETENT", kNumber, "50",
      "For a single thrust / brake lever: where brake ends and thrust begins, in % of its travel\n"
      "Default: 50" },
    { "advanced_controls", "lever_detent_width", "AC6_ADV_LEVER_DETENT_WIDTH", kNumber, "4",
      "For a single thrust / brake lever: a neutral zone around the detent, in % of its travel\n"
      "Default: 4" },
    { "advanced_controls", "look_deadzone", "AC6_ADV_LOOK_DEADZONE", kNumber, "5", "The dead zone around the centre of the camera axes, in %\n"
                                                                                   "Default: 5" },
    { "advanced_controls", "look_full", "AC6_ADV_LOOK_FULL", kNumber, "100", "Where the camera axes reach full deflection, in % of the axis\n"
                                                                             "Default: 100" },
    { "advanced_controls", "look_curve", "AC6_ADV_LOOK_CURVE", kNumber, "0", "The camera response: 0 is straight, 100 a cube, finer near the centre\n"
                                                                             "Default: 0" },
    { "advanced_controls", "look_invert", "AC6_ADV_LOOK_INVERT", kBool, "true", "Reverses the camera's axes\n"
                                                                                "Default: Off" },

    { "game", "language", "AC6_LANGUAGE", kText, "\"english\"",
      "The game's language\n"
      "English and Japanese: With dubbing\n"
      "Default: English" },
    { "game", "profile_name", "AC6_PROFILE_NAME", kText, "\"Player\"", "Your profile's name in the game\n"
                                                                       "Letters, digits and spaces, up to 15 characters\n"
                                                                       "Default: Player" },
    { "game", "dlc", "AC6_DLC", kBool, "true", "Loads the DLC found in the dlc folder\n"
                                               "Default: On" },
    { "game", "audio", "AC6_AUDIO", kBool, "true", "The game's sound\n"
                                                   "Default: On" },
    { "game", "overlay_text_size", "AC6_UI_TEXT_SIZE", kNumber, "1.25",
      "The size of the settings overlay's text\n"
      "Auto is at least the system's text size\n"
      "Default: Auto" },
    { "game", "check_updates", "AC6_CHECK_UPDATES", kBool, "true",
      "Looks for a new release on GitHub at each start, in the background, and offers it here\n"
      "Only for release download\n"
      "Self-built binary doesn't update itself\n"
      "Default: On" },

    // The captures' master switch: off by default, so no scene is captured by accident.
    { "bug_reports", "captures", "AC6_CAPTURES", kBool, "true",
      "Turns on the F11 and F12 captures and the recordings kept for them\n"
      "Only for developers or debugging\n"
      "Default: Off" },
    { "bug_reports", "frame_ring", "AC6_RING", kNumber, "5",
      "F11 saves the last this many seconds as pictures, 60 a second\n"
      "Default: 0 (off)" },
    { "bug_reports", "frame_ring_height", "AC6_RING_HEIGHT", kNumber, "720", "The F11 pictures' height, in pixels\n"
                                                                             "Default: 720" },
    { "bug_reports", "pipeline_ring", "AC6_PIPELINE_RING", kNumber, "5",
      "F12 saves the last this many seconds as a replayable capture (about 2 GB of RAM for 5 s)\n"
      "At 0, F12 captures the next 2 frames\n"
      "Default: 0" },
    { "bug_reports", "frame_times", "AC6_FRAME_TIMES", kBool, "true", "Writes frame time statistics to the log every 2 seconds\n"
                                                                      "Default: Off" },
    { "bug_reports", "fps_counter", "AC6_FPS_COUNTER", kBool, "true",
      "Shows the frame rate and frame time in the top-right corner\n"
      "Default: Off" },
    { "bug_reports", "render_stats", "AC6_RENDER_STATS", kBool, "true", "Writes renderer statistics to the log every 120 frames\n"
                                                                        "Default: Off" },
    // Modding (gpu/texture_mods.h): texture dumps and replacements by content hash.
    { "modding", "dump_textures", "AC6_DUMP_TEXTURES", kBool, "true",
      "Saves every texture the game uses to dump/textures/<hash>.dds next to the game, once each\n"
      "Default: Off" },
    { "modding", "texture_replacements", "AC6_TEXTURE_MODS", kBool, "false",
      "Replaces textures with mods/<mod>/textures/<hash>.dds next to the game, at any resolution\n"
      "F5 reads the files again, so an edited texture shows at once\n"
      "Default: On" },
    { "modding", "package_mods", "AC6_PACKAGE_MODS", kBool, "false",
      "Loads the mods' game data from mods/<mod>/packages/ when the game starts\n"
      "Default: On" },
    // Camera jitter at unlocked frame rates: the attitude smoothing on / off, and its trace.
    { "bug_reports", "attitude_smoothing", "AC6_ATTITUDE_SMOOTHING", kBool, "false",
      "With an unlocked frame rate, turns the player aircraft by a smoothed frame time to reduce jitter\n"
      "Default: On" },
    { "bug_reports", "frame_smoothing", "AC6_FRAME_SMOOTHING", kBool, "false",
      "With VSync on, the game moves on by a smoothed frame time, even when frames show a refresh apart\n"
      "Default: On" },
    { "bug_reports", "attitude_trace", "AC6_ATT_TRACE", kBool, "true",
      "Writes the player aircraft's rotation step and the camera's direction to the log every frame\n"
      "Default: Off" },
    // The graphics driver's paths (plume reads the variables when it makes the device) and the GPU debug mode
    // (gpu/video.cpp).
    { "bug_reports", "pipeline_libraries", "AC6_PIPELINE_LIBRARIES", kBool, "false",
      "Builds each graphics pipeline from ready-made parts the first time it is needed\n"
      "Off: whole pipelines are built instead, a little slower to build\n"
      "Try Off if the game crashes or freezes with your graphics driver\n"
      "Default: On" },
    { "bug_reports", "dynamic_rendering", "AC6_DYNAMIC_RENDERING", kBool, "false",
      "Draws with Vulkan's dynamic rendering\n"
      "Off: classic render passes instead, and whole pipelines (Pipeline libraries need it)\n"
      "Try Off if the game crashes, freezes or draws wrongly with your graphics driver\n"
      "Default: On" },
    // The renderer's kept copies of guest vertex data (gpu/renderer.cpp: UseVertexStream, and the streams the shaders
    // read by address).
    { "bug_reports", "vertex_cache", "AC6_VERTEX_CACHE", kBool, "false",
      "Keeps the models' vertex data on the graphics card from frame to frame, copying only what the game changes\n"
      "Off: copied again every frame, costs CPU\n"
      "Try Off if parts of aircraft or other models go missing after playing for a while\n"
      "Default: On" },
    { "bug_reports", "stream_cache", "AC6_STREAM_CACHE", kBool, "false",
      "Keeps the vertex data the game's shaders read directly on the graphics card from frame to frame, copying only what the game changes\n"
      "Off: copied again every frame, costs CPU\n"
      "Try Off if parts of aircraft or other models go missing after playing for a while\n"
      "Default: On" },
    { "bug_reports", "gpu_debug", "AC6_GPU_DEBUG", kBool, "true",
      "Marks every draw so the log can name the one where the graphics card fails, and draws one frame at a time\n"
      "Much slower: only for reproducing a crash, a freeze or a rendering bug, then sending ac6_runtime.log\n"
      "Default: Off" },
};

std::vector<std::string> s_report;   // for Log(): the log isn't open yet when Load runs
std::filesystem::path s_path;          // the settings file Load read (or wrote)
std::vector<std::string> s_applied;    // the variables Load set

// The settings overlay's view of each setting (src/ui/overlay.cpp): its tab, its label, the values it offers
// ("value=Label|value=Label", "" for free text or a range), the range for numbers, and the default (as the variable's
// text: what an unset setting means). One row per entry of kSettings.
struct Ui
{
    const char* key;
    const char* tab;
    const char* label;
    const char* choices;
    double min, max, step;
    const char* def;
    const char* group;
};

const Ui kUi[] = {
    { "monitor", "Display", "Monitor", "=Primary", 0, 0, 0, "", "SCREEN" },   // + the connected monitors (overlay)
    { "resolution", "Display", "Resolution", "native=Native", 0, 0, 0, "native", "SCREEN" },   // + the monitor's modes (overlay)
    { "fullscreen", "Display", "Display mode", "off=Windowed|borderless=Borderless|exclusive=Exclusive", 0, 0, 0, "borderless", "SCREEN" },
    { "refresh", "Display", "Refresh rate", "0=Highest", 0, 0, 0, "0", "SCREEN" },   // + the mode's rates (overlay)
    { "render_resolution", "Graphics", "Render scale", "", 0.5, 4, 0.05, "1", "IMAGE" },
    { "widescreen", "Display", "Aspect ratio", "1=Auto|0=16:9", 0, 0, 0, "1", "SCREEN" },
    { "cutscene_widescreen", "Display", "Cutscene aspect ratio", "1=Auto|0=16:9", 0, 0, 0, "0", "SCREEN" },
    { "hud_edges", "Game", "HUD position", "", 0, 1, 0.01, "0", "HUD" },
    { "hud_custom", "Game", "Custom HUD colours",
      "off=Off|custom=Custom|protanopia=Protanopia|deuteranopia=Deuteranopia|tritanopia=Tritanopia", 0, 0, 0, "off", "HUD" },
    { "hud_colour", "Game", "HUD colour", "", 0, 0, 0, "", "HUD" },   // a colour picker (ui/overlay.cpp)
    { "hud_friendly", "Game", "Friendly colour", "", 0, 0, 0, "", "HUD" },
    { "hud_enemy", "Game", "Enemy/Alert colour", "", 0, 0, 0, "", "HUD" },
    { "fov_offset", "Game", "Extended field of view", "", 0, 40, 1, "0", "GENERAL" },
    { "vsync", "Display", "VSync", "", 0, 0, 0, "1", "FRAME RATE" },
    { "frame_rate", "Display", "Frame rate cap",
      "=Match display|console=Console (30 / 60)|60=60|120=120|240=240|0=Uncapped", 0, 0, 0, "", "FRAME RATE" },
    { "cutscene_unlock", "Display", "Unlock cutscene FPS", "", 0, 0, 0, "1", "FRAME RATE" },   // on by default
    { "menu_unlock", "Display", "Unlock menu FPS", "", 0, 0, 0, "1", "FRAME RATE" },
    { "aa", "Graphics", "Anti-aliasing", "off=Off|smaa=SMAA|taa=TAA|smaa+taa=SMAA + TAA|fsr=FSR 3.1", 0, 0, 0, "off", "IMAGE" },
    // The Graphics tab's image settings. FSR's qualities are listed from the fastest. MSAA offers Off, the console's
    // 2x, 4x and 8x. Max anisotropy has no row (tab "" hides it, its variable still applies). The anisotropic
    // override's 0 keeps the console's filtering.
    { "fsr_quality", "Graphics", "FSR quality",
      "ultra=Ultra performance|performance=Performance|balanced=Balanced|quality=Quality|native=Native", 0, 0, 0, "native", "IMAGE" },
    { "msaa", "Graphics", "MSAA", "1=Off|console=Console (2x)|4=4x|8=8x", 0, 0, 0, "console", "IMAGE" },
    { "max_anisotropy", "", "Max anisotropy", "0=Off|2=2x|4=4x|8=8x|16=16x", 0, 0, 0, "16", "TEXTURES AND DETAIL" },
    { "anisotropic_override", "Graphics", "Anisotropic filtering", "0=Console|2=2x|4=4x|8=8x|16=16x", 0, 0, 0, "0", "TEXTURES AND DETAIL" },
    { "volumetrics", "Debug", "Volumetric effects (unfinished)", "", 0, 0, 0, "0", "EXPERIMENTAL" },
    { "volumetric_kinds", "Debug", "Volumetric effect kinds", "", 0, 0, 0, "smoke,fire,cloud,trail", "EXPERIMENTAL" },
    { "lod_scale", "Graphics", "Level of detail scale", "", 1, 8, 0.25, "1", "TEXTURES AND DETAIL" },
    // Aircraft edges fix and round bloom are in the Debug tab. The depth of field setting offers Console (the default)
    // and Enhanced. Deband is off by default.
    { "centroid", "Debug", "Aircraft edges fix", "", 0, 0, 0, "1", "RENDERING" },
    { "deband", "Graphics", "Deband", "", 0, 4, 0.1, "0", "IMAGE" },
    { "dof_fix", "Graphics", "Depth of field", "0=Console|1=Enhanced", 0, 0, 0, "0", "EFFECTS" },
    // Bloom: Console or Enhanced. The lens effects below it (motion blur, film grain, vignette, chromatic
    // aberration) are off by default and in every preset.
    { "bloom", "Graphics", "Bloom", "console=Console|enhanced=Enhanced", 0, 0, 0, "console", "EFFECTS" },
    { "motion_blur", "Graphics", "Motion blur", "", 0, 1, 0.05, "0", "EFFECTS" },   // (with TAA or FSR)
    { "film_grain", "Graphics", "Film grain", "", 0, 1, 0.05, "0", "EFFECTS" },
    { "vignette", "Graphics", "Vignette", "", 0, 1, 0.05, "0", "EFFECTS" },
    { "chromatic_aberration", "Graphics", "Chromatic aberration", "", 0, 1, 0.05, "0", "EFFECTS" },
    // Off / On, off by default: not the console's picture. The Enhanced preset turns it on.
    { "ambient_occlusion", "Graphics", "Ambient occlusion", "", 0, 0, 0, "0", "EFFECTS" },
    { "shadow_detail", "Graphics", "Sharp cutscene shadows", "", 0, 0, 0, "0", "EFFECTS" },
    // Off / On, off by default, at the next start. The Enhanced preset turns it on.
    { "planar_reflection", "Graphics", "Planar reflection", "", 0, 0, 0, "0", "EFFECTS" },
    { "wide_blurs", "Debug", "Round bloom (widescreen)", "", 0, 0, 0, "1", "RENDERING" },
    { "sdr_10bit", "Graphics", "10-bit output", "", 0, 0, 0, "0", "IMAGE" },
    { "mode", "Display", "HDR", "sdr=Off|hdr=On", 0, 0, 0, "sdr", "HDR" },
    // PsychoV-30 first and the default. ACES and the highlight curve not offered.
    { "hdr_tonemap", "Display", "HDR tone mapping", "psychov30=PsychoV30|renodrt=RenoDRT", 0, 0, 0, "psychov30", "HDR" },
    { "hdr_paper_white", "Display", "HDR paper white (nits)", "", 80, 500, 1, "203", "HDR" },
    { "hdr_peak", "Display", "HDR peak brightness (nits)", "", 200, 4000, 10, "1000", "HDR" },   // default: the display's (overlay)
    // (UI brightness, listed below the peak)
    { "hdr_ui_brightness", "Display", "UI brightness (nits)", "", 80, 1000, 1, "250", "HDR" },
    // HDR from the game's scene: always on and not shown in the overlay (tab ""). The file's hdr_scene and
    // AC6_HDR_SCENE can still turn it off.
    { "hdr_scene", "", "HDR from the game's scene", "", 0, 0, 0, "1", "HDR" },
    // (The RenoDX-style grading below in a window of its own, the overlay's "HDR grading" row.)
    { "hdr_exposure", "", "Exposure", "", 0.5, 2, 0.01, "1", "HDR GRADING" },
    { "hdr_highlights", "", "Highlights", "", 0, 100, 1, "50", "HDR GRADING" },
    { "hdr_shadows", "", "Shadows", "", 0, 100, 1, "50", "HDR GRADING" },
    { "hdr_contrast", "", "Contrast", "", 0, 100, 1, "50", "HDR GRADING" },
    { "hdr_saturation", "", "Saturation", "", 0, 100, 1, "50", "HDR GRADING" },
    { "hdr_hue_correction", "", "Hue correction", "", 0, 100, 1, "50", "HDR GRADING" },
    { "hdr_gamut_compression", "", "Gamut compression", "", 0, 100, 1, "100", "HDR GRADING" },
    { "hdr_compression", "", "Highlight compression", "", 0, 2, 0.01, "1", "HDR GRADING" },
    // (Cloud brightness, a lift after the fact, hidden: Cloud light lights them instead.)
    { "hdr_clouds", "", "Cloud brightness", "", 1, 6, 0.05, "1", "HDR GRADING" },
    // (The HDR tuning sliders below are hidden from the overlay. Their settings-file keys and variables still apply.)
    { "hdr_cloud_light", "", "Cloud light", "", 0.5, 4, 0.05, "1", "HDR GRADING" },
    { "hdr_cloud_edge", "", "Cloud edge light", "", 0, 8, 0.05, "1.5", "HDR GRADING" },
    { "hdr_aircraft_light", "", "Aircraft light", "", 0.5, 4, 0.05, "1.5", "HDR GRADING" },
    { "hdr_effects", "", "Effects brightness", "", 0, 4, 0.05, "1", "HDR GRADING" },
    { "hdr_afterburner", "", "Afterburner glow", "", 0, 6, 0.05, "3", "HDR GRADING" },
    { "hdr_lit_highlights", "", "Lit highlights", "", 0, 100, 1, "100", "HDR GRADING" },
    { "hdr_stripes", "Debug", "HDR brightness stripes", "off=Off|output=Display brightness|scene=Game's values", 0, 0, 0, "off", "RENDERING" },
    { "keyboard_mouse", "Controls", "Keyboard and mouse", "", 0, 0, 0, "0", "GENERAL" },
    { "cursor_hide_seconds", "Controls", "Hide idle cursor after (s)", "", 0, 30, 0.5, "3", "GENERAL" },
    { "mouse_controls", "Controls", "Mouse controls", "aircraft=Aircraft|camera=Camera", 0, 0, 0, "aircraft", "MOUSE" },
    { "steer_model", "Controls", "Steering model", "velocity=Velocity|position=Position", 0, 0, 0, "velocity", "MOUSE" },
    { "sensitivity_x", "Controls", "Sensitivity X (roll)", "", 0.25, 3, 0.05, "1", "MOUSE" },
    { "sensitivity_y", "Controls", "Sensitivity Y (pitch)", "", 0.25, 3, 0.05, "1", "MOUSE" },
    { "invert_x", "Controls", "Invert X", "", 0, 0, 0, "0", "MOUSE" },
    { "invert_y", "Controls", "Invert Y", "", 0, 0, 0, "1", "MOUSE" },
    { "touchpad_back", "Controls", "Touchpad as Back (PlayStation)", "", 0, 0, 0, "0", "CONTROLLER" },
    { "button_style", "Controls", "Button style", "xbox=Xbox|playstation=PlayStation", 0, 0, 0, "xbox", "CONTROLLER" },
    { "adaptive_triggers", "Controls", "Adaptive triggers (DualSense)", "", 0, 0, 0, "1", "CONTROLLER" },
    { "analog_yaw", "Controls", "Analog yaw (reverse throttle/yaw)", "", 0, 0, 0, "0", "CONTROLLER" },
    { "custom_flight", "Advanced controls", "Advanced controls", "", 0, 0, 0, "0", "ADVANCED CONTROLS" },
    { "pitch_deadzone", "Advanced controls", "Pitch dead zone (%)", "", 0, 50, 1, "5", "PITCH" },
    { "pitch_full", "Advanced controls", "Pitch full at (%)", "", 50, 100, 1, "100", "PITCH" },
    { "pitch_curve", "Advanced controls", "Pitch curve (%)", "", 0, 100, 5, "0", "PITCH" },
    { "pitch_invert", "Advanced controls", "Invert pitch", "", 0, 0, 0, "0", "PITCH" },
    { "roll_deadzone", "Advanced controls", "Roll dead zone (%)", "", 0, 50, 1, "5", "ROLL" },
    { "roll_full", "Advanced controls", "Roll full at (%)", "", 50, 100, 1, "100", "ROLL" },
    { "roll_curve", "Advanced controls", "Roll curve (%)", "", 0, 100, 5, "0", "ROLL" },
    { "roll_invert", "Advanced controls", "Invert roll", "", 0, 0, 0, "0", "ROLL" },
    { "yaw_deadzone", "Advanced controls", "Yaw dead zone (%)", "", 0, 50, 1, "5", "YAW" },
    { "yaw_full", "Advanced controls", "Yaw full at (%)", "", 50, 100, 1, "100", "YAW" },
    { "yaw_curve", "Advanced controls", "Yaw curve (%)", "", 0, 100, 5, "0", "YAW" },
    { "yaw_invert", "Advanced controls", "Invert yaw", "", 0, 0, 0, "0", "YAW" },
    { "throttle_deadzone", "Advanced controls", "Throttle dead zone (%)", "", 0, 50, 1, "2", "THROTTLE" },
    { "throttle_full", "Advanced controls", "Throttle full at (%)", "", 50, 100, 1, "100", "THROTTLE" },
    { "throttle_curve", "Advanced controls", "Throttle curve (%)", "", 0, 100, 5, "0", "THROTTLE" },
    { "lever_detent", "Advanced controls", "Lever detent (%)", "", 0, 100, 1, "50", "THROTTLE" },
    { "lever_detent_width", "Advanced controls", "Lever detent width (%)", "", 0, 20, 1, "4", "THROTTLE" },
    { "look_deadzone", "Advanced controls", "Camera dead zone (%)", "", 0, 50, 1, "5", "CAMERA" },
    { "look_full", "Advanced controls", "Camera full at (%)", "", 50, 100, 1, "100", "CAMERA" },
    { "look_curve", "Advanced controls", "Camera curve (%)", "", 0, 100, 5, "0", "CAMERA" },
    { "look_invert", "Advanced controls", "Invert camera", "", 0, 0, 0, "0", "CAMERA" },
    // Each language in its own name.
    { "language", "Game", "Language",
      "english=English|japanese=日本語|german=Deutsch|french=Français|spanish=Español|italian=Italiano", 0, 0, 0, "english", "GENERAL" },
    { "profile_name", "Game", "Profile name", "", 0, 0, 0, "Player", "GENERAL" },
    { "dlc", "Game", "Use DLC", "", 0, 0, 0, "1", "DOWNLOADABLE CONTENT" },
    { "audio", "Game", "Audio", "", 0, 0, 0, "1", "GENERAL" },
    { "overlay_text_size", "Game", "Settings text size", "=Auto|1=100%|1.25=125%|1.5=150%", 0, 0, 0, "", "GENERAL" },
    { "check_updates", "Game", "Check for updates at start", "", 0, 0, 0, "1", "UPDATES" },
    { "captures", "Debug", "Captures (F11 / F12)", "", 0, 0, 0, "0", "CAPTURE AND LOGS" },
    { "frame_ring", "Debug", "Frame ring (F11), seconds", "", 0, 30, 0.5, "0", "CAPTURE AND LOGS" },
    { "frame_ring_height", "Debug", "Frame ring picture height", "", 360, 2160, 10, "720", "CAPTURE AND LOGS" },
    { "pipeline_ring", "Debug", "Replayable capture (F12), seconds", "", 0, 10, 0.5, "0", "CAPTURE AND LOGS" },
    { "frame_times", "Debug", "Log frame times", "", 0, 0, 0, "0", "CAPTURE AND LOGS" },
    { "fps_counter", "Debug", "FPS counter", "", 0, 0, 0, "0", "FRAME RATE" },
    { "render_stats", "Debug", "Log render stats", "", 0, 0, 0, "0", "CAPTURE AND LOGS" },
    { "attitude_smoothing", "Debug", "Aircraft rotation smoothing", "", 0, 0, 0, "1", "FRAME RATE" },
    { "dump_textures", "Debug", "Dump textures", "", 0, 0, 0, "0", "MODDING" },
    { "texture_replacements", "Debug", "Texture replacements", "", 0, 0, 0, "1", "MODDING" },
    { "package_mods", "Debug", "Game data mods", "", 0, 0, 0, "1", "MODDING" },
    { "frame_smoothing", "Debug", "Frame time smoothing (VSync)", "", 0, 0, 0, "1", "FRAME RATE" },
    { "attitude_trace", "Debug", "Log camera every frame", "", 0, 0, 0, "0", "CAPTURE AND LOGS" },
    { "pipeline_libraries", "Debug", "Pipeline libraries", "", 0, 0, 0, "1", "GRAPHICS DRIVER" },
    { "dynamic_rendering", "Debug", "Dynamic rendering", "", 0, 0, 0, "1", "GRAPHICS DRIVER" },
    { "vertex_cache", "Debug", "Vertex cache", "", 0, 0, 0, "1", "GRAPHICS DRIVER" },
    { "stream_cache", "Debug", "Vertex stream cache", "", 0, 0, 0, "1", "GRAPHICS DRIVER" },
    { "gpu_debug", "Debug", "GPU debug mode", "", 0, 0, 0, "0", "GRAPHICS DRIVER" },
};
static_assert(std::size(kUi) == std::size(kSettings));

// A new settings file: only the settings changed from their defaults go in it (the overlay's Apply writes them, and
// takes a setting set back to its default out again), so a default changed in a later release reaches everyone.
std::string Template()
{
    return "# Ace Combat 6 settings changed from their defaults (the settings overlay, F4, writes them).\n";
}

// A TOML value as the variable's text (nothing for a value of the wrong kind).
bool ToText(const toml::node& node, Kind kind, std::string& out)
{
    if (const auto b = node.value<bool>(); b && node.is_boolean())
        out = *b ? "1" : "0";
    else if (node.is_integer())
        out = std::to_string(*node.value<int64_t>());
    else if (node.is_floating_point())
        out = std::format("{}", *node.value<double>());
    else if (node.is_string())
        out = *node.value<std::string>();
    else
        return false;
    if (kind == kBool && !node.is_boolean())
        return false;
    return true;
}

std::string Alias(const char* aliases, const std::string& value)
{
    for (std::string_view list = aliases ? aliases : ""; !list.empty();)
    {
        const size_t comma = list.find(',');
        const std::string_view pair = list.substr(0, comma);
        if (const size_t eq = pair.find('='); eq != std::string_view::npos && pair.substr(0, eq) == value)
            return std::string(pair.substr(eq + 1));
        list = comma == std::string_view::npos ? std::string_view() : list.substr(comma + 1);
    }
    return value;
}

void Apply(const char* env, const std::string& value, std::string_view name)
{
    if (const char* now = std::getenv(env))
    {
        s_report.push_back(std::format("{} = {} not applied: {}={} is set in the environment", name, value, env, now));
        return;
    }
    platform::SetEnvironment(env, value.c_str());
    s_applied.push_back(env);
    s_report.push_back(std::format("{} -> {}={}", name, env, value));
}

} // namespace

void Load(const std::filesystem::path& directory)
{
    std::filesystem::path path = directory / "ac6_settings.toml";
    if (const char* v = std::getenv("AC6_SETTINGS"); v && *v)
        path = v;
    s_path = path;
    std::error_code error;
    if (!std::filesystem::exists(path, error))
    {
        std::ofstream(path, std::ios::binary) << Template();
        s_report.push_back(std::format("wrote {} (a new file: the settings changed from their defaults go in it)", platform::PathUtf8(path)));
        return;
    }
    toml::parse_result parsed = toml::parse_file(platform::PathUtf8(path));
    if (!parsed)
    {
        s_report.push_back(std::format("{}: {} (line {}); no settings applied", platform::PathUtf8(path), parsed.error().description(),
            parsed.error().source().begin.line));
        return;
    }
    const toml::table& t = parsed.table();
    s_report.push_back(std::format("read {}", platform::PathUtf8(path)));
    for (const auto& [sectionKey, sectionNode] : t)
    {
        const toml::table* section = sectionNode.as_table();
        const std::string_view sectionName = sectionKey.str();
        if (!section)
        {
            s_report.push_back(std::format("{}: not a section, ignored", sectionName));
            continue;
        }
        if (IsInputSection(sectionName))
            continue;   // the keyboard and mouse's (input/kbm.cpp)
        for (const auto& [key, node] : *section)
        {
            const std::string name = std::format("{}.{}", sectionName, key.str());
            std::string value;
            if (sectionName == "env")
            {
                if (!key.str().starts_with("AC6_") || !ToText(node, kText, value))
                    s_report.push_back(std::format("{}: not an AC6_* variable with a value, ignored", name));
                else
                    Apply(std::string(key.str()).c_str(), value, name);
                continue;
            }
            const Setting* setting = nullptr;
            for (const Setting& s : kSettings)
                if (sectionName == s.section && key.str() == s.key)
                    setting = &s;
            if (!setting)
            {
                if (sectionName != "mouse")   // (the rest of [mouse] is input/kbm.cpp's)
                    s_report.push_back(std::format("{}: unknown setting, ignored", name));
                continue;
            }
            if (!ToText(node, setting->kind, value))
            {
                s_report.push_back(std::format("{}: wrong kind of value ({}), ignored", name,
                    setting->kind == kBool ? "true or false" : setting->kind == kNumber ? "a number" : "text"));
                continue;
            }
            Apply(setting->env, Alias(setting->aliases, value), name);
        }
    }
}

void Log()
{
    for (const std::string& line : s_report)
        log::Info("Settings: {}", line);
    s_report.clear();
}

// The language packs (game/localization.h) as the Language setting's last choices, the first one its default: a player
// who installed a translation wants it. The help's Default line names it too.
static void AddLanguagePacks(Entry& entry)
{
    const std::vector<localization::Pack>& packs = localization::Packs();
    if (packs.empty())
        return;
    static const std::string choices = [&] {
        std::string list = entry.choices;
        for (const localization::Pack& p : packs)
            list += "|" + p.id + "=" + p.name;
        return list;
    }();
    static const std::string help = [&] {
        std::string text = entry.help;
        if (const size_t at = text.rfind("\nDefault:"); at != std::string::npos)
            text.resize(at);
        return text + "\nLanguage packs come from the localization folder\nDefault: " + packs.front().name;
    }();
    entry.choices = choices.c_str();
    entry.help = help.c_str();
    entry.def = packs.front().id.c_str();
}

std::vector<Entry> Entries()
{
    std::vector<Entry> out;
    for (const Setting& s : kSettings)
    {
        const Ui* ui = nullptr;
        for (const Ui& u : kUi)
            if (std::string_view(u.key) == s.key)
                ui = &u;
        if (!ui)
            continue;
        out.push_back({ s.section, s.key, s.env, s.kind == kBool ? ValueKind::Bool : s.kind == kNumber ? ValueKind::Number : ValueKind::Text,
            s.help, ui->tab, ui->label, ui->choices, ui->min, ui->max, ui->step, ui->def, ui->group });
        if (std::string_view(s.key) == "language")
            AddLanguagePacks(out.back());
    }
    return out;
}

std::filesystem::path Path()
{
    return s_path;
}

namespace {

std::mutex s_liveMutex;
std::map<std::string, std::string, std::less<>> s_live;   // the overlay's values ("" = unset)
std::atomic<uint64_t> s_generation{ 0 };

} // namespace

std::string Get(const char* env)
{
    {
        std::lock_guard lock(s_liveMutex);
        if (const auto it = s_live.find(std::string_view(env)); it != s_live.end())
            return it->second;
    }
    const char* v = std::getenv(env);
    return v ? v : "";
}

void SetLive(const char* env, const std::string& value)
{
    {
        std::lock_guard lock(s_liveMutex);
        s_live[env] = value;
    }
    s_generation.fetch_add(1, std::memory_order_release);
}

uint64_t Generation()
{
    return s_generation.load(std::memory_order_acquire);
}

bool ProfileNameChar(unsigned int c)
{
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == ' ';
}

std::string CleanProfileName(std::string_view name)
{
    std::string out;
    for (const char c : name)   // a multi-byte character's bytes are all above 0x7F: dropped whole
        if (ProfileNameChar(static_cast<unsigned char>(c)))
            out += c;
    const size_t first = out.find_first_not_of(' ');
    if (first == std::string::npos)
        return {};
    out = out.substr(first, kProfileNameMax);
    out.erase(out.find_last_not_of(' ') + 1);
    return out;
}

std::string VariableText(const Entry& entry, const std::string& value)
{
    for (const Setting& s : kSettings)
        if (std::string_view(s.key) == entry.key && std::string_view(s.section) == entry.section)
            return Alias(s.aliases, value);
    return value;
}

bool SetByFile(const char* env)
{
    return std::find(s_applied.begin(), s_applied.end(), env) != s_applied.end();
}

bool FromEnvironment(const Entry& entry)
{
    return std::getenv(entry.env) && std::find(s_applied.begin(), s_applied.end(), entry.env) == s_applied.end();
}

void ForgetApplied()
{
    for (const std::string& name : s_applied)
        platform::UnsetEnvironment(name.c_str());
    s_applied.clear();
}

std::string FileError()
{
    std::error_code ec;
    if (!std::filesystem::exists(s_path, ec))
        return {};
    toml::parse_result parsed = toml::parse_file(platform::PathUtf8(s_path));
    if (parsed)
        return {};
    return std::format("line {}: {}", parsed.error().source().begin.line, parsed.error().description());
}

std::string FileValue(const Entry& entry)
{
    // The file parsed again only when it changed (its time or size): the overlay reads every setting at each opening.
    static std::mutex mutex;
    static std::optional<toml::parse_result> cached;
    static std::filesystem::file_time_type cachedTime;
    static uintmax_t cachedSize = 0;
    std::lock_guard lock(mutex);
    std::error_code ec;
    const auto time = std::filesystem::last_write_time(s_path, ec);
    const uintmax_t size = ec ? 0 : std::filesystem::file_size(s_path, ec);
    if (ec)
        return {};
    if (!cached || time != cachedTime || size != cachedSize)
    {
        cached = toml::parse_file(platform::PathUtf8(s_path));
        cachedTime = time;
        cachedSize = size;
    }
    const toml::parse_result& parsed = *cached;
    if (!parsed)
        return {};
    const toml::node* node = parsed.table()[entry.section][entry.key].node();
    std::string value;
    if (!node || !ToText(*node, entry.kind == ValueKind::Bool ? kBool : entry.kind == ValueKind::Number ? kNumber : kText, value))
        return {};
    return value;
}

namespace {

// The value as TOML: booleans and numbers bare, text quoted.
std::string TomlValue(const Entry& entry, const std::string& value)
{
    if (entry.kind == ValueKind::Bool)
        return value == "1" || value == "true" ? "true" : "false";
    if (entry.kind == ValueKind::Number)
    {
        // A number bare, a word a number setting also takes (frame_rate = "console") quoted. Empty (a default that is no
        // number: Match display, the primary monitor, Auto) as an empty string: written as 0 it was the same as a choice
        // of 0 (frame_rate's Uncapped), which was then taken for the default and not saved.
        char* end = nullptr;
        if (value.empty())
            return "\"\"";
        std::strtod(value.c_str(), &end);
        if (end && *end == 0)
            return value;
    }
    std::string out = "\"";
    for (const char c : value)
    {
        if (c == '"' || c == '\\')
            out += '\\';
        out += c;
    }
    return out + "\"";
}

// A line of the file that sets `key` (live or commented out: "key =", "# key ="). The comment's `#` kept apart.
bool IsKeyLine(std::string_view line, std::string_view key, bool& commented)
{
    size_t i = line.find_first_not_of(" \t");
    if (i == std::string_view::npos)
        return false;
    commented = line[i] == '#';
    if (commented)
        i = line.find_first_not_of(" \t", i + 1);
    if (i == std::string_view::npos || line.substr(i, key.size()) != key)
        return false;
    i = line.find_first_not_of(" \t", i + key.size());
    return i != std::string_view::npos && line[i] == '=';
}

} // namespace

bool IsInputSection(std::string_view section)
{
    return section == "menu_keys" || section == "flight_keys" || section == "advanced_bindings";   // (advanced_bindings: input/advanced.cpp. [mouse] holds settings)
}

bool Save(const std::vector<std::pair<Entry, std::string>>& values, std::string& error)
{
    std::vector<TomlLine> lines;
    for (const auto& [entry, value] : values)
        lines.push_back({ entry.section, entry.key,
            value.empty() || TomlValue(entry, value) == TomlValue(entry, entry.def) ? std::string() : TomlValue(entry, value),
            TomlValue(entry, entry.def) });
    return SaveLines(lines, error);
}

bool SaveLines(const std::vector<TomlLine>& values, std::string& error)
{
    // An existing file read whole, or nothing is written: rebuilt from nothing it would lose every other setting.
    std::error_code ec;
    const bool exists = std::filesystem::exists(s_path, ec);
    std::ifstream in(s_path, std::ios::binary);
    if (exists && !in)
    {
        error = std::format("couldn't read {}", platform::PathUtf8(s_path));
        return false;
    }
    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (exists && in.bad())
    {
        error = std::format("couldn't read {}", platform::PathUtf8(s_path));
        return false;
    }
    in.close();
    // A file that doesn't parse applies nothing. Changes written into it wouldn't apply either.
    if (exists)
        if (toml::parse_result parsed = toml::parse(text); !parsed)
        {
            error = std::format("{} has an error on line {} ({}): fix it, or delete the file for a fresh one",
                platform::PathUtf8(s_path.filename()), parsed.error().source().begin.line, parsed.error().description());
            return false;
        }
    const bool crlf = text.find("\r\n") != std::string::npos;
    std::vector<std::string> lines;
    for (size_t at = 0; at <= text.size();)
    {
        const size_t end = text.find('\n', at);
        std::string line = text.substr(at, end == std::string::npos ? std::string::npos : end - at);
        if (!line.empty() && line.back() == '\r')
            line.pop_back();
        lines.push_back(std::move(line));
        if (end == std::string::npos)
            break;
        at = end + 1;
    }
    for (const TomlLine& entry : values)
    {
        // The setting's line in its section: replaced (uncommented), or taken out for "" (the default applies).
        const std::string header = std::format("[{}]", entry.section);
        size_t section = lines.size();
        for (size_t i = 0; i < lines.size(); i++)
            if (lines[i] == header)
                section = i;
        if (section == lines.size() && entry.value.empty())
            continue;
        if (section == lines.size())
        {
            if (!lines.empty() && !lines.back().empty())
                lines.push_back("");   // (one blank line before the section)
            lines.push_back(header);
            section = lines.size() - 1;
        }
        size_t at = lines.size(), sectionEnd = lines.size();
        for (size_t i = section + 1; i < lines.size(); i++)
        {
            if (!lines[i].empty() && lines[i][0] == '[')
            {
                sectionEnd = i;
                break;
            }
            bool commented = false;
            if (IsKeyLine(lines[i], entry.key, commented) && (at == lines.size() || !commented))
                at = i;
        }
        const std::string line = entry.value.empty() ? std::format("# {} = {}", entry.key, entry.def)
                                                     : std::format("{} = {}", entry.key, entry.value);
        bool atCommented = false;
        if (at < lines.size() && entry.value.empty() && IsKeyLine(lines[at], entry.key, atCommented) && atCommented)
            continue;   // (not set: nothing to take out)
        // Back to its default: the line goes, unless it belongs to an older file's commented template (its help line
        // below it, "#   ..."), where it is commented out again.
        if (at < lines.size() && entry.value.empty() && !(at + 1 < lines.size() && lines[at + 1].starts_with("#   ")))
        {
            lines.erase(lines.begin() + std::ptrdiff_t(at));
            // A section left with nothing but blank lines goes too.
            size_t end = section + 1;
            while (end < lines.size() && !(!lines[end].empty() && lines[end][0] == '['))
                end++;
            bool empty = true;
            for (size_t i = section + 1; i < end; i++)
                empty = empty && lines[i].find_first_not_of(" \t") == std::string::npos;
            if (empty)
            {
                size_t from = section;
                while (from > 0 && lines[from - 1].empty())
                    from--;
                lines.erase(lines.begin() + std::ptrdiff_t(from), lines.begin() + std::ptrdiff_t(end));
            }
            continue;
        }
        if (at < lines.size())
        {
            // A comment after the value is kept ("key = value   # ..."): what follows its last quote or bracket.
            std::string comment;
            const std::string& old = lines[at];
            const size_t valueAt = old.find('=');
            const size_t close = old.find_last_of("\"]");
            if (valueAt != std::string::npos)
                if (const size_t hash = old.find('#', close != std::string::npos && close > valueAt ? close : valueAt);
                    hash != std::string::npos && hash > valueAt)
                {
                    size_t space = hash;
                    while (space > valueAt + 1 && (old[space - 1] == ' ' || old[space - 1] == '\t'))
                        space--;
                    comment = old.substr(space);
                }
            lines[at] = line + comment;
        }
        else if (!entry.value.empty())
        {
            // After the section's last line (its blank lines before the next section kept).
            size_t insertAt = sectionEnd;
            while (insertAt > section + 1 && lines[insertAt - 1].empty())
                insertAt--;
            lines.insert(lines.begin() + std::ptrdiff_t(insertAt), line);
        }
    }
    std::string out;
    for (size_t i = 0; i < lines.size(); i++)
        out += lines[i] + (i + 1 < lines.size() ? (crlf ? "\r\n" : "\n") : "");
    // Written beside it and moved over it: a crash mid-write never leaves half a file.
    const std::filesystem::path temp = std::filesystem::path(s_path).concat(".new");
    {
        std::ofstream f(temp, std::ios::binary | std::ios::trunc);
        f << out;
        f.close();   // (the buffered tail is written here: checked too, or a short file would replace the good one)
        if (!f)
        {
            error = std::format("could not write {}", platform::PathUtf8(temp));
            std::error_code removeError;
            std::filesystem::remove(temp, removeError);
            return false;
        }
    }
    ec.clear();
    std::filesystem::rename(temp, s_path, ec);
    if (ec)
    {
        error = std::format("could not replace {}: {}", platform::PathUtf8(s_path), ec.message());
        return false;
    }
    return true;
}

} // namespace ac6::settings
