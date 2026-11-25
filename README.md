<h1 align="center">AC6:GracemeriaRecomp</h1>

<p align="center">
    <a href="https://github.com/Dipshet/GracemeriaRecomp/releases/latest"><b>Download</b></a>
    ·
    <a href="docs/modding/lua.md"><b>Modding Guide</b></a>
</p>

---

GracemeriaRecomp is an unofficial native PC port of Ace Combat 6: Fires of Liberation, originally released on Xbox 360, made through static recompilation. It is the successor of the [AC6Recomp](https://github.com/sal063/AC6_recomp) project: redesigned from the ground up, with a focus on performance, graphical enhancements, and QoL improvements.

**This project does not contain any game data. You must provide your own legally obtained copy of the game to play or build GracemeriaRecomp.**

[XenonRecomp](https://github.com/hedge-dev/XenonRecomp) translates the game's PowerPC code into C++, compiled into a native program. A runtime written for this game stands in for the console: its memory, files, audio and controllers. A Vulkan renderer draws the game's graphics directly, and its shaders are translated into SPIR-V by the port's own translator, built on [XenosRecomp](https://github.com/hedge-dev/XenosRecomp)'s. Both tools were made for [Unleashed Recompiled](https://github.com/hedge-dev/UnleashedRecomp), from which parts of this
port's runtime are adapted.

> [!CAUTION]
> Only download GracemeriaRecomp from this repository's
> [Releases](https://github.com/Dipshet/GracemeriaRecomp/releases/latest) page. Builds from anywhere else may contain dangerous modifications.

## Quick Access

- [System Requirements](#system-requirements)
- [How to Install](#how-to-install)
- [Features](#features)
- [Controls](#controls)
- [Known Issues](#known-issues)
- [FAQ](#faq)
- [Building](#building)
- [Credits](#credits)
- [Licence](#licence)

## System Requirements

**Game:** Ace Combat 6: Fires of Liberation, the North American release (4E4D07D1), as a disc image (`.iso`). The
European and Japanese releases are **not supported**.

| | Windows | Linux |
|-|-|-|
| **Operating system** | Windows 10 or 11, 64-bit | A modern 64-bit distro |
| **CPU** | AVX2: Intel Haswell (4th generation Core, 2013) or newer, AMD Excavator (2015) or Ryzen<br>Older CPUs with SSE4.2: the compatibility build | AVX2: Intel Haswell (4th generation Core, 2013) or newer, AMD Excavator (2015) or Ryzen |
| **GPU** | NVIDIA, AMD, or Intel with a Vulkan 1.2 driver<br>Intel: 6th generation Core (Skylake, 2015) or newer | NVIDIA, AMD, or Intel with a Vulkan 1.2 driver<br>Intel: 6th generation Core (Skylake, 2015) or newer |

> [!WARNING]
> If your CPU doesn't support AVX2, you must download the compatibility release (the file ending in
> `-windows-x64-compat.zip`). The main release won't start on your CPU. The compatibility release is for Windows only.

## How to Install

1) Make a disc image (`.iso`) of your own legally owned copy of the game.

2) Download [the latest release](https://github.com/Dipshet/GracemeriaRecomp/releases/latest) for your system and
   extract it into a folder: `-windows-x64.zip` for Windows, `-windows-x64-compat.zip` for Windows on a CPU without
   AVX2, `-linux-x64.tar.gz` for Linux.

3) Put your disc image in that folder (file name can be anything).

4) Start `AceCombat6.exe` for Windows, or `AceCombat6` for Linux.

> [!WARNING]
> Don't install the game in `Program Files`, or in a folder Windows Security's Controlled folder access protects
> (such as Documents or Desktop). The game can't write there: it can't create your save data, and your settings
> aren't kept. Use a folder you own, for example `C:\Games\GracemeriaRecomp`.

> [!TIP]
> To install downloadable content, place them extracted inside a `dlc` folder next to the game.

## Features

| Feature | Description |
|-|-|
| **Settings overlay** | Press **F4**, or click both sticks on a controller (**LS + RS**), to open the settings overlay. Works with a controller, mouse or keyboard. Most settings apply without a restart, and a graphics preset (Console, Performance or Enhanced) sets up the graphics in one go. |
| **Display** | Any resolution and aspect ratio (even in portrait), with the menus, cutscenes and gameplay widened. Render scale and FSR 3.1 upscaling. HDR output. Adjustable HUD position. |
| **Gameplay** | Extended FOV, coloured HUD and colourblind options. |
| **High frame rates** | Menus, cutscenes and gameplay run at your display's refresh rate, or uncapped, instead of the console's 30 FPS. |
| **Image quality** | MSAA override<br>SMAA, TAA, and AMD FSR 3.1<br>Anisotropic filtering<br>10-bit output and debanding |
| **Optional effects** | Planar reflections on water<br>Enhanced bloom<br>Ambient occlusion (XeGTAO)<br>Object motion blur<br>Film grain, vignette and chromatic aberration<br>Sharper character shadows in cutscenes |
| **Controls** | DualSense adaptive triggers, similar to Ace Combat 8<br>Flight sticks and HOTAS: any device's axes and buttons bound to the game's controls, each axis with its own curve and dead zone<br>Keyboard and mouse controls in the style of Ace Combat 7, with mouse steering |
| **Languages** | English and Japanese sub and dub<br>German, French, Spanish, and Italian sub (experimental) |
| **Mod support** | Mods go in a `mods` folder next to the game<br>**Textures:** replace any texture with your own, at any resolution. The overlay's texture browser can display and dump any texture that is shown in the frame.<br>**Game files:** replace any of the game's data files.<br>**Lua mods:** scripts that hook any of the game's functions, read and write its memory, call it, read the controllers, keyboard and mouse, and draw over the picture ([guide](docs/modding/lua.md), [examples](examples/mods))<br>**Plugins:** native code that hooks any of the game's functions, for example the [PSM mod](https://github.com/Dipshet/GracemeriaRecomp) |
| **Self Update** | The game checks for a newer version at start and installs it from the overlay's Game tab. |

## Controls

Controllers work with no setup. **Keyboard and mouse controls are off by default:** turn them on in the overlay's
Controls tab, where every binding can be changed. The default mouse and keyboard control scheme follows Ace Combat 7.

### Flight

| Action | Keys | | Action | Keys |
|-|-|-|-|-|
| Pitch down / up | `1` / `3` | | Fire machine gun | `Mouse1`, `Left Ctrl` |
| Roll left / right | `A` / `D` | | Fire missile | `Mouse2`, `Space` |
| Yaw left / right | `Q` / `E` | | Change weapon | `Mouse wheel`, `C` |
| Accelerate | `W` | | Switch targets | `Tab` |
| Decelerate | `S` | | Change view | `V` |
| High-G turn | `2` | | Radar / map | `R` |
| Autopilot | `Z`, `X` | | Landing gear | `G` |
| Pause | `Escape` | | Wingman / map filter | Arrow keys |

The **mouse** steers the aircraft. Hold `Left Alt` to look around with the mouse instead. **Camera:** `Numpad 8` / `2`
/ `4` / `6`.

### Menus

| Action | Keys |
|-|-|
| Move | `W` `A` `S` `D` or arrow keys |
| Confirm | `Space`, `Mouse1` |
| Cancel | `Escape`, `Backspace`, `Mouse2` |
| Start | `Enter` |
| Back | `Tab` |
| X / Y | `R` / `F` |
| Previous / next page (LB / RB) | `Q` / `E` |

### Hotkeys

| Key | Action |
|-|-|
| F4 | Settings overlay (also **LS + RS** on a controller) |
| F5 | Reload the texture replacements from their files |
| F11 | (Debug) Save the last few seconds as pictures |
| F12 | (Debug) Save a replayable capture of the last few seconds |

## Known Issues

Before reporting a problem, check the [issues](https://github.com/Dipshet/GracemeriaRecomp/issues) and the [Discord](https://discord.gg/9yjxBkHmZT) server. When reporting an issue, attach `ac6_runtime.log` from the game's folder and how to replicate it.

**Crashes, freezes or wrong drawing with your graphics driver:** the settings overlay's (F4) Debug tab, under Graphics
driver, has Pipeline libraries and Dynamic rendering to turn off (try them one at a time), and GPU debug mode: much
slower, but the log then names the draw where the graphics card fails. Reproduce the problem with it on and attach
`ac6_runtime.log`.

- **Only the North American release** is supported.
- **European languages are experimental:** German, French, Spanish and Italian use older text on the North American
  disc that the game was never released or tested with. Some text is in the wrong language, button glyphs are drawn
  over text, the HUD has no accented letters, and the game may crash or misbehave.
  If something goes wrong, switch the language to English and see whether it happens again before reporting it.
- **Old AMD drivers (before 2022):** in the cockpit view, the canopy can show through the instrument panel. Turn
  Dynamic rendering off (Debug tab, Graphics driver).
- **DualSense adaptive triggers on Linux** may need access to the controller's hidraw device (Steam's or
  game-devices-udev's udev rules).
- **Online play** is not available, yet.

## FAQ

### Where are my saves and settings?

Next to the game: saves in `saves`, settings in `ac6_settings.toml` (written by the overlay), the log in
`ac6_runtime.log`.

### How do I update?

The game checks for updates at start. Install one from the overlay's Game tab (*Updates*). Your saves, settings and
mods are kept.

### Can I use the European or Japanese release?

No. Only the North American (NTSC-U) release is supported. Japanese, German, French, Spanish, and Italian language can be enabled in the settings (the European ones are experimental, see Known Issues).

### How do I transfer my old save files from AC6Recomp?

Run the game once first and create a save, then close the game. Navigate to `Documents\ac6recomp\B13EBABEBABEBABE\4E4D07D1\00000001` and copy the `sav_acecombat6` folder. Then, paste it inside the `saves\E0000AC600000001\00000001` folder.

## Building

Put the game's `default.xex`, taken from your disc image, in
the repository's top folder.

### Windows

Needs [Git](https://git-scm.com/) and [CMake](https://cmake.org/) 3.20 or newer, on the PATH. The first build
downloads the rest into `toolchain\` (about 225 MB):
[llvm-mingw](https://github.com/mstorsjo/llvm-mingw),
[Ninja](https://github.com/ninja-build/ninja) and [DXC](https://github.com/microsoft/DirectXShaderCompiler).

```bat
build.bat
```

The first build checks out SDL and Dear ImGui, fetches and patches the FidelityFX SDK, builds XenonRecomp, recompiles
the game's code from your `default.xex` and compiles everything. The result is `build\AceCombat6.exe`,
and `build\ac6.exe`, the same program with a console. Later builds only rebuild what changed.

`build_compat.bat` then builds the x86-64-v2 compatibility executables into `build-compat`

### Linux

`build_linux.sh [build folder]` builds with the system's clang. It needs Git, clang, lld, CMake, Ninja, curl and the
headers SDL3 builds against (X11, Wayland, audio). It downloads DXC into `toolchain/dxc-linux` unless `dxc` is on the
PATH or in `AC6_DXC`. Such a build needs your system's glibc or newer. For one that runs on any distro with glibc
2.31 or newer, the Steam Deck included, build against Valve's Steam Runtime SDK: `tools/linux/fetch_sniper.sh`
downloads it into `toolchain/sniper` (1.2 GB) and prints its path, then
`AC6_SNIPER_SYSROOT=<that path> ./build_linux.sh`.

### Layout

| Folder | |
|-|-|
| `GracemeriaRecomp` | The runtime: the stand-in for the console (kernel, memory, threads, files, audio, input), the Vulkan renderer, the settings overlay, mods and language packs |
| `GracemeriaRecompLib` | The recompiled game code: XenonRecomp's config (`config`) and the C++ it writes (`ppc`, made by the build) |
| `tools` | XenonRecomp (adapted for this port), XexDump, the build's CMake scripts, the patches to the FidelityFX SDK and FFmpeg, an IDA script that exports symbols for mods |
| `docs` | The modding guide |
| `examples` | Example Lua mods, copied into `build/example_mods` by the build |
| `thirdparty` | The libraries the runtime uses (`thirdparty/README.md` lists versions, licences and changes) |
| `toolchain` | The toolchain's download script and compiler settings |

## Credits

- [XenonRecomp](https://github.com/hedge-dev/XenonRecomp), the recompiler, and
  [Unleashed Recompiled](https://github.com/hedge-dev/UnleashedRecomp), whose runtime this port adapts in places
- [XenosRecomp](https://github.com/hedge-dev/XenosRecomp), whose shader decoding the port's shader translator is
  built on
- [plume](https://github.com/renderbag/plume), the rendering interface the renderer is built on
- [SDL](https://github.com/libsdl-org/SDL), [FFmpeg](https://ffmpeg.org) (XMA and WMA Pro audio),
  [Dear ImGui](https://github.com/ocornut/imgui) (the overlay), [AMD FidelityFX](https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK)
  (FSR 3.1), [XeGTAO](https://github.com/GameTechDev/XeGTAO), [SMAA](https://github.com/iryoku/smaa),
  [RenoDX](https://github.com/clshortfuse/renodx) (HDR tone mappers), [LuaJIT](https://luajit.org) (Lua mods),
  [o1heap](https://github.com/pavel-kirienko/o1heap),
  [xxHash](https://github.com/Cyan4973/xxHash) and the [Share Tech](https://fonts.google.com/specimen/Share+Tech) fonts
- [Ostrich](https://ostrichmods.com/), the game's icon and the PlayStation button glyphs

## Licence

GPL-3.0 (see [LICENSE](LICENSE)). Third-party code keeps its own licences: see `thirdparty/README.md`.

Ace Combat is a trademark of Bandai Namco Entertainment. This project is not affiliated with or endorsed by Bandai
Namco Entertainment, Project Aces or Microsoft.
