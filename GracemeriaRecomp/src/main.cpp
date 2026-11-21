// AC6 native runtime entry point: reserve guest memory, set up heaps, mount the disc image, load
// default.xex (from the image unless one is given explicitly), and run the game's entry point on the
// host main thread.
#include "apu/xma.h"
#include "profiler.h"
#include "cpu/guest_thread.h"
#include "crash.h"
#include "fs/dlc.h"
#include "fs/vfs.h"
#include "game/game_modes.h"
#include "game/packages.h"
#include "plugin/plugins.h"
#include "mods/mods.h"
#include "gpu/capture.h"
#include "gpu/renderer.h"
#include "gpu/video.h"
#include "heap.h"
#include "kernel/clock.h"
#include "loader.h"
#include "log.h"
#include "memory.h"
#include "platform/platform.h"
#include "settings.h"
#include "system_info.h"
#include "ui/overlay.h"
#include "update/updater.h"

#include <cstdlib>
#include <filesystem>
#include <string_view>

#if defined(_WIN32)
#include <windows.h>
#else
#include <pthread.h>
#endif

using namespace ac6;

namespace {

std::filesystem::path ExecutableDir()
{
    return platform::ExecutableDirectory();
}

// An explicitly given loose XEX (argument or AC6_XEX), for development, empty otherwise.
std::filesystem::path ExplicitXex(int argc, char** argv)
{
    // (Other arguments, a launcher's "-fullscreen" or an .iso dropped on the executable, are ignored.)
    if (argc > 1 && !std::string_view(argv[1]).starts_with("--"))
    {
        const std::filesystem::path given(argv[1]);
        std::string ext = platform::PathUtf8(given.extension());
        for (char& c : ext)
            c = char(std::tolower(uint8_t(c)));
        std::error_code ec;
        if (ext == ".xex" && std::filesystem::is_regular_file(given, ec))
            return given;
        log::Warn("Argument '{}' ignored (not a .xex file)", argv[1]);
    }
    if (const char* env = std::getenv("AC6_XEX"); env && *env)
        return env;
    return {};
}

std::filesystem::path FindLooseXex()
{
    for (auto dir = ExecutableDir(); !dir.empty() && dir != dir.root_path(); dir = dir.parent_path())
        if (std::filesystem::exists(dir / "default.xex"))
            return dir / "default.xex";
    return {};
}

// The entry point a default.xex declares (XEX2 optional header 0x00010100), 0 if unreadable.
uint32_t XexEntryPoint(const std::vector<uint8_t>& xex)
{
    auto be32 = [&](size_t at) { return at + 4 <= xex.size() ? uint32_t(xex[at]) << 24 | xex[at + 1] << 16 | xex[at + 2] << 8 | xex[at + 3] : 0u; };
    if (be32(0) != 0x58455832)   // 'XEX2'
        return 0;
    const uint32_t count = be32(0x14);
    for (uint32_t i = 0; i < count && i < 64; i++)
        if (be32(0x18 + i * 8) == 0x00010100)
            return be32(0x18 + i * 8 + 4);
    return 0;
}

// Whether this executable was built from that game executable: its entry point is one of the recompiled functions.
bool IsRecompiledEntry(uint32_t entry)
{
    for (size_t i = 0; PPCFuncMappings[i].guest != 0; i++)
        if (PPCFuncMappings[i].guest == entry && PPCFuncMappings[i].host)
            return true;
    return false;
}

// The disc image mounted as game:, d: and \Device\Cdrom0: the first *.iso next to the executable or above it whose
// default.xex is the build this executable was recompiled from. Other releases (such as the European one, whose
// default.xex is version 9) are skipped: their code isn't in this executable.
std::shared_ptr<fs::Device> MountDisc()
{
    const auto isos = fs::FindIsos();
    if (isos.empty())
    {
        log::Warn("No disc image (*.iso) found next to the executable or in a parent folder");
        return nullptr;
    }
    std::shared_ptr<fs::Device> device;
    bool unreadable = false;
    for (const auto& iso : isos)
    {
        auto candidate = fs::OpenIso(iso);
        if (!candidate)
        {
            unreadable = true;
            continue;
        }
        uint32_t entry = 0;
        if (const fs::Entry* e = candidate->Resolve("default.xex"); e && !e->directory)
        {
            std::vector<uint8_t> head(size_t(std::min<uint64_t>(e->size, 0x1000)));
            candidate->Read(*e, 0, head.data(), head.size());
            entry = XexEntryPoint(head);
        }
        if (IsRecompiledEntry(entry))
        {
            fs::SetIso(iso);
            device = std::move(candidate);
            break;
        }
        log::Warn("Skipping {}: its default.xex (entry point {:08X}) is another release than this executable was built from",
            platform::PathUtf8(iso.filename()), entry);
    }
    if (!device)
        crash::Shutdown(1, unreadable ? "the disc image can't be read: it's incomplete, damaged or not an Xbox 360 image (the log says which)"
                                      : "this release of the game isn't supported (the disc image is another version)");
    fs::Mount("game", device);
    fs::Mount("d", device);
    fs::Mount("\\Device\\Cdrom0", device);
    return device;
}

bool LoadGame(int argc, char** argv, const std::shared_ptr<fs::Device>& disc)
{
    if (auto xex = ExplicitXex(argc, argv); !xex.empty())
        return LoadXex(xex, g_module);
    if (disc)
    {
        if (const fs::Entry* e = disc->Resolve("default.xex"); e && !e->directory)
        {
            std::vector<uint8_t> file(e->size);
            disc->Read(*e, 0, file.data(), file.size());
            return LoadXex(std::move(file), "default.xex (disc)", g_module);
        }
        log::Error("The disc image has no default.xex");
        return false;
    }
    auto loose = FindLooseXex();
    if (loose.empty())
    {
        log::Error("Nothing to run: put the game's .iso next to the executable");
        return false;
    }
    return LoadXex(loose, g_module);
}

#if defined(_WIN32)
#define AC6_PLATFORM_NAME "Windows"
#else
#define AC6_PLATFORM_NAME "Linux"
#endif

int Run(int argc, char** argv)
{
    // Started again by the settings overlay: the previous copy ends first, and its log is kept (ac6_runtime.previous.log:
    // what led to the restart).
    if (platform::WaitForRestart())
    {
        std::error_code ec;
        std::filesystem::rename(ExecutableDir() / "ac6_runtime.log", ExecutableDir() / "ac6_runtime.previous.log", ec);
    }
    // The settings file first: it sets the AC6_* variables everything below reads.
    settings::Load(ExecutableDir());
    log::Init(ExecutableDir() / "ac6_runtime.log");
#if defined(_WIN32)
    SetConsoleOutputCP(CP_UTF8);
#endif
    // Which build runs: the x86-64-v2 compatibility one is started by the main one on older CPUs.
#if defined(AC6_X86_V2)
    log::Info("AC6 runtime {} ({} build, x86-64-v2: the compatibility version)", update::Version(), AC6_PLATFORM_NAME);
#else
    log::Info("AC6 runtime {} ({} build, x86-64-v3)", update::Version(), AC6_PLATFORM_NAME);
#endif
#ifdef AC6_WINDOWED_APP
    crash::ShowFatalErrors(true);
    gpu::g_video.framePacing = true;   // the game's own present interval: 30 FPS missions
#endif

    if (!g_memory.Init() || !g_heap.Init())
        crash::Shutdown(1, "couldn't reserve the game's memory");
    // OS, CPU, RAM, GPUs and their drivers: after the guest memory, whose preferred address the
    // graphics driver's DLLs (loaded by the GPU query) could otherwise take.
    system_info::Log();
    settings::Log();

    // ac6.exe --replay <capture> [--repeat N]: re-render a frame capture (gpu/capture.h) without the game.
    if (argc > 2 && std::string_view(argv[1]) == "--replay")
    {
        crash::Install();
        gpu::g_video.background = true;
        gpu::g_video.nativeByDefault = false;   // (1280x720, not the desktop's size: replays stay pixel-identical)
        ui::SuppressStartupHint();              // (no overlay hint over the replayed frames)
        if (!gpu::g_video.Init() || !gpu::InitRenderer())
            crash::Shutdown(1, "no Vulkan device for the replay");
        const uint32_t repeat = argc > 4 && std::string_view(argv[3]) == "--repeat" ? uint32_t(std::atoi(argv[4])) : 1;
        const int code = gpu::capture::Replay(argv[2], repeat);
        crash::Shutdown(code, "replay finished");
    }

    // GPU register space: the D3D library reads a handful of registers directly (PPC_MM_*).
    g_memory.AddBackedRange(kGpuRegsBegin, kGpuRegsEnd);
    apu::XmaInit();   // publishes the XMA context array in the register block

    crash::Install();
    profiler::Start();
    crash::StartWatchdog();
    crash::StartHardwareWatch();
    kernel::InitClock();

    // AceCombat6.exe --extract-packages [--items] [folder] [first] [count]: the disc's data packages unpacked with the game's own
    // decompressor, without starting the game (game/packages.h).
    const bool extract = argc > 1 && std::string_view(argv[1]) == "--extract-packages";
    if (!extract)
        mods::LoadEarly();      // Lua mods (mods/mods.h): found, their early.lua run before the window and the device
    if (!extract && gpu::g_video.Init())   // without a usable Vulkan device the game still runs (headless)
        gpu::InitRenderer();

    auto disc = MountDisc();
    if (!LoadGame(argc, argv, disc))
        crash::Shutdown(1, "the game image was not found: put the game's .iso next to the executable");
    kernel::InitVariableImports();
    if (extract)
    {
        std::vector<std::string> rest;
        bool items = false;
        for (int i = 2; i < argc; i++)
            if (std::string_view(argv[i]) == "--items")
                items = true;
            else
                rest.push_back(argv[i]);
        const std::filesystem::path folder = rest.size() > 0 ? std::filesystem::path(rest[0]) : ExecutableDir() / "dump" / "packages";
        const int code = game::ExtractPackages(folder, rest.size() > 1 ? std::atoi(rest[1].c_str()) : 0,
            rest.size() > 2 ? std::atoi(rest[2].c_str()) : -1, items);
        crash::Shutdown(code, "packages extracted");
    }
    game::LoadPackageMods();    // mods' packages built before the game reads DATA.TBL (game/packages.h)
    plugin::Load();             // mods' plugins (plugin/ac6_plugin.h), before the game's code runs
    mods::LoadInit();           // Lua mods' init.lua, before the game's code runs
    game::InstallModeHooks();   // logs the game's mode changes
    fs::StartDlcScan(g_module.titleId);   // checked while the game boots (the game lists DLC after the save loads)
    // Updates (release downloads only): checked in the background, not in scripted runs (auto-start macros, background
    // test windows).
    update::Start(!std::getenv("AC6_AUTO_START") && !std::getenv("AC6_BACKGROUND"));

    log::Info("Running entry point {:08X} on the main thread", g_module.entryPoint);
    uint32_t result = GuestThread::RunOnCurrentThread({ g_module.entryPoint });
    log::Info("Entry point returned {:08X}", result);
    crash::Shutdown(0, "entry point returned");
}

} // namespace

int main(int argc, char** argv)
{
#if defined(_WIN32)
    return Run(argc, argv);   // (16 MB stacks come from the -Wl,--stack link flag in CMakeLists.txt)
#else
    // Recompiled code nests deeply on the host stack: every thread gets 16 MB (Linux's default is the 8 MB of the
    // stack limit), and the program runs on such a thread rather than the main one, whose stack is the limit's.
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setstacksize(&attr, 16 * 1024 * 1024);
    pthread_setattr_default_np(&attr);
    struct Args { int argc; char** argv; } args{ argc, argv };
    pthread_t thread;
    if (pthread_create(&thread, &attr, [](void* p) -> void* {
            auto* a = static_cast<Args*>(p);
            Run(a->argc, a->argv);
            return nullptr;
        }, &args) != 0)
        return 1;
    pthread_join(thread, nullptr);   // (Run ends the process itself)
    return 0;
#endif
}
