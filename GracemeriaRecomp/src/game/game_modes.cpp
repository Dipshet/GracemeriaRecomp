// The game's modes: the front end and the mission are a chain of mode tasks (CModeTask*: title, main menu, briefing,
// aircraft select, loading, game, debriefing, ...). A mode picks its successor from the transition table at 0x82691A78
// (function pointers, by mode and outcome) and stores it in the mode manager ([0x8293B930] +16), which then calls it:
// each entry is a 20-instruction factory that allocates the next mode task and constructs it. The factories are only
// called through pointers, so their function-table entries are wrapped here: every mode change is logged with its
// frame, and the current mode is known (game::CurrentMode, for diagnostics and the scripted input of input/input.cpp).
#include "game/game_modes.h"

#include "crash.h"
#include "endian.h"
#include "gpu/renderer.h"
#include "log.h"
#include "memory.h"
#include "ppc.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <format>
#include <string>
#include <string_view>
#include <utility>

namespace ac6::game {

namespace {

struct Mode
{
    uint32_t factory;
    const char* name;
};

// The factories (0x821BB5F0-0x821BCD30, from the table at 0x82691A78), named by the class their constructor installs (its RTTI).
constexpr Mode kModes[] = {
    { 0x821BB5F0, "CModeTaskTitleMovie" },
    { 0x821BB640, "CSelectSaveLoadManager" },
    { 0x821BB690, "CModeTaskTitle" },
    { 0x821BB6E0, "CModeTaskCheckDLContentsMainSelect" },
    { 0x821BB730, "CModeTaskMarketPlace" },
    { 0x821BB780, "CModeTaskMainSelect" },
    { 0x821BB7D0, "CModeTaskCheckDLContentsInvited" },
    { 0x821BB820, "CModeTaskInvitationSessionSearch" },
    { 0x821BB870, "CModeTaskDemoOpening" },
    { 0x821BB8C0, "CModeTaskMissionTitle" },
    { 0x821BB910, "CModeTaskFreeMissionSelect" },
    { 0x821BB960, "CModeTaskTutorialSelect" },
    { 0x821BB9B0, "CModeTaskOptionSelect" },
    { 0x821BBA00, "CSelectSaveLoadManager" },
    { 0x821BBA50, "CModeTaskCreditMainSelect" },
    { 0x821BBAA0, "CModeTaskGallerySelect" },
    { 0x821BBAF0, "CModeTaskQuickSessionSearch" },
    { 0x821BBB40, "CModeTaskCustomSessionSearch" },
    { 0x821BBB90, "CModeTaskCreateSession" },
    { 0x821BBBE0, "CModeTaskRanking" },
    { 0x821BBC38, "CModeTaskBriefing" },
    { 0x821BBC90, "CModeTaskMissionSelect" },
    { 0x821BBCE0, "CModeTaskTutorialSelect_CampaignMission" },
    { 0x821BBD30, "CModeTaskAircraftSelect" },
    { 0x821BBD88, "CModeTaskHangar" },
    { 0x821BBDE0, "CModeTaskTutorialSelect_FromMissionMenu" },
    { 0x821BBE30, "CModeTaskGallerySelectMission" },
    { 0x821BBE80, "CModeTaskOptionSelectMission" },
    { 0x821BBED0, "CModeTaskGameDataSelectMission" },
    { 0x821BBF20, "CModeTaskLoading" },
    { 0x821BBF70, "CModeTaskCheckDLContentsSelectAircraft" },
    { 0x821BBFC0, "CModeTaskGame" },
    { 0x821BC010, "CModeTaskDebriefing" },
    { 0x821BC060, "CModeTaskDemoEnding" },
    { 0x821BC0B0, "CModeTaskDemoIntermission" },
    { 0x821BC100, "CModeTaskUnlock" },
    { 0x821BC150, "CSelectSaveLoadManager" },
    { 0x821BC1A0, "CModeTaskCredit" },
    { 0x821BC1F0, "CModeTaskBattleStatisticsClear" },
    { 0x821BC240, "CModeTaskGalleryCampaignResultMission" },
    { 0x821BC290, "CModeTaskBattleStatisticsMission" },
    { 0x821BC2E0, "CModeTaskGalleryMedalMission" },
    { 0x821BC330, "CModeTaskGalleryAssaultRecordMission" },
    { 0x821BC380, "CModeTaskGalleryReplayViewerMission" },
    { 0x821BC3D0, "CModeTaskGallerySceneViewerMission" },
    { 0x821BC420, "CModeTaskGalleryMusicPlayerMission" },
    { 0x821BC470, "CModeTaskLoadingReplayMission" },
    { 0x821BC4C0, "CModeTaskCheckDLContentsReplayViewerMission" },
    { 0x821BC510, "CModeTaskDemoSceneViewerMission" },
    { 0x821BC560, "CModeTaskGameReplayMission" },
    { 0x821BC5B0, "CModeTaskCheckDLContentsFreeMission" },
    { 0x821BC600, "CModeTaskLoadingTutorialMissionFromMain" },
    { 0x821BC650, "CModeTaskLoadingTutorialMissionFromMission" },
    { 0x821BC6A0, "CModeTaskLoadingTutorialMissionFromCampaign" },
    { 0x821BC6F0, "CModeTaskGameTutorial" },
    { 0x821BC740, "CModeTaskGameTutorial_FromMissionMenu" },
    { 0x821BC790, "CModeTaskGameTutorial_CampaignMission" },
    { 0x821BC7E0, "CModeTaskLobbyClientQuick" },
    { 0x821BC830, "CModeTaskLobbyClientCustom" },
    { 0x821BC880, "CModeTaskLobbyHost" },
    { 0x821BC8D0, "CModeTaskLoadingOnlineMission" },
    { 0x821BC920, "CModeTaskGameOnline" },
    { 0x821BC970, "CModeTaskDebriefingOnline" },
    { 0x821BC9C0, "CModeTaskGalleryCampaignResult" },
    { 0x821BCA10, "CModeTaskBattleStatistics" },
    { 0x821BCA60, "CModeTaskGalleryMedal" },
    { 0x821BCAB0, "CModeTaskGalleryAssaultRecord" },
    { 0x821BCB00, "CModeTaskGalleryReplayViewer" },
    { 0x821BCB50, "CModeTaskGallerySceneViewer" },
    { 0x821BCBA0, "CModeTaskGalleryMusicPlayer" },
    { 0x821BCBF0, "CModeTaskLoadingReplay" },
    { 0x821BCC40, "CModeTaskCheckDLContentsReplayViewer" },
    { 0x821BCC90, "CModeTaskDemoSceneViewer" },
    { 0x821BCCE0, "CModeTaskGameReplay" },
    { 0x821BCD30, "CModeTaskFirstLoadInvited" },
};
constexpr size_t kModeCount = std::size(kModes);

std::array<PPCFunc*, kModeCount> s_factory{};
std::array<int, kModeCount> s_jump{};   // AC6_MODE_JUMP: the mode built in this one's place, or -1
std::atomic<int> s_current{ -1 };
std::atomic<uint64_t> s_enteredFrame{ 0 };
const auto s_start = std::chrono::steady_clock::now();

// A mode by its class name, with or without "CModeTask" (or the leading "C"), or by its factory's address.
int FindMode(std::string_view name)
{
    if (name.starts_with("0x") || name.starts_with("0X"))
    {
        const uint32_t factory = uint32_t(std::strtoul(std::string(name).c_str(), nullptr, 16));
        for (size_t i = 0; i < kModeCount; i++)
            if (kModes[i].factory == factory)
                return int(i);
        return -1;
    }
    for (size_t i = 0; i < kModeCount; i++)
    {
        const std::string_view n = kModes[i].name;
        if (n == name || (n.starts_with("CModeTask") && n.substr(9) == name) || n.substr(1) == name)
            return int(i);
    }
    return -1;
}

// AC6_MODE_JUMP=<mode>=<mode>[,...]: when the game asks for the first mode, the second is built instead (for testing,
// e.g. MainSelect=Loading).
void ParseJumps()
{
    s_jump.fill(-1);
    const char* v = std::getenv("AC6_MODE_JUMP");
    if (!v)
        return;
    std::string_view list = v;
    while (!list.empty())
    {
        const size_t comma = list.find(',');
        const std::string_view item = list.substr(0, comma);
        list = comma == std::string_view::npos ? std::string_view{} : list.substr(comma + 1);
        const size_t eq = item.find('=');
        const int from = eq == std::string_view::npos ? -1 : FindMode(item.substr(0, eq));
        const int to = eq == std::string_view::npos ? -1 : FindMode(item.substr(eq + 1));
        if (from < 0 || to < 0)
        {
            log::Warn("AC6_MODE_JUMP: unknown mode in '{}'", item);
            continue;
        }
        s_jump[size_t(from)] = to;
        log::Info("Mode jump: {} -> {}", kModes[from].name, kModes[to].name);
    }
}

template<size_t I> void Factory(PPCContext& __restrict ctx, uint8_t* base)
{
    const int built = s_jump[I] >= 0 ? s_jump[I] : int(I);
    // AC6_MODE_WORD=<n> (with a jump): the game mode word the main menu sets (singleton +120: 1 campaign, 2 free mission).
    static const char* modeWord = std::getenv("AC6_MODE_WORD");
    if (built != int(I) && modeWord)
        *g_memory.As<be<uint32_t>>(*g_memory.As<be<uint32_t>>(0x826E4E54) + 120) = uint32_t(std::atoi(modeWord));
    s_factory[size_t(built)](ctx, base);
    // The mode manager (0x821BA5D0) builds each new mode twice: a probe instance it destroys at once (it reads whether
    // the mode needs the heaps reset, +16 == 2), then a few frames later the real one. Logged once.
    const int previous = s_current.exchange(built);
    const uint64_t frame = gpu::RendererFrame(), last = s_enteredFrame.exchange(frame);
    if (previous == built && frame - last < 60)
        return;
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - s_start).count();
    log::Info("Mode: {} (frame {}, {:.1f} s; from {}){}", kModes[built].name, frame, seconds,
        previous >= 0 ? kModes[previous].name : "boot", built != int(I) ? std::format(" in place of {}", kModes[I].name) : "");
    // AC6_MODE_TRACE=1: also the guest call chain that made the change (finding the code that drives the modes).
    static const bool trace = [] { const char* v = std::getenv("AC6_MODE_TRACE"); return v && *v == '1'; }();
    if (trace)
        log::Info("  mode change stack:{}", crash::HostStack());
}

template<size_t... I> void Wrap(std::index_sequence<I...>)
{
    ((s_factory[I] = g_memory.Function(kModes[I].factory), g_memory.Function(kModes[I].factory) = &Factory<I>), ...);
}

} // namespace

void InstallModeHooks()
{
    ParseJumps();
    Wrap(std::make_index_sequence<kModeCount>());
}

const char* CurrentMode()
{
    const int i = s_current.load();
    return i >= 0 ? kModes[i].name : "";
}

uint64_t ModeEnteredFrame()
{
    return s_enteredFrame.load();
}

bool ModeIs(std::string_view name)
{
    const int i = s_current.load();
    return i >= 0 && FindMode(name) >= 0 && std::string_view(kModes[i].name) == kModes[FindMode(name)].name;
}

bool FastBoot()
{
    // The logos and the "PROJECT ACES" screens (~1500 frames, 25 s) are frame-timed and skip nothing on Start. From the
    // first mode on (the title movie, which has a soundtrack, and the save dialog after it) the game follows real time,
    // so it runs normally.
    static const bool enabled = [] { const char* v = std::getenv("AC6_FAST_BOOT"); return v && *v == '1'; }();
    return enabled && s_current.load() < 0;
}

} // namespace ac6::game
