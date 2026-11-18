#pragma once
// Updates from the project's GitHub releases. Only a release build updates itself: one compiled with AC6_RELEASE (the
// CMake option, which the release archives' builds have). Any other build never checks or replaces anything.
//
// The check is one request to GitHub's releases API, in the background. Installing downloads the platform's archive,
// checks it against the SHA-256 GitHub publishes for it, unpacks it, and swaps the files in next to the executable
// (the replaced ones are kept as <name>.old until the next start). The new files run from the next start: the
// settings overlay offers to restart the game.
// Settings, saves, mods and caches are not in the archive and are never touched.
#include <string>

namespace ac6::update {

enum class State
{
    Off,           // not a release install (built from source): no updates
    Idle,          // a release install, not checked yet
    Checking,
    UpToDate,
    Available,     // a newer release, not installed yet
    Downloading,
    Installing,
    Installed,     // the new files are in place: they run after a restart
    Failed,
};

struct Status
{
    State state = State::Off;
    std::string latest;     // the newest release's version, once checked
    std::string error;      // what went wrong (Failed)
    float progress = -1.0f; // downloading: 0 to 1, or -1 when the size isn't known
};

// This build's version (the project version in CMakeLists.txt).
const char* Version();
// At the start of a session: removes the files the last update replaced, and checks for a release in the background
// when this is a release install, the check_updates setting is on, and `automatic` (not a replay or a scripted run).
void Start(bool automatic);
// A check again, in the background (not while one runs or while installing).
void Check();
// Downloads and installs the newest release, in the background (State::Available only).
void Install();
Status Get();

} // namespace ac6::update
