#pragma once
// Downloadable content.
//
// Packages are found in the "dlc" folders next to the disc image and next to the executable (or in AC6_DLC_DIR
// alone), in any layout: raw STFS containers (CON/LIVE/PIRS, recognised by their header) and extracted package
// folders (named after the content, 40-42 hex digits, or under <title id>/00000002/ as Xenia lays them out).
// Every package is checked before the game sees it, and one that would not load is listed with the reason
// instead of being handed over: the container's structure, the files of known packages against the sizes of
// known-good dumps (dlc_manifest.inc), unreadable, empty and zero-filled files, archives that weren't
// extracted, packages for another title, title updates. Damaged contents (blocks failing the container's
// hash tree, a file of the right size whose SHA-1 differs from the known-good one) are reported but don't
// stop it: the package is mounted (PACK08's common dump has one damaged skin, the rest works). The report
// goes to the log and to dlc_report.txt beside it. AC6_DLC=0 hides all DLC, AC6_DLC_VERIFY=0 mounts
// packages whose contents fail the checks anyway.
//
// The game enumerates marketplace content (type 2) after the save loads, mounts each package read-only
// (XamContentCreateEx) and lists its files. Licences aren't checked by the game, and every licence bit
// is reported as held.

#include "fs/vfs.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace ac6::fs {

struct DlcPackage
{
    std::string name;              // the content file name the game is given (at most 42 characters)
    std::string displayName;       // UTF-8
    std::string label;             // the known package's label (colour packs: PACK00-PACK10), if any
    std::filesystem::path path;    // the container file or the package folder
    bool container = false;
    bool known = false;            // in the known-good list, so its files were verified
    uint32_t files = 0;
    uint64_t bytes = 0;
    std::vector<std::string> errors;     // why it isn't mounted (empty: it is)
    std::vector<std::string> damage;     // damaged contents: mounted anyway, reported
    std::vector<std::string> warnings;
    std::shared_ptr<Device> device;      // set when it can be mounted

    bool Usable() const { return device != nullptr; }
};

// Starts the scan on a worker thread (the disc's title id decides which containers belong to the game).
void StartDlcScan(uint32_t titleId);
// Every package found, usable or not, sorted by name (waits for the scan).
const std::vector<DlcPackage>& DlcPackages();
// Whether the scan is over (DlcPackages won't wait), true when none was started.
bool DlcScanFinished();
// A package by the name the game was given (case-insensitive), or null.
const DlcPackage* FindDlc(std::string_view name);

// The folder the scan reads first (the overlay's Open DLC folder): AC6_DLC_DIR, else "dlc" next to the
// disc image, else next to the executable.
std::filesystem::path DlcFolder();

} // namespace ac6::fs
