#pragma once
// The game's data packages.
//
// DATA.TBL lists 926 packages stored in DATA00.PAC / DATA01.PAC, each XORed with bytes of pi's binary fraction and then
// deflated or stored. The extractor unpacks them natively (game/packages.cpp, game/inflate.cpp):
// AceCombat6.exe --extract-packages [--items] [folder] [first] [count] writes each package as <folder>/<index>.bin
// (default folder: dump/packages next to the executable), without starting the game, --items also writes its items,
// <folder>/<index>/<item>.<kind> (an FHM inside: a folder of its own items), the names mods use.
//
// A package is an FHM: "FHM " 01 01 00 10, eight zero bytes, then at 0x10 the item count n and four tables of n
// big-endian words: the items' offsets, sizes, and two more (kept as they are).

#include <cstdint>
#include <filesystem>
#include <vector>

namespace ac6::game {

// Runs on the calling thread (made a guest thread). Returns the process exit code.
int ExtractPackages(const std::filesystem::path& folder, int first, int count, bool items);

// Mods' packages: mods/<mod>/packages/<index>.bin (a whole package) or <index>/<item>.<kind> (its items, nested FHMs
// as folders). Built before the game starts (not with AC6_PACKAGE_MODS=0, the setting modding.package_mods) and handed
// to the game when it unpacks them.
void LoadPackageMods();
// Package `index` from the disc, unpacked natively into `out` (any thread).
bool UnpackPackage(uint32_t index, std::vector<uint8_t>& out);
// DATA.TBL as the game reads it (NtReadFile): the sizes of modded packages that grew raised, so the game makes room.
void PatchPackageTable(uint8_t* data, uint64_t offset, uint64_t length);

} // namespace ac6::game
