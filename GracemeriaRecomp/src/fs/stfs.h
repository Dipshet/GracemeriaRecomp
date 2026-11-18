#pragma once
// Xbox 360 content packages (STFS: CON/LIVE/PIRS containers), read-only, with their hash tree verified.
// Written from the format's public description. See stfs.cpp for the layout.

#include "fs/vfs.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace ac6::fs {

struct StfsHeader
{
    uint32_t magic = 0;            // 'CON ', 'LIVE' or 'PIRS'
    uint32_t headerSize = 0;
    uint32_t contentType = 0;      // 2 = marketplace content, 0xB0000 = title update, 1 = saved game...
    uint32_t titleId = 0;
    uint32_t descriptorType = 0;   // 0 = STFS, 1 = SVOD
    uint32_t licenseBits = 0;      // OR of the licence entries in use
    uint32_t licenseEntries = 0;   // how many are in use
    std::array<uint8_t, 20> contentId{};   // the header's SHA-1
    std::string displayName;       // UTF-8 (English)
};

// Reads the header of `path` if it is a container. False (and `error` empty) for files that aren't,
// false with `error` set for containers too damaged to read.
bool ReadStfsHeader(const std::filesystem::path& path, StfsHeader& header, std::string& error);

struct StfsOpenResult
{
    std::shared_ptr<Device> device;      // null when the container can't be read at all
    std::vector<std::string> errors;     // corruption: the package shouldn't be mounted
    std::vector<std::string> damage;     // a file's blocks that don't match their hashes: it reads, its contents may be off
    std::vector<std::string> warnings;   // worth knowing, harmless to the game
    uint64_t verifiedBlocks = 0;
};

// Parses the container's file table and checks every hash it can: the header's, the hash tables'
// and each file block's.
StfsOpenResult OpenStfs(const std::filesystem::path& path, const StfsHeader& header);

// SHA-1 (BCrypt on Windows, a built-in implementation elsewhere), shared with the folder checks.
std::array<uint8_t, 20> Sha1(const void* data, size_t size);
std::string Hex(const uint8_t* data, size_t size);

} // namespace ac6::fs
