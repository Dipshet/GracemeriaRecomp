#pragma once
// Save data on the host (saves are high-risk).
//
// Layout under <exe dir>/saves (or AC6_SAVE_DIR):
//   <XUID>/<content type>/<file name>/          the container, mounted writable while open
//   <XUID>/<content type>/<file name>.meta       display name (UTF-8)
//   <XUID>/<content type>/<file name>.thumb      thumbnail bytes as the game gave them
//   <XUID>/<content type>/<file name>.backup/    previous versions of each file (newest = .1)
//   <XUID>/<content type>/<file name>.staging/   files being written (temp copies).
//   .trash/<timestamp>-<file name>/              deleted containers (never destroyed by the runtime)
//
// File writes never touch the committed file directly: a writer works on a staging copy, and on close
// the copy is flushed to disk, the previous version is copied aside, the copy replaces the file with an
// atomic, write-through rename, and only then does the copied-aside version become the newest backup. A
// crash mid-save leaves the old file intact.

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace ac6::fs {

std::filesystem::path SaveRoot();
std::filesystem::path ContentPath(uint64_t xuid, uint32_t contentType, const std::string& fileName);

struct ContentEntry
{
    std::string fileName;
    std::string displayName;   // UTF-8
};

std::vector<ContentEntry> ListContent(uint64_t xuid, uint32_t contentType);
// False when it couldn't be written (a read-only folder, a full disk).
bool WriteContentMeta(const std::filesystem::path& content, const std::string& displayName);
std::string ReadContentMeta(const std::filesystem::path& content);
void WriteThumbnail(const std::filesystem::path& content, const void* data, size_t size);
bool TrashContent(const std::filesystem::path& content);

// Staging path for writing `target` (inside the container at `content`).
std::filesystem::path StagingPathFor(const std::filesystem::path& content, const std::filesystem::path& target);
// Flushes `staged`, backs up the current `target` and atomically replaces it. Returns false (and keeps
// both files) on failure.
bool CommitStaged(const std::filesystem::path& content, const std::filesystem::path& staged, const std::filesystem::path& target);
// Tests: AC6_SAVE_FAULT=<kind>[,<kind>...] (from the environment only, never the settings file) makes a save step fail
// on purpose: "copy" (the staging copy at open), "write" (every write comes up short), "flush", "replace" (every replace
// of the real file), "replace-busy" (the first 3 replaces, as a program briefly holding the file would).
bool FaultInjected(const char* kind);
// Removes a container's staging folder once it holds no files (called when the container closes).
// Leftover .tmp files (a write interrupted by a crash) are kept: the committed file is intact anyway.
void CleanStaging(const std::filesystem::path& content);
// Backs up and removes `target` (file deletes go through the backup folder too).
bool DeleteWithBackup(const std::filesystem::path& content, const std::filesystem::path& target);

// The container folder a host path belongs to (the mounted root), for staging/backups.
std::filesystem::path ContainerOf(const std::filesystem::path& hostPath);

} // namespace ac6::fs
