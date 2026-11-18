#pragma once
// XEX loading: decrypt/decompress with XenonRecomp's loader (XenonUtils, MIT), map the image into
// guest memory, wire variable imports, fill the function table.

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace ac6 {

struct TlsInfo
{
    uint32_t slotCount = 0;
    uint32_t rawDataAddress = 0;
    uint32_t dataSize = 0;
    uint32_t rawDataSize = 0;
};

struct VariableImport
{
    std::string library;
    uint32_t ordinal = 0;
    std::string name;      // empty if the ordinal is unknown
    uint32_t slot = 0;     // guest address of the import record (receives the variable's address)
    uint32_t storage = 0;  // guest address of the storage the runtime allocated
};

struct LoadedModule
{
    uint32_t imageBase = 0;
    uint32_t imageSize = 0;
    uint32_t entryPoint = 0;
    uint32_t defaultStackSize = 0;
    uint32_t titleId = 0;          // from the execution info (0x4E4D07D1: Ace Combat 6, NTSC-U)
    TlsInfo tls;
    std::vector<VariableImport> variables;
    uint32_t functionImports = 0;
    std::vector<uint8_t> xexHeader; // raw XEX header (XamGetExecutionId reads the execution ID from it)
};

bool LoadXex(const std::filesystem::path& path, LoadedModule& out);
bool LoadXex(std::vector<uint8_t> file, const std::string& name, LoadedModule& out);

// Guest address of the storage backing a variable import, or 0 if the game doesn't import it.
uint32_t FindVariableImport(const LoadedModule& module, const char* name);

extern LoadedModule g_module;

} // namespace ac6
