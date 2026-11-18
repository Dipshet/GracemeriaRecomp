#include "loader.h"

#include "heap.h"
#include "log.h"
#include "memory.h"

#include <cstring>
#include <unordered_map>

#include <file.h>
#include <image.h>
#include <xex.h>

// Ordinal -> name tables defined in XenonUtils/xex.cpp.
extern std::unordered_map<size_t, const char*> XamExports;
extern std::unordered_map<size_t, const char*> XboxKernelExports;

namespace ac6 {

LoadedModule g_module;

namespace {

constexpr uint32_t kVariableStorageSize = 0x100; // per variable import, known ones are initialised later

struct Xex2TlsInfo
{
    be<uint32_t> slotCount;
    be<uint32_t> rawDataAddress;
    be<uint32_t> dataSize;
    be<uint32_t> rawDataSize;
};

void WireVariableImports(const uint8_t* xex, LoadedModule& out)
{
    auto* imports = static_cast<const Xex2ImportHeader*>(getOptHeaderPtr(xex, XEX_HEADER_IMPORT_LIBRARIES));
    if (!imports)
        return;

    std::vector<std::string> libraries;
    auto* strings = reinterpret_cast<const char*>(imports + 1);
    size_t offset = 0;
    for (uint32_t i = 0; i < imports->numImports; i++)
    {
        libraries.emplace_back(strings + offset);
        offset += ((libraries.back().size() + 1) + 3) & ~size_t(3);
    }

    auto* library = reinterpret_cast<const Xex2ImportLibrary*>(
        reinterpret_cast<const uint8_t*>(imports) + sizeof(Xex2ImportHeader) + imports->sizeOfStringTable);
    for (const auto& libName : libraries)
    {
        const auto* names = libName == "xam.xex" ? &XamExports : libName == "xboxkrnl.exe" ? &XboxKernelExports : nullptr;
        // Records come as an IAT slot (in data), followed by a code thunk if the import is a function.
        // A slot with no thunk after it is a variable import.
        auto* descriptors = reinterpret_cast<const Xex2ImportDescriptor*>(library + 1);
        const uint32_t n = library->numberOfImports;
        for (uint32_t i = 0; i < n; i++)
        {
            uint32_t slot = descriptors[i].firstThunk;
            if (slot >= PPC_CODE_BASE && slot < PPC_CODE_BASE + PPC_CODE_SIZE)
                continue; // a thunk, consumed together with its IAT slot below

            // XenonUtils byte-swapped the record in place, so the original big-endian value
            // (type << 24 | hint << 16 | ordinal) reads back as a little-endian word.
            uint32_t record = *g_memory.As<uint32_t>(slot);
            uint32_t ordinal = record & 0xFFFF;
            std::string name;
            if (names)
                if (auto it = names->find(ordinal); it != names->end())
                    name = it->second;
            if (name.starts_with("__imp__"))
                name = name.substr(7);

            uint32_t next = (i + 1 < n) ? uint32_t(descriptors[i + 1].firstThunk) : 0;
            if (next >= PPC_CODE_BASE && next < PPC_CODE_BASE + PPC_CODE_SIZE)
            {
                // Function import: the IAT slot points at the thunk, which the function table maps to __imp__<name>.
                *g_memory.As<be<uint32_t>>(slot) = next;
                out.functionImports++;
                continue;
            }

            VariableImport v;
            v.library = libName;
            v.ordinal = ordinal;
            v.slot = slot;
            v.name = name;
            void* storage = g_heap.AllocPhysical(kVariableStorageSize, 0x10);
            std::memset(storage, 0, kVariableStorageSize);
            v.storage = g_memory.MapVirtual(storage);
            *g_memory.As<be<uint32_t>>(slot) = v.storage;
            out.variables.push_back(v);
        }
        library = reinterpret_cast<const Xex2ImportLibrary*>(
            reinterpret_cast<const uint8_t*>(library + 1) + library->numberOfImports * sizeof(Xex2ImportDescriptor));
    }
}

void FillFunctionTable()
{
    // The table lives right after the image (ppc_config.h): one host pointer per guest instruction.
    uint32_t tableBegin = static_cast<uint32_t>(PPC_IMAGE_BASE + PPC_IMAGE_SIZE);
    uint64_t tableSize = PPC_CODE_SIZE * 2;
    g_memory.Commit(tableBegin, tableSize);

    size_t n = 0;
    for (size_t i = 0; PPCFuncMappings[i].guest != 0; i++)
    {
        if (PPCFuncMappings[i].host)
        {
            g_memory.Function(PPCFuncMappings[i].guest) = PPCFuncMappings[i].host;
            n++;
        }
    }
    log::Info("Function table: {} functions at {:08X}-{:08X}", n, tableBegin, tableBegin + tableSize);
}

} // namespace

bool LoadXex(const std::filesystem::path& path, LoadedModule& out)
{
    auto file = LoadFile(path);
    if (file.empty())
    {
        log::Error("Cannot read {}", reinterpret_cast<const char*>(path.u8string().c_str()));
        return false;
    }
    return LoadXex(std::move(file), reinterpret_cast<const char*>(path.filename().u8string().c_str()), out);
}

bool LoadXex(std::vector<uint8_t> file, const std::string& name, LoadedModule& out)
{

    // Decrypt + decompress (XenonUtils). Also rewrites function thunks, which the recompiled code never runs.
    Image image = Image::ParseImage(file.data(), file.size());
    if (!image.data || image.base != PPC_IMAGE_BASE)
    {
        log::Error("Failed to load {} (base {:08X})", name, image.base);
        return false;
    }

    out.imageBase = static_cast<uint32_t>(image.base);
    out.imageSize = image.size;
    out.entryPoint = static_cast<uint32_t>(image.entry_point);

    if (auto* stack = static_cast<const be<uint32_t>*>(getOptHeaderPtr(file.data(), XEX_HEADER_DEFAULT_STACK_SIZE)))
        out.defaultStackSize = *stack;
    if (auto* tls = static_cast<const Xex2TlsInfo*>(getOptHeaderPtr(file.data(), XEX_HEADER_TLS_INFO)))
        out.tls = { tls->slotCount, tls->rawDataAddress, tls->dataSize, tls->rawDataSize };
    // Execution info: media id, version, base version, title id (big-endian words).
    if (auto* info = static_cast<const be<uint32_t>*>(getOptHeaderPtr(file.data(), XEX_HEADER_EXECUTION_INFO)))
        out.titleId = info[3];

    auto* header = reinterpret_cast<const Xex2Header*>(file.data());
    out.xexHeader.assign(file.begin(), file.begin() + header->headerSize);

    // Map the image (image + function table range are backed).
    g_memory.AddBackedRange(out.imageBase, PPC_IMAGE_BASE + PPC_IMAGE_SIZE + PPC_CODE_SIZE * 2);
    g_memory.Commit(out.imageBase, out.imageSize);
    std::memcpy(g_memory.Translate(out.imageBase), image.data.get(), out.imageSize);

    WireVariableImports(file.data(), out);
    FillFunctionTable();

    log::Info("Loaded {}: base {:08X} size {:08X} entry {:08X} stack {:#x}", name,
        out.imageBase, out.imageSize, out.entryPoint, out.defaultStackSize);
    log::Info("TLS: {} slots, raw data {:08X} ({:#x} of {:#x} bytes)", out.tls.slotCount, out.tls.rawDataAddress,
        out.tls.rawDataSize, out.tls.dataSize);
    log::Info("Imports: {} functions, {} variables", out.functionImports, out.variables.size());
    for (auto& v : out.variables)
        log::Info("Variable import {}!{} (#{}) slot {:08X} -> {:08X}", v.library, v.name.empty() ? "?" : v.name,
            v.ordinal, v.slot, v.storage);
    return true;
}

uint32_t FindVariableImport(const LoadedModule& module, const char* name)
{
    for (auto& v : module.variables)
        if (v.name == name)
            return v.storage;
    return 0;
}

} // namespace ac6
