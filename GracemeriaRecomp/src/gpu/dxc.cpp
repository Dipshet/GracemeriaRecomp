#include "gpu/dxc.h"

#include "log.h"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <vector>
#include <mutex>

#if !defined(_WIN32)
// Linux: not ported. Only the HLSL backend and the shader patches use DXC: the game's shaders are
// written as SPIR-V directly.
namespace ac6::gpu {

bool DxcAvailable() { return false; }

bool CompileHlslToSpirv(const std::string&, bool, std::vector<uint32_t>&, std::string& errors)
{
    errors = "DXC unavailable (Windows-only)";
    return false;
}

bool CompileHlslToSpirv(const std::string&, const wchar_t*, bool, std::vector<uint32_t>&, std::string& errors)
{
    errors = "DXC unavailable (Windows-only)";
    return false;
}

} // namespace ac6::gpu
#else
#include <windows.h>
#include <unknwn.h>   // WIN32_LEAN_AND_MEAN leaves out the COM base interfaces dxcapi.h needs
#include <objidl.h>
#include <dxcapi.h>   // (AC6_DXC_INCLUDE: toolchain/dxc/inc or the Vulkan SDK's Include/dxc)

// mingw's __uuidof needs explicit GUID declarations for the interfaces used.
__CRT_UUID_DECL(IDxcBlob, 0x8BA5FB08, 0x5195, 0x40e2, 0xAC, 0x58, 0x0D, 0x98, 0x9C, 0x3A, 0x01, 0x02)
__CRT_UUID_DECL(IDxcBlobUtf8, 0x3DA636C9, 0xBA71, 0x4024, 0xA3, 0x01, 0x30, 0xCB, 0xF1, 0x25, 0x30, 0x5B)
__CRT_UUID_DECL(IDxcResult, 0x58346CDA, 0xDDE7, 0x4497, 0x94, 0x61, 0x6F, 0x87, 0xAF, 0x5E, 0x06, 0x59)
__CRT_UUID_DECL(IDxcCompiler3, 0x228B4687, 0x5A6A, 0x4730, 0x90, 0x0C, 0x97, 0x02, 0xB2, 0x20, 0x3F, 0x54)

namespace ac6::gpu {

namespace {

DxcCreateInstanceProc s_create = nullptr;
std::once_flag s_loadOnce;

// Only the HLSL backend needs DXC (AC6_SHADER_BACKEND=hlsl, AC6_SHADER_PATCH): guest shaders are
// written as SPIR-V directly and the renderer's own are compiled at build time. The DLL
// is looked for next to the executable, then in the Vulkan SDK (VULKAN_SDK, or the newest in C:\VulkanSDK).
void Load()
{
    HMODULE module = LoadLibraryW(L"dxcompiler.dll");
    if (!module)
    {
        std::vector<std::filesystem::path> candidates;
        if (const char* sdk = std::getenv("VULKAN_SDK"))
            candidates.push_back(std::filesystem::path(sdk) / "Bin" / "dxcompiler.dll");
        std::error_code ec;
        std::vector<std::filesystem::path> installed;
        for (const auto& e : std::filesystem::directory_iterator("C:/VulkanSDK", ec))
            installed.push_back(e.path() / "Bin" / "dxcompiler.dll");
        std::sort(installed.rbegin(), installed.rend());
        candidates.insert(candidates.end(), installed.begin(), installed.end());
        for (const auto& path : candidates)
            if (std::filesystem::exists(path, ec) && (module = LoadLibraryW(path.wstring().c_str())))
                break;
    }
    if (module)
        s_create = reinterpret_cast<DxcCreateInstanceProc>(GetProcAddress(module, "DxcCreateInstance"));
    if (!s_create)
        log::Error("dxcompiler.dll not found (next to the executable or in a Vulkan SDK): the HLSL shader backend can't compile");
}

template<typename T>
struct ComPtr
{
    T* p = nullptr;
    ~ComPtr() { if (p) p->Release(); }
    T** operator&() { return &p; }
    T* operator->() const { return p; }
    explicit operator bool() const { return p != nullptr; }
};

} // namespace

bool DxcAvailable()
{
    std::call_once(s_loadOnce, Load);
    return s_create != nullptr;
}

bool CompileHlslToSpirv(const std::string& source, bool pixelShader, std::vector<uint32_t>& spirv, std::string& errors)
{
    return CompileHlslToSpirv(source, pixelShader ? L"-T ps_6_0" : L"-T vs_6_0", !pixelShader, spirv, errors);
}

bool CompileHlslToSpirv(const std::string& source, const wchar_t* target, bool invertY, std::vector<uint32_t>& spirv, std::string& errors)
{
    if (!DxcAvailable())
    {
        errors = "DXC unavailable";
        return false;
    }
    thread_local ComPtr<IDxcCompiler3> compiler;
    if (!compiler && FAILED(s_create(CLSID_DxcCompiler, IID_PPV_ARGS(&compiler))))
    {
        errors = "DxcCreateInstance failed";
        return false;
    }

    // XenosRecomp's flags for its SPIR-V output (see its dxc_compiler.cpp).
    const wchar_t* args[] = {
        target,
        L"-HV 2021",
        L"-all-resources-bound",
        L"-spirv",
        L"-fvk-use-dx-layout",
        L"-Qstrip_debug",
        L"-fvk-invert-y",   // vertex shaders only (dropped below otherwise)
    };
    const uint32_t argCount = invertY ? 7 : 6;

    DxcBuffer buffer{ source.data(), source.size(), DXC_CP_UTF8 };
    ComPtr<IDxcResult> result;
    if (FAILED(compiler->Compile(&buffer, args, argCount, nullptr, IID_PPV_ARGS(&result))) || !result)
    {
        errors = "Compile call failed";
        return false;
    }
    HRESULT status = S_OK;
    result->GetStatus(&status);
    if (FAILED(status))
    {
        ComPtr<IDxcBlobUtf8> messages;
        if (SUCCEEDED(result->GetOutput(DXC_OUT_ERRORS, IID_PPV_ARGS(&messages), nullptr)) && messages)
            errors.assign(messages->GetStringPointer(), messages->GetStringLength());
        return false;
    }
    ComPtr<IDxcBlob> object;
    if (FAILED(result->GetOutput(DXC_OUT_OBJECT, IID_PPV_ARGS(&object), nullptr)) || !object)
    {
        errors = "no object output";
        return false;
    }
    spirv.resize(object->GetBufferSize() / 4);
    std::memcpy(spirv.data(), object->GetBufferPointer(), spirv.size() * 4);
    return true;
}

} // namespace ac6::gpu
#endif
