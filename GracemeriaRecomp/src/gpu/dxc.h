#pragma once
// HLSL -> SPIR-V through the DirectX Shader Compiler, loaded at runtime (dxcompiler.dll next to the
// executable, else from a Vulkan SDK). Only the HLSL shader backend uses it (AC6_SHADER_BACKEND=hlsl,
// AC6_SHADER_PATCH): by default the guest shaders are translated to SPIR-V directly.

#include <cstdint>
#include <string>
#include <vector>

namespace ac6::gpu {

bool DxcAvailable();
// Returns false and fills `errors` on failure. One compiler instance per calling thread.
bool CompileHlslToSpirv(const std::string& source, bool pixelShader, std::vector<uint32_t>& spirv, std::string& errors);
// Any target profile (e.g. L"-T cs_6_0"), invertY flips SV_Position.y (vertex shaders).
bool CompileHlslToSpirv(const std::string& source, const wchar_t* target, bool invertY, std::vector<uint32_t>& spirv, std::string& errors);

} // namespace ac6::gpu
