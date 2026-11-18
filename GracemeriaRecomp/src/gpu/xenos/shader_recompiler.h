// Vendored from XenosRecomp (https://github.com/hedge-dev/XenosRecomp @ 990d03b, MIT, see
// LICENSE-XenosRecomp.md). Changes for AC6: wrapped in namespace xenos, std::format instead of {fmt},
// vertex inputs from the vfetch instructions (VertexInput).
#pragma once

#include "shader.h"
#include "shader_code.h"

#include "xenos_prelude.h"

#include <vector>

namespace xenos {

struct StringBuffer
{
    std::string out;

    template<class... Args>
    void print(fmt::format_string<Args...> fmt, Args&&... args)
    {
        fmt::vformat_to(std::back_inserter(out), fmt.get(), fmt::make_format_args(args...));
    }

    template<class... Args>
    void println(fmt::format_string<Args...> fmt, Args&&... args)
    {
        fmt::vformat_to(std::back_inserter(out), fmt.get(), fmt::make_format_args(args...));
        out += '\n';
    }
};

// AC6: vertex shader inputs come from the (D3D-patched) vfetch instructions themselves, one per
// distinct fetch constant + offset + format, instead of from the container's semantic list (whose
// order no longer matches the instructions once D3D has re-sorted them for a vertex declaration).
struct VertexInput
{
    uint32_t location = 0;
    uint32_t fetchSlot = 0;      // vertex fetch constant: constIndex * 3 + constIndexSelect
    int32_t offset = 0;          // dwords from the vertex start
    uint32_t stride = 0;         // dwords (0 for mini fetches: taken from the full fetch of the same slot)
    uint32_t format = 0;         // GPU vertex format
    bool isSigned = false;
    bool isInteger = false;
};

struct ShaderRecompiler : StringBuffer
{
    uint32_t indentation = 0;
    bool isPixelShader = false;
    const uint8_t* constantTableData = nullptr;
    std::unordered_map<uint32_t, VertexElement> vertexElements;
    std::vector<VertexInput> vertexInputs;
    std::unordered_map<uint64_t, uint32_t> vertexInputIndex;
    std::unordered_map<uint32_t, std::string> interpolators;
    std::unordered_map<uint32_t, const ConstantInfo*> float4Constants;
    std::unordered_map<uint32_t, const char*> boolConstants;
    std::unordered_map<uint32_t, const char*> samplers;
    std::unordered_map<uint32_t, uint32_t> ifEndLabels;
    uint32_t specConstantsMask = 0;
    bool usesMemExport = false;   // AC6: an ALLOC of type memory (memexport) was seen
    // AC6: vertex fetches at a computed index (not the untouched r0.x vertex index) read their stream
    // directly. These are the streams they read (bit s = stream s = fetch constant 95 - s).
    uint32_t indexedFetchStreams = 0;
    bool r0Written = false;               // something before this point in program order wrote r0.x
    bool lastFetchIndexed = false;        // the last full vfetch (mini fetches reuse its index)
    // AC6: masked execution in pixel shaders with jumps: lanes that take a forward
    // jump become inactive until its target instead of leaving the quad. Kills honour it.
    bool maskedExecution = false;
    bool allowMaskedExecution = true;
    // AC6: texture fetches whose instruction sets its own filters (mag, min, mip or anisotropy instead of
    // "use the fetch constant's"): fetch slot | fields << 8 (mag 2 bits, min 2, mip 2, aniso 3, as in the
    // instruction, 3/7 = keep the fetch constant's). Entry k samples through g_OverrideSampler(k).
    std::vector<uint32_t> samplerOverrides;
    static constexpr uint32_t kMaxSamplerOverrides = 8;
    std::vector<const char*> maskedOutputs;   // the pixel shader's outputs (saved with the registers)
    std::string lastFetchIndex;
    uint32_t lastFetchStride = 0;

#ifdef UNLEASHED_RECOMP
    bool hasMtxProjection = false;
    bool hasMtxPrevInvViewProjection = false;
#endif

    void indent()
    {
        for (uint32_t i = 0; i < indentation; i++)
            out += '\t';
    }

    void printDstSwizzle(uint32_t dstSwizzle, bool operand);
    void printDstSwizzle01(uint32_t dstRegister, uint32_t dstSwizzle);

    void recompile(const VertexFetchInstruction& instr, uint32_t address);
    void recompile(const TextureFetchInstruction& instr, bool bicubic);
    void recompile(const AluInstruction& instr);

    // AC6: `variant` selects the container's shader record (D3D loads record k from the offset at
    // container +0x18 + 8k, record 0 is shaderOffset).
    void recompile(const uint8_t* shaderData, const std::string_view& include, uint32_t variant = 0);
};

} // namespace xenos
