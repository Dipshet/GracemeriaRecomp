// Vendored from XenosRecomp (https://github.com/hedge-dev/XenosRecomp @ 990d03b, MIT, see
// LICENSE-XenosRecomp.md). Changes for AC6: wrapped in namespace xenos, std::format instead of {fmt}, recompile() takes the container bytes from the runtime.
// The stages link by interpolator register (TEXCOORD0-15), not by the containers' declared semantics.
// Vertex inputs come from the vfetch instructions (fetch constant, offset, format), not the container's semantics.
#include "shader_recompiler.h"
#include "shader_common.h"

#include <cmath>

namespace xenos {


static constexpr char SWIZZLES[] = 
{ 
    'x',
    'y', 
    'z', 
    'w', 
    '0', 
    '1',
    '_',
    '_'
};

static constexpr const char* USAGE_TYPES[] =
{
    "float4", // POSITION
    "float4", // BLENDWEIGHT
    "uint4", // BLENDINDICES
    "uint4", // NORMAL
    "float4", // PSIZE
    "float4", // TEXCOORD
    "uint4", // TANGENT
    "uint4", // BINORMAL
    "float4", // TESSFACTOR
    "float4", // POSITIONT
    "float4", // COLOR
    "float4", // FOG
    "float4", // DEPTH
    "float4", // SAMPLE
};

static constexpr const char* USAGE_VARIABLES[] =
{
    "Position",
    "BlendWeight",
    "BlendIndices",
    "Normal",
    "PointSize",
    "TexCoord",
    "Tangent",
    "Binormal",
    "TessFactor",
    "PositionT",
    "Color",
    "Fog",
    "Depth",
    "Sample"
};

static constexpr const char* USAGE_SEMANTICS[] =
{
    "POSITION",
    "BLENDWEIGHT",
    "BLENDINDICES",
    "NORMAL",
    "PSIZE",
    "TEXCOORD",
    "TANGENT",
    "BINORMAL",
    "TESSFACTOR",
    "POSITIONT",
    "COLOR",
    "FOG",
    "DEPTH",
    "SAMPLE"
};

struct DeclUsageLocation
{
    DeclUsage usage;
    uint32_t usageIndex;
    uint32_t location;
};

// NOTE: These are specialized Vulkan locations for Unleashed Recompiled. Change as necessary. Likely not going to work with other games.
static constexpr DeclUsageLocation USAGE_LOCATIONS[] =
{
    { DeclUsage::Position, 0, 0 },
    { DeclUsage::Normal, 0, 1 },
    { DeclUsage::Tangent, 0, 2 },
    { DeclUsage::Binormal, 0, 3 },
    { DeclUsage::TexCoord, 0, 4 },
    { DeclUsage::TexCoord, 1, 5 },
    { DeclUsage::TexCoord, 2, 6 },
    { DeclUsage::TexCoord, 3, 7 },
    { DeclUsage::Color, 0, 8 },
    { DeclUsage::BlendIndices, 0, 9 },
    { DeclUsage::BlendWeight, 0, 10 },
    { DeclUsage::Color, 1, 11 },
    { DeclUsage::TexCoord, 4, 12 },
    { DeclUsage::TexCoord, 5, 13 },
    { DeclUsage::TexCoord, 6, 14 },
    { DeclUsage::TexCoord, 7, 15 },
    { DeclUsage::Position, 1, 15 },
};

static constexpr std::pair<DeclUsage, size_t> INTERPOLATORS[] =
{
    { DeclUsage::TexCoord, 0 },
    { DeclUsage::TexCoord, 1 },
    { DeclUsage::TexCoord, 2 },
    { DeclUsage::TexCoord, 3 },
    { DeclUsage::TexCoord, 4 },
    { DeclUsage::TexCoord, 5 },
    { DeclUsage::TexCoord, 6 },
    { DeclUsage::TexCoord, 7 },
    { DeclUsage::TexCoord, 8 },
    { DeclUsage::TexCoord, 9 },
    { DeclUsage::TexCoord, 10 },
    { DeclUsage::TexCoord, 11 },
    { DeclUsage::TexCoord, 12 },
    { DeclUsage::TexCoord, 13 },
    { DeclUsage::TexCoord, 14 },
    { DeclUsage::TexCoord, 15 },
};
// AC6: the stages link by interpolator register, as the GPU does: register N is TEXCOORDN in both. Xbox
// D3D patches a vertex shader's exports to the registers its pixel shader reads, so the vertex shader's
// own declaration table names the wrong semantics for patched code (a COLOR0 export moved to register 4,
// which the table declares as TEXCOORD13, would leave the pixel shader's COLOR0 at zero).

static constexpr std::string_view TEXTURE_DIMENSIONS[] = 
{
    "2D",
    "3D", 
    "Cube" 
};

static FetchDestinationSwizzle getDestSwizzle(uint32_t dstSwizzle, uint32_t index)
{
    return FetchDestinationSwizzle((dstSwizzle >> (index * 3)) & 0x7);
}

void ShaderRecompiler::printDstSwizzle(uint32_t dstSwizzle, bool operand)
{
    for (size_t i = 0; i < 4; i++)
    {
        const auto swizzle = getDestSwizzle(dstSwizzle, i);
        if (swizzle >= FetchDestinationSwizzle::X && swizzle <= FetchDestinationSwizzle::W)
            out += SWIZZLES[operand ? uint32_t(swizzle) : i];
    }
}

void ShaderRecompiler::printDstSwizzle01(uint32_t dstRegister, uint32_t dstSwizzle)
{
    for (size_t i = 0; i < 4; i++)
    {
        const auto swizzle = getDestSwizzle(dstSwizzle, i);
        if (swizzle == FetchDestinationSwizzle::Zero)
        {
            indent();
            println("r{}.{} = 0.0;", dstRegister, SWIZZLES[i]);
        }
        else if (swizzle == FetchDestinationSwizzle::One)
        {
            indent();
            println("r{}.{} = 1.0;", dstRegister, SWIZZLES[i]);
        }
    }
}

void ShaderRecompiler::recompile(const VertexFetchInstruction& instr, uint32_t address)
{
    if (instr.isPredicated)
    {
        indent();
        println("if ({}p0)", instr.predicateCondition ? "" : "!");

        indent();
        out += "{\n";
        ++indentation;
    }

    // AC6 diagnostics: which fetch constant / offset / format this vfetch reads.
    indent();
    println("// vfetch @{} const {}.{} offset {} stride {} format {} mini {} index r{}.{}{} exp {}", address, uint32_t(instr.constIndex),
        uint32_t(instr.constIndexSelect), int32_t(instr.offset), uint32_t(instr.stride), uint32_t(instr.format), uint32_t(instr.isMiniFetch),
        uint32_t(instr.srcRegister), uint32_t(instr.srcSwizzle), r0Written ? " (r0.x written)" : "", int32_t(instr.expAdjust));
    println("// vfetch words {:08X} {:08X} {:08X}", reinterpret_cast<const uint32_t*>(&instr)[0], reinterpret_cast<const uint32_t*>(&instr)[1],
        reinterpret_cast<const uint32_t*>(&instr)[2]);

    // AC6: a fetch whose swizzle only writes constants (0 / 1) or nothing has no assignment.
    const size_t assignmentStart = out.size();
    bool fetchesComponents = false;
    for (size_t i = 0; i < 4; i++)
    {
        const auto swizzle = getDestSwizzle(instr.dstSwizzle, i);
        fetchesComponents |= swizzle >= FetchDestinationSwizzle::X && swizzle <= FetchDestinationSwizzle::W;
    }

    indent();
    print("r{}.", instr.dstRegister);
    printDstSwizzle(instr.dstSwizzle, false);

    out += " = ";
    // AC6: the exponent adjust scales the fetched value by 2^expAdjust (terrain tiles store their vertex
    // offsets as small integers scaled this way).
    const size_t expressionStart = out.size();
    auto applyExpAdjust = [&] {
        if (instr.expAdjust != 0)
        {
            out.insert(expressionStart, "(");
            out += fmt::format(" * {})", std::ldexp(1.0, int32_t(instr.expAdjust)));
        }
    };

    // AC6: the input is whatever this instruction fetches (fetch constant, offset, format), see
    // VertexInput in shader_recompiler.h.
    const uint32_t fetchSlot = instr.constIndex * 3 + instr.constIndexSelect;
    // The number-format bits only apply to fixed-point formats (float formats often have them set).
    const bool isFloatFormat = instr.format == 31 || instr.format == 32 || (instr.format >= 36 && instr.format <= 38) || instr.format == 57;
    const bool isSigned = !isFloatFormat && instr.formatCompAll != 0, isInteger = !isFloatFormat && instr.numFormatAll != 0;

    // AC6: the input assembler only serves fetches at the vertex index (r0.x before anything writes r0).
    // Others (particles derive a corner and a particle index from the vertex index) read the stream
    // themselves at the index they computed. Mini fetches reuse the index of the last full fetch.
    if (!instr.isMiniFetch)
    {
        lastFetchIndexed = instr.srcRegister != 0 || instr.srcSwizzle != 0 || instr.srcRegisterAm || r0Written;
        // Truncated indices are floored as on the console. Shaders compute them as whole numbers in float
        // and rely on the console's arithmetic landing on the right side of them: the terrain's ring slot
        // floor(40 * frac((v + 1) * rcp(40))) needs its truncating reciprocal (xeRcp),
        // and the city's buildings add their own +0.001 to v * (1 / N). Rounded indices are floor(x + 0.5), as on the
        // console (not round-half-to-even).
        lastFetchIndex = instr.isIndexRounded
            ? fmt::format("uint(max(floor(r{}.{} + 0.5), 0.0))", uint32_t(instr.srcRegister), SWIZZLES[instr.srcSwizzle])
            : fmt::format("uint(max(floor(r{}.{}), 0.0))", uint32_t(instr.srcRegister), SWIZZLES[instr.srcSwizzle]);
        lastFetchStride = instr.stride;
    }
    const uint32_t stream = 95 - fetchSlot;
    if (lastFetchIndexed && fetchSlot <= 95 && stream < 16)
    {
        indexedFetchStreams |= 1u << stream;
        const std::string args = fmt::format("{}, {}, {}, {}", stream, lastFetchIndex, int32_t(instr.offset), lastFetchStride);
        switch (instr.format)
        {
        case 7:
            print("xeFetch2_10_10_10(xeFetchWord({}, 0), {}, {})", args, isSigned, isInteger);
            break;
        case 16:
            print("xeFetch10_11_11(xeFetchWord({}, 0), {}, {})", args, isSigned, isInteger);
            break;
        case 17:
            print("xeFetch11_11_10(xeFetchWord({}, 0), {}, {})", args, isSigned, isInteger);
            break;
        default:
            print("xeFetchIndexed({}, {}, {}, {})", args, uint32_t(instr.format), isSigned, isInteger);
            break;
        }
        applyExpAdjust();
        out += '.';
        printDstSwizzle(instr.dstSwizzle, true);
        out += ";\n";
        if (!fetchesComponents)
            out.resize(assignmentStart);
        printDstSwizzle01(instr.dstRegister, instr.dstSwizzle);
        if (instr.dstRegister == 0 && getDestSwizzle(instr.dstSwizzle, 0) != FetchDestinationSwizzle::Keep)
            r0Written = true;
        if (instr.isPredicated)
        {
            --indentation;
            indent();
            out += "}\n";
        }
        return;
    }
    const uint64_t key = (uint64_t(fetchSlot) << 40) | (uint64_t(uint32_t(int32_t(instr.offset)) & 0xFFFFFF) << 16) |
        (uint64_t(instr.format) << 2) | (uint64_t(isSigned) << 1) | uint64_t(isInteger);
    auto [inputIt, inserted] = vertexInputIndex.emplace(key, uint32_t(vertexInputs.size()));
    if (inserted)
    {
        vertexInputs.push_back({ uint32_t(vertexInputs.size()), fetchSlot, int32_t(instr.offset), instr.isMiniFetch ? 0u : uint32_t(instr.stride),
            uint32_t(instr.format), isSigned, isInteger });
    }
    else if (!instr.isMiniFetch && vertexInputs[inputIt->second].stride == 0)
    {
        vertexInputs[inputIt->second].stride = instr.stride;
    }
    const uint32_t input = inputIt->second;
    switch (instr.format)
    {
    case 7:
        print("xeFetch2_10_10_10(iFetch{}.x, {}, {})", input, isSigned, isInteger);
        break;
    case 16:
        print("xeFetch10_11_11(iFetch{}.x, {}, {})", input, isSigned, isInteger);
        break;
    case 17:
        print("xeFetch11_11_10(iFetch{}.x, {}, {})", input, isSigned, isInteger);
        break;
    default:
        if (isInteger)
            print("float4(iFetch{})", input);
        else
            print("iFetch{}", input);
        break;
    }
    applyExpAdjust();

    out += '.';
    printDstSwizzle(instr.dstSwizzle, true);

    out += ";\n";
    if (!fetchesComponents)
        out.resize(assignmentStart);

    printDstSwizzle01(instr.dstRegister, instr.dstSwizzle);
    if (instr.dstRegister == 0 && getDestSwizzle(instr.dstSwizzle, 0) != FetchDestinationSwizzle::Keep)
        r0Written = true;

    if (instr.isPredicated)
    {
        --indentation;
        indent();
        out += "}\n";
    }
}

void ShaderRecompiler::recompile(const TextureFetchInstruction& instr, bool bicubic)
{
    if (instr.dstRegister == 0 && getDestSwizzle(instr.dstSwizzle, 0) != FetchDestinationSwizzle::Keep)
        r0Written = true;
    // AC6: getCompTexLOD is translated too (upstream drops it, leaving its destination stale).
    if (instr.opcode != FetchOpcode::TextureFetch && instr.opcode != FetchOpcode::GetTextureWeights &&
        instr.opcode != FetchOpcode::GetTextureComputedLod)
        return;

    // AC6: in pixel shaders a predicated fetch runs for every pixel and only its result is masked, as on
    // the console (a predicate masks results, the quad keeps executing): inside a real branch the texture
    // derivatives are undefined, and the LOD falls to the finest mip on predicate seams.
    const bool maskedFetch = instr.isPredicated && isPixelShader && allowMaskedExecution;
    if (maskedFetch)
    {
        indent();
        out += "{\n";
        ++indentation;
        indent();
        println("float4 fetchSave = r{};", instr.dstRegister);
    }
    else if (instr.isPredicated)
    {
        indent();
        println("if ({}p0)", instr.predCondition ? "" : "!");

        indent();
        out += "{\n";
        ++indentation;
    }

    auto printSrcRegister = [&](size_t componentCount)
        {
            print("r{}.", instr.srcRegister);

            for (size_t i = 0; i < componentCount; i++)
                out += SWIZZLES[((instr.srcSwizzle >> (i * 2))) & 0x3];
        };

    std::string constName;
    const char* constNamePtr = nullptr;
#ifdef UNLEASHED_RECOMP
    bool subtractFromOne = false;
#endif

    auto findResult = samplers.find(instr.constIndex);
    if (findResult != samplers.end())
    {
        constNamePtr = findResult->second;

    #ifdef UNLEASHED_RECOMP
        subtractFromOne = hasMtxPrevInvViewProjection && strcmp(constNamePtr, "sampZBuffer") == 0;
    #endif
    }
    else
    {
        constName = fmt::format("s{}", instr.constIndex);
        constNamePtr = constName.c_str();
    }

#ifdef UNLEASHED_RECOMP
    if (instr.constIndex == 0 && instr.dimension == TextureDimension::Texture2D)
    {
        indent();
        print("pixelCoord = getPixelCoord({}_Texture2DDescriptorIndex, ", constNamePtr);
        printSrcRegister(2);
        out += ");\n";
    }
#endif

    indent();
    print("r{}.", instr.dstRegister);
    printDstSwizzle(instr.dstSwizzle, false);

    out += " = ";
    switch (instr.opcode)
    {
    case FetchOpcode::TextureFetch:
    {
    #ifdef UNLEASHED_RECOMP
        if (subtractFromOne)
            out += "1.0 - ";
    #endif

        out += "tfetch";
        break;
    }
    case FetchOpcode::GetTextureWeights:
    {
        out += "getWeights";
        break;
    }
    case FetchOpcode::GetTextureComputedLod:
    {
        out += "getComputedLod";
        break;
    }
    }

    std::string_view dimension;
    uint32_t componentCount = 0;

    switch (instr.dimension)
    {
    case TextureDimension::Texture1D:
        dimension = "1D";
        componentCount = 1;
        break;
    case TextureDimension::Texture2D:
        dimension = "2D";
        componentCount = 2;
        break;
    case TextureDimension::Texture3D:
        dimension = "3D";
        componentCount = 3;
        break;
    case TextureDimension::TextureCube:
        dimension = "Cube";
        componentCount = 3;
        break;
    }

    out += dimension;

#ifdef UNLEASHED_RECOMP
    if (bicubic)
        out += "Bicubic";
#endif

    // AC6: the instruction's own filters, where it sets them, replace the fetch constant's (the sea's
    // shader samples its shore mask without anisotropy past a computed LOD: with the fetch constant's 8:1
    // the far mask stays sharp and its thresholds leave dotted lines along the mesh rows).
    std::string samplerIndex = fmt::format("{}_SamplerDescriptorIndex", constNamePtr);
    // (getCompTexLOD without anisotropy: as the SPIR-V backend.)
    const uint32_t anisoField = instr.opcode == FetchOpcode::GetTextureComputedLod ? 0u : uint32_t(instr.anisoFilter);
    const uint32_t filterFields = instr.magFilter | instr.minFilter << 2 | instr.mipFilter << 4 | anisoField << 6;
    if (filterFields != (3 | 3 << 2 | 3 << 4 | 7 << 6))
    {
        const uint32_t slot = instr.constIndex + (isPixelShader ? 0 : 16);
        const uint32_t key = slot | filterFields << 8 | (instr.opcode == FetchOpcode::GetTextureComputedLod ? 1u << 24 : 0u);
        auto it = std::find(samplerOverrides.begin(), samplerOverrides.end(), key);
        if (it == samplerOverrides.end() && samplerOverrides.size() < kMaxSamplerOverrides)
            it = samplerOverrides.insert(samplerOverrides.end(), key);
        if (it != samplerOverrides.end())
            samplerIndex = fmt::format("g_OverrideSampler({})", uint32_t(it - samplerOverrides.begin()) + (isPixelShader ? 0 : kMaxSamplerOverrides));
    }
    print("({0}_Texture{1}DescriptorIndex, {2}, ", constNamePtr, dimension, samplerIndex);
    printSrcRegister(componentCount);

    switch (instr.dimension)
    {
    case TextureDimension::Texture2D:
        print(", float2({}, {})", instr.offsetX * 0.5f, instr.offsetY * 0.5f);
        break;
    case TextureDimension::TextureCube:
        out += ", cubeMapData";
        break;
    }

    out += ").";

    printDstSwizzle(instr.dstSwizzle, true);

    out += ";\n";

    printDstSwizzle01(instr.dstRegister, instr.dstSwizzle);

    if (maskedFetch)
    {
        indent();
        println("if ({}p0)", instr.predCondition ? "!" : "");
        indent();
        println("\tr{0} = fetchSave;", instr.dstRegister);
    }
    if (instr.isPredicated)
    {
        --indentation;
        indent();
        out += "}\n";
    }
}

// The comparison of a predicate push (setp_*_push) against 0.
static const char* SetpPushCompare(AluVectorOpcode opcode)
{
    switch (opcode)
    {
    case AluVectorOpcode::SetpNePush: return "!=";
    case AluVectorOpcode::SetpGtPush: return ">";
    case AluVectorOpcode::SetpGePush: return ">=";
    default: return "==";
    }
}

void ShaderRecompiler::recompile(const AluInstruction& instr)
{
    // AC6: r0.x stops being the vertex index once written (see the vertex fetch recompile).
    if (!instr.exportData && ((instr.vectorDest == 0 && (instr.vectorWriteMask & 1)) || (instr.scalarDest == 0 && (instr.scalarWriteMask & 1))))
        r0Written = true;

    if (instr.isPredicated)
    {
        indent();
        println("if ({}p0)", instr.predicateCondition ? "" : "!");

        indent(); 
        out += "{\n";
        ++indentation;
    }

    enum
    {
        VECTOR_0,
        VECTOR_1,
        VECTOR_2,
        SCALAR_0,
        SCALAR_1,
        SCALAR_CONSTANT_0,
        SCALAR_CONSTANT_1
    };

    auto op = [&](size_t operand)
        {
            size_t reg = 0;
            size_t swizzle = 0;
            bool select = true;
            bool negate = false;
            bool abs = false;

            switch (operand)
            {
            case SCALAR_CONSTANT_0:
                reg = instr.src3Register;
                swizzle = instr.src3Swizzle;
                select = false;
                negate = instr.src3Negate;
                abs = instr.absConstants;
                break;

            case SCALAR_CONSTANT_1:
                reg = (uint32_t(instr.scalarOpcode) & 1) | (instr.src3Select << 1) | (instr.src3Swizzle & 0x3C);
                swizzle = instr.src3Swizzle;
                select = true;
                negate = instr.src3Negate;
                abs = instr.absConstants;
                break;

            default:
                switch (operand)
                {
                case VECTOR_0:
                    reg = instr.src1Register;
                    swizzle = instr.src1Swizzle;
                    select = instr.src1Select;
                    negate = instr.src1Negate;
                    break;
                case VECTOR_1:
                    reg = instr.src2Register;
                    swizzle = instr.src2Swizzle;
                    select = instr.src2Select;
                    negate = instr.src2Negate;
                    break;
                case VECTOR_2:
                case SCALAR_0:
                case SCALAR_1:
                    reg = instr.src3Register;
                    swizzle = instr.src3Swizzle;
                    select = instr.src3Select;
                    negate = instr.src3Negate;
                    break;
                }

                if (select)
                {
                    abs = (reg & 0x80) != 0;
                    reg &= 0x3F;
                }
                else
                {
                    abs = instr.absConstants;
                }

                break;
            }

            std::string regFormatted;

            if (select)
            {
                regFormatted = fmt::format("r{}", reg);
            }
            else
            {
                auto findResult = float4Constants.find(reg);
                if (findResult != float4Constants.end())
                {
                    const char* constantName = reinterpret_cast<const char*>(constantTableData + findResult->second->name);
                    if (findResult->second->registerCount > 1)
                    {
                    #ifdef UNLEASHED_RECOMP
                        if (hasMtxProjection && strcmp(constantName, "g_MtxProjection") == 0)
                        {
                            regFormatted = fmt::format("(iterationIndex == 0 ? mtxProjectionReverseZ[{0}] : mtxProjection[{0}])",
                                reg - findResult->second->registerIndex);
                        }
                        else
                    #endif
                        {
                            regFormatted = fmt::format("{}({}{})", constantName,
                                reg - findResult->second->registerIndex, instr.const0Relative ? (instr.constAddressRegisterRelative ? " + a0" : " + aL") : "");
                        }
                    }
                    else
                    {
                        assert(!instr.const0Relative && !instr.const1Relative);
                        regFormatted = constantName;
                    }
                }
                else
                {
                    assert(!instr.const0Relative && !instr.const1Relative);
                    regFormatted = fmt::format("c{}", reg);
                }
            }

            std::string result;

            if (negate)
                result += '-';

            if (abs)
                result += "abs(";

            result += regFormatted;
            result += '.';

            switch (operand)
            {
            case VECTOR_0:
            case VECTOR_1:
            case VECTOR_2:
            {
                uint32_t mask;

                switch (instr.vectorOpcode)
                {
                case AluVectorOpcode::Dp2Add:
                    mask = (operand == VECTOR_2) ? 0b1 : 0b11;
                    break;

                case AluVectorOpcode::Dp3:
                    mask = 0b111;
                    break;

                case AluVectorOpcode::Dp4:
                case AluVectorOpcode::Max4:
                // AC6: these read fixed components, not the written ones: cube its
                // direction from z, w, x. The kills compare all four. The predicate pushes take the
                // predicate from w and the result from x.
                case AluVectorOpcode::Cube:
                case AluVectorOpcode::KillEq:
                case AluVectorOpcode::KillGt:
                case AluVectorOpcode::KillGe:
                case AluVectorOpcode::KillNe:
                case AluVectorOpcode::SetpEqPush:
                case AluVectorOpcode::SetpNePush:
                case AluVectorOpcode::SetpGtPush:
                case AluVectorOpcode::SetpGePush:
                    mask = 0b1111;
                    break;

                default:
                {
                    // AC6: an export's components written with constant 1 aren't vector results: the
                    // operands line up with the components the vector op actually writes.
                    const uint32_t written = instr.exportData ? instr.vectorWriteMask & ~instr.scalarWriteMask : instr.vectorWriteMask;
                    mask = written != 0 ? written : 0b1;
                    break;
                }
                }

                for (size_t i = 0; i < 4; i++)
                {
                    if ((mask >> i) & 0x1)
                        result += SWIZZLES[((swizzle >> (i * 2)) + i) & 0x3];
                }

                break;
            }

            case SCALAR_0:
            case SCALAR_CONSTANT_0:
                result += SWIZZLES[((swizzle >> 6) + 3) & 0x3];
                break;

            case SCALAR_1:
            case SCALAR_CONSTANT_1:
                result += SWIZZLES[swizzle & 0x3];
                break;
            }

            if (abs)
                result += ")";

            return result;
        };

    switch (instr.vectorOpcode)
    {
    case AluVectorOpcode::KillEq:
        indent();
        println("{}clip(any({} == {}) ? -1 : 1);", maskedExecution ? "if (!xeInactive) " : "", op(VECTOR_0), op(VECTOR_1));
        break;
    
    case AluVectorOpcode::KillGt:
        indent();
        println("{}clip(any({} > {}) ? -1 : 1);", maskedExecution ? "if (!xeInactive) " : "", op(VECTOR_0), op(VECTOR_1));
        break;
    
    case AluVectorOpcode::KillGe:
        indent();
        println("{}clip(any({} >= {}) ? -1 : 1);", maskedExecution ? "if (!xeInactive) " : "", op(VECTOR_0), op(VECTOR_1));
        break;
    
    case AluVectorOpcode::KillNe:
        indent();
        println("{}clip(any({} != {}) ? -1 : 1);", maskedExecution ? "if (!xeInactive) " : "", op(VECTOR_0), op(VECTOR_1));
        break;
    }

    bool closeIfBracket = false;

    std::string_view exportRegister;
    if (instr.exportData)
    {
        if (isPixelShader)
        {
            switch (ExportRegister(instr.vectorDest))
            {
            case ExportRegister::PSColor0:
                exportRegister = "oC0";
                break;        
            case ExportRegister::PSColor1:
                exportRegister = "oC1";
                break;        
            case ExportRegister::PSColor2:
                exportRegister = "oC2";
                break;            
            case ExportRegister::PSColor3:
                exportRegister = "oC3";
                break;           
            case ExportRegister::PSDepth:
                exportRegister = "oDepth";
                break;
            }
        }
        else
        {
            switch (ExportRegister(instr.vectorDest))
            {
            case ExportRegister::VSPointSizeEdgeFlagKillVertex:
                // AC6: point size (x, the diameter in pixels), edge flag, kill.
                exportRegister = "oPts";
                break;

            case ExportRegister::VSPosition:
                exportRegister = "oPos";

            #ifdef UNLEASHED_RECOMP
                if (hasMtxProjection)
                {
                    indent();
                    out += "if ((g_SpecConstants() & SPEC_CONSTANT_REVERSE_Z) == 0 || iterationIndex == 0)\n";
                    indent();
                    out += "{\n";
                    ++indentation;

                    closeIfBracket = true;
                }
            #endif

                break;

            default:
            {
                // AC6: an export the container doesn't declare (no consumer) goes to a dummy instead of
                // dereferencing end().
                auto findResult = interpolators.find(instr.vectorDest);
                exportRegister = findResult != interpolators.end() ? std::string_view(findResult->second) : std::string_view("oUnused");
                break;
            }
            }
        }
    }

    if (instr.vectorOpcode >= AluVectorOpcode::SetpEqPush && instr.vectorOpcode <= AluVectorOpcode::SetpGePush)
    {
        // AC6: p0 = src0.w == 0 && src1.w <op> 0. The result (below) tests x the same way.
        indent();
        println("p0 = ({}).w == 0.0 && ({}).w {} 0.0;", op(VECTOR_0), op(VECTOR_1), SetpPushCompare(instr.vectorOpcode));
    }
    else if (instr.vectorOpcode >= AluVectorOpcode::MaxA)
    {
        indent();
        println("a0 = (int)clamp(floor(({}).w + 0.5), -256.0, 255.0);", op(VECTOR_0));
    }

    // AC6: both halves of an ALU instruction read their sources before either writes. The vector
    // result is emitted after the scalar one is computed (into ps), so a scalar source that is also the
    // vector destination still reads the old value.
    const size_t vectorCodeStart = out.size();
    uint32_t vectorWriteMask = instr.vectorWriteMask;
    if (instr.exportData)
        vectorWriteMask &= ~instr.scalarWriteMask;

    if (vectorWriteMask != 0)
    {
        indent();
        if (!exportRegister.empty())
        {
            out += exportRegister;
            out += '.';
        }
        else
        {
            print("r{}.", instr.vectorDest);
        }

        for (size_t i = 0; i < 4; i++)
        {
            if ((vectorWriteMask >> i) & 0x1)
                out += SWIZZLES[i];
        }

        out += " = ";

        if (instr.vectorSaturate)
            out += "saturate(";

        switch (instr.vectorOpcode)
        {
        case AluVectorOpcode::Add:
            print("{} + {}", op(VECTOR_0), op(VECTOR_1));
            break;

        case AluVectorOpcode::Mul:
            print("{} * {}", op(VECTOR_0), op(VECTOR_1));
            break;

        case AluVectorOpcode::Max:
        case AluVectorOpcode::MaxA:
            print("max({}, {})", op(VECTOR_0), op(VECTOR_1));
            break;

        case AluVectorOpcode::Min:
            print("min({}, {})", op(VECTOR_0), op(VECTOR_1));
            break;

        case AluVectorOpcode::Seq:
            print("{} == {}", op(VECTOR_0), op(VECTOR_1));
            break;

        case AluVectorOpcode::Sgt:
            print("{} > {}", op(VECTOR_0), op(VECTOR_1));
            break;

        case AluVectorOpcode::Sge:
            print("{} >= {}", op(VECTOR_0), op(VECTOR_1));
            break;

        case AluVectorOpcode::Sne:
            print("{} != {}", op(VECTOR_0), op(VECTOR_1));
            break;

        case AluVectorOpcode::Frc:
            print("frac({})", op(VECTOR_0));
            break;

        case AluVectorOpcode::Trunc:
            print("trunc({})", op(VECTOR_0));
            break;

        case AluVectorOpcode::Floor:
            print("floor({})", op(VECTOR_0));
            break;

        case AluVectorOpcode::Mad:
            print("{} * {} + {}", op(VECTOR_0), op(VECTOR_1), op(VECTOR_2));
            break;

        case AluVectorOpcode::CndEq:
            print("select({} == 0.0, {}, {})", op(VECTOR_0), op(VECTOR_1), op(VECTOR_2));
            break;

        case AluVectorOpcode::CndGe:
            print("select({} >= 0.0, {}, {})", op(VECTOR_0), op(VECTOR_1), op(VECTOR_2));
            break;

        case AluVectorOpcode::CndGt:
            print("select({} > 0.0, {}, {})", op(VECTOR_0), op(VECTOR_1), op(VECTOR_2));
            break;

        case AluVectorOpcode::Dp4:
        case AluVectorOpcode::Dp3:
            print("dot({}, {})", op(VECTOR_0), op(VECTOR_1));
            break;

        case AluVectorOpcode::Dp2Add:
            print("dot({}, {}) + {}", op(VECTOR_0), op(VECTOR_1), op(VECTOR_2));
            break;

        case AluVectorOpcode::Cube:
            // AC6: the direction is the first operand with its swizzle and modifiers (conventionally r.zzxy):
            // x, y, z = its z, w, x components.
            print("cube(({}).zwxy, cubeMapData)", op(VECTOR_0));
            break;

        case AluVectorOpcode::Max4:
            print("max4({})", op(VECTOR_0));
            break;

        case AluVectorOpcode::SetpEqPush:
        case AluVectorOpcode::SetpNePush:
        case AluVectorOpcode::SetpGtPush:
        case AluVectorOpcode::SetpGePush:
            print("(({0}).x == 0.0 && ({1}).x {2} 0.0 ? 0.0 : ({0}).x + 1.0)", op(VECTOR_0), op(VECTOR_1), SetpPushCompare(instr.vectorOpcode));
            break;

        case AluVectorOpcode::KillEq:
            print("any({} == {})", op(VECTOR_0), op(VECTOR_1));
            break;

        case AluVectorOpcode::KillGt:
            print("any({} > {})", op(VECTOR_0), op(VECTOR_1));
            break;

        case AluVectorOpcode::KillGe:
            print("any({} >= {})", op(VECTOR_0), op(VECTOR_1));
            break;

        case AluVectorOpcode::KillNe:
            print("any({} != {})", op(VECTOR_0), op(VECTOR_1));
            break;

        case AluVectorOpcode::Dst:
            print("dst({}, {})", op(VECTOR_0), op(VECTOR_1));
            break;
        }

        if (instr.vectorSaturate)
            out += ')';

        out += ";\n";
    }

    std::string vectorCode = out.substr(vectorCodeStart);
    out.resize(vectorCodeStart);

    if (instr.scalarOpcode != AluScalarOpcode::RetainPrev)
    {
        if (instr.scalarOpcode >= AluScalarOpcode::SetpEq && instr.scalarOpcode <= AluScalarOpcode::SetpRstr)
        {
            indent();
            out += "p0 = ";

            switch (instr.scalarOpcode)
            {
            case AluScalarOpcode::SetpEq:
                print("{} == 0.0", op(SCALAR_0));
                break;

            case AluScalarOpcode::SetpNe:
                print("{} != 0.0", op(SCALAR_0));
                break;

            case AluScalarOpcode::SetpGt:
                print("{} > 0.0", op(SCALAR_0));
                break;

            case AluScalarOpcode::SetpGe:
                print("{} >= 0.0", op(SCALAR_0));
                break;

            case AluScalarOpcode::SetpInv:
                print("{} == 1.0", op(SCALAR_0));
                break;

            case AluScalarOpcode::SetpPop:
                print("{} - 1.0 <= 0.0", op(SCALAR_0));
                break;

            case AluScalarOpcode::SetpClr:
                out += "false";
                break;

            case AluScalarOpcode::SetpRstr:
                print("{} == 0.0", op(SCALAR_0));
                break;
            }

            out += ";\n";
        }

        indent();
        out += "ps = ";
        if (instr.scalarSaturate)
            out += "saturate(";

        switch (instr.scalarOpcode)
        {
        case AluScalarOpcode::Adds:
            print("{} + {}", op(SCALAR_0), op(SCALAR_1));
            break;

        case AluScalarOpcode::AddsPrev:
            print("{} + ps", op(SCALAR_0));
            break;

        case AluScalarOpcode::Muls:
            print("{} * {}", op(SCALAR_0), op(SCALAR_1));
            break;

        case AluScalarOpcode::MulsPrev:
        case AluScalarOpcode::MulsPrev2:
            print("{} * ps", op(SCALAR_0));
            break;

        case AluScalarOpcode::Maxs:
        case AluScalarOpcode::MaxAs:
        case AluScalarOpcode::MaxAsf:
            print("max({}, {})", op(SCALAR_0), op(SCALAR_1));
            break;

        case AluScalarOpcode::Mins:
            print("min({}, {})", op(SCALAR_0), op(SCALAR_1));
            break;

        case AluScalarOpcode::Seqs:
            print("{} == 0.0", op(SCALAR_0));
            break;

        case AluScalarOpcode::Sgts:
            print("{} > 0.0", op(SCALAR_0));
            break;

        case AluScalarOpcode::Sges:
            print("{} >= 0.0", op(SCALAR_0));
            break;

        case AluScalarOpcode::Snes:
            print("{} != 0.0", op(SCALAR_0));
            break;

        case AluScalarOpcode::Frcs:
            print("frac({})", op(SCALAR_0));
            break;

        case AluScalarOpcode::Truncs:
            print("trunc({})", op(SCALAR_0));
            break;

        case AluScalarOpcode::Floors:
            print("floor({})", op(SCALAR_0));
            break;

        case AluScalarOpcode::Exp:
            print("exp2({})", op(SCALAR_0));
            break;

        case AluScalarOpcode::Logc:
        case AluScalarOpcode::Log:
            print("clamp(log2({}), FLT_MIN, FLT_MAX)", op(SCALAR_0));
            break;

        case AluScalarOpcode::Rcpc:
        case AluScalarOpcode::Rcpf:
        case AluScalarOpcode::Rcp:
            print("clamp(xeRcp({}), FLT_MIN, FLT_MAX)", op(SCALAR_0));
            break;

        case AluScalarOpcode::Rsqc:
        case AluScalarOpcode::Rsqf:
        case AluScalarOpcode::Rsq:
            print("clamp(rsqrt({}), FLT_MIN, FLT_MAX)", op(SCALAR_0));
            break;

        case AluScalarOpcode::Subs:
            print("{} - {}", op(SCALAR_0), op(SCALAR_1));
            break;

        case AluScalarOpcode::SubsPrev:
            print("{} - ps", op(SCALAR_0));
            break;

        case AluScalarOpcode::SetpEq:
        case AluScalarOpcode::SetpNe:
        case AluScalarOpcode::SetpGt:
        case AluScalarOpcode::SetpGe:
            out += "p0 ? 0.0 : 1.0";
            break;

        case AluScalarOpcode::SetpInv:
            // AC6: src == 1 (the "if" was skipped by one level) enters the "else" with a zero predicate
            // count. Upstream leaves it at 1, which skips every predicate push nested in the "else".
            print("p0 ? 0.0 : ({0} == 0.0 ? 1.0 : {0})", op(SCALAR_0));
            break;

        case AluScalarOpcode::SetpPop:
            print("p0 ? 0.0 : ({} - 1.0)", op(SCALAR_0));
            break;

        case AluScalarOpcode::SetpClr:
            out += "FLT_MAX";
            break;

        case AluScalarOpcode::SetpRstr:
            print("p0 ? 0.0 : {}", op(SCALAR_0));
            break;

        case AluScalarOpcode::KillsEq:
            print("{} == 0.0", op(SCALAR_0));
            break;

        case AluScalarOpcode::KillsGt:
            print("{} > 0.0", op(SCALAR_0));
            break;

        case AluScalarOpcode::KillsGe:
            print("{} >= 0.0", op(SCALAR_0));
            break;

        case AluScalarOpcode::KillsNe:
            print("{} != 0.0", op(SCALAR_0));
            break;

        case AluScalarOpcode::KillsOne:
            print("{} == 1.0", op(SCALAR_0));
            break;

        case AluScalarOpcode::Sqrt:
            print("sqrt({})", op(SCALAR_0));
            break;

        case AluScalarOpcode::Mulsc0:
        case AluScalarOpcode::Mulsc1:
            print("{} * {}", op(SCALAR_CONSTANT_0), op(SCALAR_CONSTANT_1));
            break;

        case AluScalarOpcode::Addsc0:
        case AluScalarOpcode::Addsc1:
            print("{} + {}", op(SCALAR_CONSTANT_0), op(SCALAR_CONSTANT_1));
            break;

        case AluScalarOpcode::Subsc0:
        case AluScalarOpcode::Subsc1:
            print("{} - {}", op(SCALAR_CONSTANT_0), op(SCALAR_CONSTANT_1));
            break;

        case AluScalarOpcode::Sin:
            print("sin({})", op(SCALAR_0));
            break;

        case AluScalarOpcode::Cos:
            print("cos({})", op(SCALAR_0));
            break;
        }

        if (instr.scalarSaturate)
            out += ')';

        out += ";\n";

        switch (instr.scalarOpcode)
        {
        case AluScalarOpcode::MaxAs:
            indent();
            println("a0 = (int)clamp(floor({} + 0.5), -256.0, 255.0);", op(SCALAR_0));
            break;     
        case AluScalarOpcode::MaxAsf:
            indent();
            println("a0 = (int)clamp(floor({}), -256.0, 255.0);", op(SCALAR_0));
            break;
        }
    }

    out += vectorCode;

    uint32_t scalarWriteMask = instr.scalarWriteMask;
    if (instr.exportData)
        scalarWriteMask &= ~instr.vectorWriteMask;

    if (scalarWriteMask != 0)
    {
        indent();
        if (!exportRegister.empty())
        {
            out += exportRegister;
            out += '.';
        }
        else
        {
            print("r{}.", instr.scalarDest);
        }

        for (size_t i = 0; i < 4; i++)
        {
            if ((scalarWriteMask >> i) & 0x1)
                out += SWIZZLES[i];
        }

        out += " = ps;\n";
    }

    if (instr.exportData)
    {
        uint32_t zeroMask = instr.scalarDestRelative ? (0b1111 & ~(instr.vectorWriteMask | instr.scalarWriteMask)) : 0;
        uint32_t oneMask = instr.vectorWriteMask & instr.scalarWriteMask;

        for (size_t i = 0; i < 4; i++)
        {
            uint32_t mask = 1 << i;
            if (zeroMask & mask)
            {
                indent();
                println("{}.{} = 0.0;", exportRegister, SWIZZLES[i]);
            }
            else if (oneMask & mask)
            {
                indent();
                println("{}.{} = 1.0;", exportRegister, SWIZZLES[i]);
            }
        }
    }

    if (instr.scalarOpcode >= AluScalarOpcode::KillsEq && instr.scalarOpcode <= AluScalarOpcode::KillsOne)
    {
        indent();
        out += maskedExecution ? "if (!xeInactive) clip(ps != 0.0 ? -1 : 1);\n" : "clip(ps != 0.0 ? -1 : 1);\n";
    }

    if (closeIfBracket)
    {
        --indentation;
        indent();
        out += "}\n";
    }

    if (instr.isPredicated)
    {
        --indentation;
        indent();
        out += "}\n";
    }
}

void ShaderRecompiler::recompile(const uint8_t* shaderData, const std::string_view& include, uint32_t variant)
{
    const auto shaderContainer = reinterpret_cast<const ShaderContainer*>(shaderData);

    assert((shaderContainer->flags & 0xFFFFFF00) == 0x102A1100);
    assert(shaderContainer->constantTableOffset != NULL);

    out += include;
    out += '\n';

    isPixelShader = (shaderContainer->flags & 0x1) == 0;

    const auto constantTableContainer = reinterpret_cast<const ConstantTableContainer*>(shaderData + shaderContainer->constantTableOffset);
    constantTableData = reinterpret_cast<const uint8_t*>(&constantTableContainer->constantTable);

    out += "#ifdef __spirv__\n\n";

#ifdef UNLEASHED_RECOMP
    bool isMetaInstancer = false;
    bool hasIndexCount = false;
#endif

    for (uint32_t i = 0; i < constantTableContainer->constantTable.constants; i++)
    {
        const auto constantInfo = reinterpret_cast<const ConstantInfo*>(
            constantTableData + constantTableContainer->constantTable.constantInfo + i * sizeof(ConstantInfo));

        const char* constantName = reinterpret_cast<const char*>(constantTableData + constantInfo->name);

    #ifdef UNLEASHED_RECOMP
        if (!isPixelShader)
        {
            if (strcmp(constantName, "g_MtxProjection") == 0)
                hasMtxProjection = true;
            else if (strcmp(constantName, "g_InstanceTypes") == 0)
                isMetaInstancer = true;
            else if (strcmp(constantName, "g_IndexCount") == 0)
                hasIndexCount = true;
        }
        else
        {
            if (strcmp(constantName, "g_MtxPrevInvViewProjection") == 0)
                hasMtxPrevInvViewProjection = true;
        }
    #endif

        switch (constantInfo->registerSet)
        {
        case RegisterSet::Float4:
        {
            const char* shaderName = isPixelShader ? "Pixel" : "Vertex";

            if (constantInfo->registerCount > 1)
            {
                uint32_t tailCount = (isPixelShader ? 224 : 256) - constantInfo->registerIndex;

                println("#define {}(INDEX) select((INDEX) < {}, vk::RawBufferLoad<float4>(g_PushConstants.{}ShaderConstants + ({} + min(INDEX, {})) * 16, 0x10), 0.0)",
                    constantName, tailCount, shaderName, constantInfo->registerIndex.get(), tailCount - 1);
            }
            else
            {
                println("#define {} vk::RawBufferLoad<float4>(g_PushConstants.{}ShaderConstants + {}, 0x10)",
                    constantName, shaderName, constantInfo->registerIndex * 16);
            }
            
            for (uint16_t j = 0; j < constantInfo->registerCount; j++)
                float4Constants.emplace(constantInfo->registerIndex + j, constantInfo);

            break;
        }

        case RegisterSet::Sampler:
        {
            // AC6: vertex shader samplers are hardware fetch constants 16+ (the shader fetches from 16 +
            // its register index), so the shared constants hold 32 slots per kind (128 bytes each).
            const uint32_t slot = constantInfo->registerIndex + (isPixelShader ? 0 : 16);
            for (size_t j = 0; j < std::size(TEXTURE_DIMENSIONS); j++)
            {
                println("#define {}_Texture{}DescriptorIndex vk::RawBufferLoad<uint>(g_PushConstants.SharedConstants + {})",
                    constantName, TEXTURE_DIMENSIONS[j], j * 128 + slot * 4);
            }

            println("#define {}_SamplerDescriptorIndex vk::RawBufferLoad<uint>(g_PushConstants.SharedConstants + {})",
                constantName, std::size(TEXTURE_DIMENSIONS) * 128 + slot * 4);

            samplers.emplace(slot, constantName);
            break;
        }

        }
    }

    out += "\n#else\n\n";

    println("cbuffer {}ShaderConstants : register(b{}, space4)", isPixelShader ? "Pixel" : "Vertex", isPixelShader ? 1 : 0);
    out += "{\n";

    for (uint32_t i = 0; i < constantTableContainer->constantTable.constants; i++)
    {
        const auto constantInfo = reinterpret_cast<const ConstantInfo*>(
            constantTableData + constantTableContainer->constantTable.constantInfo + i * sizeof(ConstantInfo));

        if (constantInfo->registerSet == RegisterSet::Float4)
        {
            const char* constantName = reinterpret_cast<const char*>(constantTableData + constantInfo->name);

            print("\tfloat4 {}", constantName);

            if (constantInfo->registerCount > 1)
                print("[{}]", constantInfo->registerCount.get());

            println(" : packoffset(c{});", constantInfo->registerIndex.get());

            if (constantInfo->registerCount > 1)
            {
                uint32_t tailCount = (isPixelShader ? 224 : 256) - constantInfo->registerIndex;
                println("#define {0}(INDEX) select((INDEX) < {1}, {0}[min(INDEX, {2})], 0.0)", constantName, tailCount, tailCount - 1);
            }
        }
    }

    out += "};\n\n";

    out += "cbuffer SharedConstants : register(b2, space4)\n";
    out += "{\n";

    for (uint32_t i = 0; i < constantTableContainer->constantTable.constants; i++)
    {
        const auto constantInfo = reinterpret_cast<const ConstantInfo*>(
            constantTableData + constantTableContainer->constantTable.constantInfo + i * sizeof(ConstantInfo));

        if (constantInfo->registerSet == RegisterSet::Sampler)
        {
            const char* constantName = reinterpret_cast<const char*>(constantTableData + constantInfo->name);

            for (size_t j = 0; j < std::size(TEXTURE_DIMENSIONS); j++)
            {
                println("\tuint {}_Texture{}DescriptorIndex : packoffset(c{}.{});",
                    constantName, TEXTURE_DIMENSIONS[j], j * 4 + constantInfo->registerIndex / 4, SWIZZLES[constantInfo->registerIndex % 4]);
            }

            println("\tuint {}_SamplerDescriptorIndex : packoffset(c{}.{});",
                constantName, 4 * std::size(TEXTURE_DIMENSIONS) + constantInfo->registerIndex / 4, SWIZZLES[constantInfo->registerIndex % 4]);
        }
    }

    out += "\tDEFINE_SHARED_CONSTANTS();\n";
    out += "};\n\n";

    out += "#endif\n";

    for (uint32_t i = 0; i < constantTableContainer->constantTable.constants; i++)
    {
        const auto constantInfo = reinterpret_cast<const ConstantInfo*>(
            constantTableData + constantTableContainer->constantTable.constantInfo + i * sizeof(ConstantInfo));

        if (constantInfo->registerSet == RegisterSet::Bool)
        {
            const char* constantName = reinterpret_cast<const char*>(constantTableData + constantInfo->name);
            println("\t#define {} (1 << {})", constantName, constantInfo->registerIndex + (isPixelShader ? 16 : 0));
            boolConstants.emplace(constantInfo->registerIndex, constantName);
        }
    }

    out += '\n';

    const uint32_t shaderRecordOffset = variant == 0 ? uint32_t(shaderContainer->shaderOffset)
        : uint32_t(reinterpret_cast<const be<uint32_t>*>(shaderData + 0x18)[variant * 2]);
    const auto shader = reinterpret_cast<const Shader*>(shaderData + shaderRecordOffset);

    out += "#ifndef __spirv__\n";

    if (isPixelShader)
        out += "[shader(\"pixel\")]\n";
    else
        out += "[shader(\"vertex\")]\n";

    out += "#endif\n";

    out += "void main(\n";

    if (isPixelShader)
    {
        out += "\tin float4 iPos : SV_Position,\n";

        for (auto& [usage, usageIndex] : INTERPOLATORS)
            println("\tin float4 i{0}{1} : {2}{1},", USAGE_VARIABLES[uint32_t(usage)], usageIndex, USAGE_SEMANTICS[uint32_t(usage)]);
        out += "\tin float4 iPointCoord : TEXCOORD16,\n";   // AC6: point sprite coordinates

        out += "#ifdef __spirv__\n";
        out += "\tin bool iFace : SV_IsFrontFace\n";
        out += "#else\n";
        out += "\tin uint iFace : SV_IsFrontFace\n";
        out += "#endif\n";

        auto pixelShader = reinterpret_cast<const PixelShader*>(shader);
        maskedOutputs.clear();
        if (pixelShader->outputs & PIXEL_SHADER_OUTPUT_COLOR0) maskedOutputs.push_back("oC0");
        if (pixelShader->outputs & PIXEL_SHADER_OUTPUT_COLOR1) maskedOutputs.push_back("oC1");
        if (pixelShader->outputs & PIXEL_SHADER_OUTPUT_COLOR2) maskedOutputs.push_back("oC2");
        if (pixelShader->outputs & PIXEL_SHADER_OUTPUT_COLOR3) maskedOutputs.push_back("oC3");
        if (pixelShader->outputs & PIXEL_SHADER_OUTPUT_DEPTH) maskedOutputs.push_back("oDepth");
        if (pixelShader->outputs & PIXEL_SHADER_OUTPUT_COLOR0)
            out += ",\n\tout float4 oC0 : SV_Target0";
        if (pixelShader->outputs & PIXEL_SHADER_OUTPUT_COLOR1)
            out += ",\n\tout float4 oC1 : SV_Target1";
        if (pixelShader->outputs & PIXEL_SHADER_OUTPUT_COLOR2)
            out += ",\n\tout float4 oC2 : SV_Target2";
        if (pixelShader->outputs & PIXEL_SHADER_OUTPUT_COLOR3)
            out += ",\n\tout float4 oC3 : SV_Target3";
        if (pixelShader->outputs & PIXEL_SHADER_OUTPUT_DEPTH)
            out += ",\n\tout float oDepth : SV_Depth";
    }
    else
    {
        // AC6: inputs are declared after the body has been translated (see the vfetch recompile).
        out += "@@AC6_VERTEX_INPUTS@@";
    #ifndef UNLEASHED_RECOMP
        out += "\tin uint iVertexId : SV_VertexID,\n";
    #endif

    #ifdef UNLEASHED_RECOMP
        if (hasIndexCount)
        {
            out += "\tin uint iVertexId : SV_VertexID,\n";
            out += "\tin uint iInstanceId : SV_InstanceID,\n";
        }
    #endif

        out += "\tout float4 oPos : SV_Position";

        for (auto& [usage, usageIndex] : INTERPOLATORS)
            print(",\n\tout float4 o{0}{1} : {2}{1}", USAGE_VARIABLES[uint32_t(usage)], usageIndex, USAGE_SEMANTICS[uint32_t(usage)]);
        // AC6: point sprite coordinates, after the 16 interpolators in both stages.
        out += ",\n\tout float4 oPointCoord : TEXCOORD16";
    }

    out += ")\n";
    out += "{\n";

#ifdef UNLEASHED_RECOMP
    if (hasMtxProjection)
    {
        specConstantsMask |= SPEC_CONSTANT_REVERSE_Z;

        out += "\toPos = 0.0;\n";

        out += "\tfloat4x4 mtxProjection = float4x4(g_MtxProjection(0), g_MtxProjection(1), g_MtxProjection(2), g_MtxProjection(3));\n";
        out += "\tfloat4x4 mtxProjectionReverseZ = mul(mtxProjection, float4x4(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1, 0, 0, 0, 1, 1));\n";

        out += "\t[unroll] for (int iterationIndex = 0; iterationIndex < 2; iterationIndex++)\n";
        out += "\t{\n";
    }
#endif

    if (shaderContainer->definitionTableOffset != NULL)
    {
        auto definitionTable = reinterpret_cast<const DefinitionTable*>(shaderData + shaderContainer->definitionTableOffset);
        auto definitions = definitionTable->definitions;
        while (*definitions != 0)
        {
            auto definition = reinterpret_cast<const Float4Definition*>(definitions);
            auto value = reinterpret_cast<const be<uint32_t>*>(shaderData + shaderContainer->virtualSize + definition->physicalOffset);
            for (uint16_t i = 0; i < (definition->count + 3) / 4; i++)
            {
                println("\tfloat4 c{} = asfloat(uint4(0x{:X}, 0x{:X}, 0x{:X}, 0x{:X}));",
                    definition->registerIndex + i - (isPixelShader ? 256 : 0), value[0].get(), value[1].get(), value[2].get(), value[3].get());

                value += 4;
            }
            definitions += 2;
        }
        ++definitions;
        while (*definitions != 0)
        {
            auto definition = reinterpret_cast<const Int4Definition*>(definitions);
            for (uint16_t i = 0; i < definition->count; i++)
            {
                union
                {
                    uint32_t value;
                    struct
                    {
                        int8_t x;
                        int8_t y;
                        int8_t z;
                        int8_t w;
                    };
                };

                value = definition->values[i].get();

                println("\tint4 i{} = int4({}, {}, {}, {});",
                    (definition->registerIndex - 8992) / 4 + i, x, y, z, w);
            }
            definitions += 2;
            definitions += definition->count;
        }

        out += "\n";
    }

    bool printedRegisters[32]{};

    uint32_t interpolatorCount = (shader->interpolatorInfo >> 5) & 0x1F;

    for (uint32_t i = 0; i < interpolatorCount; i++)
    {
        union
        {
            Interpolator interpolator;
            uint32_t value;
        };
    
        if (isPixelShader)
        {
            value = reinterpret_cast<const PixelShader*>(shader)->interpolators[i];
            // AC6: at the centroid when the draw's sample control asks for it.
            println("\tfloat4 r{0} = (g_InterpolatorCentroid & (1u << {0})) != 0 ? EvaluateAttributeCentroid(iTexCoord{0}) : iTexCoord{0};",
                uint32_t(interpolator.reg));
            printedRegisters[interpolator.reg] = true;
        }
    }

    // AC6: every interpolator export reaches the pixel stage by its register, declared or not.
    if (!isPixelShader)
        for (uint32_t reg = 0; reg < 16; reg++)
            interpolators.emplace(reg, fmt::format("oTexCoord{}", reg));

    if (!isPixelShader)
    {
    #ifdef UNLEASHED_RECOMP
        if (!hasMtxProjection)
            out += "\toPos = 0.0;\n";
    #endif

        for (auto& [usage, usageIndex] : INTERPOLATORS)
            println("\to{}{} = 0.0;", USAGE_VARIABLES[uint32_t(usage)], usageIndex);
        out += "\tfloat4 oUnused = 0.0;\n";
        out += "\toPointCoord = 0.0;\n";
        out += "\tfloat4 oPts = float4(-1.0, 0.0, 0.0, 0.0);   // point size not written\n";

        out += "\n";
    }

    for (size_t i = 0; i < 32; i++)
    {
        if (!printedRegisters[i])
        {
            print("\tfloat4 r{} = ", i);
            if (isPixelShader && i == ((shader->fieldC >> 8) & 0xFF))
            {
                // AC6: in guest pixels (g_PixelPosScale is 0.5 on 4x MSAA surfaces drawn at twice the size), zw are
                // the point sprite coordinates, zero for other primitives.
                out += "float4((iPos.xy * g_PixelPosScale - 0.5) * float2(iFace ? 1.0 : -1.0, 1.0), iPointCoord.xy);\n";
            }
        #ifdef UNLEASHED_RECOMP
            else if (!isPixelShader && hasIndexCount && i == 0)
            {
                out += "float4(iVertexId + g_IndexCount.x * iInstanceId, 0.0, 0.0, 0.0);\n";
            }
        #else
            else if (!isPixelShader && i == 0)
            {
                // AC6: the GPU starts a vertex shader with the vertex index in r0.x. Point lists are drawn as
                // six vertices per point.
                out += "float4(float(((g_SpecConstants() & SPEC_CONSTANT_POINT_LIST) ? iVertexId / 6 : iVertexId) + g_VertexIndexBase), 0.0, 0.0, 0.0);\n";
            }
        #endif
            else
            {
                out += "0.0;\n";
            }
        }
    }

    out += "\tint a0 = 0;\n";
    out += "\tint aL = 0;\n";
    out += "\tbool p0 = false;\n";
    out += "\tfloat ps = 0.0;\n";
    if (isPixelShader)
    {
#ifdef UNLEASHED_RECOMP
        out += "\tfloat2 pixelCoord = 0.0;\n";
#endif
        out += "\tCubeMapData cubeMapData = (CubeMapData)0;\n";
    }

    const be<uint32_t>* code = reinterpret_cast<const be<uint32_t>*>(shaderData + shaderContainer->virtualSize + shader->physicalOffset);

    union
    {
        ControlFlowInstruction controlFlow[2];
        struct
        {
            uint32_t code0;
            uint32_t code1;
            uint32_t code2;
            uint32_t code3;
        };
    };

    auto controlFlowCode = code;
    uint32_t instrAddress = 0;
    uint32_t instrSize = shader->size;
    bool simpleControlFlow = true;

    while (instrAddress < instrSize)
    {
        code0 = controlFlowCode[0];
        code1 = controlFlowCode[1] & 0xFFFF;
        code2 = (controlFlowCode[1] >> 16) | (controlFlowCode[2] << 16);
        code3 = controlFlowCode[2] >> 16;

        for (auto& cfInstr : controlFlow)
        {
            uint32_t address = 0;

            switch (cfInstr.opcode)
            {
            case ControlFlowOpcode::Exec:
            case ControlFlowOpcode::ExecEnd:
                address = cfInstr.exec.address;
                break;

            case ControlFlowOpcode::CondExec:
            case ControlFlowOpcode::CondExecEnd:
            case ControlFlowOpcode::CondExecPredClean:
            case ControlFlowOpcode::CondExecPredCleanEnd:
                address = cfInstr.condExec.address;
                break;

            case ControlFlowOpcode::CondExecPred:
            case ControlFlowOpcode::CondExecPredEnd:
                address = cfInstr.condExecPred.address;
                break;

            case ControlFlowOpcode::CondJmp:
            {
                // AC6: conditional jumps go through the pc-switch form. Structured ifs can't represent the
                // compiler's if/else (a jump over the if body, then one over the else body, whose ranges
                // cross): the else body would end up nested inside the if body and never run.
                simpleControlFlow = false;
                break;
            }
            }

            if (address != 0)
                instrSize = std::min<uint32_t>(instrSize, address * 12);
        }

        controlFlowCode += 3;
        instrAddress += 12;
    }

    if (simpleControlFlow)
    {
        out += '\n';
        indentation = 1;
    }
    else
    {
        out += "\n\tuint pc = 0;\n";
        for (uint32_t id = 0; id < 32; id++)
            println("\tint loopCount{} = 0;", id);
        // AC6: pixel shaders execute jumped-over code masked, as on the console, instead of branching
        // per pixel: a lane that takes a forward jump to case T sets skip = T and
        // stays in the quad, inactive. Each exec block saves the state it may write and restores it for
        // inactive lanes. Texture derivatives stay defined across the jump's seam.
        maskedExecution = isPixelShader && allowMaskedExecution;
        if (maskedExecution)
        {
            out += "\tuint skip = 0;\n";
            out += "\tbool xeInactive = false;\n";
            for (uint32_t i = 0; i < 32; i++)
                println("\tfloat4 save_r{} = 0.0;", i);
            out += "\tint save_a0 = 0;\n\tbool save_p0 = false;\n\tfloat save_ps = 0.0;\n";
            for (const char* o : maskedOutputs)
                println("\t{0} = 0.0;\n\t{1} save_{0} = 0.0;", o, std::string_view(o) == "oDepth" ? "float" : "float4");
        }
        out += "\twhile (true)\n";
        out += "\t{\n";
        out += "\t\tswitch (pc)\n";
        out += "\t\t{\n";
    }

    controlFlowCode = code;
    instrAddress = 0;
    uint32_t pc = 0;

    while (instrAddress < instrSize)
    {
        code0 = controlFlowCode[0];
        code1 = controlFlowCode[1] & 0xFFFF;
        code2 = (controlFlowCode[1] >> 16) | (controlFlowCode[2] << 16);
        code3 = controlFlowCode[2] >> 16;

        for (auto& cfInstr : controlFlow)
        {
            if (!simpleControlFlow)
            {
                indentation = 3;
                println("\t\tcase {}:", pc);
                if (maskedExecution && pc != 0)
                {
                    println("\t\t\tif (skip == {}) skip = 0;", pc);
                    out += "\t\t\txeInactive = skip != 0;\n";
                }
            }
            else
            {
                auto findResult = ifEndLabels.find(pc);
                if (findResult != ifEndLabels.end())
                {
                    for (uint32_t i = 0; i < findResult->second; i++)
                    {
                        --indentation;
                        indent();
                        out += "}\n";
                    }
                }
            }

            ++pc;

            uint32_t address = 0;
            uint32_t count = 0;
            uint32_t sequence = 0;
            bool shouldReturn = false;
            bool shouldCloseCurlyBracket = false;

            switch (cfInstr.opcode)
            {
            case ControlFlowOpcode::Exec:
            case ControlFlowOpcode::ExecEnd:
                address = cfInstr.exec.address;
                count = cfInstr.exec.count;
                sequence = cfInstr.exec.sequence;
                shouldReturn = (cfInstr.opcode == ControlFlowOpcode::ExecEnd);
                break;

            case ControlFlowOpcode::CondExec:
            case ControlFlowOpcode::CondExecEnd:
            case ControlFlowOpcode::CondExecPredClean:
            case ControlFlowOpcode::CondExecPredCleanEnd:
                address = cfInstr.condExec.address;
                count = cfInstr.condExec.count;
                sequence = cfInstr.condExec.sequence;
                shouldReturn = (cfInstr.opcode == ControlFlowOpcode::CondExecEnd || cfInstr.opcode == ControlFlowOpcode::CondExecPredCleanEnd);   // AC6: upstream tests CondExecEnd twice
                break;

            case ControlFlowOpcode::CondExecPred:
            case ControlFlowOpcode::CondExecPredEnd:
                address = cfInstr.condExecPred.address;
                count = cfInstr.condExecPred.count;
                sequence = cfInstr.condExecPred.sequence;
                shouldReturn = (cfInstr.opcode == ControlFlowOpcode::CondExecPredEnd);
                break;

            // AC6 diagnostics: allocation type 3 is memory export (shader writes to memory), which
            // isn't translated.
            case ControlFlowOpcode::Alloc:
                if (cfInstr.alloc.allocType == 3)
                {
                    usesMemExport = true;
                    indent();
                    out += "// MEMEXPORT alloc (not translated)\n";
                }
                break;

            case ControlFlowOpcode::LoopStart:
                if (simpleControlFlow)
                {
                    indent();
                #ifdef UNLEASHED_RECOMP
                    print("[unroll] ");
                #endif
                    // AC6: loop constants are (count, start, step) as in D3D9.
                    println("aL = i{}.y;", uint32_t(cfInstr.loopStart.loopId));
                    indent();
                    println("for (int loop{0} = 0; loop{0} < i{0}.x; loop{0}++, aL += i{0}.z)", uint32_t(cfInstr.loopStart.loopId));
                    indent();
                    out += "{\n";
                    ++indentation;
                }
                else
                {
                    // AC6: count and start from the loop constant. An empty loop skips to its end.
                    println("\t\t\tloopCount{0} = i{0}.x;", uint32_t(cfInstr.loopStart.loopId));
                    println("\t\t\taL = i{}.y;", uint32_t(cfInstr.loopStart.loopId));
                    println("\t\t\tif (loopCount{} <= 0)", uint32_t(cfInstr.loopStart.loopId));
                    out += "\t\t\t{\n";
                    println("\t\t\t\tpc = {};", uint32_t(cfInstr.loopStart.address));
                    out += "\t\t\t\tcontinue;\n";
                    out += "\t\t\t}\n";
                }
                break;

            case ControlFlowOpcode::LoopEnd:
                if (simpleControlFlow)
                {
                    // AC6: a predicated break leaves the loop when p0 matches the condition.
                    if (cfInstr.loopEnd.isPredicatedBreak)
                    {
                        indent();
                        println("if ({}p0) break;", cfInstr.loopEnd.condition ? "" : "!");
                    }
                    --indentation;
                    indent();
                    out += "}\n";
                }
                else
                {
                    if (cfInstr.loopEnd.isPredicatedBreak)
                        println("\t\t\tif ({}p0) loopCount{} = 0;", cfInstr.loopEnd.condition ? "" : "!", uint32_t(cfInstr.loopEnd.loopId));
                    println("\t\t\taL += i{}.z;", uint32_t(cfInstr.loopEnd.loopId));
                    println("\t\t\tif (--loopCount{} > 0)", uint32_t(cfInstr.loopEnd.loopId));
                    out += "\t\t\t{\n";
                    println("\t\t\t\tpc = {};", uint32_t(cfInstr.loopEnd.address));
                    out += "\t\t\t\tcontinue;\n";
                    out += "\t\t\t}\n";
                }
                break;

            case ControlFlowOpcode::CondJmp:
            {
                if (maskedExecution && uint32_t(cfInstr.condJmp.address) > pc - 1)
                {
                    // Forward: active lanes that take it go inactive until the target.
                    std::string condition;
                    if (cfInstr.condJmp.isUnconditional)
                        condition = "";
                    else if (cfInstr.condJmp.isPredicated)
                        condition = fmt::format(" && {}p0", cfInstr.condJmp.condition ? "" : "!");
                    else if (auto b = boolConstants.find(cfInstr.condJmp.boolAddress); b != boolConstants.end())
                        condition = fmt::format(" && (g_Booleans & {}) {}= 0", b->second, cfInstr.condJmp.condition ? "!" : "=");
                    else
                        condition = fmt::format(" && b{} {}= 0", uint32_t(cfInstr.condJmp.boolAddress), cfInstr.condJmp.condition ? "!" : "=");
                    println("\t\t\tif (skip == 0{}) skip = {};", condition, uint32_t(cfInstr.condJmp.address));
                    break;
                }
                if (cfInstr.condJmp.isUnconditional)
                {
                    assert(!simpleControlFlow);
                    println("\t\t\tpc = {};", uint32_t(cfInstr.condJmp.address));
                    out += "\t\t\tcontinue;\n";
                }
                else
                {
                    indent();
                    if (cfInstr.condJmp.isPredicated)
                    {
                        println("if ({}p0)", cfInstr.condJmp.condition ^ simpleControlFlow ? "" : "!");
                    }
                    else
                    {
                        auto findResult = boolConstants.find(cfInstr.condJmp.boolAddress);
                        if (findResult != boolConstants.end())
                            println("if ((g_Booleans & {}) {}= 0)", findResult->second, cfInstr.condJmp.condition ^ simpleControlFlow ? "!" : "=");
                        else
                            println("if (b{} {}= 0)", uint32_t(cfInstr.condJmp.boolAddress), cfInstr.condJmp.condition ^ simpleControlFlow ? "!" : "=");
                    }

                    if (simpleControlFlow)
                    {
                        indent();
                        out += "{\n";
                        ++indentation;
                    }
                    else
                    {
                        out += "\t\t\t{\n";
                        println("\t\t\t\tpc = {};", uint32_t(cfInstr.condJmp.address));
                        out += "\t\t\t\tcontinue;\n";
                        out += "\t\t\t}\n";
                    }
                }
                break;
            }
            }

            auto instructionCode = code + address * 3;

            if (maskedExecution && count > 0)
            {
                std::string save = "\t\t\t";
                for (uint32_t r = 0; r < 32; r++)
                    save += fmt::format("save_r{0} = r{0}; ", r);
                save += "save_a0 = a0; save_p0 = p0; save_ps = ps;";
                for (const char* o : maskedOutputs)
                    save += fmt::format(" save_{0} = {0};", o);
                out += save + "\n";
            }

            for (uint32_t i = 0; i < count; i++)
            {
                union
                {
                    VertexFetchInstruction vertexFetch;
                    TextureFetchInstruction textureFetch;
                    AluInstruction alu;
                    struct
                    {
                        uint32_t code0;
                        uint32_t code1;
                        uint32_t code2;
                    };
                };
            
                code0 = instructionCode[0];
                code1 = instructionCode[1];
                code2 = instructionCode[2];

                // AC6: a predicated exec block ((p0) exec) runs its instructions under the block's predicate.
                // AC6's compiler predicates nearly every instruction inside such a block
                // itself, with the same condition. The few it doesn't would otherwise run unconditionally.
                if (cfInstr.opcode == ControlFlowOpcode::CondExecPred || cfInstr.opcode == ControlFlowOpcode::CondExecPredEnd)
                {
                    const uint32_t condition = cfInstr.condExecPred.condition;
                    if ((sequence & 0x1) != 0)
                    {
                        if (!(code1 >> 31))
                            code1 |= 1u << 31, code2 = (code2 & 0x7FFFFFFFu) | condition << 31;
                    }
                    else if (!((code1 >> 28) & 1))
                    {
                        code1 |= 1u << 28;
                        code1 = (code1 & ~(1u << 27)) | condition << 27;
                    }
                }
            
                if ((sequence & 0x1) != 0)
                {
                    if (vertexFetch.opcode == FetchOpcode::VertexFetch)
                    {
                        recompile(vertexFetch, address + i);
                    }
                    else
                    {
                    #ifdef UNLEASHED_RECOMP
                        if (textureFetch.constIndex == 10) // g_GISampler
                        {
                            specConstantsMask |= SPEC_CONSTANT_BICUBIC_GI_FILTER;

                            indent();
                            out += "if (g_SpecConstants() & SPEC_CONSTANT_BICUBIC_GI_FILTER)";
                            indent();
                            out += '{';

                            ++indentation;
                            recompile(textureFetch, true);
                            --indentation;

                            indent();
                            out += "}";
                            indent();
                            out += "else";
                            indent();
                            out += '{';

                            ++indentation;
                            recompile(textureFetch, false);
                            --indentation;

                            indent();
                            out += '}';
                        }
                        else
                    #endif
                        {
                            recompile(textureFetch, false);
                        }
                    }
                }
                else
                {
                    recompile(alu);
                }
            
                sequence >>= 2;
                instructionCode += 3;
            }

            if (maskedExecution && count > 0)
            {
                std::string restore = "\t\t\tif (xeInactive) { ";
                for (uint32_t r = 0; r < 32; r++)
                    restore += fmt::format("r{0} = save_r{0}; ", r);
                restore += "a0 = save_a0; p0 = save_p0; ps = save_ps;";
                for (const char* o : maskedOutputs)
                    restore += fmt::format(" {0} = save_{0};", o);
                out += restore + " }\n";
            }

            if (shouldReturn && maskedExecution)
                out += "\t\t\tif (!xeInactive)\n\t\t\t{\n";
            if (shouldReturn)
            {
                if (isPixelShader)
                {
                    specConstantsMask |= SPEC_CONSTANT_ALPHA_TEST;

                    indent();
                    out += "[branch] if (g_SpecConstants() & SPEC_CONSTANT_ALPHA_TEST)";
                    indent();
                    out += '{';

                    indent();
                    out += "\tclip(oC0.w - g_AlphaThreshold);\n";

                    indent();
                    out += "}";

                #ifdef UNLEASHED_RECOMP
                    specConstantsMask |= SPEC_CONSTANT_ALPHA_TO_COVERAGE;

                    indent();
                    out += "else if (g_SpecConstants() & SPEC_CONSTANT_ALPHA_TO_COVERAGE)";
                    indent();
                    out += '{';

                    indent();
                    out += "\toC0.w *= 1.0 + computeMipLevel(pixelCoord) * 0.25;\n";
                    indent();
                    out += "\toC0.w = 0.5 + (oC0.w - g_AlphaThreshold) / max(fwidth(oC0.w), 1e-6);\n";

                    indent();
                    out += '}';
                #endif
                }
                else
                {
                #ifdef UNLEASHED_RECOMP
                    if (!hasMtxProjection)
                #endif
                    {
                        out += "\toPos.xy += g_HalfPixelOffset * oPos.w;\n";
                    }
                    // AC6: a point list's point becomes a screen-aligned square: this vertex is
                    // corner iVertexId % 6 of its two triangles, moved by the point's radius in clip space. The
                    // pixel shader gets the corner's point coordinates (left to right, top to bottom).
                    out += "\t[branch] if (g_SpecConstants() & SPEC_CONSTANT_POINT_LIST)\n\t{\n"
                           "\t\tconst uint pointCorner = iVertexId % 6;\n"
                           "\t\tconst float2 pointUv = float2(pointCorner == 1 || pointCorner == 4 || pointCorner == 5 ? 1.0 : 0.0,\n"
                           "\t\t\tpointCorner == 2 || pointCorner == 3 || pointCorner == 5 ? 1.0 : 0.0);\n"
                           "\t\tconst float4 pointParams = g_PointParams;\n"
                           "\t\tconst float2 pointDiameter = oPts.x >= 0.0 ? clamp(oPts.x, pointParams.x, pointParams.y).xx : g_PointConstantDiameter;\n"
                           "\t\toPos.xy += (pointUv * 2.0 - 1.0) * float2(1.0, -1.0) * pointDiameter * pointParams.zw * oPos.w;\n"
                           "\t\toPointCoord = float4(pointUv, 0.0, 0.0);\n"
                           "\t}\n";
                }

                if (simpleControlFlow)
                {
                    indent();
                #ifdef UNLEASHED_RECOMP
                    if (hasMtxProjection)
                    {
                        out += "continue;\n";
                    }
                    else
                #endif
                    {
                        out += "return;\n";
                    }
                }
                else
                {
                    out += "\t\t\tbreak;\n";
                }
            }
            if (shouldReturn && maskedExecution)
                out += "\t\t\t}\n";

            if (shouldCloseCurlyBracket)
            {
                --indentation;
                indent();
                out += "}\n";
            }
        }

        controlFlowCode += 3;
        instrAddress += 12;
    }

    if (!simpleControlFlow)
    {
        out += "\t\t\tbreak;\n";
        out += "\t\t}\n";
        out += "\t\tbreak;\n";
        out += "\t}\n";
    }

#ifdef UNLEASHED_RECOMP
    if (hasMtxProjection)
        out += "\t}\n";

    if (!isPixelShader && hasMtxProjection)
        out += "\toPos.xy += g_HalfPixelOffset * oPos.w;\n";
#endif

    out += "}";

    if (!isPixelShader)
    {
        std::string inputs;
        for (const VertexInput& input : vertexInputs)
        {
            const bool packed = input.format == 7 || input.format == 16 || input.format == 17;
            const char* type = packed ? "uint4" : input.isInteger ? (input.isSigned ? "int4" : "uint4") : "float4";
            inputs += fmt::format("\t[[vk::location({0})]] in {1} iFetch{0} : TEXCOORD{0}, // fetch {2} offset {3} format {4}\n",
                input.location, type, input.fetchSlot, input.offset, input.format);
        }
        const size_t at = out.find("@@AC6_VERTEX_INPUTS@@");
        if (at != std::string::npos)
            out.replace(at, sizeof("@@AC6_VERTEX_INPUTS@@") - 1, inputs);
    }
}

} // namespace xenos
