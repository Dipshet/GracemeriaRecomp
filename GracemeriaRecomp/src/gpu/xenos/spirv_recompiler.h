#pragma once
// Xenos microcode -> SPIR-V, without HLSL or a compiler in between. It translates what ShaderRecompiler (the HLSL
// backend, kept for comparison and AC6_SHADER_PATCH) translates, the same way, plus the renderer's own changes (the
// options below, the draw flags, TAA's motion vectors). The module has the HLSL backend's interface: descriptor sets
// 0-3 (2D/3D/cube texture heaps, samplers), 24 bytes of push constants (vertex, pixel and shared constant buffer
// addresses), specialization constant 0, vertex inputs at their VertexInput locations, interpolators at locations 0-15
// and the point sprite coordinates at 16. On top of it: the constants as uniform buffers at set 4 (unless
// AC6_UNIFORM_CONSTANTS=0), the clip position for the motion at location 17 and the motion at colour output 1.

#include "shader_recompiler.h"
#include "spirv_builder.h"

#include <array>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace xenos {

struct SpirvRecompiler
{
    // Results, as ShaderRecompiler reports them.
    bool isPixelShader = false;
    uint32_t specConstantsMask = 0;
    bool usesMemExport = false;
    uint32_t indexedFetchStreams = 0;
    uint32_t textureSlots = 0;        // fetch slots the texture fetches read (diagnostics: not in the shader cache)
    // The float constant registers the shader reads: [constantFirst, constantEnd) (empty if none).
    uint32_t constantFirst = UINT32_MAX, constantEnd = 0;
    std::vector<uint32_t> samplerOverrides;
    std::vector<VertexInput> vertexInputs;
    bool allowMaskedExecution = true;
    uint32_t identityFetchSlots = 0;   // 2D fetches sampling 1:1 at the pixel instead (the game's EDRAM un-swizzles)
    // The depth of field above 1x: 2D fetches of a rendered image snapped to its guest texel's centre,
    // or averaged over 4 samples at -0.75, -0.25, 0.25 and 0.75 guest texels along x / y.
    uint32_t guestTexelSnapSlots = 0, densifyXSlots = 0, densifyYSlots = 0;
    // Screen blurs on wide outputs: 2D fetches whose texel offsets are in guest texels at the vertical
    // scale on both axes (square on screen: a guest texel of a rendered image is k times wider than tall there).
    uint32_t squareOffsetSlots = 0;
    // The map's atlas pages: 2D fetches of these slots can be filtered by the shader itself, the
    // taps of an anisotropic footprint kept inside the pixel's own tile (TileFiltered), when the draw asks for it.
    uint32_t tileAtlasSlots = 0;
    // HDR from the scene: in the game's grade pass a saturate of a colour (a vector write of 3 or more
    // components) keeps values above 1 (max(x, 0)). Others (its vignette factor) stay clamped.
    bool rangeSaturate = false;
    // The game's lighting in float: the saturate feeding a colour's scale
    // to the game's code range (a vector multiply by the literal 240/255) keeps values above 1 (max(x, 0)), and so does a
    // saturated colour export. The decoders' emitter test (the literal 241/255) is never taken.
    bool sceneFloat = false;
    bool liftWhiteMin = true;   // sceneFloat: minimums against 240/255 lifted too (AC6_SCENE_FLOAT_MIN=0: kept)
    // The sun's shaders: their colour 0 written decoded when the draw asks (SPEC_CONSTANT_EMITTER_DECODE).
    bool emitterDecode = false;
    // The sea's reflection of the scene: the map's pixel shaders blend it into what they fetch from
    // their sky cube (SeaReflection), where the draw gives one.
    bool seaReflection = false;
    // The game's light inputs: register -> kind (shaders.cpp NoteLightConstants). A static read of
    // one is multiplied by the shared block's relighting factor of its kind (x, y, z. Kind 8, the light cap: x only).
    std::unordered_map<uint32_t, uint8_t> lightRegisters;
    bool uniformConstants = true;      // constants from uniform buffers, else buffer addresses
    std::vector<uint32_t> spirv;
    std::string warnings;   // things the HLSL backend would have failed to compile (translated as zero)

    void recompile(const uint8_t* shaderData, uint32_t variant = 0);
    size_t unclampedCount() const { return unclampAlu.size() + unminAlu.size(); }

    struct Value
    {
        uint32_t id = 0;
        uint32_t n = 0;   // float components
    };

private:
    SpirvBuilder b;
    const uint8_t* constantTableData = nullptr;
    std::unordered_map<uint32_t, const ConstantInfo*> float4Constants;
    std::unordered_map<uint32_t, uint32_t> boolConstants;            // register -> g_Booleans bit
    std::unordered_map<uint32_t, uint32_t> samplerSlots;             // fetch constant -> shared constant slot
    std::unordered_map<uint32_t, std::array<uint32_t, 4>> literalConstants;   // c register -> bits
    std::unordered_map<uint32_t, std::array<int32_t, 4>> intConstants;        // i register
    std::unordered_map<uint64_t, uint32_t> vertexInputIndex;
    std::vector<uint32_t> vertexInputVariables;
    std::vector<uint32_t> interfaceVariables;

    // Module-level objects.
    uint32_t pushConstants = 0, specConstants = 0, function = 0;
    uint32_t constantBuffers[3] = {};
    uint32_t heaps[4] = {};   // 2D, 3D, cube, samplers
    uint32_t vertexIndexInput = 0, fragCoordInput = 0, frontFacingInput = 0, pointCoordInput = 0;
    uint32_t motionInput = 0, motionOutput = 0;   // TAA's motion vectors (pixel shaders)
    uint32_t interpolatorInputs[16] = {};

    // Function variables.
    uint32_t r[32] = {}, a0 = 0, aL = 0, p0 = 0, ps = 0;
    uint32_t oPos = 0, oTexCoord[16] = {}, oUnused = 0, oPointCoord = 0, oPts = 0, oMotionPos = 0;
    uint32_t oC[4] = {}, oDepth = 0;
    uint32_t cubeDirections[2] = {}, cubeIndex = 0;   // the HLSL backend's CubeMapData
    uint32_t pc = 0, skip = 0, xeInactive = 0, loopCount[32] = {};
    uint32_t saveR[32] = {}, saveA0 = 0, saveP0 = 0, savePs = 0, saveOC[4] = {}, saveODepth = 0;

    // Control flow.
    bool simpleControlFlow = true;
    bool maskedExecution = false;
    uint32_t loopHeader = 0, loopMerge = 0, switchMerge = 0, loopContinue = 0;
    struct SimpleLoop { uint32_t header, merge, continueTarget, counter, loopId; };
    std::vector<SimpleLoop> simpleLoops;

    bool r0Written = false;
    std::unordered_set<uint32_t> unclampAlu;   // sceneFloat: the ALU instructions (by address) whose saturate is lifted
    // sceneFloat: the vector minimums against the literal 240/255 (by address), with the operand that is the value.
    std::unordered_map<uint32_t, uint32_t> unminAlu;
    uint32_t aluAddress = 0;                   // the instruction being translated
    bool lastFetchIndexed = false;
    uint32_t lastFetchRegister = 0, lastFetchComponent = 0, lastFetchStride = 0;
    bool lastFetchRounded = false;
    uint32_t outputs = 0;   // pixel shader outputs (PIXEL_SHADER_OUTPUT_*)

    // Helpers.
    uint32_t F(float v) { return b.ConstF32(v); }
    Value Vec4Zero() { return { b.ConstVec(4, 0.0f), 4 }; }
    Value Swizzle(Value v, const std::vector<uint32_t>& components);
    Value SmallerScale2(Value scale);
    Value Extract(Value v, uint32_t component);
    Value Splat(Value scalar, uint32_t n);
    Value Fit(Value v, uint32_t n);   // HLSL's implicit scalar promotion / vector truncation
    Value Arith(spv::Op op, Value a, Value b);
    Value Unary(spv::Op op, Value a);
    Value Glsl(GLSLstd450 inst, std::initializer_list<Value> args);
    uint32_t Compare(spv::Op op, Value a, Value b);   // bool (vector)
    Value BoolToFloat(uint32_t condition, uint32_t n);
    Value Select(uint32_t condition, Value ifTrue, Value ifFalse);
    uint32_t BoolType(uint32_t n) { return b.TypeVector(b.Bool(), n); }
    uint32_t And(uint32_t x, uint32_t y) { return b.Emit(spv::OpLogicalAnd, b.Bool(), { x, y }); }
    uint32_t Or(uint32_t x, uint32_t y) { return b.Emit(spv::OpLogicalOr, b.Bool(), { x, y }); }
    uint32_t Not(uint32_t x) { return b.Emit(spv::OpLogicalNot, b.Bool(), { x }); }
    Value LoadVar(uint32_t var, uint32_t n) { return { b.Load(b.Vec(n), var), n }; }
    void StoreMasked(uint32_t var, uint32_t mask, Value value);
    void StoreScalarVar(uint32_t var, Value value) { b.Store(var, Extract(value, 0).id); }

    // Guest memory reads through buffer addresses.
    uint32_t PushAddress(uint32_t member);
    uint32_t RawLoad(uint32_t type, uint32_t address, uint32_t alignment);
    uint32_t AddressPlus(uint32_t base, uint32_t offsetU64);
    // Constants as uniform buffers: set 4, binding 0 the vertex shader's float constants, 1 the
    // pixel shader's, 2 the shared block.
    uint32_t ConstantBuffer(uint32_t binding);
    uint32_t SharedU32(uint32_t offset);
    // The value with its bits ORed with a shared word that is always 0: unchanged, but the driver's compiler can't
    // know anything about it (its range from a vertex format, say).
    Value Unknowable(Value v);
    Value SharedVec(uint32_t offset, uint32_t n);
    Value ResolutionScale();
    Value TileFiltered(uint32_t image, uint32_t sampler, Value uv, uint32_t key);
    Value SeaReflection(Value cube);
    uint32_t tileCacheVar = 0, tileCacheUseVar = 0, tileCacheKey = ~0u;
    Value ConstantRegister(uint32_t reg, bool relative, bool addressRelative);

    // Textures.
    uint32_t Heap(uint32_t which);
    uint32_t LoadImage(uint32_t which, uint32_t index);
    uint32_t LoadSampler(uint32_t index);

    uint32_t SpecBits(uint32_t mask);   // (g_SpecConstants & mask) != 0
    void KillIf(uint32_t condition);
    void BeginPredicate(bool condition, std::vector<SpirvBuilder::IfScope>& scopes);

    void DeclareInterface();
    void WriteOutputs();
    void WriteMotion();
    void WriteReactive();   // FSR's reactive mask (SPEC_CONSTANT_REACTIVE)
    void EmitterDecode();
    void ExitPixelOrVertex();
    Value XeRcp(Value x);
    Value FetchComponent(uint32_t word, uint32_t shift, uint32_t bits, bool isSigned, bool isInteger);
    Value FetchPacked(uint32_t format, uint32_t word, bool isSigned, bool isInteger);
    uint32_t FetchWord(uint32_t stream, uint32_t index, int32_t offset, uint32_t stride, uint32_t word);

    void recompile(const VertexFetchInstruction& instr);
    void recompile(const TextureFetchInstruction& instr);
    void recompile(const AluInstruction& instr);
};

} // namespace xenos
