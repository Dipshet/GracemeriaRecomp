// Xenos microcode -> SPIR-V. Mirrors shader_recompiler.cpp (the HLSL backend) section by
// section: the same register model (r0-r31, a0, aL, p0, ps as function variables), the same control flow
// (structured, or the pc switch loop when the shader jumps), the same masked execution in pixel shaders, and
// the same operations as DXC lowers the HLSL helpers in shader_common.h (max/min as NMax/NMin, ordered
// comparisons, saturate as FClamp, discard as OpKill). Only the code paths AC6's shaders use matter. The
// HLSL backend's compile failures (undeclared names) are translated as zero and reported in `warnings`.
#include "spirv_recompiler.h"
#include "shader_common.h"

#include <array>
#include <bit>
#include <cmath>
#include <format>

namespace xenos {

namespace {

// Shared constants block (renderer.cpp SharedConstants, shader_common.h).
constexpr uint32_t kSharedSamplerIndices = 384, kSharedBooleans = 512, kSharedHalfPixelOffset = 520, kSharedAlphaThreshold = 528,
    kSharedVertexIndexBase = 532, kSharedStreamAddress = 544, kSharedStreamDwords = 672, kSharedPixelPosScale = 736,
    kSharedOverrideSampler = 740, kSharedPointParams = 804, kSharedPointConstantDiameter = 820, kSharedInterpolatorCentroid = 828,
    kSharedDrawFlags = 832, kSharedResolutionScale = 836, kSharedScaledTextures = 840, kSharedResolutionScaleY = 844,
    kSharedFallbackAddress = 848, kSharedDepthBias = 536, kSharedMotion = 856, kSharedAltScaledTextures = 920, kSharedAltScale = 924,
    kSharedRelight = 932, kSharedTileTaps = 972, kSharedCloudRim = 976, kSharedSeaImage = 980, kSharedSea = 984,
    kSharedZero = 540;   // (SharedConstants::zero: always 0)
// A motion output with no motion known (renderer.cpp kMotionUnknown).
constexpr float kMotionUnknown = 30000.0f;

FetchDestinationSwizzle DestSwizzle(uint32_t dstSwizzle, uint32_t index)
{
    return FetchDestinationSwizzle((dstSwizzle >> (index * 3)) & 0x7);
}

bool IsFetched(FetchDestinationSwizzle s)
{
    return s >= FetchDestinationSwizzle::X && s <= FetchDestinationSwizzle::W;
}

} // namespace

using Value = SpirvRecompiler::Value;

// --- Value helpers -------------------------------------------------------------------------------------

Value SpirvRecompiler::Extract(Value v, uint32_t component)
{
    if (v.n == 1)
        return v;
    if (component >= v.n)
    {
        warnings += std::format("component {} of a {}-component value; ", component, v.n);
        return { F(0.0f), 1 };
    }
    return { b.Emit(spv::OpCompositeExtract, b.F32(), { v.id, component }), 1 };
}

Value SpirvRecompiler::Swizzle(Value v, const std::vector<uint32_t>& components)
{
    if (components.size() == 1)
        return Extract(v, components[0]);
    if (v.n == 1)
    {
        SpirvBuilder::Words parts(components.size(), v.id);
        return { b.Emit(spv::OpCompositeConstruct, b.Vec(uint32_t(components.size())), parts), uint32_t(components.size()) };
    }
    bool identity = components.size() == v.n;
    for (size_t i = 0; i < components.size() && identity; i++)
        identity = components[i] == i;
    if (identity)
        return v;
    SpirvBuilder::Words w = { v.id, v.id };
    for (uint32_t c : components)
    {
        if (c >= v.n)
        {
            warnings += std::format("component {} of a {}-component value; ", c, v.n);
            c = 0;
        }
        w.push_back(c);
    }
    return { b.Emit(spv::OpVectorShuffle, b.Vec(uint32_t(components.size())), w), uint32_t(components.size()) };
}

// The smaller of a per-axis scale's two components, on both (a guest texel's size along the axis the screen
// doesn't extend).
Value SpirvRecompiler::SmallerScale2(Value scale)
{
    return Splat(Glsl(GLSLstd450FMin, { Extract(scale, 0), Extract(scale, 1) }), 2);
}

Value SpirvRecompiler::Splat(Value scalar, uint32_t n)
{
    if (n == 1)
        return Extract(scalar, 0);
    scalar = Extract(scalar, 0);
    return { b.Emit(spv::OpCompositeConstruct, b.Vec(n), SpirvBuilder::Words(n, scalar.id)), n };
}

Value SpirvRecompiler::Fit(Value v, uint32_t n)
{
    if (v.n == n)
        return v;
    if (v.n == 1)
        return Splat(v, n);
    if (v.n > n)
    {
        std::vector<uint32_t> c(n);
        for (uint32_t i = 0; i < n; i++)
            c[i] = i;
        return Swizzle(v, c);
    }
    warnings += std::format("{}-component value widened to {}; ", v.n, n);
    SpirvBuilder::Words parts = { v.id };
    for (uint32_t i = v.n; i < n; i++)
        parts.push_back(F(0.0f));
    return { b.Emit(spv::OpCompositeConstruct, b.Vec(n), parts), n };
}

namespace {
uint32_t CommonWidth(uint32_t a, uint32_t b) { return a == 1 ? b : b == 1 ? a : std::min(a, b); }
}

Value SpirvRecompiler::Arith(spv::Op op, Value x, Value y)
{
    const uint32_t n = CommonWidth(x.n, y.n);
    x = Fit(x, n);
    y = Fit(y, n);
    return { b.Emit(op, b.Vec(n), { x.id, y.id }), n };
}

Value SpirvRecompiler::Unary(spv::Op op, Value x)
{
    return { b.Emit(op, b.Vec(x.n), { x.id }), x.n };
}

Value SpirvRecompiler::Glsl(GLSLstd450 inst, std::initializer_list<Value> args)
{
    uint32_t n = 1;
    for (const Value& v : args)
        n = std::max(n, v.n);
    SpirvBuilder::Words w = { b.Glsl(), uint32_t(inst) };
    for (const Value& v : args)
        w.push_back(Fit(v, n).id);
    return { b.Emit(spv::OpExtInst, b.Vec(n), w), n };
}

uint32_t SpirvRecompiler::Compare(spv::Op op, Value x, Value y)
{
    const uint32_t n = CommonWidth(x.n, y.n);
    x = Fit(x, n);
    y = Fit(y, n);
    return b.Emit(op, BoolType(n), { x.id, y.id });
}

Value SpirvRecompiler::BoolToFloat(uint32_t condition, uint32_t n)
{
    return { b.Emit(spv::OpSelect, b.Vec(n), { condition, b.ConstVec(n, 1.0f), b.ConstVec(n, 0.0f) }), n };
}

Value SpirvRecompiler::Select(uint32_t condition, Value ifTrue, Value ifFalse)
{
    const uint32_t n = std::max(ifTrue.n, ifFalse.n);
    ifTrue = Fit(ifTrue, n);
    ifFalse = Fit(ifFalse, n);
    return { b.Emit(spv::OpSelect, b.Vec(n), { condition, ifTrue.id, ifFalse.id }), n };
}

void SpirvRecompiler::StoreMasked(uint32_t var, uint32_t mask, Value value)
{
    mask &= 0xF;
    const uint32_t count = std::popcount(mask);
    if (count == 0)
        return;
    if (mask == 0xF)
    {
        b.Store(var, Fit(value, 4).id);
        return;
    }
    const uint32_t old = b.Load(b.Vec(4), var);
    if (count == 1)
    {
        const uint32_t component = std::countr_zero(mask);
        b.Store(var, b.Emit(spv::OpCompositeInsert, b.Vec(4), { Extract(Fit(value, 1), 0).id, old, component }));
        return;
    }
    value = Fit(value, count);
    SpirvBuilder::Words w = { old, value.id };
    uint32_t k = 0;
    for (uint32_t i = 0; i < 4; i++)
        w.push_back((mask >> i) & 1 ? 4 + k++ : i);
    b.Store(var, b.Emit(spv::OpVectorShuffle, b.Vec(4), w));
}

// --- Guest memory through buffer addresses ----------------------------------------------------------------

uint32_t SpirvRecompiler::PushAddress(uint32_t member)
{
    const uint32_t pointer = b.Emit(spv::OpAccessChain, b.TypePointer(spv::StorageClassPushConstant, b.U64()), { pushConstants, b.ConstI32(int32_t(member)) });
    return b.Load(b.U64(), pointer);
}

uint32_t SpirvRecompiler::RawLoad(uint32_t type, uint32_t address, uint32_t alignment)
{
    const uint32_t pointer = b.Emit(spv::OpConvertUToPtr, b.TypePointer(spv::StorageClassPhysicalStorageBuffer, type), { address });
    return b.Emit(spv::OpLoad, type, { pointer, uint32_t(spv::MemoryAccessAlignedMask), alignment });
}

uint32_t SpirvRecompiler::AddressPlus(uint32_t base, uint32_t offsetU64)
{
    return b.Emit(spv::OpIAdd, b.U64(), { base, offsetU64 });
}

// --- Constants: uniform buffers --------------------------------------------------------------------------

// The constants as uniform buffers: set 4, binding 0 the vertex shader's float constants (256
// registers), 1 the pixel shader's (224), 2 the shared block (1 KB as uint4s), each a dynamic uniform buffer into the
// upload ring, bound per draw. NVIDIA reads uniform-buffer constants from its constant cache, as instruction operands.
// Raw buffer-address loads go through the memory pipeline (its queue, long-scoreboard waits, 64-bit addresses in
// registers): a large share of the shaders' time at 4K. The push constants carry the blocks'
// addresses for the HLSL backend, and for this one with AC6_UNIFORM_CONSTANTS=0 (uniformConstants false: the loads
// through the addresses).
uint32_t SpirvRecompiler::ConstantBuffer(uint32_t binding)
{
    if (!constantBuffers[binding])
    {
        const uint32_t element = binding == 2 ? b.TypeVector(b.U32(), 4) : b.Vec(4);
        const uint32_t array = b.TypeArray(element, binding == 0 ? 256 : binding == 1 ? 224 : 64);
        b.Decorate(array, spv::DecorationArrayStride, { 16 });
        const uint32_t block = b.TypeStruct({ array });
        b.Decorate(block, spv::DecorationBlock);
        b.MemberDecorate(block, 0, spv::DecorationOffset, { 0 });
        const uint32_t variable = b.GlobalVariable(b.TypePointer(spv::StorageClassUniform, block), spv::StorageClassUniform);
        b.Decorate(variable, spv::DecorationDescriptorSet, { 4 });
        b.Decorate(variable, spv::DecorationBinding, { binding });
        constantBuffers[binding] = variable;
    }
    return constantBuffers[binding];
}

uint32_t SpirvRecompiler::SharedU32(uint32_t offset)
{
    if (!uniformConstants)
        return RawLoad(b.U32(), AddressPlus(PushAddress(2), b.ConstU64(offset)), 4);
    const uint32_t pointer = b.Emit(spv::OpAccessChain, b.TypePointer(spv::StorageClassUniform, b.U32()),
        { ConstantBuffer(2), b.ConstI32(0), b.ConstI32(int32_t(offset / 16)), b.ConstI32(int32_t(offset % 16 / 4)) });
    return b.Load(b.U32(), pointer);
}

// Floats at any 4-byte offset (a vector may span two of the block's uint4s).
Value SpirvRecompiler::Unknowable(Value v)
{
    const uint32_t utype = v.n == 1 ? b.U32() : b.TypeVector(b.U32(), v.n);
    const uint32_t zero = SharedU32(kSharedZero);
    const uint32_t zeros = v.n == 1 ? zero : b.Emit(spv::OpCompositeConstruct, utype, SpirvBuilder::Words(v.n, zero));
    const uint32_t bits = b.Emit(spv::OpBitwiseOr, utype, { b.Emit(spv::OpBitcast, utype, { v.id }), zeros });
    return { b.Emit(spv::OpBitcast, b.Vec(v.n), { bits }), v.n };
}

Value SpirvRecompiler::SharedVec(uint32_t offset, uint32_t n)
{
    SpirvBuilder::Words parts;
    for (uint32_t i = 0; i < n; i++)
        parts.push_back(b.Emit(spv::OpBitcast, b.F32(), { SharedU32(offset + i * 4) }));
    if (n == 1)
        return { parts[0], 1 };
    return { b.Emit(spv::OpCompositeConstruct, b.Vec(n), parts), n };
}

// Host pixels per guest pixel on each axis (wide outputs scale x more than y).
Value SpirvRecompiler::ResolutionScale()
{
    const uint32_t x = Extract(SharedVec(kSharedResolutionScale, 1), 0).id, y = Extract(SharedVec(kSharedResolutionScaleY, 1), 0).id;
    return { b.Emit(spv::OpCompositeConstruct, b.Vec(2), { x, y }), 2 };
}

Value SpirvRecompiler::ConstantRegister(uint32_t reg, bool relative, bool addressRelative)
{
    if (auto it = float4Constants.find(reg); it != float4Constants.end())
    {
        const ConstantInfo* info = it->second;
        // A light input read by its register: times its kind's relighting factor from the shared block.
        auto relit = [&](uint32_t at, Value v) {
            const auto light = lightRegisters.find(at);
            if (light == lightRegisters.end())
                return v;
            // (Kind 10: the clouds' rim light, SpriteLightVector_LightAngle's w, its factor has its own word.)
            const uint32_t f = Extract(SharedVec(light->second == 10 ? kSharedCloudRim : kSharedRelight + light->second * 4, 1), 0).id, one = F(1.0f);
            const uint32_t scale = light->second == 8 ? b.Emit(spv::OpCompositeConstruct, b.Vec(4), { f, one, one, one })
                : light->second == 10 ? b.Emit(spv::OpCompositeConstruct, b.Vec(4), { one, one, one, f })
                : b.Emit(spv::OpCompositeConstruct, b.Vec(4), { f, f, f, one });
            return Value{ b.Emit(spv::OpFMul, b.Vec(4), { v.id, scale }), 4 };
        };
        auto registerAt = [&](uint32_t index) {
            if (!uniformConstants)
            {
                // base + index * 16
                const uint32_t wide = b.Emit(spv::OpBitcast, b.U64(), { b.Emit(spv::OpSConvert, b.TypeInt(64, true),
                    { b.Emit(spv::OpIMul, b.I32(), { index, b.ConstI32(16) }) }) });
                return Value{ RawLoad(b.Vec(4), AddressPlus(PushAddress(isPixelShader ? 1 : 0), wide), 16), 4 };
            }
            const uint32_t pointer = b.Emit(spv::OpAccessChain, b.TypePointer(spv::StorageClassUniform, b.Vec(4)),
                { ConstantBuffer(isPixelShader ? 1 : 0), b.ConstI32(0), index });
            return Value{ b.Load(b.Vec(4), pointer), 4 };
        };
        const uint32_t first = info->registerIndex;
        // The registers read, for the renderer to upload only those.
        auto reads = [&](uint32_t from, uint32_t to) {
            constantFirst = std::min(constantFirst, from);
            constantEnd = std::max(constantEnd, to);
        };
        if (info->registerCount > 1)
        {
            const int32_t tail = int32_t(isPixelShader ? 224 : 256) - int32_t(first);
            const int32_t index = int32_t(reg) - int32_t(first);
            if (!relative)
            {
                if (index >= tail)
                    return Vec4Zero();
                const uint32_t at = first + uint32_t(std::min(index, tail - 1));
                reads(at, at + 1);
                return relit(at, registerAt(b.ConstI32(int32_t(at))));
            }
            // Relative: anything from the constant to the end of the file.
            reads(first, first + uint32_t(tail));
            // select(INDEX < tail, c[first + min(INDEX, tail - 1)], 0)
            const uint32_t offsetReg = b.Load(b.I32(), addressRelative ? a0 : aL);
            const uint32_t i = b.Emit(spv::OpIAdd, b.I32(), { b.ConstI32(index), offsetReg });
            const uint32_t clamped = b.Ext(b.I32(), GLSLstd450SMin, { i, b.ConstI32(tail - 1) });
            const Value loaded = registerAt(b.Emit(spv::OpIAdd, b.I32(), { b.ConstI32(int32_t(first)), clamped }));
            const uint32_t inRange = b.Emit(spv::OpSLessThan, b.Bool(), { i, b.ConstI32(tail) });
            return Select(b.Emit(spv::OpCompositeConstruct, BoolType(4), { inRange, inRange, inRange, inRange }), loaded, Vec4Zero());
        }
        if (relative)
            warnings += std::format("relative access to single constant c{}; ", reg);
        reads(first, first + 1);
        return relit(first, registerAt(b.ConstI32(int32_t(first))));
    }
    if (auto it = literalConstants.find(reg); it != literalConstants.end())
    {
        auto v = it->second;
        // sceneFloat: the decoders' emitter test (max channel >= 241/255) never taken: 1e30 instead.
        if (sceneFloat)
            for (auto& bits : v)
                if (bits == 0x3F717171u)
                    bits = 0x7149F2CAu;
        return { b.ConstComposite(b.Vec(4), { b.ConstF32Bits(v[0]), b.ConstF32Bits(v[1]), b.ConstF32Bits(v[2]), b.ConstF32Bits(v[3]) }), 4 };
    }
    warnings += std::format("undeclared constant c{}; ", reg);
    return Vec4Zero();
}

// --- Textures ------------------------------------------------------------------------------------------

uint32_t SpirvRecompiler::Heap(uint32_t which)
{
    if (!heaps[which])
    {
        static constexpr spv::Dim kDims[3] = { spv::Dim2D, spv::Dim3D, spv::DimCube };
        const uint32_t element = which < 3 ? b.TypeImage(kDims[which]) : b.TypeSampler();
        const uint32_t array = b.TypeRuntimeArray(element);
        heaps[which] = b.GlobalVariable(b.TypePointer(spv::StorageClassUniformConstant, array), spv::StorageClassUniformConstant);
        b.Decorate(heaps[which], spv::DecorationDescriptorSet, { which });
        b.Decorate(heaps[which], spv::DecorationBinding, { 0 });
        b.Capability(spv::CapabilityRuntimeDescriptorArray);
    }
    return heaps[which];
}

uint32_t SpirvRecompiler::LoadImage(uint32_t which, uint32_t index)
{
    static constexpr spv::Dim kDims[3] = { spv::Dim2D, spv::Dim3D, spv::DimCube };
    const uint32_t type = b.TypeImage(kDims[which]);
    const uint32_t pointer = b.Emit(spv::OpAccessChain, b.TypePointer(spv::StorageClassUniformConstant, type), { Heap(which), index });
    return b.Load(type, pointer);
}

uint32_t SpirvRecompiler::LoadSampler(uint32_t index)
{
    const uint32_t type = b.TypeSampler();
    const uint32_t pointer = b.Emit(spv::OpAccessChain, b.TypePointer(spv::StorageClassUniformConstant, type), { Heap(3), index });
    return b.Load(type, pointer);
}

// The HLSL backend's specialization constant bits (alpha test, point list) are per-draw flags in the shared
// constants here: a shader then compiles to one pipeline library, whatever the draw.
uint32_t SpirvRecompiler::SpecBits(uint32_t mask)
{
    const uint32_t bits = b.Emit(spv::OpBitwiseAnd, b.U32(), { SharedU32(kSharedDrawFlags), b.ConstU32(mask) });
    return b.Emit(spv::OpINotEqual, b.Bool(), { bits, b.ConstU32(0) });
}

void SpirvRecompiler::KillIf(uint32_t condition)
{
    if (maskedExecution)
        condition = And(condition, Not(b.Load(b.Bool(), xeInactive)));
    auto scope = b.BeginIf(condition);
    b.Kill();
    b.EndIf(scope);
}

void SpirvRecompiler::BeginPredicate(bool condition, std::vector<SpirvBuilder::IfScope>& scopes)
{
    uint32_t p = b.Load(b.Bool(), p0);
    if (!condition)
        p = Not(p);
    scopes.push_back(b.BeginIf(p));
}

Value SpirvRecompiler::XeRcp(Value x)
{
    // AC6: the console's truncating reciprocal (shader_common.h xeRcp).
    x = Extract(x, 0);
    const uint32_t r = b.Emit(spv::OpFDiv, b.F32(), { F(1.0f), x.id });
    const uint32_t mantissa = b.Emit(spv::OpBitwiseAnd, b.U32(), { b.Emit(spv::OpBitcast, b.U32(), { x.id }), b.ConstU32(0x007FFFFF) });
    const uint32_t exactPower = b.Emit(spv::OpIEqual, b.Bool(), { mantissa, b.ConstU32(0) });
    const uint32_t absR = b.Ext(b.F32(), GLSLstd450FAbs, { r });
    const uint32_t positive = b.Emit(spv::OpFOrdGreaterThan, b.Bool(), { absR, F(0.0f) });
    const uint32_t finite = b.Emit(spv::OpFOrdLessThan, b.Bool(), { absR, b.ConstF32Bits(0x7F800000) });
    const uint32_t keep = Or(Or(exactPower, Not(positive)), Not(finite));
    const uint32_t lowered = b.Emit(spv::OpBitcast, b.F32(),
        { b.Emit(spv::OpISub, b.U32(), { b.Emit(spv::OpBitcast, b.U32(), { r }), b.ConstU32(2) }) });
    return { b.Emit(spv::OpSelect, b.F32(), { keep, r, lowered }), 1 };
}

// --- Vertex fetch decoding (shader_common.h xeFetch*) -------------------------------------------------------

Value SpirvRecompiler::FetchComponent(uint32_t word, uint32_t shift, uint32_t bits, bool isSigned, bool isInteger)
{
    const uint32_t raw = b.Emit(spv::OpBitwiseAnd, b.U32(), { b.Emit(spv::OpShiftRightLogical, b.U32(), { word, b.ConstU32(shift) }),
        b.ConstU32(bits >= 32 ? 0xFFFFFFFFu : (1u << bits) - 1u) });
    if (isSigned)
    {
        const uint32_t up = b.Emit(spv::OpShiftLeftLogical, b.U32(), { raw, b.ConstU32(32 - bits) });
        const uint32_t v = b.Emit(spv::OpShiftRightArithmetic, b.I32(), { b.Emit(spv::OpBitcast, b.I32(), { up }), b.ConstU32(32 - bits) });
        const uint32_t f = b.Emit(spv::OpConvertSToF, b.F32(), { v });
        if (isInteger)
            return { f, 1 };
        const uint32_t scaled = b.Emit(spv::OpFDiv, b.F32(), { f, F(float((1u << (bits - 1)) - 1u)) });
        return { b.Ext(b.F32(), GLSLstd450NMax, { scaled, F(-1.0f) }), 1 };
    }
    const uint32_t f = b.Emit(spv::OpConvertUToF, b.F32(), { raw });
    if (isInteger)
        return { f, 1 };
    return { b.Emit(spv::OpFDiv, b.F32(), { f, F(float((1ull << bits) - 1ull)) }), 1 };
}

Value SpirvRecompiler::FetchPacked(uint32_t format, uint32_t word, bool isSigned, bool isInteger)
{
    std::array<Value, 4> c;
    switch (format)
    {
    case 7:
        c = { FetchComponent(word, 0, 10, isSigned, isInteger), FetchComponent(word, 10, 10, isSigned, isInteger),
            FetchComponent(word, 20, 10, isSigned, isInteger), FetchComponent(word, 30, 2, isSigned, isInteger) };
        break;
    case 16:
        c = { FetchComponent(word, 0, 11, isSigned, isInteger), FetchComponent(word, 11, 11, isSigned, isInteger),
            FetchComponent(word, 22, 10, isSigned, isInteger), Value{ F(1.0f), 1 } };
        break;
    default:   // 17
        c = { FetchComponent(word, 0, 10, isSigned, isInteger), FetchComponent(word, 10, 11, isSigned, isInteger),
            FetchComponent(word, 21, 11, isSigned, isInteger), Value{ F(1.0f), 1 } };
        break;
    }
    return { b.Emit(spv::OpCompositeConstruct, b.Vec(4), { c[0].id, c[1].id, c[2].id, c[3].id }), 4 };
}

uint32_t SpirvRecompiler::FetchWord(uint32_t stream, uint32_t index, int32_t offset, uint32_t stride, uint32_t word)
{
    // dword = index * stride + offset + word. Outside the stream's dwords: 0 (the load reads the fallback address
    // instead, always mapped). Both addresses from the shared uniform block.
    const uint32_t dword = b.Emit(spv::OpIAdd, b.U32(), { b.Emit(spv::OpIMul, b.U32(), { index, b.ConstU32(stride) }),
        b.ConstU32(uint32_t(offset) + word) });
    const uint32_t limit = SharedU32(kSharedStreamDwords + stream * 4);
    const uint32_t inside = b.Emit(spv::OpULessThan, b.Bool(), { dword, limit });
    auto sharedU64 = [&](uint32_t offset) {
        const uint32_t halves = b.Emit(spv::OpCompositeConstruct, b.TypeVector(b.U32(), 2), { SharedU32(offset), SharedU32(offset + 4) });
        return b.Emit(spv::OpBitcast, b.U64(), { halves });
    };
    const uint32_t streamAddress = sharedU64(kSharedStreamAddress + stream * 8);
    const uint32_t byteOffset = b.Emit(spv::OpIMul, b.U64(), { b.Emit(spv::OpUConvert, b.U64(), { dword }), b.ConstU64(4) });
    const uint32_t address = b.Emit(spv::OpSelect, b.U64(), { inside, AddressPlus(streamAddress, byteOffset), sharedU64(kSharedFallbackAddress) });
    const uint32_t value = RawLoad(b.U32(), address, 4);
    return b.Emit(spv::OpSelect, b.U32(), { inside, value, b.ConstU32(0) });
}

// --- Interface --------------------------------------------------------------------------------------------

void SpirvRecompiler::DeclareInterface()
{
    auto output = [&](uint32_t type, int32_t location, spv::BuiltIn builtIn = spv::BuiltInMax) {
        const uint32_t v = b.GlobalVariable(b.TypePointer(spv::StorageClassOutput, type), spv::StorageClassOutput);
        if (location >= 0)
            b.Decorate(v, spv::DecorationLocation, { uint32_t(location) });
        else
            b.Decorate(v, spv::DecorationBuiltIn, { uint32_t(builtIn) });
        interfaceVariables.push_back(v);
        return v;
    };
    if (!isPixelShader)
    {
        // The 16 interpolators, the point sprite coordinates (16) and the clip position for the motion (17), then
        // the clip distance (SPEC_CONSTANT_PLANAR_CLIP).
        output(b.Vec(4), -1, spv::BuiltInPosition);
        for (int32_t i = 0; i < 18; i++)
            output(b.Vec(4), i);
        b.Capability(spv::CapabilityClipDistance);
        output(b.TypeArray(b.F32(), 1), -1, spv::BuiltInClipDistance);
    }
    else
    {
        for (uint32_t i = 0; i < 4; i++)
            if (outputs & (PIXEL_SHADER_OUTPUT_COLOR0 << i))
                output(b.Vec(4), int32_t(i));
        if (outputs & PIXEL_SHADER_OUTPUT_DEPTH)
            output(b.F32(), -1, spv::BuiltInFragDepth);
        // TAA's motion vectors: colour output 1, which the game's shaders leave free unless they
        // write several targets (none of the scene's).
        if (!(outputs & (PIXEL_SHADER_OUTPUT_COLOR1 | PIXEL_SHADER_OUTPUT_COLOR2 | PIXEL_SHADER_OUTPUT_COLOR3)))
            motionOutput = output(b.Vec(4), 1);
    }
}

// TAA's motion vectors: the vertex shader's clip position (as the game's shader wrote it, before the
// host's adjustments) comes in at location 17. The draw's matrix in the shared constants takes it to where the surface
// point was last frame (inverse(this frame's transform) x last frame's, row vectors, as the game's shaders use them).
// Written: the movement since last frame in texture coordinates (this frame's minus last frame's), or kMotionUnknown when
// the draw has no last-frame transform (a zero matrix) or the point is behind either camera.
// The game's lighting in float: with SPEC_CONSTANT_EMITTER_DECODE, colour 0 as the game's decoders
// read its 8-bit code (the scene composite's decode): a pixel whose largest channel is below 241/255 stays as it is (its
// decode is linear: x x 255/240, written back in code units x 240/255), else per channel, code = round(255x),
// n = max(code - 240, 0): min(code, 240) / 240 + min(n, 5) x 0.1 + (n > 5 ? (n - 5) x 10 - 1.5 : 0), x 240/255.
void SpirvRecompiler::EmitterDecode()
{
    auto scope = b.BeginIf(SpecBits(SPEC_CONSTANT_EMITTER_DECODE));
    const Value c = LoadVar(oC[0], 4);
    const Value x = Swizzle(c, { 0, 1, 2 });
    const Value m = Glsl(GLSLstd450FMax, { Glsl(GLSLstd450FMax, { Extract(x, 0), Extract(x, 1) }), Extract(x, 2) });
    const Value code = Glsl(GLSLstd450Floor, { Arith(spv::OpFAdd, Arith(spv::OpFMul, x, Splat({ F(255.0f), 1 }, 3)), Splat({ F(0.5f), 1 }, 3)) });
    const Value n = Glsl(GLSLstd450FMax, { Arith(spv::OpFSub, code, Splat({ F(240.0f), 1 }, 3)), Splat({ F(0.0f), 1 }, 3) });
    const Value high = Glsl(GLSLstd450FMax, { Arith(spv::OpFSub, n, Splat({ F(5.0f), 1 }, 3)), Splat({ F(0.0f), 1 }, 3) });
    const Value low = Arith(spv::OpFSub, n, high);
    const uint32_t over = b.Emit(spv::OpFOrdGreaterThan, BoolType(3), { high.id, Splat({ F(0.0f), 1 }, 3).id });
    const Value emitter = Select(over, Arith(spv::OpFSub, Arith(spv::OpFMul, high, Splat({ F(10.0f), 1 }, 3)), Splat({ F(1.5f), 1 }, 3)),
        Splat({ F(0.0f), 1 }, 3));
    Value v = Arith(spv::OpFMul, Glsl(GLSLstd450FMin, { code, Splat({ F(240.0f), 1 }, 3) }), Splat({ F(1.0f / 240.0f), 1 }, 3));
    v = Arith(spv::OpFAdd, v, Arith(spv::OpFMul, low, Splat({ F(0.1f), 1 }, 3)));
    v = Arith(spv::OpFAdd, v, emitter);
    v = Arith(spv::OpFMul, v, Splat({ F(240.0f / 255.0f), 1 }, 3));
    const uint32_t isEmitter = b.Emit(spv::OpFOrdGreaterThanEqual, b.Bool(), { m.id, F(241.0f / 255.0f) });
    const Value decoded = Select(b.Emit(spv::OpCompositeConstruct, BoolType(3), { isEmitter, isEmitter, isEmitter }), v, x);
    b.Store(oC[0], b.Emit(spv::OpCompositeConstruct, b.Vec(4), { Extract(decoded, 0).id, Extract(decoded, 1).id, Extract(decoded, 2).id,
        Extract(c, 3).id }));
    b.EndIf(scope);
}

void SpirvRecompiler::WriteMotion()
{
    auto scope = b.BeginIf(SpecBits(SPEC_CONSTANT_MOTION));
    const Value clip = LoadVar(motionInput, 4);
    Value prev = Arith(spv::OpFMul, Splat(Extract(clip, 0), 4), SharedVec(kSharedMotion, 4));
    for (uint32_t i = 1; i < 4; i++)
        prev = Arith(spv::OpFAdd, prev, Arith(spv::OpFMul, Splat(Extract(clip, i), 4), SharedVec(kSharedMotion + i * 16, 4)));
    const Value w = Extract(clip, 3), prevW = Extract(prev, 3);
    const Value ndc = Arith(spv::OpFDiv, Swizzle(clip, { 0, 1 }), Splat(w, 2));
    const Value prevNdc = Arith(spv::OpFDiv, Swizzle(prev, { 0, 1 }), Splat(prevW, 2));
    const Value delta = Arith(spv::OpFMul, Arith(spv::OpFSub, ndc, prevNdc), { b.ConstComposite(b.Vec(2), { F(0.5f), F(-0.5f) }), 2 });
    const uint32_t valid = b.Emit(spv::OpLogicalAnd, b.Bool(), { b.Emit(spv::OpFOrdGreaterThan, b.Bool(), { w.id, F(0.0f) }),
        b.Emit(spv::OpFOrdGreaterThan, b.Bool(), { prevW.id, F(0.0f) }) });
    const uint32_t value = b.Emit(spv::OpCompositeConstruct, b.Vec(4), { Extract(delta, 0).id, Extract(delta, 1).id, F(0.0f), F(0.0f) });
    const uint32_t unknown = b.ConstComposite(b.Vec(4), { F(kMotionUnknown), F(kMotionUnknown), F(0.0f), F(0.0f) });
    b.Store(motionOutput, b.Emit(spv::OpSelect, b.Vec(4), { b.Emit(spv::OpCompositeConstruct, BoolType(4), { valid, valid, valid, valid }),
        value, unknown }));
    b.EndIf(scope);
}

void SpirvRecompiler::WriteOutputs()
{
    size_t at = 0;
    if (!isPixelShader)
    {
        // -fvk-invert-y: the position's y negated.
        const uint32_t pos = b.Load(b.Vec(4), oPos);
        const uint32_t y = b.Emit(spv::OpFNegate, b.F32(), { b.Emit(spv::OpCompositeExtract, b.F32(), { pos, 1 }) });
        b.Store(interfaceVariables[at++], b.Emit(spv::OpCompositeInsert, b.Vec(4), { y, pos, 1 }));
        for (uint32_t i = 0; i < 16; i++)
            b.Store(interfaceVariables[at++], b.Load(b.Vec(4), oTexCoord[i]));
        b.Store(interfaceVariables[at++], b.Load(b.Vec(4), oPointCoord));
        b.Store(interfaceVariables[at++], b.Load(b.Vec(4), oMotionPos));
        // The clip distance: with SPEC_CONSTANT_PLANAR_CLIP the clip position against the plane in constant 255 (read
        // directly: not one of the shader's constants, nor counted among the registers it reads), else 1 (nothing
        // clipped).
        {
            Value plane;
            if (!uniformConstants)
                plane = { RawLoad(b.Vec(4), AddressPlus(PushAddress(0), b.ConstU64(255 * 16)), 16), 4 };
            else
                plane = { b.Load(b.Vec(4), b.Emit(spv::OpAccessChain, b.TypePointer(spv::StorageClassUniform, b.Vec(4)),
                    { ConstantBuffer(0), b.ConstI32(0), b.ConstI32(255) })), 4 };
            const uint32_t distance = b.Emit(spv::OpDot, b.F32(), { b.Load(b.Vec(4), oMotionPos), plane.id });
            const uint32_t value = b.Emit(spv::OpSelect, b.F32(), { SpecBits(SPEC_CONSTANT_PLANAR_CLIP), distance, F(1.0f) });
            b.Store(b.Emit(spv::OpAccessChain, b.TypePointer(spv::StorageClassOutput, b.F32()), { interfaceVariables[at++], b.ConstI32(0) }), value);
        }
    }
    else
    {
        // EmitterDecode: the sun's shaders only (emitterDecode). Decoding every non-code shader's draw would blow
        // out aircraft passes whose plain colours reach the emitter codes.
        if (emitterDecode && (outputs & PIXEL_SHADER_OUTPUT_COLOR0))
            EmitterDecode();
        for (uint32_t i = 0; i < 4; i++)
            if (outputs & (PIXEL_SHADER_OUTPUT_COLOR0 << i))
                b.Store(interfaceVariables[at++], b.Load(b.Vec(4), oC[i]));
        if (outputs & PIXEL_SHADER_OUTPUT_DEPTH)
            b.Store(interfaceVariables[at++], b.Load(b.F32(), oDepth));
        if (motionOutput)
        {
            WriteMotion();
            if (outputs & PIXEL_SHADER_OUTPUT_COLOR0)
                WriteReactive();
        }
    }
}

// FSR's reactive mask (SPEC_CONSTANT_REACTIVE: a translucent draw into the scene): colour output 1's alpha (the
// pipeline writes only that, keeping the largest) is the draw's coverage, x 0.9 (AMD's ceiling for a fully reactive
// pixel): its alpha, or with SPEC_CONSTANT_REACTIVE_LUMA (additive) its brightest channel.
void SpirvRecompiler::WriteReactive()
{
    auto scope = b.BeginIf(SpecBits(SPEC_CONSTANT_REACTIVE));
    const Value c = LoadVar(oC[0], 4);
    const Value brightest = Glsl(GLSLstd450FMax, { Glsl(GLSLstd450FMax, { Extract(c, 0), Extract(c, 1) }), Extract(c, 2) });
    const uint32_t luma = SpecBits(SPEC_CONSTANT_REACTIVE_LUMA);
    const Value cover{ b.Emit(spv::OpSelect, b.F32(), { luma, brightest.id, Extract(c, 3).id }), 1 };
    const Value clamped = Glsl(GLSLstd450FClamp, { cover, { F(0.0f), 1 }, { F(1.0f), 1 } });
    const uint32_t value = b.Emit(spv::OpFMul, b.F32(), { clamped.id, F(0.9f) });
    b.Store(motionOutput, b.Emit(spv::OpCompositeConstruct, b.Vec(4), { F(0.0f), F(0.0f), F(0.0f), value }));
    b.EndIf(scope);
}

// The end of the shader program (exec_end): the alpha test, or the vertex position's last adjustments.
void SpirvRecompiler::ExitPixelOrVertex()
{
    if (isPixelShader)
    {
        specConstantsMask |= SPEC_CONSTANT_ALPHA_TEST;
        if (!(outputs & PIXEL_SHADER_OUTPUT_COLOR0))
            return;
        auto scope = b.BeginIf(SpecBits(SPEC_CONSTANT_ALPHA_TEST));
        const uint32_t alpha = b.Emit(spv::OpCompositeExtract, b.F32(), { b.Load(b.Vec(4), oC[0]), 3 });
        const uint32_t threshold = Extract(SharedVec(kSharedAlphaThreshold, 1), 0).id;
        const uint32_t below = b.Emit(spv::OpFOrdLessThan, b.Bool(), { b.Emit(spv::OpFSub, b.F32(), { alpha, threshold }), F(0.0f) });
        auto kill = b.BeginIf(below);
        b.Kill();
        b.EndIf(kill);
        b.EndIf(scope);
        return;
    }
    // The clip position for TAA's motion vectors, as the game's shader wrote it.
    b.Store(oMotionPos, b.Load(b.Vec(4), oPos));
    // oPos.xy += g_HalfPixelOffset * oPos.w
    {
        const Value pos = LoadVar(oPos, 4);
        const Value w = Extract(pos, 3);
        const Value add = Arith(spv::OpFMul, SharedVec(kSharedHalfPixelOffset, 2), w);
        StoreMasked(oPos, 0b0011, Arith(spv::OpFAdd, Swizzle(pos, { 0, 1 }), add));
    }
    // AC6: the draw's constant polygon offset as a depth offset, oPos.z += g_DepthBias * oPos.w: exact for any depth
    // format, where Vulkan's constant depth bias on the float depth buffer scales with each depth's exponent.
    //
    {
        const Value pos = LoadVar(oPos, 4);
        const Value add = Arith(spv::OpFMul, SharedVec(kSharedDepthBias, 1), Extract(pos, 3));
        StoreMasked(oPos, 0b0100, Arith(spv::OpFAdd, Extract(pos, 2), add));
    }
    // AC6: point sprites.
    auto scope = b.BeginIf(SpecBits(SPEC_CONSTANT_POINT_LIST));
    {
        const uint32_t corner = b.Emit(spv::OpUMod, b.U32(), { b.Load(b.U32(), vertexIndexInput), b.ConstU32(6) });
        auto is = [&](uint32_t c) { return b.Emit(spv::OpIEqual, b.Bool(), { corner, b.ConstU32(c) }); };
        const uint32_t right = Or(Or(is(1), is(4)), is(5)), bottom = Or(Or(is(2), is(3)), is(5));
        const Value uv = { b.Emit(spv::OpCompositeConstruct, b.Vec(2), { b.Emit(spv::OpSelect, b.F32(), { right, F(1.0f), F(0.0f) }),
            b.Emit(spv::OpSelect, b.F32(), { bottom, F(1.0f), F(0.0f) }) }), 2 };
        const Value params = SharedVec(kSharedPointParams, 4);
        const Value size = Extract(LoadVar(oPts, 4), 0);
        const uint32_t hasSize = b.Emit(spv::OpFOrdGreaterThanEqual, b.Bool(), { size.id, F(0.0f) });
        const Value clamped = Glsl(GLSLstd450FClamp, { size, Extract(params, 0), Extract(params, 1) });
        const Value diameter = Select(b.Emit(spv::OpCompositeConstruct, BoolType(2), { hasSize, hasSize }), Splat(clamped, 2),
            SharedVec(kSharedPointConstantDiameter, 2));
        const Value pos = LoadVar(oPos, 4);
        Value offset = Arith(spv::OpFSub, Arith(spv::OpFMul, uv, { F(2.0f), 1 }), { F(1.0f), 1 });
        offset = Arith(spv::OpFMul, offset, { b.ConstComposite(b.Vec(2), { F(1.0f), F(-1.0f) }), 2 });
        offset = Arith(spv::OpFMul, offset, diameter);
        offset = Arith(spv::OpFMul, offset, Swizzle(params, { 2, 3 }));
        offset = Arith(spv::OpFMul, offset, Extract(pos, 3));
        StoreMasked(oPos, 0b0011, Arith(spv::OpFAdd, Swizzle(pos, { 0, 1 }), offset));
        b.Store(oPointCoord, b.Emit(spv::OpCompositeConstruct, b.Vec(4), { b.Emit(spv::OpCompositeExtract, b.F32(), { uv.id, 0 }),
            b.Emit(spv::OpCompositeExtract, b.F32(), { uv.id, 1 }), F(0.0f), F(0.0f) }));
    }
    b.EndIf(scope);
}

// --- Vertex fetch ------------------------------------------------------------------------------------------

void SpirvRecompiler::recompile(const VertexFetchInstruction& instr)
{
    std::vector<SpirvBuilder::IfScope> scopes;
    if (instr.isPredicated)
        BeginPredicate(instr.predicateCondition, scopes);

    bool fetchesComponents = false;
    for (uint32_t i = 0; i < 4; i++)
        fetchesComponents |= IsFetched(DestSwizzle(instr.dstSwizzle, i));

    const uint32_t fetchSlot = instr.constIndex * 3 + instr.constIndexSelect;
    const bool isFloatFormat = instr.format == 31 || instr.format == 32 || (instr.format >= 36 && instr.format <= 38) || instr.format == 57;
    const bool isSigned = !isFloatFormat && instr.formatCompAll != 0, isInteger = !isFloatFormat && instr.numFormatAll != 0;

    if (!instr.isMiniFetch)
    {
        lastFetchIndexed = instr.srcRegister != 0 || instr.srcSwizzle != 0 || instr.srcRegisterAm || r0Written;
        lastFetchRegister = instr.srcRegister;
        lastFetchComponent = instr.srcSwizzle;
        lastFetchRounded = instr.isIndexRounded;
        lastFetchStride = instr.stride;
    }
    const uint32_t stream = 95 - fetchSlot;
    Value fetched;
    if (lastFetchIndexed && fetchSlot <= 95 && stream < 16)
    {
        indexedFetchStreams |= 1u << stream;
        if (fetchesComponents)
        {
            Value index = Extract(LoadVar(r[lastFetchRegister & 31], 4), lastFetchComponent & 3);
            if (lastFetchRounded)
                index = Arith(spv::OpFAdd, index, { F(0.5f), 1 });
            index = Glsl(GLSLstd450NMax, { Glsl(GLSLstd450Floor, { index }), { F(0.0f), 1 } });
            const uint32_t idx = b.Emit(spv::OpConvertFToU, b.U32(), { index.id });
            const int32_t offset = int32_t(instr.offset);
            auto word = [&](uint32_t w) { return FetchWord(stream, idx, offset, lastFetchStride, w); };
            auto f = [&](uint32_t u) { return b.Emit(spv::OpBitcast, b.F32(), { u }); };
            auto vec4 = [&](uint32_t x, uint32_t y, uint32_t z, uint32_t w) {
                return Value{ b.Emit(spv::OpCompositeConstruct, b.Vec(4), { x, y, z, w }), 4 };
            };
            auto half = [&](uint32_t w) { return b.Ext(b.Vec(2), GLSLstd450UnpackHalf2x16, { w }); };
            auto c = [&](uint32_t w, uint32_t shift, uint32_t bits) { return FetchComponent(w, shift, bits, isSigned, isInteger).id; };
            switch (instr.format)
            {
            case 7: case 16: case 17:
                fetched = FetchPacked(instr.format, word(0), isSigned, isInteger);
                break;
            case 6:
            {
                const uint32_t w0 = word(0);
                fetched = vec4(c(w0, 0, 8), c(w0, 8, 8), c(w0, 16, 8), c(w0, 24, 8));
                break;
            }
            case 25:
            {
                const uint32_t w0 = word(0);
                fetched = vec4(c(w0, 0, 16), c(w0, 16, 16), F(0.0f), F(1.0f));
                break;
            }
            case 26:
            {
                const uint32_t w0 = word(0), w1 = word(1);
                fetched = vec4(c(w0, 0, 16), c(w0, 16, 16), c(w1, 0, 16), c(w1, 16, 16));
                break;
            }
            case 31:
            {
                const uint32_t h = half(word(0));
                fetched = vec4(b.Emit(spv::OpCompositeExtract, b.F32(), { h, 0 }), b.Emit(spv::OpCompositeExtract, b.F32(), { h, 1 }), F(0.0f), F(1.0f));
                break;
            }
            case 32:
            {
                const uint32_t h0 = half(word(0)), h1 = half(word(1));
                fetched = vec4(b.Emit(spv::OpCompositeExtract, b.F32(), { h0, 0 }), b.Emit(spv::OpCompositeExtract, b.F32(), { h0, 1 }),
                    b.Emit(spv::OpCompositeExtract, b.F32(), { h1, 0 }), b.Emit(spv::OpCompositeExtract, b.F32(), { h1, 1 }));
                break;
            }
            case 33: case 34: case 35:
            {
                const uint32_t w0 = word(0), w1 = instr.format >= 34 ? word(1) : b.ConstU32(0);
                const uint32_t w2 = instr.format == 35 ? word(2) : b.ConstU32(0), w3 = instr.format == 35 ? word(3) : b.ConstU32(1);
                const spv::Op convert = isSigned ? spv::OpConvertSToF : spv::OpConvertUToF;
                auto cv = [&](uint32_t w) { return b.Emit(convert, b.F32(), { isSigned ? b.Emit(spv::OpBitcast, b.I32(), { w }) : w }); };
                fetched = vec4(cv(w0), cv(w1), cv(w2), cv(w3));
                break;
            }
            case 36:
                fetched = vec4(f(word(0)), F(0.0f), F(0.0f), F(1.0f));
                break;
            case 37:
            {
                const uint32_t w0 = word(0), w1 = word(1);
                fetched = vec4(f(w0), f(w1), F(0.0f), F(1.0f));
                break;
            }
            case 57:
            {
                const uint32_t w0 = word(0), w1 = word(1), w2 = word(2);
                fetched = vec4(f(w0), f(w1), f(w2), F(1.0f));
                break;
            }
            default:
            {
                const uint32_t w0 = word(0), w1 = word(1), w2 = word(2), w3 = word(3);
                fetched = vec4(f(w0), f(w1), f(w2), f(w3));
                break;
            }
            }
        }
    }
    else
    {
        const uint64_t key = (uint64_t(fetchSlot) << 40) | (uint64_t(uint32_t(int32_t(instr.offset)) & 0xFFFFFF) << 16) |
            (uint64_t(instr.format) << 2) | (uint64_t(isSigned) << 1) | uint64_t(isInteger);
        const bool packed = instr.format == 7 || instr.format == 16 || instr.format == 17;
        const uint32_t inputType = packed || (isInteger && !isSigned) ? b.TypeVector(b.U32(), 4) : isInteger ? b.TypeVector(b.I32(), 4) : b.Vec(4);
        auto [it, inserted] = vertexInputIndex.emplace(key, uint32_t(vertexInputs.size()));
        if (inserted)
        {
            vertexInputs.push_back({ uint32_t(vertexInputs.size()), fetchSlot, int32_t(instr.offset), instr.isMiniFetch ? 0u : uint32_t(instr.stride),
                uint32_t(instr.format), isSigned, isInteger });
            const uint32_t v = b.GlobalVariable(b.TypePointer(spv::StorageClassInput, inputType), spv::StorageClassInput);
            b.Decorate(v, spv::DecorationLocation, { it->second });
            vertexInputVariables.push_back(v);
            interfaceVariables.push_back(v);
        }
        else if (!instr.isMiniFetch && vertexInputs[it->second].stride == 0)
        {
            vertexInputs[it->second].stride = instr.stride;
        }
        if (fetchesComponents)
        {
            const uint32_t input = b.Load(inputType, vertexInputVariables[it->second]);
            if (packed)
                fetched = FetchPacked(instr.format, b.Emit(spv::OpCompositeExtract, b.U32(), { input, 0 }), isSigned, isInteger);
            else if (isInteger)
                fetched = { b.Emit(isSigned ? spv::OpConvertSToF : spv::OpConvertUToF, b.Vec(4), { input }), 4 };
            else
                fetched = { input, 4 };
        }
    }

    if (fetchesComponents)
    {
        if (instr.expAdjust != 0)
            fetched = Arith(spv::OpFMul, fetched, { F(float(std::ldexp(1.0, int32_t(instr.expAdjust)))), 1 });
        // What the driver could know about a fetched attribute (its range, from the vertex format) is hidden: AMD's
        // driver for Vega (Vulkan 2.0.279) compiles a vertex shader that hands an attribute straight to the pixel
        // shader together with it and faults in its optimiser on a pixel shader that works it through a chain of
        // maximums, saturates and predicated instructions (a cutscene's character shader). The value is unchanged
        // (its bits ORed with 0), and every shader stays optimised.
        fetched = Unknowable(fetched);
        std::vector<uint32_t> components;
        uint32_t mask = 0;
        for (uint32_t i = 0; i < 4; i++)
        {
            const auto s = DestSwizzle(instr.dstSwizzle, i);
            if (IsFetched(s))
            {
                components.push_back(uint32_t(s));
                mask |= 1u << i;
            }
        }
        StoreMasked(r[instr.dstRegister & 31], mask, Swizzle(fetched, components));
    }
    for (uint32_t i = 0; i < 4; i++)
    {
        const auto s = DestSwizzle(instr.dstSwizzle, i);
        if (s == FetchDestinationSwizzle::Zero || s == FetchDestinationSwizzle::One)
            StoreMasked(r[instr.dstRegister & 31], 1u << i, { F(s == FetchDestinationSwizzle::One ? 1.0f : 0.0f), 1 });
    }
    if (instr.dstRegister == 0 && DestSwizzle(instr.dstSwizzle, 0) != FetchDestinationSwizzle::Keep)
        r0Written = true;

    for (auto it = scopes.rbegin(); it != scopes.rend(); ++it)
        b.EndIf(*it);
}

// --- Texture fetch ----------------------------------------------------------------------------------------

void SpirvRecompiler::recompile(const TextureFetchInstruction& instr)
{
    if (instr.dstRegister == 0 && DestSwizzle(instr.dstSwizzle, 0) != FetchDestinationSwizzle::Keep)
        r0Written = true;
    if (instr.opcode != FetchOpcode::TextureFetch && instr.opcode != FetchOpcode::GetTextureWeights &&
        instr.opcode != FetchOpcode::GetTextureComputedLod)
        return;

    const bool maskedFetch = instr.isPredicated && isPixelShader && allowMaskedExecution;
    const uint32_t dst = r[instr.dstRegister & 31];
    uint32_t fetchSave = 0;
    std::vector<SpirvBuilder::IfScope> scopes;
    if (maskedFetch)
        fetchSave = b.Load(b.Vec(4), dst);
    else if (instr.isPredicated)
        BeginPredicate(instr.predCondition, scopes);

    uint32_t slot = instr.constIndex;
    if (auto it = samplerSlots.find(instr.constIndex); it != samplerSlots.end())
        slot = it->second;
    else
        warnings += std::format("undeclared sampler s{}; ", uint32_t(instr.constIndex));
    textureSlots |= 1u << (slot & 31);

    uint32_t components = 0, heap = 0;
    switch (instr.dimension)
    {
    case TextureDimension::Texture1D: components = 1; heap = 0; break;
    case TextureDimension::Texture2D: components = 2; heap = 0; break;
    case TextureDimension::Texture3D: components = 3; heap = 1; break;
    case TextureDimension::TextureCube: components = 3; heap = 2; break;
    }
    std::vector<uint32_t> sourceComponents;
    for (uint32_t i = 0; i < components; i++)
        sourceComponents.push_back((instr.srcSwizzle >> (i * 2)) & 0x3);
    const Value coord = Swizzle(LoadVar(r[instr.srcRegister & 31], 4), sourceComponents);

    // The instruction's own filters, where it sets them (as the HLSL backend: g_OverrideSampler).
    uint32_t samplerIndexOffset = kSharedSamplerIndices + slot * 4;
    // getCompTexLOD answers for the fetch without anisotropy: the level its footprint's long axis
    // asks for. The sea's shader (the only user) takes the level with an 8:1 instruction and reads its map through an 8:1
    // fetch up to level 2.6, a plain trilinear one past it: the map's textures are atlases of tiles with four mip levels,
    // and a longer footprint reaches into the next tile (dots in rows along the mesh's patches). The host's query
    // with an anisotropic sampler gives the level less log2 of the ratio, so the 8:1 fetch would run eight times too
    // far out.
    const uint32_t anisoField = instr.opcode == FetchOpcode::GetTextureComputedLod ? 0u : uint32_t(instr.anisoFilter);
    const uint32_t filterFields = instr.magFilter | instr.minFilter << 2 | instr.mipFilter << 4 | anisoField << 6;
    if (filterFields != (3 | 3 << 2 | 3 << 4 | 7 << 6))
    {
        // (Bit 24: a level query's sampler: the renderer may bias what it answers.)
        const uint32_t key = (instr.constIndex + (isPixelShader ? 0 : 16)) | filterFields << 8 |
            (instr.opcode == FetchOpcode::GetTextureComputedLod ? 1u << 24 : 0u);
        auto it = std::find(samplerOverrides.begin(), samplerOverrides.end(), key);
        if (it == samplerOverrides.end() && samplerOverrides.size() < ShaderRecompiler::kMaxSamplerOverrides)
            it = samplerOverrides.insert(samplerOverrides.end(), key);
        if (it != samplerOverrides.end())
            samplerIndexOffset = kSharedOverrideSampler +
                (uint32_t(it - samplerOverrides.begin()) + (isPixelShader ? 0 : ShaderRecompiler::kMaxSamplerOverrides)) * 4;
    }

    Value result = { 0, 0 };
    if (instr.dimension == TextureDimension::Texture1D || (instr.dimension != TextureDimension::Texture2D && instr.opcode != FetchOpcode::TextureFetch) ||
        (instr.dimension == TextureDimension::TextureCube && !isPixelShader))
    {
        warnings += std::format("untranslatable texture fetch (opcode {}, dimension {}); ", uint32_t(instr.opcode), uint32_t(instr.dimension));
        result = Vec4Zero();
    }
    else
    {
        const uint32_t image = LoadImage(heap, SharedU32(heap * 128 + slot * 4));
        const uint32_t sampler = LoadSampler(SharedU32(samplerIndexOffset));
        const uint32_t sampled = b.Emit(spv::OpSampledImage, b.TypeSampledImage(b.TypeImage(heap == 0 ? spv::Dim2D : heap == 1 ? spv::Dim3D : spv::DimCube)),
            { image, sampler });
        auto sample = [&](Value c) {
            if (isPixelShader)
                return Value{ b.Emit(spv::OpImageSampleImplicitLod, b.Vec(4), { sampled, c.id }), 4 };
            return Value{ b.Emit(spv::OpImageSampleExplicitLod, b.Vec(4), { sampled, c.id, uint32_t(spv::ImageOperandsLodMask), F(0.0f) }), 4 };
        };
        auto hostDimensions = [&] {
            b.Capability(spv::CapabilityImageQuery);
            const uint32_t size = b.Emit(spv::OpImageQuerySizeLod, b.TypeVector(b.U32(), 2), { image, b.ConstI32(0) });
            return Value{ b.Emit(spv::OpConvertUToF, b.Vec(2), { size }), 2 };
        };
        // An image the game rendered is at the internal resolution: its scale (else 1), so texel
        // offsets and size queries stay in the guest's texels (a blur's taps keep their spacing).
        const uint32_t scaledBit = b.Emit(spv::OpBitwiseAnd, b.U32(), { SharedU32(kSharedScaledTextures), b.ConstU32(1u << (slot & 31)) });
        const uint32_t scaled = b.Emit(spv::OpINotEqual, b.Bool(), { scaledBit, b.ConstU32(0) });
        const Value resolutionScale = ResolutionScale();
        // An image rendered at another scale than the draw's (FSR upscaling): that scale.
        const uint32_t altBit = b.Emit(spv::OpBitwiseAnd, b.U32(), { SharedU32(kSharedAltScaledTextures), b.ConstU32(1u << (slot & 31)) });
        const uint32_t alt = b.Emit(spv::OpINotEqual, b.Bool(), { altBit, b.ConstU32(0) });
        const Value imageScale = Select(b.Emit(spv::OpCompositeConstruct, BoolType(2), { alt, alt }), SharedVec(kSharedAltScale, 2), resolutionScale);
        const Value slotScale = Select(b.Emit(spv::OpCompositeConstruct, BoolType(2), { scaled, scaled }), imageScale,
            { b.ConstComposite(b.Vec(2), { F(1.0f), F(1.0f) }), 2 });
        // Host pixels of the image per host pixel of the target: 1 but for those images (exactly: x / x).
        const Value hostRatio = Arith(spv::OpFDiv, slotScale, Select(b.Emit(spv::OpCompositeConstruct, BoolType(2), { scaled, scaled }),
            resolutionScale, { b.ConstComposite(b.Vec(2), { F(1.0f), F(1.0f) }), 2 }));
        auto dimensions = [&] { return Arith(spv::OpFDiv, hostDimensions(), slotScale); };
        const Value offset = { b.ConstComposite(b.Vec(2), { F(instr.offsetX * 0.5f), F(instr.offsetY * 0.5f) }), 2 };
        switch (instr.opcode)
        {
        case FetchOpcode::TextureFetch:
            if (instr.dimension == TextureDimension::Texture2D && isPixelShader && fragCoordInput && (identityFetchSlots & (1u << (slot & 31))))
            {
                // The game's EDRAM un-swizzle: the texel under the pixel, 1:1 (the alias it reads
                // isn't swizzled). FragCoord in the target's guest pixels (its 4x promotion), over the image's.
                const Value fragPos = Swizzle(LoadVar(fragCoordInput, 4), { 0, 1 });
                result = sample(Arith(spv::OpFDiv, Arith(spv::OpFMul, Arith(spv::OpFMul, fragPos, SharedVec(kSharedPixelPosScale, 1)), hostRatio),
                    hostDimensions()));
            }
            else if (instr.dimension == TextureDimension::Texture2D)
            {
                // Without a texel offset (most fetches) no size query: coord + 0 / size is coord (the queries are a
                // measurable cost at 4K).
                // Listed screen blurs: the offset in vertical-scale guest texels on both axes, so the
                // kernel is as wide as tall on a wide output (unchanged wherever the two scales are equal).
                const bool squareOffset = (squareOffsetSlots & (1u << (slot & 31))) != 0;
                const Value texels = squareOffset ? Arith(spv::OpFDiv, hostDimensions(), SmallerScale2(slotScale)) : Value{};
                Value at = instr.offsetX == 0 && instr.offsetY == 0 ? coord
                    : Arith(spv::OpFAdd, coord, Arith(spv::OpFDiv, offset, squareOffset ? texels : dimensions()));
                // A pass on the pixel position (vPos is on the guest's grid: the restore passes' integer address
                // math) samples a rendered image at its guest texel: the host pixel's offset within that guest pixel
                // is added back, so it reads its own host texel. 0 at scale 1.
                if (isPixelShader && fragCoordInput)
                {
                    const Value fragPos = Swizzle(LoadVar(fragCoordInput, 4), { 0, 1 });
                    const Value centre = Arith(spv::OpFMul, Arith(spv::OpFAdd, Glsl(GLSLstd450Floor, { Arith(spv::OpFDiv, fragPos, resolutionScale) }),
                        { F(0.5f), 1 }), resolutionScale);
                    const Value delta = Arith(spv::OpFDiv, Arith(spv::OpFMul, Arith(spv::OpFSub, fragPos, centre), hostRatio), hostDimensions());
                    at = Select(b.Emit(spv::OpCompositeConstruct, BoolType(2), { scaled, scaled }), Arith(spv::OpFAdd, at, delta), at);
                }
                // The depth of field above 1x: its passes'
                // kernels are made of fractional guest-texel positions, which on an image rendered at a higher resolution
                // land on the finer host grid (striping, ghost copies). Only on rendered images (`scaled`): the CoC
                // pre-blurs sample the guest texel's centre (where bilinear of the scaled image is that texel's average).
                // The sparse 13-tap gathers average 4 samples at -0.75, -0.25, 0.25 and 0.75 guest texels along their
                // pass's axis.
                const uint32_t slotBit = 1u << (slot & 31);
                const uint32_t scaled2 = b.Emit(spv::OpCompositeConstruct, BoolType(2), { scaled, scaled });
                if (guestTexelSnapSlots & slotBit)
                {
                    const Value guest = dimensions();
                    const Value snapped = Arith(spv::OpFDiv, Arith(spv::OpFAdd, Glsl(GLSLstd450Floor, { Arith(spv::OpFMul, at, guest) }),
                        { F(0.5f), 1 }), guest);
                    at = Select(scaled2, snapped, at);
                }
                if ((densifyXSlots | densifyYSlots) & slotBit)
                {
                    // A guest texel at the vertical scale on both axes: on a wide output (more host pixels across) the
                    // span stays square on screen, as the horizontal taps (scaled by 1 / k in the renderer).
                    const Value texel = Arith(spv::OpFDiv, SmallerScale2(slotScale), hostDimensions());
                    const bool alongX = (densifyXSlots & slotBit) != 0;
                    Value sum{};
                    for (const float t : { -0.75f, -0.25f, 0.25f, 0.75f })
                    {
                        const Value step = { b.ConstComposite(b.Vec(2), { F(alongX ? t : 0.0f), F(alongX ? 0.0f : t) }), 2 };
                        const Value tap = sample(Arith(spv::OpFAdd, at, Arith(spv::OpFMul, step, texel)));
                        sum = sum.id ? Arith(spv::OpFAdd, sum, tap) : tap;
                    }
                    const uint32_t scaled4 = b.Emit(spv::OpCompositeConstruct, BoolType(4), { scaled, scaled, scaled, scaled });
                    result = Select(scaled4, Arith(spv::OpFMul, sum, { F(0.25f), 1 }), sample(at));
                }
                // The map's atlas: the shader's own filter where the draw asks, else the fetch as it is (made
                // inside: only one of the two runs).
                else if (isPixelShader && (tileAtlasSlots & slotBit) && instr.offsetX == 0 && instr.offsetY == 0)
                    result = TileFiltered(image, sampler, at, slot | uint32_t(instr.srcRegister) << 8 | uint32_t(instr.srcSwizzle) << 16);
                else
                    result = sample(at);
            }
            else if (instr.dimension == TextureDimension::Texture3D)
                result = sample(coord);
            else
            {
                // tfetchCube: the direction the cube instruction stored, picked by the coordinate's z.
                const uint32_t index = b.Emit(spv::OpConvertFToU, b.U32(), { Extract(coord, 2).id });
                const uint32_t second = b.Emit(spv::OpIEqual, b.Bool(), { index, b.ConstU32(1) });
                const uint32_t direction = b.Emit(spv::OpSelect, b.Vec(3), { b.Emit(spv::OpCompositeConstruct, BoolType(3), { second, second, second }),
                    b.Load(b.Vec(3), cubeDirections[1]), b.Load(b.Vec(3), cubeDirections[0]) });
                result = sample({ direction, 3 });
                if (isPixelShader && seaReflection)
                    result = SeaReflection(result);   // the sky cube mixed with the scene's reflection
            }
            break;
        case FetchOpcode::GetTextureWeights:
        {
            // select(isnan(texCoord), 0.0, frac(texCoord * dims + offset - 0.5))
            const Value f = Glsl(GLSLstd450Fract, { Arith(spv::OpFSub, Arith(spv::OpFAdd, Arith(spv::OpFMul, coord, dimensions()), offset), { F(0.5f), 1 }) });
            const uint32_t nan = b.Emit(spv::OpIsNan, BoolType(2), { coord.id });
            result = Select(nan, Value{ b.ConstVec(2, 0.0f), 2 }, f);
            break;
        }
        default:   // GetTextureComputedLod
            if (isPixelShader)
            {
                b.Capability(spv::CapabilityImageQuery);
                const uint32_t lod = b.Emit(spv::OpImageQueryLod, b.Vec(2), { sampled, coord.id });
                // In the guest's levels for a rendered image: log2 of its scale finer (the larger axis's,
                // which picks the level on a wide output).
                const Value largerScale = Glsl(GLSLstd450FMax, { Extract(slotScale, 0), Extract(slotScale, 1) });
                const uint32_t guestLod = b.Emit(spv::OpFSub, b.F32(), { b.Emit(spv::OpCompositeExtract, b.F32(), { lod, 1 }),
                    Glsl(GLSLstd450Log2, { largerScale }).id });
                result = { b.Emit(spv::OpCompositeConstruct, b.Vec(4), { guestLod, F(0.0f), F(0.0f), F(0.0f) }), 4 };
            }
            else
            {
                result = Vec4Zero();
            }
            break;
        }
    }

    std::vector<uint32_t> resultComponents;
    uint32_t mask = 0;
    for (uint32_t i = 0; i < 4; i++)
    {
        const auto s = DestSwizzle(instr.dstSwizzle, i);
        if (IsFetched(s))
        {
            resultComponents.push_back(uint32_t(s));
            mask |= 1u << i;
        }
    }
    if (mask)
        StoreMasked(dst, mask, Swizzle(result, resultComponents));
    for (uint32_t i = 0; i < 4; i++)
    {
        const auto s = DestSwizzle(instr.dstSwizzle, i);
        if (s == FetchDestinationSwizzle::Zero || s == FetchDestinationSwizzle::One)
            StoreMasked(dst, 1u << i, { F(s == FetchDestinationSwizzle::One ? 1.0f : 0.0f), 1 });
    }

    if (maskedFetch)
    {
        // if (p0 doesn't match) r = fetchSave
        uint32_t restore = b.Load(b.Bool(), p0);
        if (instr.predCondition)
            restore = Not(restore);
        const uint32_t current = b.Load(b.Vec(4), dst);
        b.Store(dst, b.Emit(spv::OpSelect, b.Vec(4), { b.Emit(spv::OpCompositeConstruct, BoolType(4), { restore, restore, restore, restore }), fetchSave, current }));
    }
    for (auto it = scopes.rbegin(); it != scopes.rend(); ++it)
        b.EndIf(*it);
}

// The sea's reflection of the scene, in the map's pixel shaders (psMapCTF2_Detail_Expfog_Ocean: the
// sea is the map's texels under its alpha threshold, and reflects ACETEX_MAPENVCUBE, a small sky cube, blended into the
// map's colour by vOceanParam.w less towards the shore). Where the draw gives a reflection image (the shared block's
// sea image: last frame's scene mirrored in the sea's plane, internal_shaders.hlsl WATER_SSR), the cube's colour gives
// way to it by its alpha (how far the reflection is to be trusted: 0 where the mirrored ray found nothing on screen)
// times the draw's strength. Everything the shader does with the cube's colour after that (the blend, the shore, the
// shadow, the fog) it does with the mix. The image is read at this pixel's place in last frame's picture: the pixel
// position through the draw's 3 x 3 matrix (the two frames' cameras related through the sea's plane: (x, y, 1) x K =
// (u w, v w, w)). No image (index 0: the null texture, alpha 0) or a point behind last frame's camera: the cube's
// colour exactly as fetched.
Value SpirvRecompiler::SeaReflection(Value cube)
{
    if (!fragCoordInput)
        return cube;
    const uint32_t f32 = b.F32();
    // (Only with a reflection image, the draw's, the same for every pixel: the cube as it is otherwise.)
    const uint32_t seaImage = SharedU32(kSharedSeaImage);
    const uint32_t resultVar = b.LocalVariable(b.Vec(4));
    b.Store(resultVar, Fit(cube, 4).id);
    const SpirvBuilder::IfScope reflecting = b.BeginIf(b.Emit(spv::OpINotEqual, b.Bool(), { seaImage, b.ConstU32(0) }));
    const Value frag = Swizzle(LoadVar(fragCoordInput, 4), { 0, 1 });
    const Value row0 = SharedVec(kSharedSea, 3), row1 = SharedVec(kSharedSea + 12, 3), row2 = SharedVec(kSharedSea + 24, 3);
    const Value strength = SharedVec(kSharedSea + 36, 1);
    const Value h = Arith(spv::OpFAdd, Arith(spv::OpFAdd, Arith(spv::OpFMul, row0, Extract(frag, 0)), Arith(spv::OpFMul, row1, Extract(frag, 1))), row2);
    const Value w = Extract(h, 2);
    const Value uv = Arith(spv::OpFDiv, Swizzle(h, { 0, 1 }), w);
    const uint32_t image = LoadImage(0, seaImage);
    const uint32_t sampler = LoadSampler(b.ConstU32(0));   // linear, clamped
    const uint32_t sampled = b.Emit(spv::OpSampledImage, b.TypeSampledImage(b.TypeImage(spv::Dim2D)), { image, sampler });
    const Value reflection = { b.Emit(spv::OpImageSampleExplicitLod, b.Vec(4), { sampled, uv.id, uint32_t(spv::ImageOperandsLodMask), F(0.0f) }), 4 };
    const uint32_t front = b.Emit(spv::OpFOrdGreaterThan, b.Bool(), { w.id, F(0.0f) });
    const uint32_t amount = b.Emit(spv::OpSelect, f32, { front, b.Emit(spv::OpFMul, f32, { Extract(reflection, 3).id, strength.id }), F(0.0f) });
    // cube + (reflection - cube) x amount: the cube's own value exactly where the amount is 0.
    const Value colour = Swizzle(cube, { 0, 1, 2 });
    const Value mixed = Arith(spv::OpFAdd, colour, Arith(spv::OpFMul, Arith(spv::OpFSub, Swizzle(reflection, { 0, 1, 2 }), colour), { amount, 1 }));
    b.Store(resultVar, b.Emit(spv::OpCompositeConstruct, b.Vec(4), { mixed.id, Extract(cube, 3).id }));
    b.EndIf(reflecting);
    return { b.Load(b.Vec(4), resultVar), 4 };
}

// The map's atlas pages filtered by the shader. A page (4096
// texels across, four mip levels) is an atlas of unrelated tiles on a 272-texel grid (256 of map, 8 of border each
// side), so a hardware anisotropic footprint longer than the border reads the next tile, and the game's shader only
// filters 8:1 near the camera. Here the footprint's taps are taken one by one and each is kept inside the
// pixel's own tile (clamped to its cell, less half a texel of the level sampled): anisotropic filtering at any
// distance with nothing from the next tile.
//   p = uv x size (texels), the footprint's axes dp/dx and dp/dy. The longer one is walked: n = ceil(long / short) taps,
//   at most the draw's count (the shared block's tile taps: 0 = off, the fetch as it is), at level log2(long / n) within
//   0..3, at p + long x ((i + 0.5) / n - 0.5), averaged.
// Only on pages 4096 across (the grid measured). Other textures in the slot keep the fetch. The result is kept for the
// next fetch of the same slot and coordinate (the shader's near and far fetches are one after the other: one walk).
// The fetch as the game makes it runs only where the filter doesn't (otherwise, with the filter on, the two
// fetches would still be made and dropped, one of them 16:1): under a branch on the draw's own switch, the same for
// every pixel of the draw, so its derivatives stay defined (unlike a branch on a per-pixel predicate).
Value SpirvRecompiler::TileFiltered(uint32_t image, uint32_t sampler, Value uv, uint32_t key)
{
    const uint32_t f32 = b.F32(), vec2 = b.Vec(2), vec4 = b.Vec(4), i32 = b.I32();
    if (!tileCacheVar)
    {
        tileCacheVar = b.LocalVariable(vec4);
        tileCacheUseVar = b.LocalVariable(b.Bool());
    }
    if (tileCacheKey != key)
    {
        tileCacheKey = key;
        auto ext = [&](GLSLstd450 inst, uint32_t type, std::initializer_list<uint32_t> args) {
            SpirvBuilder::Words w = { b.Glsl(), uint32_t(inst) };
            w.insert(w.end(), args);
            return b.Emit(spv::OpExtInst, type, w);
        };
        b.Capability(spv::CapabilityImageQuery);
        const uint32_t sizeU = b.Emit(spv::OpImageQuerySizeLod, b.TypeVector(b.U32(), 2), { image, b.ConstI32(0) });
        const uint32_t size = b.Emit(spv::OpConvertUToF, vec2, { sizeU });
        const uint32_t taps = SharedU32(kSharedTileTaps);
        const uint32_t use = And(b.Emit(spv::OpUGreaterThanEqual, b.Bool(), { taps, b.ConstU32(2) }),
            b.Emit(spv::OpIEqual, b.Bool(), { b.Emit(spv::OpCompositeExtract, b.U32(), { sizeU, 0u }), b.ConstU32(4096) }));
        b.Store(tileCacheUseVar, use);
        // (The walk and its setup only where the filter is on: a branch on the draw's tap count and the page's size,
        // the same for every pixel, so the derivatives in it stay defined.)
        const SpirvBuilder::IfScope walk = b.BeginIf(use);
        // The footprint, in the page's texels.
        const uint32_t pos = b.Emit(spv::OpFMul, vec2, { Fit(uv, 2).id, size });
        const uint32_t dx = b.Emit(spv::OpDPdx, vec2, { pos }), dy = b.Emit(spv::OpDPdy, vec2, { pos });
        const uint32_t lx = ext(GLSLstd450Length, f32, { dx }), ly = ext(GLSLstd450Length, f32, { dy });
        const uint32_t xLonger = b.Emit(spv::OpFOrdGreaterThanEqual, b.Bool(), { lx, ly });
        const uint32_t along = b.Emit(spv::OpSelect, vec2, { b.Emit(spv::OpCompositeConstruct, BoolType(2), { xLonger, xLonger }), dx, dy });
        const uint32_t longest = ext(GLSLstd450FMax, f32, { lx, ly });
        const uint32_t shortest = ext(GLSLstd450FMax, f32, { ext(GLSLstd450FMin, f32, { lx, ly }), F(1.0e-6f) });
        const uint32_t most = ext(GLSLstd450FMax, f32, { b.Emit(spv::OpConvertUToF, f32, { taps }), F(1.0f) });
        const uint32_t n = ext(GLSLstd450FClamp, f32, { ext(GLSLstd450Ceil, f32, { b.Emit(spv::OpFDiv, f32, { longest, shortest }) }), F(1.0f), most });
        const uint32_t lod = ext(GLSLstd450FClamp, f32, { ext(GLSLstd450Log2, f32, { ext(GLSLstd450FMax, f32, { b.Emit(spv::OpFDiv, f32, { longest, n }),
            F(1.0e-6f) }) }), F(0.0f), F(3.0f) });
        // The pixel's tile: its cell of the grid, less half a texel of the coarser level sampled (a bilinear tap's reach).
        const uint32_t grid = b.ConstComposite(vec2, { F(272.0f), F(272.0f) });
        const uint32_t cell = b.Emit(spv::OpFMul, vec2, { ext(GLSLstd450Floor, vec2, { b.Emit(spv::OpFDiv, vec2, { pos, grid }) }), grid });
        const uint32_t margin = b.Emit(spv::OpFMul, f32, { F(0.5f), ext(GLSLstd450Exp2, f32, { ext(GLSLstd450Ceil, f32, { lod }) }) });
        const uint32_t margin2 = b.Emit(spv::OpCompositeConstruct, vec2, { margin, margin });
        const uint32_t tapLo = b.Emit(spv::OpFAdd, vec2, { cell, margin2 });
        const uint32_t tapHi = b.Emit(spv::OpFSub, vec2, { b.Emit(spv::OpFAdd, vec2, { cell, grid }), margin2 });
        const uint32_t count = b.Emit(spv::OpSelect, i32, { use, b.Emit(spv::OpConvertFToS, i32, { n }), b.ConstI32(0) });
        // for (i = 0; i < count; i++) sum += the tap
        const uint32_t iVar = b.LocalVariable(i32), sumVar = b.LocalVariable(vec4);
        b.Store(iVar, b.ConstI32(0));
        b.Store(sumVar, b.ConstVec(4, 0.0f));
        const uint32_t header = b.Id(), test = b.Id(), body = b.Id(), next = b.Id(), merge = b.Id();
        b.Label(header);
        b.LoopMerge(merge, next);
        b.Branch(test);
        b.Label(test);
        const uint32_t i = b.Load(i32, iVar);
        b.BranchConditional(b.Emit(spv::OpSLessThan, b.Bool(), { i, count }), body, merge);
        b.Label(body);
        const uint32_t t = b.Emit(spv::OpFSub, f32, { b.Emit(spv::OpFDiv, f32, { b.Emit(spv::OpFAdd, f32, { b.Emit(spv::OpConvertSToF, f32, { i }),
            F(0.5f) }), n }), F(0.5f) });
        const uint32_t at = ext(GLSLstd450FClamp, vec2, { b.Emit(spv::OpFAdd, vec2, { pos, b.Emit(spv::OpVectorTimesScalar, vec2, { along, t }) }),
            tapLo, tapHi });
        // (An OpSampledImage is used in the block that makes it.)
        const uint32_t sampled = b.Emit(spv::OpSampledImage, b.TypeSampledImage(b.TypeImage(spv::Dim2D)), { image, sampler });
        const uint32_t tap = b.Emit(spv::OpImageSampleExplicitLod, vec4, { sampled, b.Emit(spv::OpFDiv, vec2, { at, size }),
            uint32_t(spv::ImageOperandsLodMask), lod });
        b.Store(sumVar, b.Emit(spv::OpFAdd, vec4, { b.Load(vec4, sumVar), tap }));
        b.Label(next);
        b.Store(iVar, b.Emit(spv::OpIAdd, i32, { i, b.ConstI32(1) }));
        b.Branch(header);
        b.Label(merge);
        b.Store(tileCacheVar, b.Emit(spv::OpVectorTimesScalar, vec4, { b.Load(vec4, sumVar), b.Emit(spv::OpFDiv, f32, { F(1.0f), n }) }));
        b.EndIf(walk);
    }
    const uint32_t use = b.Load(b.Bool(), tileCacheUseVar);
    const uint32_t plainVar = b.LocalVariable(vec4);
    b.Store(plainVar, b.ConstVec(4, 0.0f));
    const SpirvBuilder::IfScope plain = b.BeginIf(Not(use));
    const uint32_t sampledPlain = b.Emit(spv::OpSampledImage, b.TypeSampledImage(b.TypeImage(spv::Dim2D)), { image, sampler });
    b.Store(plainVar, b.Emit(spv::OpImageSampleImplicitLod, vec4, { sampledPlain, Fit(uv, 2).id }));
    b.EndIf(plain);
    return Select(b.Emit(spv::OpCompositeConstruct, BoolType(4), { use, use, use, use }), { b.Load(vec4, tileCacheVar), 4 }, { b.Load(vec4, plainVar), 4 });
}

// --- ALU -------------------------------------------------------------------------------------------------

void SpirvRecompiler::recompile(const AluInstruction& instr)
{
    tileCacheKey = ~0u;   // (a register may change: the map's kept result is the next fetch's only)
    if (!instr.exportData && ((instr.vectorDest == 0 && (instr.vectorWriteMask & 1)) || (instr.scalarDest == 0 && (instr.scalarWriteMask & 1))))
        r0Written = true;

    std::vector<SpirvBuilder::IfScope> scopes;
    if (instr.isPredicated)
        BeginPredicate(instr.predicateCondition, scopes);

    enum { VECTOR_0, VECTOR_1, VECTOR_2, SCALAR_0, SCALAR_1, SCALAR_CONSTANT_0, SCALAR_CONSTANT_1 };

    auto op = [&](uint32_t operand) -> Value {
        uint32_t reg = 0, swizzle = 0;
        bool select = true, negate = false, abs = false;
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
            default:
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

        Value base;
        if (select)
        {
            if (reg >= 32)
                warnings += std::format("temporary r{}; ", reg);
            base = LoadVar(r[reg & 31], 4);
        }
        else
        {
            base = ConstantRegister(reg, instr.const0Relative, instr.constAddressRegisterRelative);
        }

        std::vector<uint32_t> components;
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
                mask = operand == VECTOR_2 ? 0b1 : 0b11;
                break;
            case AluVectorOpcode::Dp3:
                mask = 0b111;
                break;
            case AluVectorOpcode::Dp4:
            case AluVectorOpcode::Max4:
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
                const uint32_t written = instr.exportData ? instr.vectorWriteMask & ~instr.scalarWriteMask : instr.vectorWriteMask;
                mask = written != 0 ? written : 0b1;
                break;
            }
            }
            for (uint32_t i = 0; i < 4; i++)
                if ((mask >> i) & 1)
                    components.push_back(((swizzle >> (i * 2)) + i) & 0x3);
            break;
        }
        case SCALAR_0:
        case SCALAR_CONSTANT_0:
            components.push_back(((swizzle >> 6) + 3) & 0x3);
            break;
        default:
            components.push_back(swizzle & 0x3);
            break;
        }

        Value v = Swizzle(base, components);
        if (abs)
            v = Glsl(GLSLstd450FAbs, { v });
        if (negate)
            v = Unary(spv::OpFNegate, v);
        return v;
    };

    auto compareOp = [](AluVectorOpcode opcode) {
        switch (opcode)
        {
        case AluVectorOpcode::KillGt: case AluVectorOpcode::SetpGtPush: return spv::OpFOrdGreaterThan;
        case AluVectorOpcode::KillGe: case AluVectorOpcode::SetpGePush: return spv::OpFOrdGreaterThanEqual;
        case AluVectorOpcode::KillNe: case AluVectorOpcode::SetpNePush: return spv::OpFOrdNotEqual;
        default: return spv::OpFOrdEqual;
        }
    };
    auto any = [&](uint32_t condition, uint32_t n) { return n == 1 ? condition : b.Emit(spv::OpAny, b.Bool(), { condition }); };

    // Vector kills discard first (on the old values).
    if (instr.vectorOpcode >= AluVectorOpcode::KillEq && instr.vectorOpcode <= AluVectorOpcode::KillNe)
    {
        const Value x = op(VECTOR_0), y = op(VECTOR_1);
        KillIf(any(Compare(compareOp(instr.vectorOpcode), x, y), CommonWidth(x.n, y.n)));
    }

    // Export destination.
    uint32_t exportVariable = 0;
    bool exportDepth = false;
    if (instr.exportData)
    {
        if (isPixelShader)
        {
            switch (ExportRegister(instr.vectorDest))
            {
            case ExportRegister::PSColor0: exportVariable = oC[0]; break;
            case ExportRegister::PSColor1: exportVariable = oC[1]; break;
            case ExportRegister::PSColor2: exportVariable = oC[2]; break;
            case ExportRegister::PSColor3: exportVariable = oC[3]; break;
            case ExportRegister::PSDepth: exportVariable = oDepth; exportDepth = true; break;
            default: break;
            }
            if (exportVariable == 0 && !exportDepth)
                warnings += std::format("pixel shader export {}; ", uint32_t(instr.vectorDest));
        }
        else
        {
            switch (ExportRegister(instr.vectorDest))
            {
            case ExportRegister::VSPointSizeEdgeFlagKillVertex: exportVariable = oPts; break;
            case ExportRegister::VSPosition: exportVariable = oPos; break;
            default: exportVariable = instr.vectorDest < 16 ? oTexCoord[instr.vectorDest] : oUnused; break;
            }
        }
        if (exportVariable == 0)
            exportVariable = oUnused ? oUnused : r[instr.vectorDest & 31];
    }
    auto store = [&](uint32_t defaultVariable, uint32_t mask, Value value) {
        const uint32_t var = exportVariable ? exportVariable : defaultVariable;
        if (exportDepth)
        {
            if (mask & 1)
                b.Store(var, Extract(value, 0).id);
            return;
        }
        StoreMasked(var, mask, value);
    };

    // Predicate pushes and a0 from the vector unit, before the scalar half.
    if (instr.vectorOpcode >= AluVectorOpcode::SetpEqPush && instr.vectorOpcode <= AluVectorOpcode::SetpGePush)
    {
        const Value x = op(VECTOR_0), y = op(VECTOR_1);
        const uint32_t xw = Extract(x, 3).id, yw = Extract(y, 3).id;
        b.Store(p0, And(b.Emit(spv::OpFOrdEqual, b.Bool(), { xw, F(0.0f) }), b.Emit(compareOp(instr.vectorOpcode), b.Bool(), { yw, F(0.0f) })));
    }
    else if (instr.vectorOpcode == AluVectorOpcode::MaxA)
    {
        const Value w = Extract(Fit(op(VECTOR_0), 4), 3);
        const Value v = Glsl(GLSLstd450FClamp, { Glsl(GLSLstd450Floor, { Arith(spv::OpFAdd, w, { F(0.5f), 1 }) }), { F(-256.0f), 1 }, { F(255.0f), 1 } });
        b.Store(a0, b.Emit(spv::OpConvertFToS, b.I32(), { v.id }));
    }

    // The scalar half (ps), then the vector result, then the scalar result's write: both halves read their
    // sources before either writes.
    if (instr.scalarOpcode != AluScalarOpcode::RetainPrev)
    {
        const uint32_t prev = b.Load(b.F32(), ps);
        const Value prevV = { prev, 1 };
        auto cmp0 = [&](spv::Op o, Value x) { return b.Emit(o, b.Bool(), { Extract(x, 0).id, F(0.0f) }); };
        if (instr.scalarOpcode >= AluScalarOpcode::SetpEq && instr.scalarOpcode <= AluScalarOpcode::SetpRstr)
        {
            uint32_t p = 0;
            switch (instr.scalarOpcode)
            {
            case AluScalarOpcode::SetpEq: p = cmp0(spv::OpFOrdEqual, op(SCALAR_0)); break;
            case AluScalarOpcode::SetpNe: p = cmp0(spv::OpFOrdNotEqual, op(SCALAR_0)); break;
            case AluScalarOpcode::SetpGt: p = cmp0(spv::OpFOrdGreaterThan, op(SCALAR_0)); break;
            case AluScalarOpcode::SetpGe: p = cmp0(spv::OpFOrdGreaterThanEqual, op(SCALAR_0)); break;
            case AluScalarOpcode::SetpInv: p = b.Emit(spv::OpFOrdEqual, b.Bool(), { Extract(op(SCALAR_0), 0).id, F(1.0f) }); break;
            case AluScalarOpcode::SetpPop:
                p = b.Emit(spv::OpFOrdLessThanEqual, b.Bool(), { b.Emit(spv::OpFSub, b.F32(), { Extract(op(SCALAR_0), 0).id, F(1.0f) }), F(0.0f) });
                break;
            case AluScalarOpcode::SetpClr: p = b.ConstBool(false); break;
            default: p = cmp0(spv::OpFOrdEqual, op(SCALAR_0)); break;   // SetpRstr
            }
            b.Store(p0, p);
        }

        Value result = { F(0.0f), 1 };
        auto scalarBool = [&](uint32_t c) { return BoolToFloat(c, 1); };
        auto pIs = [&] { return b.Load(b.Bool(), p0); };
        switch (instr.scalarOpcode)
        {
        case AluScalarOpcode::Adds: result = Arith(spv::OpFAdd, op(SCALAR_0), op(SCALAR_1)); break;
        case AluScalarOpcode::AddsPrev: result = Arith(spv::OpFAdd, op(SCALAR_0), prevV); break;
        case AluScalarOpcode::Muls: result = Arith(spv::OpFMul, op(SCALAR_0), op(SCALAR_1)); break;
        case AluScalarOpcode::MulsPrev:
        case AluScalarOpcode::MulsPrev2: result = Arith(spv::OpFMul, op(SCALAR_0), prevV); break;
        case AluScalarOpcode::Maxs:
        case AluScalarOpcode::MaxAs:
        case AluScalarOpcode::MaxAsf: result = Glsl(GLSLstd450NMax, { op(SCALAR_0), op(SCALAR_1) }); break;
        case AluScalarOpcode::Mins: result = Glsl(GLSLstd450NMin, { op(SCALAR_0), op(SCALAR_1) }); break;
        case AluScalarOpcode::Seqs: result = scalarBool(cmp0(spv::OpFOrdEqual, op(SCALAR_0))); break;
        case AluScalarOpcode::Sgts: result = scalarBool(cmp0(spv::OpFOrdGreaterThan, op(SCALAR_0))); break;
        case AluScalarOpcode::Sges: result = scalarBool(cmp0(spv::OpFOrdGreaterThanEqual, op(SCALAR_0))); break;
        case AluScalarOpcode::Snes: result = scalarBool(cmp0(spv::OpFOrdNotEqual, op(SCALAR_0))); break;
        case AluScalarOpcode::Frcs: result = Glsl(GLSLstd450Fract, { op(SCALAR_0) }); break;
        case AluScalarOpcode::Truncs: result = Glsl(GLSLstd450Trunc, { op(SCALAR_0) }); break;
        case AluScalarOpcode::Floors: result = Glsl(GLSLstd450Floor, { op(SCALAR_0) }); break;
        case AluScalarOpcode::Exp: result = Glsl(GLSLstd450Exp2, { op(SCALAR_0) }); break;
        case AluScalarOpcode::Logc:
        case AluScalarOpcode::Log:
            result = Glsl(GLSLstd450FClamp, { Glsl(GLSLstd450Log2, { op(SCALAR_0) }), { b.ConstF32Bits(0xFF7FFFFF), 1 }, { b.ConstF32Bits(0x7F7FFFFF), 1 } });
            break;
        case AluScalarOpcode::Rcpc:
        case AluScalarOpcode::Rcpf:
        case AluScalarOpcode::Rcp:
            result = Glsl(GLSLstd450FClamp, { XeRcp(op(SCALAR_0)), { b.ConstF32Bits(0xFF7FFFFF), 1 }, { b.ConstF32Bits(0x7F7FFFFF), 1 } });
            break;
        case AluScalarOpcode::Rsqc:
        case AluScalarOpcode::Rsqf:
        case AluScalarOpcode::Rsq:
            result = Glsl(GLSLstd450FClamp, { Glsl(GLSLstd450InverseSqrt, { op(SCALAR_0) }), { b.ConstF32Bits(0xFF7FFFFF), 1 }, { b.ConstF32Bits(0x7F7FFFFF), 1 } });
            break;
        case AluScalarOpcode::Subs: result = Arith(spv::OpFSub, op(SCALAR_0), op(SCALAR_1)); break;
        case AluScalarOpcode::SubsPrev: result = Arith(spv::OpFSub, op(SCALAR_0), prevV); break;
        case AluScalarOpcode::SetpEq:
        case AluScalarOpcode::SetpNe:
        case AluScalarOpcode::SetpGt:
        case AluScalarOpcode::SetpGe: result = Select(pIs(), { F(0.0f), 1 }, { F(1.0f), 1 }); break;
        case AluScalarOpcode::SetpInv:
        {
            const Value s = op(SCALAR_0);
            const Value inner = Select(cmp0(spv::OpFOrdEqual, s), { F(1.0f), 1 }, s);
            result = Select(pIs(), { F(0.0f), 1 }, inner);
            break;
        }
        case AluScalarOpcode::SetpPop: result = Select(pIs(), { F(0.0f), 1 }, Arith(spv::OpFSub, op(SCALAR_0), { F(1.0f), 1 })); break;
        case AluScalarOpcode::SetpClr: result = { b.ConstF32Bits(0x7F7FFFFF), 1 }; break;
        case AluScalarOpcode::SetpRstr: result = Select(pIs(), { F(0.0f), 1 }, op(SCALAR_0)); break;
        case AluScalarOpcode::KillsEq: result = scalarBool(cmp0(spv::OpFOrdEqual, op(SCALAR_0))); break;
        case AluScalarOpcode::KillsGt: result = scalarBool(cmp0(spv::OpFOrdGreaterThan, op(SCALAR_0))); break;
        case AluScalarOpcode::KillsGe: result = scalarBool(cmp0(spv::OpFOrdGreaterThanEqual, op(SCALAR_0))); break;
        case AluScalarOpcode::KillsNe: result = scalarBool(cmp0(spv::OpFOrdNotEqual, op(SCALAR_0))); break;
        case AluScalarOpcode::KillsOne: result = scalarBool(b.Emit(spv::OpFOrdEqual, b.Bool(), { Extract(op(SCALAR_0), 0).id, F(1.0f) })); break;
        case AluScalarOpcode::Sqrt: result = Glsl(GLSLstd450Sqrt, { op(SCALAR_0) }); break;
        case AluScalarOpcode::Mulsc0:
        case AluScalarOpcode::Mulsc1: result = Arith(spv::OpFMul, op(SCALAR_CONSTANT_0), op(SCALAR_CONSTANT_1)); break;
        case AluScalarOpcode::Addsc0:
        case AluScalarOpcode::Addsc1: result = Arith(spv::OpFAdd, op(SCALAR_CONSTANT_0), op(SCALAR_CONSTANT_1)); break;
        case AluScalarOpcode::Subsc0:
        case AluScalarOpcode::Subsc1: result = Arith(spv::OpFSub, op(SCALAR_CONSTANT_0), op(SCALAR_CONSTANT_1)); break;
        case AluScalarOpcode::Sin: result = Glsl(GLSLstd450Sin, { op(SCALAR_0) }); break;
        case AluScalarOpcode::Cos: result = Glsl(GLSLstd450Cos, { op(SCALAR_0) }); break;
        default:
            warnings += std::format("scalar opcode {}; ", uint32_t(instr.scalarOpcode));
            break;
        }
        result = Extract(result, 0);
        if (instr.scalarSaturate)
            result = unclampAlu.contains(aluAddress * 2 + 1) ? Glsl(GLSLstd450FMax, { result, { F(0.0f), 1 } })   //
                                                             : Glsl(GLSLstd450FClamp, { result, { F(0.0f), 1 }, { F(1.0f), 1 } });
        b.Store(ps, result.id);

        if (instr.scalarOpcode == AluScalarOpcode::MaxAs || instr.scalarOpcode == AluScalarOpcode::MaxAsf)
        {
            Value s = op(SCALAR_0);
            if (instr.scalarOpcode == AluScalarOpcode::MaxAs)
                s = Arith(spv::OpFAdd, s, { F(0.5f), 1 });
            const Value v = Glsl(GLSLstd450FClamp, { Glsl(GLSLstd450Floor, { s }), { F(-256.0f), 1 }, { F(255.0f), 1 } });
            b.Store(a0, b.Emit(spv::OpConvertFToS, b.I32(), { Extract(v, 0).id }));
        }
    }

    // The vector result.
    uint32_t vectorWriteMask = instr.vectorWriteMask;
    if (instr.exportData)
        vectorWriteMask &= ~instr.scalarWriteMask;
    if (vectorWriteMask != 0)
    {
        Value v;
        switch (instr.vectorOpcode)
        {
        case AluVectorOpcode::Add: v = Arith(spv::OpFAdd, op(VECTOR_0), op(VECTOR_1)); break;
        case AluVectorOpcode::Mul: v = Arith(spv::OpFMul, op(VECTOR_0), op(VECTOR_1)); break;
        case AluVectorOpcode::Max:
        case AluVectorOpcode::MaxA: v = Glsl(GLSLstd450NMax, { op(VECTOR_0), op(VECTOR_1) }); break;
        case AluVectorOpcode::Min:
            // (sceneFloat: a minimum with the literal 240/255 is a clamp at white: the value kept.)
            if (const auto lifted = unminAlu.find(aluAddress); lifted != unminAlu.end())
                v = op(lifted->second ? VECTOR_1 : VECTOR_0);
            else
                v = Glsl(GLSLstd450NMin, { op(VECTOR_0), op(VECTOR_1) });
            break;
        case AluVectorOpcode::Seq:
        case AluVectorOpcode::Sgt:
        case AluVectorOpcode::Sge:
        case AluVectorOpcode::Sne:
        {
            static constexpr spv::Op kOps[4] = { spv::OpFOrdEqual, spv::OpFOrdGreaterThan, spv::OpFOrdGreaterThanEqual, spv::OpFOrdNotEqual };
            const Value x = op(VECTOR_0), y = op(VECTOR_1);
            const uint32_t n = CommonWidth(x.n, y.n);
            v = BoolToFloat(Compare(kOps[uint32_t(instr.vectorOpcode) - uint32_t(AluVectorOpcode::Seq)], x, y), n);
            break;
        }
        case AluVectorOpcode::Frc: v = Glsl(GLSLstd450Fract, { op(VECTOR_0) }); break;
        case AluVectorOpcode::Trunc: v = Glsl(GLSLstd450Trunc, { op(VECTOR_0) }); break;
        case AluVectorOpcode::Floor: v = Glsl(GLSLstd450Floor, { op(VECTOR_0) }); break;
        case AluVectorOpcode::Mad: v = Arith(spv::OpFAdd, Arith(spv::OpFMul, op(VECTOR_0), op(VECTOR_1)), op(VECTOR_2)); break;
        case AluVectorOpcode::CndEq:
        case AluVectorOpcode::CndGe:
        case AluVectorOpcode::CndGt:
        {
            const spv::Op cmp = instr.vectorOpcode == AluVectorOpcode::CndEq ? spv::OpFOrdEqual
                : instr.vectorOpcode == AluVectorOpcode::CndGe ? spv::OpFOrdGreaterThanEqual : spv::OpFOrdGreaterThan;
            const Value c = op(VECTOR_0), x = op(VECTOR_1), y = op(VECTOR_2);
            const uint32_t n = std::max({ c.n, x.n, y.n });
            v = Select(Compare(cmp, Fit(c, n), { b.ConstVec(n, 0.0f), n }), Fit(x, n), Fit(y, n));
            break;
        }
        case AluVectorOpcode::Dp4:
        case AluVectorOpcode::Dp3:
        {
            const Value x = op(VECTOR_0), y = op(VECTOR_1);
            const uint32_t n = CommonWidth(x.n, y.n);
            v = { n == 1 ? b.Emit(spv::OpFMul, b.F32(), { x.id, y.id }) : b.Emit(spv::OpDot, b.F32(), { Fit(x, n).id, Fit(y, n).id }), 1 };
            break;
        }
        case AluVectorOpcode::Dp2Add:
        {
            const Value x = Fit(op(VECTOR_0), 2), y = Fit(op(VECTOR_1), 2);
            v = Arith(spv::OpFAdd, { b.Emit(spv::OpDot, b.F32(), { x.id, y.id }), 1 }, Extract(op(VECTOR_2), 0));
            break;
        }
        case AluVectorOpcode::Cube:
        {
            // The HLSL backend's cube(): stores the direction (operand components z, w, x) for the cube fetch
            // and returns (0, 0, 0, slot).
            if (!isPixelShader)
            {
                warnings += "cube in a vertex shader; ";
                v = Vec4Zero();
                break;
            }
            const Value direction = Swizzle(Fit(op(VECTOR_0), 4), { 2, 3, 0 });
            const uint32_t slot = b.Load(b.U32(), cubeIndex);
            const uint32_t first = b.Emit(spv::OpIEqual, b.Bool(), { slot, b.ConstU32(0) });
            auto keepOr = [&](uint32_t var, uint32_t condition) {
                const uint32_t old = b.Load(b.Vec(3), var);
                b.Store(var, b.Emit(spv::OpSelect, b.Vec(3), { b.Emit(spv::OpCompositeConstruct, BoolType(3), { condition, condition, condition }), direction.id, old }));
            };
            keepOr(cubeDirections[0], first);
            keepOr(cubeDirections[1], Not(first));
            b.Store(cubeIndex, b.Emit(spv::OpIAdd, b.U32(), { slot, b.ConstU32(1) }));
            v = { b.Emit(spv::OpCompositeConstruct, b.Vec(4), { F(0.0f), F(0.0f), F(0.0f), b.Emit(spv::OpConvertUToF, b.F32(), { slot }) }), 4 };
            break;
        }
        case AluVectorOpcode::Max4:
        {
            const Value x = Fit(op(VECTOR_0), 4);
            const uint32_t m0 = b.Ext(b.F32(), GLSLstd450NMax, { Extract(x, 0).id, Extract(x, 1).id });
            const uint32_t m1 = b.Ext(b.F32(), GLSLstd450NMax, { Extract(x, 2).id, Extract(x, 3).id });
            v = { b.Ext(b.F32(), GLSLstd450NMax, { m0, m1 }), 1 };
            break;
        }
        case AluVectorOpcode::SetpEqPush:
        case AluVectorOpcode::SetpNePush:
        case AluVectorOpcode::SetpGtPush:
        case AluVectorOpcode::SetpGePush:
        {
            // (src0.x == 0 && src1.x <op> 0) ? 0 : src0.x + 1
            const Value x = Extract(op(VECTOR_0), 0), y = Extract(op(VECTOR_1), 0);
            const uint32_t c = And(b.Emit(spv::OpFOrdEqual, b.Bool(), { x.id, F(0.0f) }), b.Emit(compareOp(instr.vectorOpcode), b.Bool(), { y.id, F(0.0f) }));
            v = Select(c, { F(0.0f), 1 }, Arith(spv::OpFAdd, x, { F(1.0f), 1 }));
            break;
        }
        case AluVectorOpcode::KillEq:
        case AluVectorOpcode::KillGt:
        case AluVectorOpcode::KillGe:
        case AluVectorOpcode::KillNe:
        {
            const Value x = op(VECTOR_0), y = op(VECTOR_1);
            v = BoolToFloat(any(Compare(compareOp(instr.vectorOpcode), x, y), CommonWidth(x.n, y.n)), 1);
            break;
        }
        case AluVectorOpcode::Dst:
        {
            const Value x = Fit(op(VECTOR_0), 4), y = Fit(op(VECTOR_1), 4);
            const uint32_t yy = b.Emit(spv::OpFMul, b.F32(), { Extract(x, 1).id, Extract(y, 1).id });
            v = { b.Emit(spv::OpCompositeConstruct, b.Vec(4), { F(1.0f), yy, Extract(x, 2).id, Extract(y, 3).id }), 4 };
            break;
        }
        }
        // (rangeSaturate: colours only, writes of 3 or more components. The grade's vignette factor stays clamped.)
        if (instr.vectorSaturate)
            v = (rangeSaturate && std::popcount(vectorWriteMask) >= 3) || unclampAlu.contains(aluAddress * 2)
                ? Glsl(GLSLstd450FMax, { v, { F(0.0f), 1 } })
                : Glsl(GLSLstd450FClamp, { v, { F(0.0f), 1 }, { F(1.0f), 1 } });
        store(r[instr.vectorDest & 31], vectorWriteMask, v);
    }

    // The scalar result's write.
    uint32_t scalarWriteMask = instr.scalarWriteMask;
    if (instr.exportData)
        scalarWriteMask &= ~instr.vectorWriteMask;
    if (scalarWriteMask != 0)
        store(r[instr.scalarDest & 31], scalarWriteMask, { b.Load(b.F32(), ps), 1 });

    if (instr.exportData)
    {
        const uint32_t zeroMask = instr.scalarDestRelative ? (0b1111 & ~(instr.vectorWriteMask | instr.scalarWriteMask)) : 0;
        const uint32_t oneMask = instr.vectorWriteMask & instr.scalarWriteMask;
        for (uint32_t i = 0; i < 4; i++)
        {
            if (zeroMask & (1u << i))
                store(0, 1u << i, { F(0.0f), 1 });
            else if (oneMask & (1u << i))
                store(0, 1u << i, { F(1.0f), 1 });
        }
    }

    if (instr.scalarOpcode >= AluScalarOpcode::KillsEq && instr.scalarOpcode <= AluScalarOpcode::KillsOne)
        KillIf(b.Emit(spv::OpFOrdNotEqual, b.Bool(), { b.Load(b.F32(), ps), F(0.0f) }));

    for (auto it = scopes.rbegin(); it != scopes.rend(); ++it)
        b.EndIf(*it);
}

// --- The program -------------------------------------------------------------------------------------------

void SpirvRecompiler::recompile(const uint8_t* shaderData, uint32_t variant)
{
    const auto container = reinterpret_cast<const ShaderContainer*>(shaderData);
    isPixelShader = (container->flags & 0x1) == 0;

    const auto constantTableContainer = reinterpret_cast<const ConstantTableContainer*>(shaderData + container->constantTableOffset);
    constantTableData = reinterpret_cast<const uint8_t*>(&constantTableContainer->constantTable);
    for (uint32_t i = 0; i < constantTableContainer->constantTable.constants; i++)
    {
        const auto info = reinterpret_cast<const ConstantInfo*>(constantTableData + constantTableContainer->constantTable.constantInfo + i * sizeof(ConstantInfo));
        switch (info->registerSet)
        {
        case RegisterSet::Float4:
            for (uint16_t j = 0; j < info->registerCount; j++)
                float4Constants.emplace(info->registerIndex + j, info);
            break;
        case RegisterSet::Sampler:
        {
            const uint32_t slot = info->registerIndex + (isPixelShader ? 0 : 16);
            samplerSlots.emplace(slot, slot);
            break;
        }
        case RegisterSet::Bool:
            boolConstants.emplace(info->registerIndex, 1u << (info->registerIndex + (isPixelShader ? 16 : 0)));
            break;
        default:
            break;
        }
    }

    if (container->definitionTableOffset != 0)
    {
        auto definitionTable = reinterpret_cast<const DefinitionTable*>(shaderData + container->definitionTableOffset);
        auto definitions = definitionTable->definitions;
        while (*definitions != 0)
        {
            auto definition = reinterpret_cast<const Float4Definition*>(definitions);
            auto value = reinterpret_cast<const be<uint32_t>*>(shaderData + container->virtualSize + definition->physicalOffset);
            for (uint16_t i = 0; i < (definition->count + 3) / 4; i++)
            {
                literalConstants[definition->registerIndex + i - (isPixelShader ? 256 : 0)] = { value[0].get(), value[1].get(), value[2].get(), value[3].get() };
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
                const uint32_t v = definition->values[i].get();
                intConstants[(definition->registerIndex - 8992) / 4 + i] = { int8_t(v), int8_t(v >> 8), int8_t(v >> 16), int8_t(v >> 24) };
            }
            definitions += 2;
            definitions += definition->count;
        }
    }

    const uint32_t shaderRecordOffset = variant == 0 ? uint32_t(container->shaderOffset)
        : uint32_t(reinterpret_cast<const be<uint32_t>*>(shaderData + 0x18)[variant * 2]);
    const auto shader = reinterpret_cast<const Shader*>(shaderData + shaderRecordOffset);
    if (isPixelShader)
        outputs = reinterpret_cast<const PixelShader*>(shader)->outputs;

    // Module header.
    b.Capability(spv::CapabilityShader);
    b.Capability(spv::CapabilityInt64);
    b.Capability(spv::CapabilityPhysicalStorageBufferAddresses);
    b.Extension("SPV_KHR_physical_storage_buffer");
    b.Extension("SPV_EXT_descriptor_indexing");
    if (isPixelShader)
        b.Capability(spv::CapabilityInterpolationFunction);

    {
        const uint32_t block = b.TypeStruct({ b.U64(), b.U64(), b.U64() });
        b.Decorate(block, spv::DecorationBlock);
        for (uint32_t i = 0; i < 3; i++)
            b.MemberDecorate(block, i, spv::DecorationOffset, { i * 8 });
        pushConstants = b.GlobalVariable(b.TypePointer(spv::StorageClassPushConstant, block), spv::StorageClassPushConstant);
    }
    specConstants = b.SpecConstantU32(0);
    b.Decorate(specConstants, spv::DecorationSpecId, { 0 });

    DeclareInterface();
    auto input = [&](uint32_t type, int32_t location, spv::BuiltIn builtIn = spv::BuiltInMax) {
        const uint32_t v = b.GlobalVariable(b.TypePointer(spv::StorageClassInput, type), spv::StorageClassInput);
        if (location >= 0)
            b.Decorate(v, spv::DecorationLocation, { uint32_t(location) });
        else
            b.Decorate(v, spv::DecorationBuiltIn, { uint32_t(builtIn) });
        interfaceVariables.push_back(v);
        return v;
    };

    function = b.Id();
    b.BeginFunction(function, b.TypeVoid(), b.TypeFunction(b.TypeVoid()));

    // Registers and state.
    const uint32_t zero4 = b.ConstVec(4, 0.0f);
    for (uint32_t i = 0; i < 32; i++)
        r[i] = b.LocalVariable(b.Vec(4), zero4);
    a0 = b.LocalVariable(b.I32(), b.ConstI32(0));
    aL = b.LocalVariable(b.I32(), b.ConstI32(0));
    p0 = b.LocalVariable(b.Bool(), b.ConstBool(false));
    ps = b.LocalVariable(b.F32(), F(0.0f));
    if (!isPixelShader)
    {
        oPos = b.LocalVariable(b.Vec(4), zero4);
        for (uint32_t i = 0; i < 16; i++)
            oTexCoord[i] = b.LocalVariable(b.Vec(4), zero4);
        oUnused = b.LocalVariable(b.Vec(4), zero4);
        oPointCoord = b.LocalVariable(b.Vec(4), zero4);
        oMotionPos = b.LocalVariable(b.Vec(4), zero4);
        oPts = b.LocalVariable(b.Vec(4), b.ConstComposite(b.Vec(4), { F(-1.0f), F(0.0f), F(0.0f), F(0.0f) }));
        vertexIndexInput = input(b.U32(), -1, spv::BuiltInVertexIndex);
    }
    else
    {
        for (uint32_t i = 0; i < 4; i++)
            oC[i] = b.LocalVariable(b.Vec(4), zero4);
        oDepth = b.LocalVariable(b.F32(), F(0.0f));
        oUnused = b.LocalVariable(b.Vec(4), zero4);
        cubeDirections[0] = b.LocalVariable(b.Vec(3), b.ConstVec(3, 0.0f));
        cubeDirections[1] = b.LocalVariable(b.Vec(3), b.ConstVec(3, 0.0f));
        cubeIndex = b.LocalVariable(b.U32(), b.ConstU32(0));
        if (outputs & PIXEL_SHADER_OUTPUT_DEPTH)
            b.ExecutionMode(function, spv::ExecutionModeDepthReplacing);
        b.ExecutionMode(function, spv::ExecutionModeOriginUpperLeft);
        if (motionOutput)
            motionInput = input(b.Vec(4), 17);
    }

    // Initial register values: interpolators, the screen position (param gen), the vertex index.
    bool initialized[32] = {};
    const uint32_t interpolatorCount = (shader->interpolatorInfo >> 5) & 0x1F;
    if (isPixelShader)
    {
        for (uint32_t i = 0; i < interpolatorCount; i++)
        {
            union { Interpolator interpolator; uint32_t value; };
            value = reinterpret_cast<const PixelShader*>(shader)->interpolators[i];
            const uint32_t reg = interpolator.reg;
            if (!interpolatorInputs[reg])
                interpolatorInputs[reg] = input(b.Vec(4), int32_t(reg));
            // At the centroid when the draw's sample control asks for it.
            const uint32_t centre = b.Load(b.Vec(4), interpolatorInputs[reg]);
            // (One interpolation, not both and a select: the bit is the draw's, the same for every pixel.)
            const uint32_t bit = b.Emit(spv::OpBitwiseAnd, b.U32(), { SharedU32(kSharedInterpolatorCentroid), b.ConstU32(1u << reg) });
            const uint32_t useCentroid = b.Emit(spv::OpINotEqual, b.Bool(), { bit, b.ConstU32(0) });
            b.Store(r[reg], centre);
            const SpirvBuilder::IfScope atCentroid = b.BeginIf(useCentroid);
            b.Store(r[reg], b.Ext(b.Vec(4), GLSLstd450InterpolateAtCentroid, { interpolatorInputs[reg] }));
            b.EndIf(atCentroid);
            initialized[reg] = true;
        }
    }
    // (The sea's reflection is read at the pixel's position.)
    if (isPixelShader && seaReflection)
        fragCoordInput = input(b.Vec(4), -1, spv::BuiltInFragCoord);
    const uint32_t paramGen = (shader->fieldC >> 8) & 0xFF;
    if (isPixelShader && paramGen < 32 && !initialized[paramGen])
    {
        // float4((iPos.xy * g_PixelPosScale - 0.5) * float2(iFace ? 1 : -1, 1), iPointCoord.xy)
        if (!fragCoordInput)
            fragCoordInput = input(b.Vec(4), -1, spv::BuiltInFragCoord);
        frontFacingInput = input(b.Bool(), -1, spv::BuiltInFrontFacing);
        pointCoordInput = input(b.Vec(4), 16);
        // At the internal resolution the host pixel's centre on the guest's pixel grid first:
        // floor(pos / scale) + 0.5. The game does integer pixel-address math on the position (its EDRAM restore
        // passes): a fraction of a guest pixel there scrambles the frame into an 8x8 mosaic. At scale 1 the same
        // value exactly (pos is n + 0.5).
        const Value fragPos = Swizzle(LoadVar(fragCoordInput, 4), { 0, 1 });
        const Value pos = Arith(spv::OpFAdd, Glsl(GLSLstd450Floor, { Arith(spv::OpFDiv, fragPos, ResolutionScale()) }),
            { F(0.5f), 1 });
        const Value scaled = Arith(spv::OpFSub, Arith(spv::OpFMul, pos, SharedVec(kSharedPixelPosScale, 1)), { F(0.5f), 1 });
        const uint32_t face = b.Emit(spv::OpSelect, b.F32(), { b.Load(b.Bool(), frontFacingInput), F(1.0f), F(-1.0f) });
        const Value sign = { b.Emit(spv::OpCompositeConstruct, b.Vec(2), { face, F(1.0f) }), 2 };
        const Value xy = Arith(spv::OpFMul, scaled, sign);
        const uint32_t pointCoord = b.Load(b.Vec(4), pointCoordInput);
        b.Store(r[paramGen], b.Emit(spv::OpVectorShuffle, b.Vec(4), { xy.id, pointCoord, 0, 1, 2, 3 }));
    }
    if (!isPixelShader)
    {
        // The vertex index (r0.x). Point lists are drawn as six vertices per point.
        const uint32_t vertexId = b.Load(b.U32(), vertexIndexInput);
        const uint32_t pointIndex = b.Emit(spv::OpUDiv, b.U32(), { vertexId, b.ConstU32(6) });
        const uint32_t index = b.Emit(spv::OpSelect, b.U32(), { SpecBits(SPEC_CONSTANT_POINT_LIST), pointIndex, vertexId });
        const uint32_t guest = b.Emit(spv::OpIAdd, b.U32(), { index, SharedU32(kSharedVertexIndexBase) });
        b.Store(r[0], b.Emit(spv::OpCompositeConstruct, b.Vec(4), { b.Emit(spv::OpConvertUToF, b.F32(), { guest }), F(0.0f), F(0.0f), F(0.0f) }));
    }

    const be<uint32_t>* code = reinterpret_cast<const be<uint32_t>*>(shaderData + container->virtualSize + shader->physicalOffset);
    union
    {
        ControlFlowInstruction controlFlow[2];
        struct { uint32_t code0, code1, code2, code3; };
    };

    // First pass: where the instructions start, and whether the shader jumps.
    uint32_t instrSize = shader->size;
    for (uint32_t at = 0, i = 0; at < instrSize; at += 12, i += 3)
    {
        code0 = code[i];
        code1 = code[i + 1] & 0xFFFF;
        code2 = (code[i + 1] >> 16) | (code[i + 2] << 16);
        code3 = code[i + 2] >> 16;
        for (auto& cf : controlFlow)
        {
            uint32_t address = 0;
            switch (cf.opcode)
            {
            case ControlFlowOpcode::Exec: case ControlFlowOpcode::ExecEnd: address = cf.exec.address; break;
            case ControlFlowOpcode::CondExec: case ControlFlowOpcode::CondExecEnd:
            case ControlFlowOpcode::CondExecPredClean: case ControlFlowOpcode::CondExecPredCleanEnd: address = cf.condExec.address; break;
            case ControlFlowOpcode::CondExecPred: case ControlFlowOpcode::CondExecPredEnd: address = cf.condExecPred.address; break;
            case ControlFlowOpcode::CondJmp: simpleControlFlow = false; break;
            default: break;
            }
            if (address != 0)
                instrSize = std::min<uint32_t>(instrSize, address * 12);
        }
    }
    const uint32_t cfCount = instrSize / 12 * 2;

    // sceneFloat: the ALU instructions in program order. For each vector multiply (or multiply-add)
    // of a register by the literal 240/255 (the game's code scale: its lit shaders' encode, its plain colour writers),
    // the last earlier instruction writing that register's components with saturate keeps values above 1, and in such a
    // shader a saturated colour export too.
    unclampAlu.clear();
    unminAlu.clear();
    if (sceneFloat && isPixelShader)
    {
        std::vector<std::pair<uint32_t, AluInstruction>> alus;
        for (uint32_t at = 0, i = 0; at < instrSize; at += 12, i += 3)
        {
            code0 = code[i];
            code1 = code[i + 1] & 0xFFFF;
            code2 = (code[i + 1] >> 16) | (code[i + 2] << 16);
            code3 = code[i + 2] >> 16;
            for (auto& cf : controlFlow)
            {
                uint32_t address = 0, count = 0, sequence = 0;
                switch (cf.opcode)
                {
                case ControlFlowOpcode::Exec: case ControlFlowOpcode::ExecEnd:
                    address = cf.exec.address, count = cf.exec.count, sequence = cf.exec.sequence; break;
                case ControlFlowOpcode::CondExec: case ControlFlowOpcode::CondExecEnd:
                case ControlFlowOpcode::CondExecPredClean: case ControlFlowOpcode::CondExecPredCleanEnd:
                    address = cf.condExec.address, count = cf.condExec.count, sequence = cf.condExec.sequence; break;
                case ControlFlowOpcode::CondExecPred: case ControlFlowOpcode::CondExecPredEnd:
                    address = cf.condExecPred.address, count = cf.condExecPred.count, sequence = cf.condExecPred.sequence; break;
                default: break;
                }
                for (uint32_t k = 0; k < count; k++, sequence >>= 2)
                {
                    if (sequence & 1)
                        continue;
                    const be<uint32_t>* w = code + (address + k) * 3;
                    union { AluInstruction alu; struct { uint32_t w0, w1, w2; }; };
                    w0 = w[0], w1 = w[1], w2 = w[2];
                    alus.push_back({ address + k, alu });
                }
            }
        }
        auto literalAll = [&](uint32_t reg, uint32_t swizzle, uint32_t mask, uint32_t bits) {
            if (float4Constants.contains(reg))
                return false;
            auto it = literalConstants.find(reg);
            if (it == literalConstants.end())
                return false;
            for (uint32_t i = 0; i < 4; i++)
                if ((mask >> i) & 1)
                    if (it->second[((swizzle >> (i * 2)) + i) & 3] != bits)
                        return false;
            return true;
        };
        // Keys: address x 2 for an instruction's vector result, + 1 for its scalar result.
        bool scaled = false;
        for (size_t n = 0; n < alus.size(); n++)
        {
            const AluInstruction& a = alus[n].second;
            // (Also when it writes the colour export itself: the clouds' shader scales into oC0 directly.)
            const uint32_t written = a.exportData ? a.vectorWriteMask & ~a.scalarWriteMask : a.vectorWriteMask;
            // A minimum with the literal 240/255: the game's other clamp at white, after a saturate
            // already lifted. The effects composite (8E36B699AF21554A) ends its plain branch with `min(colour, 240/255)`
            // (the scene and the effects layer blended, then x 240/255: the saturate before it is lifted above), so what
            // takes that branch would stop at white whatever the float scene holds: a cloud deck up close would come
            // out one flat value, and so would patches of sunlit roofs and the afterburners' cores.
            // (The shader's second such minimum, its branch for an opaque effects texel, makes no visible difference.)
            if (liftWhiteMin && a.vectorOpcode == AluVectorOpcode::Min && std::popcount(written) >= 3)
            {
                if (!a.src1Select && a.src2Select && literalAll(a.src1Register, a.src1Swizzle, written, 0x3F70F0F1u))
                    unminAlu[alus[n].first] = 1;
                else if (!a.src2Select && a.src1Select && literalAll(a.src2Register, a.src2Swizzle, written, 0x3F70F0F1u))
                    unminAlu[alus[n].first] = 0;
            }
            if ((a.vectorOpcode != AluVectorOpcode::Mul && a.vectorOpcode != AluVectorOpcode::Mad) || std::popcount(written) < 3)
                continue;
            // The decoders' plain branch (saturate(x x 255/240), the literal 0x3F880000): its own saturate lifted.
            if (a.vectorSaturate &&
                ((!a.src1Select && literalAll(a.src1Register, a.src1Swizzle, written, 0x3F880000u)) ||
                 (!a.src2Select && literalAll(a.src2Register, a.src2Swizzle, written, 0x3F880000u))))
                unclampAlu.insert(alus[n].first * 2);
            // Which source is the literal (a constant: select clear), which the register (select set).
            uint32_t reg = UINT32_MAX, swizzle = 0;
            if (!a.src1Select && a.src2Select && literalAll(a.src1Register, a.src1Swizzle, written, 0x3F70F0F1u))
                reg = a.src2Register & 0x3F, swizzle = a.src2Swizzle;
            else if (!a.src2Select && a.src1Select && literalAll(a.src2Register, a.src2Swizzle, written, 0x3F70F0F1u))
                reg = a.src1Register & 0x3F, swizzle = a.src1Swizzle;
            if (reg == UINT32_MAX)
                continue;
            scaled = true;
            // Each colour component read back to the instruction that last wrote it (its vector or its scalar result:
            // the lit shaders clamp their channels one by one with scalar saturates).
            for (uint32_t i = 0; i < 3; i++)
            {
                if (!((written >> i) & 1))
                    continue;
                const uint32_t component = ((swizzle >> (i * 2)) + i) & 3;
                for (size_t m = n; m-- > 0;)
                {
                    const AluInstruction& p = alus[m].second;
                    if (p.exportData)
                        continue;
                    if (p.vectorDest == reg && ((p.vectorWriteMask >> component) & 1))
                    {
                        if (p.vectorSaturate)
                            unclampAlu.insert(alus[m].first * 2);
                        break;
                    }
                    if (p.scalarDest == reg && ((p.scalarWriteMask >> component) & 1))
                    {
                        if (p.scalarSaturate)
                            unclampAlu.insert(alus[m].first * 2 + 1);
                        break;
                    }
                }
            }
        }
        if (scaled)
            for (auto& [address, a] : alus)
                if (a.exportData && a.vectorSaturate && a.vectorDest == uint32_t(ExportRegister::PSColor0))
                    unclampAlu.insert(address * 2);
    }

    std::vector<uint32_t> caseLabels;
    if (!simpleControlFlow)
    {
        pc = b.LocalVariable(b.U32(), b.ConstU32(0));
        for (uint32_t id = 0; id < 32; id++)
            loopCount[id] = b.LocalVariable(b.I32(), b.ConstI32(0));
        maskedExecution = isPixelShader && allowMaskedExecution;
        if (maskedExecution)
        {
            skip = b.LocalVariable(b.U32(), b.ConstU32(0));
            xeInactive = b.LocalVariable(b.Bool(), b.ConstBool(false));
            for (uint32_t i = 0; i < 32; i++)
                saveR[i] = b.LocalVariable(b.Vec(4), zero4);
            saveA0 = b.LocalVariable(b.I32(), b.ConstI32(0));
            saveP0 = b.LocalVariable(b.Bool(), b.ConstBool(false));
            savePs = b.LocalVariable(b.F32(), F(0.0f));
            for (uint32_t i = 0; i < 4; i++)
                saveOC[i] = b.LocalVariable(b.Vec(4), zero4);
            saveODepth = b.LocalVariable(b.F32(), F(0.0f));
        }
        // while (true) { switch (pc) { case 0: ... } break; }
        const uint32_t switchBlock = b.Id();
        loopHeader = b.Id();
        loopMerge = b.Id();
        loopContinue = b.Id();
        switchMerge = b.Id();
        b.Label(loopHeader);
        b.LoopMerge(loopMerge, loopContinue);
        b.Branch(switchBlock);
        b.Label(switchBlock);
        const uint32_t selector = b.Load(b.U32(), pc);
        std::vector<std::pair<uint32_t, uint32_t>> cases;
        for (uint32_t i = 0; i < cfCount; i++)
        {
            caseLabels.push_back(b.Id());
            cases.push_back({ i, caseLabels.back() });
        }
        b.SelectionMerge(switchMerge);
        b.Switch(selector, switchMerge, cases);
    }

    auto booleanTest = [&](uint32_t boolAddress, bool condition) {
        // (g_Booleans & bit) != 0 when condition is set, == 0 otherwise.
        uint32_t bit = 0;
        if (auto it = boolConstants.find(boolAddress); it != boolConstants.end())
            bit = it->second;
        else
            warnings += std::format("undeclared boolean b{}; ", boolAddress);
        const uint32_t bits = b.Emit(spv::OpBitwiseAnd, b.U32(), { SharedU32(kSharedBooleans), b.ConstU32(bit) });
        return b.Emit(condition ? spv::OpINotEqual : spv::OpIEqual, b.Bool(), { bits, b.ConstU32(0) });
    };
    auto intConstant = [&](uint32_t loopId) {
        if (auto it = intConstants.find(loopId); it != intConstants.end())
            return it->second;
        warnings += std::format("undeclared loop constant i{}; ", loopId);
        return std::array<int32_t, 4>{};
    };
    auto setPcAndContinue = [&](uint32_t target) {
        b.Store(pc, b.ConstU32(target));
        b.Branch(loopContinue);
    };

    uint32_t pcIndex = 0;
    for (uint32_t at = 0, i = 0; at < instrSize; at += 12, i += 3)
    {
        code0 = code[i];
        code1 = code[i + 1] & 0xFFFF;
        code2 = (code[i + 1] >> 16) | (code[i + 2] << 16);
        code3 = code[i + 2] >> 16;
        for (auto& cf : controlFlow)
        {
            const uint32_t current = pcIndex++;
            if (!simpleControlFlow)
            {
                b.Label(caseLabels[current]);
                if (maskedExecution && current != 0)
                {
                    // if (skip == pc) skip = 0; xeInactive = skip != 0;
                    const uint32_t s = b.Load(b.U32(), skip);
                    const uint32_t reached = b.Emit(spv::OpIEqual, b.Bool(), { s, b.ConstU32(current) });
                    const uint32_t s2 = b.Emit(spv::OpSelect, b.U32(), { reached, b.ConstU32(0), s });
                    b.Store(skip, s2);
                    b.Store(xeInactive, b.Emit(spv::OpINotEqual, b.Bool(), { s2, b.ConstU32(0) }));
                }
            }

            uint32_t address = 0, count = 0, sequence = 0;
            bool shouldReturn = false;
            switch (cf.opcode)
            {
            case ControlFlowOpcode::Exec:
            case ControlFlowOpcode::ExecEnd:
                address = cf.exec.address;
                count = cf.exec.count;
                sequence = cf.exec.sequence;
                shouldReturn = cf.opcode == ControlFlowOpcode::ExecEnd;
                break;
            case ControlFlowOpcode::CondExec:
            case ControlFlowOpcode::CondExecEnd:
            case ControlFlowOpcode::CondExecPredClean:
            case ControlFlowOpcode::CondExecPredCleanEnd:
                address = cf.condExec.address;
                count = cf.condExec.count;
                sequence = cf.condExec.sequence;
                shouldReturn = cf.opcode == ControlFlowOpcode::CondExecEnd || cf.opcode == ControlFlowOpcode::CondExecPredCleanEnd;
                break;
            case ControlFlowOpcode::CondExecPred:
            case ControlFlowOpcode::CondExecPredEnd:
                address = cf.condExecPred.address;
                count = cf.condExecPred.count;
                sequence = cf.condExecPred.sequence;
                shouldReturn = cf.opcode == ControlFlowOpcode::CondExecPredEnd;
                break;
            case ControlFlowOpcode::Alloc:
                if (cf.alloc.allocType == 3)
                    usesMemExport = true;
                break;
            case ControlFlowOpcode::LoopStart:
            {
                const uint32_t loopId = cf.loopStart.loopId;
                const auto ic = intConstant(loopId);
                if (simpleControlFlow)
                {
                    // aL = i.y; for (loop = 0; loop < i.x; loop++, aL += i.z) {
                    b.Store(aL, b.ConstI32(ic[1]));
                    SimpleLoop loop{ b.Id(), b.Id(), b.Id(), b.LocalVariable(b.I32(), b.ConstI32(0)), loopId };
                    b.Store(loop.counter, b.ConstI32(0));
                    const uint32_t body = b.Id();
                    b.Label(loop.header);
                    const uint32_t more = b.Emit(spv::OpSLessThan, b.Bool(), { b.Load(b.I32(), loop.counter), b.ConstI32(ic[0]) });
                    b.LoopMerge(loop.merge, loop.continueTarget);
                    b.BranchConditional(more, body, loop.merge);
                    b.Label(body);
                    simpleLoops.push_back(loop);
                }
                else
                {
                    b.Store(loopCount[loopId], b.ConstI32(ic[0]));
                    b.Store(aL, b.ConstI32(ic[1]));
                    auto scope = b.BeginIf(b.Emit(spv::OpSLessThanEqual, b.Bool(), { b.Load(b.I32(), loopCount[loopId]), b.ConstI32(0) }));
                    setPcAndContinue(cf.loopStart.address);
                    b.EndIf(scope);
                }
                break;
            }
            case ControlFlowOpcode::LoopEnd:
            {
                const uint32_t loopId = cf.loopEnd.loopId;
                const auto ic = intConstant(loopId);
                auto predicateMatches = [&] {
                    uint32_t p = b.Load(b.Bool(), p0);
                    return cf.loopEnd.condition ? p : Not(p);
                };
                if (simpleControlFlow)
                {
                    if (simpleLoops.empty())
                    {
                        warnings += "loop end without a loop; ";
                        break;
                    }
                    const SimpleLoop loop = simpleLoops.back();
                    simpleLoops.pop_back();
                    if (cf.loopEnd.isPredicatedBreak)
                    {
                        auto scope = b.BeginIf(predicateMatches());
                        b.Branch(loop.merge);
                        b.EndIf(scope);
                    }
                    b.Branch(loop.continueTarget);
                    b.Label(loop.continueTarget);
                    b.Store(loop.counter, b.Emit(spv::OpIAdd, b.I32(), { b.Load(b.I32(), loop.counter), b.ConstI32(1) }));
                    b.Store(aL, b.Emit(spv::OpIAdd, b.I32(), { b.Load(b.I32(), aL), b.ConstI32(ic[2]) }));
                    b.Branch(loop.header);
                    b.Label(loop.merge);
                }
                else
                {
                    if (cf.loopEnd.isPredicatedBreak)
                    {
                        auto scope = b.BeginIf(predicateMatches());
                        b.Store(loopCount[loopId], b.ConstI32(0));
                        b.EndIf(scope);
                    }
                    b.Store(aL, b.Emit(spv::OpIAdd, b.I32(), { b.Load(b.I32(), aL), b.ConstI32(ic[2]) }));
                    const uint32_t left = b.Emit(spv::OpISub, b.I32(), { b.Load(b.I32(), loopCount[loopId]), b.ConstI32(1) });
                    b.Store(loopCount[loopId], left);
                    auto scope = b.BeginIf(b.Emit(spv::OpSGreaterThan, b.Bool(), { left, b.ConstI32(0) }));
                    setPcAndContinue(cf.loopEnd.address);
                    b.EndIf(scope);
                }
                break;
            }
            case ControlFlowOpcode::CondJmp:
            {
                const uint32_t target = cf.condJmp.address;
                auto jumpCondition = [&]() -> uint32_t {
                    if (cf.condJmp.isUnconditional)
                        return 0;
                    if (cf.condJmp.isPredicated)
                    {
                        const uint32_t p = b.Load(b.Bool(), p0);
                        return cf.condJmp.condition ? p : Not(p);
                    }
                    return booleanTest(cf.condJmp.boolAddress, cf.condJmp.condition);
                };
                if (maskedExecution && target > current)
                {
                    // Forward: active lanes that take it go inactive until the target.
                    uint32_t take = b.Emit(spv::OpIEqual, b.Bool(), { b.Load(b.U32(), skip), b.ConstU32(0) });
                    if (const uint32_t c = jumpCondition())
                        take = And(take, c);
                    b.Store(skip, b.Emit(spv::OpSelect, b.U32(), { take, b.ConstU32(target), b.Load(b.U32(), skip) }));
                    break;
                }
                if (const uint32_t c = jumpCondition())
                {
                    auto scope = b.BeginIf(c);
                    setPcAndContinue(target);
                    b.EndIf(scope);
                }
                else
                {
                    setPcAndContinue(target);
                }
                break;
            }
            default:
                break;
            }

            if (maskedExecution && count > 0)
            {
                for (uint32_t k = 0; k < 32; k++)
                    b.Store(saveR[k], b.Load(b.Vec(4), r[k]));
                b.Store(saveA0, b.Load(b.I32(), a0));
                b.Store(saveP0, b.Load(b.Bool(), p0));
                b.Store(savePs, b.Load(b.F32(), ps));
                for (uint32_t k = 0; k < 4; k++)
                    if (outputs & (PIXEL_SHADER_OUTPUT_COLOR0 << k))
                        b.Store(saveOC[k], b.Load(b.Vec(4), oC[k]));
                if (outputs & PIXEL_SHADER_OUTPUT_DEPTH)
                    b.Store(saveODepth, b.Load(b.F32(), oDepth));
            }

            const be<uint32_t>* instructionCode = code + address * 3;
            for (uint32_t k = 0; k < count; k++)
            {
                union
                {
                    VertexFetchInstruction vertexFetch;
                    TextureFetchInstruction textureFetch;
                    AluInstruction alu;
                    struct { uint32_t w0, w1, w2; };
                };
                w0 = instructionCode[0];
                w1 = instructionCode[1];
                w2 = instructionCode[2];
                // A predicated exec block runs its instructions under the block's predicate.
                if (cf.opcode == ControlFlowOpcode::CondExecPred || cf.opcode == ControlFlowOpcode::CondExecPredEnd)
                {
                    const uint32_t condition = cf.condExecPred.condition;
                    if ((sequence & 0x1) != 0)
                    {
                        if (!(w1 >> 31))
                            w1 |= 1u << 31, w2 = (w2 & 0x7FFFFFFFu) | condition << 31;
                    }
                    else if (!((w1 >> 28) & 1))
                    {
                        w1 |= 1u << 28;
                        w1 = (w1 & ~(1u << 27)) | condition << 27;
                    }
                }
                if ((sequence & 0x1) != 0)
                {
                    if (vertexFetch.opcode == FetchOpcode::VertexFetch)
                        recompile(vertexFetch);
                    else
                        recompile(textureFetch);
                }
                else
                {
                    aluAddress = address + k;
                    recompile(alu);
                }
                sequence >>= 2;
                instructionCode += 3;
            }

            if (maskedExecution && count > 0)
            {
                // if (xeInactive) restore everything the block may have written.
                const uint32_t inactive = b.Load(b.Bool(), xeInactive);
                auto restore = [&](uint32_t var, uint32_t save, uint32_t type, uint32_t n) {
                    const uint32_t cond = n == 1 ? inactive : b.Emit(spv::OpCompositeConstruct, BoolType(n), SpirvBuilder::Words(n, inactive));
                    b.Store(var, b.Emit(spv::OpSelect, type, { cond, b.Load(type, save), b.Load(type, var) }));
                };
                for (uint32_t k = 0; k < 32; k++)
                    restore(r[k], saveR[k], b.Vec(4), 4);
                restore(a0, saveA0, b.I32(), 1);
                restore(p0, saveP0, b.Bool(), 1);
                restore(ps, savePs, b.F32(), 1);
                for (uint32_t k = 0; k < 4; k++)
                    if (outputs & (PIXEL_SHADER_OUTPUT_COLOR0 << k))
                        restore(oC[k], saveOC[k], b.Vec(4), 4);
                if (outputs & PIXEL_SHADER_OUTPUT_DEPTH)
                    restore(oDepth, saveODepth, b.F32(), 1);
            }

            if (shouldReturn)
            {
                std::vector<SpirvBuilder::IfScope> scopes;
                if (maskedExecution)
                    scopes.push_back(b.BeginIf(Not(b.Load(b.Bool(), xeInactive))));
                const bool wasMasked = maskedExecution;
                maskedExecution = false;   // the alpha test's kill is already inside the active-lane test
                ExitPixelOrVertex();
                maskedExecution = wasMasked;
                if (simpleControlFlow)
                {
                    WriteOutputs();
                    b.Return();
                }
                else
                {
                    b.Branch(switchMerge);
                }
                for (auto it = scopes.rbegin(); it != scopes.rend(); ++it)
                    b.EndIf(*it);
            }
        }
    }

    if (!simpleControlFlow)
    {
        // } (switch) break; } (while)
        if (!b.Terminated())
            b.Branch(switchMerge);
        b.Label(switchMerge);
        b.Branch(loopMerge);
        b.Label(loopContinue);
        b.Branch(loopHeader);
        b.Label(loopMerge);
    }
    if (!b.Terminated())
    {
        WriteOutputs();
        b.Return();
    }

    b.EntryPoint(isPixelShader ? spv::ExecutionModelFragment : spv::ExecutionModelVertex, function, interfaceVariables);
    spirv = b.Finish(spv::AddressingModelPhysicalStorageBuffer64, spv::MemoryModelGLSL450);
}

} // namespace xenos
