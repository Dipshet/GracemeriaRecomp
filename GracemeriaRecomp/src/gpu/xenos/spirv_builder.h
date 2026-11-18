#pragma once
// A minimal SPIR-V module builder for the shader translator: ids, deduplicated types
// and constants, the module's sections, and one function body written block by block. Instructions are
// encoded directly (word count << 16 | opcode). Nothing is validated here (spirv-val can check the output).

#include <spirv/unified1/GLSL.std.450.h>
#include <spirv/unified1/spirv.hpp>

#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <map>
#include <set>
#include <string_view>
#include <vector>

namespace xenos {

class SpirvBuilder
{
public:
    using Words = std::vector<uint32_t>;

    uint32_t Id() { return nextId++; }

    // --- Module sections ---------------------------------------------------------------------------
    void Capability(spv::Capability c)
    {
        if (capabilitySet.insert(uint32_t(c)).second)
            Op(capabilities, spv::OpCapability, { uint32_t(c) });
    }
    void Extension(std::string_view name) { OpString(extensions, spv::OpExtension, {}, name, {}); }
    uint32_t Glsl()
    {
        if (!glslId)
        {
            glslId = Id();
            OpString(imports, spv::OpExtInstImport, { glslId }, "GLSL.std.450", {});
        }
        return glslId;
    }
    void Name(uint32_t id, std::string_view name) { OpString(debug, spv::OpName, { id }, name, {}); }
    void Decorate(uint32_t id, spv::Decoration d, std::initializer_list<uint32_t> extra = {})
    {
        Words w = { id, uint32_t(d) };
        w.insert(w.end(), extra);
        Op(annotations, spv::OpDecorate, w);
    }
    void MemberDecorate(uint32_t id, uint32_t member, spv::Decoration d, std::initializer_list<uint32_t> extra = {})
    {
        Words w = { id, member, uint32_t(d) };
        w.insert(w.end(), extra);
        Op(annotations, spv::OpMemberDecorate, w);
    }

    // --- Types (deduplicated) ------------------------------------------------------------------------
    uint32_t TypeVoid() { return Type(spv::OpTypeVoid, {}); }
    uint32_t TypeBool() { return Type(spv::OpTypeBool, {}); }
    uint32_t TypeInt(uint32_t bits, bool isSigned) { return Type(spv::OpTypeInt, { bits, isSigned ? 1u : 0u }); }
    uint32_t TypeFloat() { return Type(spv::OpTypeFloat, { 32 }); }
    uint32_t TypeVector(uint32_t component, uint32_t count) { return count == 1 ? component : Type(spv::OpTypeVector, { component, count }); }
    uint32_t TypePointer(spv::StorageClass storage, uint32_t type) { return Type(spv::OpTypePointer, { uint32_t(storage), type }); }
    uint32_t TypeFunction(uint32_t result) { return Type(spv::OpTypeFunction, { result }); }
    uint32_t TypeImage(spv::Dim dim) { return Type(spv::OpTypeImage, { TypeFloat(), uint32_t(dim), 2, 0, 0, 1, uint32_t(spv::ImageFormatUnknown) }); }
    uint32_t TypeSampler() { return Type(spv::OpTypeSampler, {}); }
    uint32_t TypeSampledImage(uint32_t image) { return Type(spv::OpTypeSampledImage, { image }); }
    uint32_t TypeRuntimeArray(uint32_t element) { return Type(spv::OpTypeRuntimeArray, { element }); }
    uint32_t TypeArray(uint32_t element, uint32_t length) { return Type(spv::OpTypeArray, { element, ConstU32(length) }); }
    // Structs are never deduplicated (they carry decorations).
    uint32_t TypeStruct(const Words& members)
    {
        const uint32_t id = Id();
        Words w = { id };
        w.insert(w.end(), members.begin(), members.end());
        Op(globals, spv::OpTypeStruct, w);
        return id;
    }

    uint32_t F32() { return TypeFloat(); }
    uint32_t U32() { return TypeInt(32, false); }
    uint32_t I32() { return TypeInt(32, true); }
    uint32_t U64() { return TypeInt(64, false); }
    uint32_t Bool() { return TypeBool(); }
    uint32_t Vec(uint32_t n) { return TypeVector(F32(), n); }

    // --- Constants (deduplicated) --------------------------------------------------------------------
    uint32_t ConstU32(uint32_t v) { return Constant(spv::OpConstant, U32(), { v }); }
    uint32_t ConstI32(int32_t v) { return Constant(spv::OpConstant, I32(), { uint32_t(v) }); }
    uint32_t ConstU64(uint64_t v) { return Constant(spv::OpConstant, U64(), { uint32_t(v), uint32_t(v >> 32) }); }
    uint32_t ConstF32(float v)
    {
        uint32_t bits;
        std::memcpy(&bits, &v, 4);
        return Constant(spv::OpConstant, F32(), { bits });
    }
    uint32_t ConstF32Bits(uint32_t bits) { return Constant(spv::OpConstant, F32(), { bits }); }
    uint32_t ConstBool(bool v) { return Constant(v ? spv::OpConstantTrue : spv::OpConstantFalse, Bool(), {}); }
    uint32_t ConstComposite(uint32_t type, const Words& parts) { return Constant(spv::OpConstantComposite, type, parts); }
    uint32_t ConstVec(uint32_t n, float v)
    {
        if (n == 1)
            return ConstF32(v);
        return ConstComposite(Vec(n), Words(n, ConstF32(v)));
    }

    // A module-scope variable.
    uint32_t GlobalVariable(uint32_t pointerType, spv::StorageClass storage)
    {
        const uint32_t id = Id();
        Op(globals, spv::OpVariable, { pointerType, id, uint32_t(storage) });
        return id;
    }
    uint32_t SpecConstantU32(uint32_t value)
    {
        const uint32_t id = Id();
        Op(globals, spv::OpSpecConstant, { U32(), id, value });
        return id;
    }

    // --- Function body -------------------------------------------------------------------------------
    // A Function-storage variable (declared in the function's first block, as SPIR-V requires).
    uint32_t LocalVariable(uint32_t valueType, uint32_t initializer = 0)
    {
        const uint32_t id = Id();
        if (initializer)
            Op(functionVariables, spv::OpVariable, { TypePointer(spv::StorageClassFunction, valueType), id, uint32_t(spv::StorageClassFunction), initializer });
        else
            Op(functionVariables, spv::OpVariable, { TypePointer(spv::StorageClassFunction, valueType), id, uint32_t(spv::StorageClassFunction) });
        return id;
    }

    // An instruction with a result: returns the result id.
    uint32_t Emit(spv::Op op, uint32_t resultType, std::initializer_list<uint32_t> operands)
    {
        EnsureBlock();
        const uint32_t id = Id();
        Words w = { resultType, id };
        w.insert(w.end(), operands);
        Op(body, op, w);
        return id;
    }
    uint32_t Emit(spv::Op op, uint32_t resultType, const Words& operands)
    {
        EnsureBlock();
        const uint32_t id = Id();
        Words w = { resultType, id };
        w.insert(w.end(), operands.begin(), operands.end());
        Op(body, op, w);
        return id;
    }
    // An instruction without a result.
    void EmitVoid(spv::Op op, std::initializer_list<uint32_t> operands) { EnsureBlock(); Op(body, op, Words(operands)); }
    void EmitVoid(spv::Op op, const Words& operands) { EnsureBlock(); Op(body, op, operands); }
    uint32_t Ext(uint32_t resultType, GLSLstd450 inst, std::initializer_list<uint32_t> operands)
    {
        Words w = { Glsl(), uint32_t(inst) };
        w.insert(w.end(), operands);
        return Emit(spv::OpExtInst, resultType, w);
    }

    uint32_t Load(uint32_t type, uint32_t pointer) { return Emit(spv::OpLoad, type, { pointer }); }
    void Store(uint32_t pointer, uint32_t value) { EmitVoid(spv::OpStore, { pointer, value }); }

    // Blocks. `terminated` is set by the terminators below. Code emitted while it is set goes into a new,
    // unreachable block (SPIR-V allows those), so callers need not track reachability themselves.
    void Label(uint32_t id)
    {
        if (!terminated)
            Op(body, spv::OpBranch, { id });
        Op(body, spv::OpLabel, { id });
        terminated = false;
    }
    void Branch(uint32_t target)
    {
        EnsureBlock();
        Op(body, spv::OpBranch, { target });
        terminated = true;
    }
    void BranchConditional(uint32_t condition, uint32_t ifTrue, uint32_t ifFalse)
    {
        EnsureBlock();
        Op(body, spv::OpBranchConditional, { condition, ifTrue, ifFalse });
        terminated = true;
    }
    void SelectionMerge(uint32_t merge) { EnsureBlock(); Op(body, spv::OpSelectionMerge, { merge, 0 }); }
    void LoopMerge(uint32_t merge, uint32_t continueTarget) { EnsureBlock(); Op(body, spv::OpLoopMerge, { merge, continueTarget, 0 }); }
    void Switch(uint32_t selector, uint32_t defaultTarget, const std::vector<std::pair<uint32_t, uint32_t>>& cases)
    {
        EnsureBlock();
        Words w = { selector, defaultTarget };
        for (auto [value, target] : cases)
            w.insert(w.end(), { value, target });
        Op(body, spv::OpSwitch, w);
        terminated = true;
    }
    void Return() { EnsureBlock(); Op(body, spv::OpReturn, {}); terminated = true; }
    void Kill() { EnsureBlock(); Op(body, spv::OpKill, {}); terminated = true; }
    bool Terminated() const { return terminated; }

    // if (condition) { ... } : BeginIf, then the body, then EndIf.
    struct IfScope { uint32_t merge; };
    IfScope BeginIf(uint32_t condition)
    {
        const uint32_t then = Id(), merge = Id();
        SelectionMerge(merge);
        BranchConditional(condition, then, merge);
        Op(body, spv::OpLabel, { then });
        terminated = false;
        return { merge };
    }
    void EndIf(IfScope scope)
    {
        if (!terminated)
            Op(body, spv::OpBranch, { scope.merge });
        Op(body, spv::OpLabel, { scope.merge });
        terminated = false;
    }

    // --- Finishing ----------------------------------------------------------------------------------
    void BeginFunction(uint32_t function, uint32_t resultType, uint32_t functionType)
    {
        Op(functionHeader, spv::OpFunction, { resultType, function, 0, functionType });
        firstLabel = Id();
        Op(body, spv::OpLabel, { firstLabel });
        terminated = false;
    }
    void EntryPoint(spv::ExecutionModel model, uint32_t function, const Words& interface)
    {
        Words w = { uint32_t(model), function };
        StringWords(w, "main");
        w.insert(w.end(), interface.begin(), interface.end());
        Op(entryPoints, spv::OpEntryPoint, w);
    }
    void ExecutionMode(uint32_t function, spv::ExecutionMode mode) { Op(executionModes, spv::OpExecutionMode, { function, uint32_t(mode) }); }

    Words Finish(uint32_t addressing, uint32_t memoryModel)
    {
        if (!terminated)
            Return();
        Words out = { spv::MagicNumber, 0x00010300, 0, nextId, 0 };   // SPIR-V 1.3
        auto add = [&out](const Words& w) { out.insert(out.end(), w.begin(), w.end()); };
        add(capabilities);
        add(extensions);
        add(imports);
        Words model;
        Op(model, spv::OpMemoryModel, { addressing, memoryModel });
        add(model);
        add(entryPoints);
        add(executionModes);
        add(debug);
        add(annotations);
        add(globals);
        add(functionHeader);
        // The first block: its label, then the function's variables, then its code.
        out.insert(out.end(), body.begin(), body.begin() + 2);
        add(functionVariables);
        out.insert(out.end(), body.begin() + 2, body.end());
        Words end;
        Op(end, spv::OpFunctionEnd, {});
        add(end);
        out[3] = nextId;
        return out;
    }

private:
    uint32_t nextId = 1;
    uint32_t glslId = 0;
    uint32_t firstLabel = 0;
    bool terminated = true;
    Words capabilities, extensions, imports, entryPoints, executionModes, debug, annotations, globals;
    Words functionHeader, functionVariables, body;
    std::set<uint32_t> capabilitySet;
    std::map<Words, uint32_t> types, constants;

    static void Op(Words& section, spv::Op op, const Words& operands)
    {
        section.push_back(uint32_t(operands.size() + 1) << 16 | uint32_t(op));
        section.insert(section.end(), operands.begin(), operands.end());
    }
    static void StringWords(Words& w, std::string_view s)
    {
        const size_t words = s.size() / 4 + 1;
        const size_t at = w.size();
        w.resize(at + words, 0);
        std::memcpy(w.data() + at, s.data(), s.size());
    }
    static void OpString(Words& section, spv::Op op, const Words& before, std::string_view s, const Words& after)
    {
        Words w = before;
        StringWords(w, s);
        w.insert(w.end(), after.begin(), after.end());
        Op(section, op, w);
    }
    void EnsureBlock()
    {
        if (terminated)
        {
            const uint32_t id = Id();
            Op(body, spv::OpLabel, { id });
            terminated = false;
        }
    }
    uint32_t Type(spv::Op op, const Words& operands)
    {
        Words key = { uint32_t(op) };
        key.insert(key.end(), operands.begin(), operands.end());
        if (auto it = types.find(key); it != types.end())
            return it->second;
        const uint32_t id = Id();
        Words w = { id };
        w.insert(w.end(), operands.begin(), operands.end());
        Op(globals, op, w);
        types.emplace(std::move(key), id);
        return id;
    }
    uint32_t Constant(spv::Op op, uint32_t type, const Words& operands)
    {
        Words key = { uint32_t(op), type };
        key.insert(key.end(), operands.begin(), operands.end());
        if (auto it = constants.find(key); it != constants.end())
            return it->second;
        const uint32_t id = Id();
        Words w = { type, id };
        w.insert(w.end(), operands.begin(), operands.end());
        Op(globals, op, w);
        constants.emplace(std::move(key), id);
        return id;
    }
};

} // namespace xenos
