#pragma once
// Guest shaders -> host SPIR-V: the shader container is rebuilt from the guest D3D shader object (virtual part copied
// in the object, microcode in physical memory) and translated to SPIR-V directly (xenos/spirv_recompiler.h), or with
// the HLSL backend (HlslBackend) to HLSL by XenosRecomp and compiled with DXC. Results are cached by content hash.

#include <plume_render_interface.h>

#include <cstdint>
#include <memory>
#include <vector>

namespace ac6::gpu {

// A vertex shader input as its vfetch instruction reads it (mirrors xenos::VertexInput).
struct ShaderVertexInput
{
    uint32_t location = 0;
    uint32_t fetchSlot = 0;      // vertex fetch constant index (stream = 95 - fetchSlot)
    int32_t offset = 0;          // dwords
    uint32_t stride = 0;         // dwords (0 = unknown, mini fetch only)
    uint32_t format = 0;         // GPU vertex format
    bool isSigned = false;
    bool isInteger = false;
};

struct HostShader
{
    uint64_t hash = 0;
    // The hash the renderer's shader lists know it by: `hash` without the state of the game's lighting in float, which
    // marks dozens of shaders (the other options' states stay in it).
    uint64_t idHash = 0;
    bool codeScale = false;   // holds the game's code constants (240/255, 241/255): its lighting in float handles it
    // Its kind among those: the lit shaders' encode (its 10/255, for the codes above 240 that stand for lit brightness
    // 1.1 to 1.5) or a decoder (the emitter test 241/255). A code shader with neither writes plain colour (the effects).
    bool litEncode = false, decoder = false;
    // The game's light inputs by their constant-table names (for the relighting): the register
    // (VS or PS file, as the shader reads it) and its kind (0 to 10, listed in shaders.cpp NoteLightConstants and
    // scaled per kind by renderer.cpp Relight).
    struct LightConstant { uint16_t reg; uint8_t kind; };
    std::vector<LightConstant> lightConstants;
    // Reads a constant named Cloud... (the cloud puffs', the flat clouds' and the background clouds'
    // shaders all read CloudFogColor_Param, nothing else in the game's shaders reads a cloud constant): a cloud's draw.
    bool cloud = false;
    // The game's name for it: its source file (the container's debug block holds the path), the folder and the name,
    // e.g. "Effect/AfterBurner". An effect's draw is named by its vertex shader (its pixel shader is shared).
    std::string source;
    bool effect = false;   // from the Effect folder, its clouds aside: particles, fire, trails, tracers, glows
    // The map's pixel shaders (the ground and the sea): they read the sea's reflection of the scene where the draw
    // gives one.
    bool seaReflection = false;
    // The terrain and what grows on it: the map's pixel shaders and the trees' (point sprites). The ambient
    // occlusion leaves their pixels alone.
    bool aoTerrain = false;
    bool pixel = false;
    bool ok = false;
    uint32_t variant = 0;               // vertex shaders: container shader record (see NoteVertexShaderVariant)
    uint32_t specConstantsMask = 0;
    std::vector<uint8_t> container;     // rebuilt guest container (virtual + physical)
    std::vector<ShaderVertexInput> vertexInputs;   // vertex shaders
    uint32_t indexedFetchStreams = 0;   // vertex shaders: streams read at a computed index (whole buffer bound)
    std::vector<uint32_t> samplerOverrides;   // fetches with their own filters (ShaderRecompiler::samplerOverrides)
    bool drawFlags = false;             // reads the SPEC_CONSTANT_* bits from the shared constants (SPIR-V backend)
    bool motionOutput = false;          // pixel shaders: writes TAA's motion at colour output 1 when asked
    // Vertex shaders: the float register of the matrix the position is transformed with, by the constant table's names
    // (TAA's motion vectors), UINT32_MAX when none is known.
    uint32_t positionMatrix = UINT32_MAX;
    // The other camera matrices it may read, by name (the planar reflection mirrors them with the position matrix):
    // InverseWorldViewProjectionMatrix (4 registers), InverseViewMatrix (its first 2 rows: the camera's right and up, for
    // sprites facing it), ViewMatrix (4), WorldInverseViewMatrix (2: those axes in the object's space). UINT32_MAX: not read.
    uint32_t inverseViewProjectionMatrix = UINT32_MAX, inverseViewMatrix = UINT32_MAX, viewMatrix = UINT32_MAX,
        worldInverseViewMatrix = UINT32_MAX;
    // The float constant registers it reads, [constantFirst, constantEnd): what a draw uploads.
    // The HLSL backend's shaders: the whole file.
    uint32_t constantFirst = 0, constantEnd = 256;
    std::vector<uint32_t> spirv;
    // The shader module, shared by every shader translating to the same SPIR-V (D3D's vertex
    // shaders patched per vertex declaration mostly translate alike), and that SPIR-V's hash: pipelines and
    // their libraries are made per module, so a shader first met in a mission finds them built.
    std::shared_ptr<plume::RenderShader> shader;
    uint64_t codeHash = 0;
    uint32_t textureSlots = 0;          // fetch slots its texture fetches read (translated this run only: diagnostics)
    // Translation options it depends on (bit 0 High quality DoF, bit 1 round bloom, 0: none) and
    // those it was translated with. Such shaders are kept per option state (in their hash) and not in the disk cache.
    uint32_t optionUse = 0, options = 0;
};

// AC6_EDRAM_SWIZZLE=1 (or the HLSL backend, which doesn't sample the game's un-swizzles 1:1): emulate the EDRAM sample
// interleave between 1x and promoted 4x surfaces.
bool EdramSwizzle();

// AC6_SHADER_BACKEND=hlsl (or AC6_SHADER_PATCH): the game's shaders through HLSL and DXC.
bool HlslBackend();

// The SPIR-V backend's constants from uniform buffers (set 4) rather than buffer addresses (AC6_UNIFORM_CONSTANTS=0
// turns it off, never with the HLSL backend).
bool UniformConstants();

// Gives `shader` its module (one per distinct SPIR-V) and code hash.
void MakeShaderModule(HostShader& shader);

// The host shader for a guest vertex/pixel shader object (translated on first use). Never null for a
// valid object. Check `ok`.
HostShader* GetShader(uint32_t guestObject, bool pixel);

// Shader warm-up: D3D finished creating a shader object (container copied, microcode
// in place). It is queued for background threads to translate and compile, so its first draw finds it
// ready. AC6_SHADER_WARMUP=0 turns it off.
void NoteShaderObjectCreated(uint32_t guestObject);

// Shaders translated and compiled so far (not from the cache), and the time it took (diagnostics).
struct ShaderCompileStats { uint64_t count = 0; int64_t ns = 0; };
ShaderCompileStats GetShaderCompileStats();

// D3D loads vertex shader record `variant` of the object's container (its loader 0x821ECEB0 picks
// it per draw). Later GetShader calls for the object translate that record.
void NoteVertexShaderVariant(uint32_t guestObject, uint32_t variant);

// After D3D's vertex shader loader ran: captures the microcode it copied into the command buffer at
// `writePointer` (its IM_LOAD_IMMEDIATE packet), which D3D patched for the current vertex declaration.
// Later GetShader calls for the object translate that copy.
void NoteVertexShaderLoad(uint32_t guestObject, uint32_t variant, uint32_t writePointer);
// D3D re-patched the object's own microcode in place for the bound declaration (sub_821ED0D0) and
// loads it from there: the last immediate copy no longer applies.
void NoteVertexShaderPatchedInPlace(uint32_t guestObject, uint32_t variant);

// Frame capture / replay (gpu/capture.h): the per-object shader notes above, as bytes, and a load
// replayed from its recorded microcode.
std::vector<uint8_t> ExportVertexShaderState();
void ImportVertexShaderState(const uint8_t* data, size_t size);
// Changes whenever that state does (the pipeline ring exports it only then).
uint64_t VertexShaderStateEpoch();
void ReplayVertexShaderLoad(uint32_t guestObject, uint32_t offset, std::vector<uint8_t> bytes);

} // namespace ac6::gpu
