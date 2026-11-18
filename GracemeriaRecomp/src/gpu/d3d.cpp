// Native renderer: hooks on the game's Xbox D3D functions.
//
// The guest D3D library keeps running and maintaining its own device state (its GPU output goes
// nowhere: see gpu/no_gpu.cpp). These hooks hand the actions (clear, draw, swap) and the render
// states the renderer translates to gpu/renderer.cpp.
#include "gpu/d3d_census.h"
#include "gpu/frame_rate.h"
#include "gpu/renderer.h"
#include "gpu/shaders.h"
#include "kernel/function.h"
#include "log.h"
#include "memory.h"

#include <algorithm>
#include <atomic>
#include <utility>

namespace ac6::gpu {

// D3DDevice_Clear(device, flags, rects, color (float4*), z, stencil)
void Clear(uint32_t device, uint32_t flags, const be<float>* rects, const be<float>* color, double z, uint32_t stencil)
{
    float c[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    if (color)
        for (int i = 0; i < 4; i++)
            c[i] = color[i];
    RendererClear(device, flags, c, float(z), stencil);
}

// D3DDevice_Resolve(device, flags, srcRect, destTexture, destPoint, destLevel, destSlice, clearColor,
// clearZ (f1), clearStencil (first stack argument), parameters)
void Resolve(PPCContext& ctx)
{
    ResolveRect rect;
    ResolvePoint point;
    const ResolveRect* rectPtr = nullptr;
    const ResolvePoint* pointPtr = nullptr;
    if (ctx.r5.u32)
    {
        auto* r = g_memory.As<be<int32_t>>(ctx.r5.u32);
        rect = { r[0], r[1], r[2], r[3] };
        rectPtr = &rect;
    }
    if (ctx.r7.u32)
    {
        auto* p = g_memory.As<be<int32_t>>(ctx.r7.u32);
        point = { p[0], p[1] };
        pointPtr = &point;
    }
    float color[4];
    const float* colorPtr = nullptr;
    if (ctx.r10.u32)
    {
        auto* c = g_memory.As<be<float>>(ctx.r10.u32);
        for (int i = 0; i < 4; i++)
            color[i] = c[i];
        colorPtr = color;
    }
    const uint32_t stencil = *g_memory.As<be<uint32_t>>(ctx.r1.u32 + 0x54);
    RendererResolve(ctx.r3.u32, ctx.r4.u32, rectPtr, ctx.r6.u32, pointPtr, ctx.r8.u32, colorPtr, float(ctx.f1.f64), stencil & 0xFF);
}

// D3DDevice_BeginTiling(device, flags, count, rects (D3DRECT[count]), clearColor, clearZ, clearStencil)
void BeginTiling(PPCContext& ctx)
{
    ResolveRect rects[16];
    const uint32_t count = std::min<uint32_t>(ctx.r5.u32, 16);
    auto* r = ctx.r6.u32 ? g_memory.As<be<int32_t>>(ctx.r6.u32) : nullptr;
    for (uint32_t i = 0; r && i < count; i++)
        rects[i] = { r[i * 4], r[i * 4 + 1], r[i * 4 + 2], r[i * 4 + 3] };
    RendererBeginTiling(r ? rects : nullptr, r ? count : 0);
}

// Census indices of the functions hooked here (gpu/d3d_census.cpp counts and probes them).
const int kCensusDrawIndexed = CensusIndex("DrawIndexedVertices");
const int kCensusResolve = CensusIndex("Resolve");
const int kCensusBeginTiling = CensusIndex("BeginTiling");
const int kCensusEndTiling = CensusIndex("EndTiling");

} // namespace ac6::gpu

// --- Actions ------------------------------------------------------------------------------------------

extern "C" PPC_FUNC(__imp__sub_821E2440);   // D3DDevice_Clear
extern "C" PPC_FUNC(__imp__sub_821F03B0);   // D3D swap (VdSwap path)
extern "C" PPC_FUNC(__imp__sub_821DEF18);   // D3DDevice_DrawVertices(device, primType, startVertex, vertexCount)
extern "C" PPC_FUNC(__imp__sub_821DEED0);   // D3D inline quad-list draw (device, -, vertexCount, vertices)
extern "C" PPC_FUNC(__imp__sub_821DE308);   // D3DDevice_SetPixelShader(device, shader)
extern "C" PPC_FUNC(__imp__sub_821DE600);   // D3DDevice_SetVertexShader(device, shader)
extern "C" PPC_FUNC(__imp__sub_821DF300);   // D3DDevice_DrawIndexedVertices(device, primType, baseVertex, startIndex, indexCount)
extern "C" PPC_FUNC(__imp__sub_821E2BB8);   // D3DDevice_Resolve
extern "C" PPC_FUNC(__imp__sub_821E4630);   // D3DDevice_BeginTiling
extern "C" PPC_FUNC(__imp__sub_821E4AD0);   // D3DDevice_EndTiling
extern "C" PPC_FUNC(__imp__sub_821ECEB0);   // vertex shader load (shader record r10 of object r5)
extern "C" PPC_FUNC(__imp__sub_821ED0D0);   // vertex shader in-place patch cache (device, shader, declaration, record)
extern "C" PPC_FUNC(__imp__sub_821E64A8);   // wait for the GPU to drain (spins on device +11000)
extern "C" PPC_FUNC(__imp__sub_821EB8B8);   // gamma ramp to the display controller: 256-entry table (r4 = ramp)
extern "C" PPC_FUNC(__imp__sub_821EB998);   // gamma ramp to the display controller: piecewise linear (r4 = ramp)
namespace ac6::gpu { void SetSwapWait(bool waiting); }
extern "C" PPC_FUNC(__imp__sub_821DE508);   // vertex shader object init (r3 = object, r4 = microcode)
extern "C" PPC_FUNC(__imp__sub_821DE2D0);   // pixel shader object init (r3 = object, r4 = microcode)
extern "C" PPC_FUNC(__imp__sub_821EC478);   // draw-time flush: dirty float constants -> ring
extern "C" PPC_FUNC(__imp__sub_821EC0D8);   // draw-time flush: dirty register block -> ring

// Display gamma ramp. D3D's SetGammaRamp/SetPWLGamma (sub_821F00C0) correct the
// game's ramp for the display (VdGetCurrentDisplayGamma) and hand it to one of these two, which write
// it to the display controller's lookup table. The console applies that table to the scanned-out image.
// The 256-entry ramp is three u16 arrays (R, G, B: the top 10 bits are used). The piecewise linear one is
// {u16 base, u16 delta}[128] per channel (R, G, B), one segment per 8 of 1024 input steps.
PPC_FUNC(sub_821EB8B8)
{
    const uint32_t ramp = ctx.r4.u32;
    __imp__sub_821EB8B8(ctx, base);
    const auto* v = ac6::g_memory.As<ac6::be<uint16_t>>(ramp);
    uint16_t table[3][256];
    for (uint32_t c = 0; c < 3; c++)
        for (uint32_t i = 0; i < 256; i++)
            table[c][i] = v[c * 256 + i];
    ac6::gpu::RendererSetGammaTable(table);
}

// Shader warm-up: AC6 creates its shader objects from its shader package as it loads
// it (sub_8234FC38: each one's container copied into the object, its microcode copied out, then these
// set the object up), long before most are drawn.
PPC_FUNC(sub_821DE508)
{
    const uint32_t object = ctx.r3.u32;
    __imp__sub_821DE508(ctx, base);
    ac6::gpu::NoteShaderObjectCreated(object);
}

PPC_FUNC(sub_821DE2D0)
{
    const uint32_t object = ctx.r3.u32;
    __imp__sub_821DE2D0(ctx, base);
    ac6::gpu::NoteShaderObjectCreated(object);
}

PPC_FUNC(sub_821EB998)
{
    const uint32_t ramp = ctx.r4.u32;
    __imp__sub_821EB998(ctx, base);
    const auto* v = ac6::g_memory.As<ac6::be<uint16_t>>(ramp);
    uint16_t pwl[3][128][2];
    for (uint32_t c = 0; c < 3; c++)
        for (uint32_t i = 0; i < 128; i++)
        {
            pwl[c][i][0] = v[c * 256 + i * 2];
            pwl[c][i][1] = v[c * 256 + i * 2 + 1];
        }
    ac6::gpu::RendererSetGammaPwl(pwl);
}

PPC_FUNC(sub_821E2440)
{
    PPCContext saved = ctx;
    __imp__sub_821E2440(ctx, base);
    ac6::HostToGuestFunction<&ac6::gpu::Clear>(saved, base);
}

// Draws are translated after the original runs: D3D patches the bound vertex shader's fetch
// instructions for the current vertex declaration inside the draw call, and the translation reads
// the patched microcode.
PPC_FUNC(sub_821DEF18)
{
    PPCContext saved = ctx;
    ac6::gpu::NoteDrawDirty(ctx.r3.u32);
    __imp__sub_821DEF18(ctx, base);
    ac6::gpu::RendererDraw(saved.r3.u32, saved.r4.u32, saved.r5.u32, saved.r6.u32);
}

// The inline quad-list draw: the vertices go into the command ring with their own
// fetch constant, so the renderer is given them directly.
PPC_FUNC(sub_821DEED0)
{
    const uint32_t device = ctx.r3.u32, count = ctx.r5.u32, vertices = ctx.r6.u32;
    ac6::gpu::NoteDrawDirty(device);
    __imp__sub_821DEED0(ctx, base);
    ac6::gpu::RendererDrawQuadsUP(device, count, vertices);
}

PPC_FUNC(sub_821DF300)
{
    ac6::gpu::CensusNote(ac6::gpu::kCensusDrawIndexed, ctx);
    PPCContext saved = ctx;
    ac6::gpu::NoteDrawDirty(ctx.r3.u32);
    __imp__sub_821DF300(ctx, base);
    ac6::gpu::RendererDrawIndexed(saved.r3.u32, saved.r4.u32, saved.r5.s32, saved.r6.u32, saved.r7.u32);
}

PPC_FUNC(sub_821E2BB8)
{
    ac6::gpu::CensusNote(ac6::gpu::kCensusResolve, ctx);
    ac6::gpu::Resolve(ctx);
    __imp__sub_821E2BB8(ctx, base);
}

// Tiling state is set before the original runs: BeginTiling clears through D3DDevice_Clear, and
// EndTiling may resolve, both of which size the surfaces by the tiles.
PPC_FUNC(sub_821E4630)
{
    ac6::gpu::CensusNote(ac6::gpu::kCensusBeginTiling, ctx);
    ac6::gpu::BeginTiling(ctx);
    __imp__sub_821E4630(ctx, base);
}

PPC_FUNC(sub_821E4AD0)
{
    ac6::gpu::CensusNote(ac6::gpu::kCensusEndTiling, ctx);
    __imp__sub_821E4AD0(ctx, base);
    ac6::gpu::RendererEndTiling();
}

// The D3D vertex shader loader: which container record a draw uses.
PPC_FUNC(sub_821ECEB0)
{
    // Its IM_LOAD_IMMEDIATE packet starts at the command buffer write pointer (device +48): the
    // microcode copy, patched by D3D for the current vertex declaration, follows 24 bytes in.
    const uint32_t device = ctx.r3.u32, object = ctx.r5.u32, variant = ctx.r10.u32;
    const uint32_t writePointer = *ac6::g_memory.As<ac6::be<uint32_t>>(device + 48);
    ac6::gpu::NoteVertexShaderVariant(object, variant);
    __imp__sub_821ECEB0(ctx, base);
    ac6::gpu::NoteVertexShaderLoad(object, variant, writePointer);
}

// Xbox D3D keeps one patched copy of each vertex shader record in the shader's own microcode: when the
// declaration (or stream strides) matches what it was patched for, or the GPU is done with it, it is
// (re-)patched in place and loaded from there instead of through the immediate copy of sub_821ECEB0.
// Only the call that binds the draw's record (at 0x821ED594) changes what the next draw runs. The other call
// (0x821ED5F4) prepares record 1 ahead of time. A mid-asm hook marks the binding call
// (GracemeriaRecompLib/config/AC6.toml: the recompiler's skip_lr leaves the return address unset).
static thread_local bool t_vertexShaderBindCall = false;

void VertexShaderBindCall()
{
    t_vertexShaderBindCall = true;
}

PPC_FUNC(sub_821ED0D0)
{
    const uint32_t object = ctx.r4.u32, variant = ctx.r6.u32;
    const bool binding = std::exchange(t_vertexShaderBindCall, false);
    __imp__sub_821ED0D0(ctx, base);
    if (binding && ctx.r3.u32 != 0)
        ac6::gpu::NoteVertexShaderPatchedInPlace(object, variant);
}

// This wait (reached e.g. when a cutscene is skipped) queues a WAIT_UNTIL, kicks the ring and spins
// until the device's count of outstanding GPU work (+11000, raised by sub_821E5FD0) drops to zero,
// which only the console GPU does. The renderer's queued calls are that work: they are waited for, then
// the count is cleared before the original runs.
PPC_FUNC(sub_821E64A8)
{
    ac6::gpu::RendererWaitIdle();   // the renderer's queued calls are the GPU's outstanding work
    auto* outstanding = ac6::g_memory.As<ac6::be<uint32_t>>(ctx.r3.u32 + 11000);
    static std::atomic<uint32_t> logged{ 0 };
    if (logged.fetch_add(1) == 0)
        ac6::log::Info("GPU drain wait (sub_821E64A8): {} outstanding, completed immediately", uint32_t(*outstanding));
    *outstanding = 0u;
    __imp__sub_821E64A8(ctx, base);
}

// The present interval (device +13580, D3DPRESENT_INTERVAL_*) as sub_821EFE78 encodes it into the swap:
// 0/ONE -> 1 vblank, TWO -> 2, THREE (4) -> 3, IMMEDIATE (0x80000000) -> 0. The game asks for TWO in
// in-engine cutscenes and missions (from the intro flyover), ONE in menus, the briefing and the hangar.
PPC_FUNC(sub_821F03B0)
{
    const uint32_t interval = ac6::g_memory.As<ac6::be<uint32_t>>(ctx.r3.u32 + 13580)->get();
    const uint32_t vblanks = interval == 2 ? 2 : interval == 4 ? 3 : interval == 0x80000000u ? 0 : 1;
    static uint32_t last = ~0u;
    if (interval != last)
    {
        ac6::log::Info("Present interval {:X}: {} vblank(s) per frame (frame {})", interval, vblanks, ac6::gpu::RendererFrame());
        last = interval;
    }
    const uint32_t paced = ac6::gpu::NoteSwap(vblanks);   // AC6_FRAME_RATE (gpu/frame_rate.h)
    ac6::gpu::SetSwapWait(true);
    __imp__sub_821F03B0(ctx, base);
    ac6::gpu::SetSwapWait(false);
    ac6::gpu::RendererPresent(paced);
    // The settings overlay holds the game while it is open. The frame clock skips that time: CNuTimer::advance's next
    // delta is measured from the previous frame's tick (g_last_frame_tick), moved on by it.
    if (const uint64_t held = ac6::gpu::RendererHoldForOverlay())
    {
        auto* last = ac6::g_memory.As<ac6::be<uint64_t>>(0x826EDD98);
        *last = last->get() + held;
    }
}

// The draw cores' flush of dirty state into the command ring: the renderer reads state from
// the device's register shadow and constant files and nothing reads the ring's register and constant packets,
// so these two copies (only ring writes and the ring pointer, +48) are skipped. The packets the renderer does
// read (the vertex shader loads, sub_821ECEB0) are written at the ring pointer as usual. AC6_GUEST_PM4=1 runs
// them.
static bool GuestPm4()
{
    static const bool enabled = [] { const char* v = std::getenv("AC6_GUEST_PM4"); return v && *v == '1'; }();
    return enabled;
}

PPC_FUNC(sub_821EC478)
{
    if (GuestPm4())
        __imp__sub_821EC478(ctx, base);
}

PPC_FUNC(sub_821EC0D8)
{
    if (GuestPm4())
        __imp__sub_821EC0D8(ctx, base);
}

// Shaders are translated at first bind so compile time isn't spent inside a draw.
PPC_FUNC(sub_821DE308)
{
    ac6::gpu::GetShader(ctx.r4.u32, true);
    __imp__sub_821DE308(ctx, base);
}

PPC_FUNC(sub_821DE600)
{
    ac6::gpu::GetShader(ctx.r4.u32, false);
    __imp__sub_821DE600(ctx, base);
}
