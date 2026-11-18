// D3D call census (diagnostics): pass-through hooks on the identified Xbox D3D functions
// that count calls and then run the original. With
// AC6_D3D_CENSUS=1 the per-second counts are logged, which shows what each boot phase draws.
// AC6_D3D_PROBE=<n> logs the arguments (and the objects they point to) of the first <n> calls of the
// render-target, resolve, tiling and indexed-draw functions, starting at the first indexed draw.
#include "gpu/d3d_census.h"
#include "kernel/function.h"
#include "log.h"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <string>
#include <thread>
#include <mutex>
#include <set>
#include "memory.h"
#include "endian.h"

#define AC6_D3D_FUNCTIONS(X)                                    \
    X(821DC538, SetSamplerState5)                               \
    X(821DC6C8, SetSamplerState4)                               \
    X(821DCF28, SetViewport)                                    \
    X(821DD028, SetScissorRect)                                 \
    X(821DD0A8, SetStreamSource)                                \
    X(821DD1C8, SetIndices)                                     \
    X(821DD260, SetRenderTarget)                                \
    X(821DD5C8, SetDepthStencilSurface)                         \
    X(821DDA30, SetPerRenderTargetState)                        \
    X(821DDF08, SetVertexShaderConstantF)                       \
    X(821DDFF0, SetPixelShaderConstantF)                        \
    X(821DE0D8, SetShaderConstantBI_1)                          \
    X(821DE138, SetShaderConstantBI_2)                          \
    X(821DE198, SetShaderConstantBI_3)                          \
    X(821DE1F0, SetShaderConstantBI_4)                          \
    X(821DE7D0, SetVertexDeclaration)                           \
    X(821E10C8, SetTexture)                                     \
    X(821E6D40, FrameMarkA)                                     \
    X(821E6D80, FrameMarkB)                                     \
    X(821FBE70, CreateTexture)                                  \
    X(821FBF70, CreateCubeTexture)

// Functions hooked by the renderer (gpu/d3d.cpp), which reports them through CensusNote.
#define AC6_D3D_EXTERNAL_FUNCTIONS(X)                           \
    X(821DF300, DrawIndexedVertices)                \
    X(821E2BB8, Resolve)                            \
    X(821E4630, BeginTiling)                        \
    X(821E4AD0, EndTiling)

namespace ac6::gpu {

namespace {

enum Index
{
#define X(addr, name) k##name,
    AC6_D3D_FUNCTIONS(X)
    AC6_D3D_EXTERNAL_FUNCTIONS(X)
#undef X
    kCount
};

const char* const kNames[] = {
#define X(addr, name) #name,
    AC6_D3D_FUNCTIONS(X)
    AC6_D3D_EXTERNAL_FUNCTIONS(X)
#undef X
};

std::atomic<uint64_t> s_counts[kCount];

std::string HexWords(uint32_t guest, uint32_t words)
{
    std::string out;
    if (guest < 0x20000 || guest >= 0xC0000000u)
        return "-";
    for (uint32_t i = 0; i < words; i++)
        out += std::format("{}{:08X}", i ? " " : "", uint32_t(*g_memory.As<be<uint32_t>>(guest + i * 4)));
    return out;
}

void Probe(int index, const PPCContext& ctx)
{
    static const long limit = [] { const char* v = std::getenv("AC6_D3D_PROBE"); return v ? std::atol(v) : 0L; }();
    static std::atomic<long> logged{ 0 };
    static std::atomic<bool> started{ false };
    if (limit <= 0)
        return;
    if (index == kDrawIndexedVertices)
        started = true;
    if (!started)
        return;
    switch (index)
    {
    case kSetRenderTarget: case kSetDepthStencilSurface: case kResolve: case kBeginTiling: case kEndTiling:
    case kDrawIndexedVertices: case kSetViewport: case kSetIndices: case kFrameMarkA: case kCreateTexture:
        break;
    default:
        return;
    }
    if (logged.fetch_add(1) >= limit)
        return;
    std::string extra;
    const uint32_t r4 = ctx.r4.u32, r5 = ctx.r5.u32, r6 = ctx.r6.u32;
    switch (index)
    {
    case kSetRenderTarget:
    case kSetDepthStencilSurface:
    {
        // Each distinct surface once (they are long-lived objects).
        static std::mutex m;
        static std::set<uint32_t> seen;
        const uint32_t surface = index == kSetRenderTarget ? r5 : r4;
        std::lock_guard lock(m);
        if (surface && seen.insert(surface).second)
            extra = "NEW surface " + HexWords(surface, 20);
        break;
    }
    case kResolve:
        extra = std::format("rect {} | dest {} | point {} | clear {} | stack {}", HexWords(r5, 4), HexWords(r6, 13), HexWords(ctx.r7.u32, 2),
            HexWords(ctx.r10.u32, 4), HexWords(ctx.r1.u32 + 0x54, 4));
        break;
    case kDrawIndexedVertices: extra = "ib " + HexWords(*g_memory.As<be<uint32_t>>(ctx.r3.u32 + 0x308C), 8); break;
    case kSetIndices: extra = "ib " + HexWords(r4, 8); break;
    case kSetViewport: extra = "vp " + HexWords(r4, 6); break;
    case kBeginTiling: extra = std::format("rects {} | stack {}", HexWords(r5, 8), HexWords(ctx.r1.u32 + 0x54, 4)); break;
    default: break;
    }
    log::Info("[probe] {}(r3={:08X} r4={:08X} r5={:08X} r6={:08X} r7={:08X} r8={:08X} r9={:08X} r10={:08X}) lr={:08X} {}",
        kNames[index], ctx.r3.u32, r4, r5, r6, ctx.r7.u32, ctx.r8.u32, ctx.r9.u32, ctx.r10.u32, ac6::GuestLinkRegister(ctx), extra);
}

struct CensusReporter
{
    CensusReporter()
    {
        const char* v = std::getenv("AC6_D3D_CENSUS");
        if (!v || !*v || *v == '0')
            return;
        std::thread([] {
            uint64_t last[kCount]{};
            for (int second = 1;; second++)
            {
                std::this_thread::sleep_for(std::chrono::seconds(1));
                std::string line;
                for (int i = 0; i < kCount; i++)
                {
                    uint64_t now = s_counts[i].load(std::memory_order_relaxed);
                    if (now != last[i])
                        line += std::format(" {}={}", kNames[i], now - last[i]);
                    last[i] = now;
                }
                log::Info("[d3d {:3}s]{}", second, line.empty() ? " (none)" : line);
            }
        }).detach();
    }
} s_reporter;

} // namespace

int CensusIndex(std::string_view name)
{
    for (int i = 0; i < kCount; i++)
        if (name == kNames[i])
            return i;
    return -1;
}

void CensusNote(int index, const PPCContext& ctx)
{
    if (index < 0)
        return;
    s_counts[index].fetch_add(1, std::memory_order_relaxed);
    Probe(index, ctx);
}

} // namespace ac6::gpu

#define X(addr, name)                                                     \
    extern "C" PPC_FUNC(__imp__sub_##addr);                                 \
    PPC_FUNC(sub_##addr)                                                    \
    {                                                                       \
        ac6::gpu::s_counts[ac6::gpu::k##name].fetch_add(1, std::memory_order_relaxed); \
        ac6::gpu::Probe(ac6::gpu::k##name, ctx);                            \
        __imp__sub_##addr(ctx, base);                                       \
    }
AC6_D3D_FUNCTIONS(X)
#undef X
