#pragma once
// Recompiled-code interface: generated config/context, plus the per-thread current context.

#include <ppc_recomp_shared.h>

namespace ac6 {

// The PPC context of the guest thread currently running on this host thread.
inline thread_local PPCContext* g_ppcContext = nullptr;

inline PPCContext* GetPPCContext() { return g_ppcContext; }
inline void SetPPCContext(PPCContext& ctx) { g_ppcContext = &ctx; }

// The guest link register, for diagnostics: 0 when the recompiler's skip_lr leaves it out.
inline uint32_t GuestLinkRegister(const PPCContext& ctx)
{
#ifdef PPC_CONFIG_SKIP_LR
    return 0;
#else
    return static_cast<uint32_t>(ctx.lr);
#endif
}

} // namespace ac6
