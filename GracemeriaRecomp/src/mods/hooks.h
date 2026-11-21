#pragma once
// Hooks on the game's functions at runtime: any recompiled function, through XenonRecomp's hook gates
// (GracemeriaRecompLib/config/AC6.toml hook_gates). Each gated function checks a slot of its own on entry (a load and
// a branch while it isn't hooked). A hooked function's calls go through its chain instead: the hooks, highest priority
// first (equal priorities in the order they were added), then the function itself.
//
// A hook gets the call (the guest registers, as the caller left them) and passes it on with CallNext: to the next hook,
// or after the last to the function. It can change the registers before and after, call CallNext more than once, or not
// at all (the call skipped: the hook sets the results itself). Hooks are added and removed at any time on any thread. A
// call in progress finishes with the chain it started with. A chain replaced by a change is freed once no call runs it,
// and a removed hook's `release` (AddHook) runs once no chain has it: memory doesn't grow as hooks come and go.
//
// The gate is in the function's body (__imp__sub_X), so a function the runtime overrides (a strong sub_X in src/game)
// is hooked where the override runs the game's own: the hooks see the game's function, under the runtime's changes.
#include "ppc.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

struct PPCHookSlot;   // (ppc_recomp_shared.h with hook gates)

namespace ac6::mods {

struct HookChain;

// A hooked call as a hook sees it.
struct HookCall
{
    PPCContext& ctx;              // the guest registers (r3-r10 arguments, r3 / f1 results)
    uint8_t* base;                // guest memory
    uint32_t function;            // the hooked function's guest address
    // Internal: the chain this call runs, the function's slot, and CallNext's position in the chain.
    const HookChain* chain;
    PPCHookSlot* slot;
    uint32_t next;
};

using HookFn = void (*)(HookCall& call, void* user);
// Frees a hook's `user` once the hook is removed and no call can still reach it (on whichever thread let go of it last:
// the one removing it, or one whose call was running it).
using HookRelease = void (*)(void* user);

// The functions that have a gate (all of the game's but the register save/restore helpers): whether `function` (a
// guest function's start address) is one, and how many there are.
bool Hookable(uint32_t function);
size_t HookableCount();
// All of them, by address.
const std::vector<uint32_t>& HookableFunctions();

// Adds a hook on `function`. `owner` names who added it (a mod, a plugin) for RemoveHooksOf and the log. Returns the
// hook's id, 0 when the function has no gate (not a function start, or the build has no gates): `release` isn't called
// then, the caller still has `user`.
uint64_t AddHook(uint32_t function, HookFn hook, void* user, int32_t priority, std::string_view owner,
    HookRelease release = nullptr);
// Removes a hook (false: no hook has that id).
bool RemoveHook(uint64_t id);
// Removes every hook `owner` added. Returns how many.
size_t RemoveHooksOf(std::string_view owner);

// From a hook: the rest of the chain (the next hook, or the function).
void CallNext(HookCall& call);

// Diagnostics: the functions hooked now, and the hooks on them (in all, or `owner`'s).
size_t HookedFunctionCount();
size_t HookCount();
size_t HookCountOf(std::string_view owner);

} // namespace ac6::mods
