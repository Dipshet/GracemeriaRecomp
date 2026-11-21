// Hooks on the game's functions at runtime (mods/hooks.h): the registry, and PPCHookEnter, where the recompiled
// functions' hook gates (ppc_recomp_shared.h PPC_HOOK_GATE) send their hooked calls.
//
// Memory. A function's chain never changes once published: a change publishes a new one. The slot holds a reference on
// its current chain, and each hooked call one on the chain it runs. A call takes its reference inside a short window the
// slot counts (`readers`, no other code in it). A change waits for that window to be empty before it drops the slot's
// reference on the chain it replaced, so no call is ever still taking a reference on a chain that is being freed. A
// chain goes with its last reference, a hook entry (shared by the chains that have it) with the last chain that has it,
// and the entry's release then frees the hook's user data.
#include "mods/hooks.h"

#include "log.h"

#include <algorithm>
#include <atomic>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

namespace ac6::mods {

struct HookEntry
{
    uint64_t id = 0;
    HookFn fn = nullptr;
    void* user = nullptr;
    int32_t priority = 0;
    std::string owner;
    HookRelease release = nullptr;

    ~HookEntry()
    {
        if (release)
            release(user);
    }
};

// A function's hooks, highest priority first.
struct HookChain
{
    std::vector<std::shared_ptr<HookEntry>> entries;
    std::atomic<uint32_t> refs{ 1 };   // the slot's while it's published, and one per call running it
};

} // namespace ac6::mods

// A gated function's slot (the generated code's PPCHookSlots[] point at these while the function has hooks). Made on
// the first hook of the function and kept: the gates of calls in progress may hold it.
struct PPCHookSlot
{
    uint32_t guest = 0;
    uint32_t ordinal = 0;
    PPCFunc* body = nullptr;
    std::atomic<ac6::mods::HookChain*> chain{ nullptr };
    std::atomic<uint32_t> readers{ 0 };   // calls taking a reference on `chain` this moment
};

namespace ac6::mods {

namespace {

struct Target
{
    uint32_t ordinal;
    PPCFunc* body;
};

std::mutex s_mutex;   // the registry (adding and removing hooks). Hooked calls don't take it.
std::unordered_map<uint32_t, Target>* s_targets = nullptr;   // guest address -> its gate (made on first use)
std::unordered_map<uint32_t, std::unique_ptr<PPCHookSlot>> s_slots;   // by guest address
std::unordered_map<uint64_t, uint32_t> s_hookFunction;                // hook id -> its function
uint64_t s_nextId = 1;
size_t s_hookCount = 0;

// The call the hooks are running the function for, on this thread: its gate lets it through once.
thread_local PPCHookSlot* t_bypass = nullptr;

const std::unordered_map<uint32_t, Target>& Targets()
{
    if (!s_targets)
    {
        s_targets = new std::unordered_map<uint32_t, Target>();
#if defined(PPC_HOOK_GATES)
        for (const PPCHookTarget* t = PPCHookTargets; t->guest != 0; t++)
            s_targets->emplace(t->guest, Target{ t->slot, t->body });
#endif
    }
    return *s_targets;
}

// A call's reference on the slot's chain (null: no hooks). The window: a change's Retire waits for it to close.
HookChain* Acquire(PPCHookSlot& slot)
{
    slot.readers.fetch_add(1, std::memory_order_seq_cst);
    HookChain* chain = slot.chain.load(std::memory_order_seq_cst);
    if (chain)
        chain->refs.fetch_add(1, std::memory_order_relaxed);
    slot.readers.fetch_sub(1, std::memory_order_seq_cst);
    return chain;
}

void Release(HookChain* chain)
{
    if (chain && chain->refs.fetch_sub(1, std::memory_order_acq_rel) == 1)
        delete chain;
}

// A chain a change replaced, and its slot.
struct Retired
{
    PPCHookSlot* slot;
    HookChain* chain;
};

// The slot's references on replaced chains, dropped once no call can be taking one on them (after s_mutex: the last
// reference frees the chain, and with it maybe hooks, whose release runs then).
void Retire(const std::vector<Retired>& retired)
{
    for (const Retired& r : retired)
    {
        while (r.slot->readers.load(std::memory_order_seq_cst) != 0)
            std::this_thread::yield();
        Release(r.chain);
    }
}

// Publishes `slot`'s new chain (empty: no hooks) and points its gate at it, or not. Under s_mutex. The chain it replaces
// goes into `retired`, for Retire.
void Publish(PPCHookSlot& slot, std::unique_ptr<HookChain> chain, std::vector<Retired>& retired)
{
    HookChain* next = chain && !chain->entries.empty() ? chain.release() : nullptr;
    HookChain* old = slot.chain.exchange(next, std::memory_order_seq_cst);
#if defined(PPC_HOOK_GATES)
    __atomic_store_n(&PPCHookSlots[slot.ordinal], next ? &slot : nullptr, __ATOMIC_RELEASE);
#endif
    if (old)
        retired.push_back({ &slot, old });
}

} // namespace

bool Hookable(uint32_t function)
{
    std::lock_guard lock(s_mutex);
    return Targets().contains(function);
}

size_t HookableCount()
{
    std::lock_guard lock(s_mutex);
    return Targets().size();
}

const std::vector<uint32_t>& HookableFunctions()
{
    static const std::vector<uint32_t> functions = [] {
        std::vector<uint32_t> v;
        {
            std::lock_guard lock(s_mutex);
            for (const auto& [guest, target] : Targets())
                v.push_back(guest);
        }
        std::sort(v.begin(), v.end());
        return v;
    }();
    return functions;
}

uint64_t AddHook(uint32_t function, HookFn hook, void* user, int32_t priority, std::string_view owner, HookRelease release)
{
    if (!hook)
        return 0;
    std::vector<Retired> retired;
    uint64_t id = 0;
    {
        std::lock_guard lock(s_mutex);
        const auto target = Targets().find(function);
        if (target == Targets().end())
        {
            log::Warn("Hooks: {} asked for {:08X}, which isn't a hookable function", owner, function);
            return 0;
        }
        auto& slot = s_slots[function];
        if (!slot)
        {
            slot = std::make_unique<PPCHookSlot>();
            slot->guest = function;
            slot->ordinal = target->second.ordinal;
            slot->body = target->second.body;
        }
        auto chain = std::make_unique<HookChain>();
        if (const HookChain* current = slot->chain.load(std::memory_order_acquire))
            chain->entries = current->entries;
        id = s_nextId++;
        auto entry = std::make_shared<HookEntry>();
        entry->id = id;
        entry->fn = hook;
        entry->user = user;
        entry->priority = priority;
        entry->owner = std::string(owner);
        entry->release = release;
        // Highest priority first, a new hook after those of its priority.
        const auto at = std::upper_bound(chain->entries.begin(), chain->entries.end(), priority,
            [](int32_t p, const std::shared_ptr<HookEntry>& e) { return p > e->priority; });
        chain->entries.insert(at, std::move(entry));
        Publish(*slot, std::move(chain), retired);
        s_hookFunction.emplace(id, function);
        s_hookCount++;
    }
    Retire(retired);
    return id;
}

bool RemoveHook(uint64_t id)
{
    std::vector<Retired> retired;
    {
        std::lock_guard lock(s_mutex);
        const auto it = s_hookFunction.find(id);
        if (it == s_hookFunction.end())
            return false;
        PPCHookSlot& slot = *s_slots.at(it->second);
        s_hookFunction.erase(it);
        auto chain = std::make_unique<HookChain>();
        if (const HookChain* current = slot.chain.load(std::memory_order_acquire))
            for (const auto& e : current->entries)
                if (e->id != id)
                    chain->entries.push_back(e);
        Publish(slot, std::move(chain), retired);
        s_hookCount--;
    }
    Retire(retired);
    return true;
}

size_t RemoveHooksOf(std::string_view owner)
{
    std::vector<Retired> retired;
    size_t removed = 0;
    {
        std::lock_guard lock(s_mutex);
        for (auto& [function, slot] : s_slots)
        {
            const HookChain* current = slot->chain.load(std::memory_order_acquire);
            if (!current)
                continue;
            auto chain = std::make_unique<HookChain>();
            for (const auto& e : current->entries)
            {
                if (e->owner == owner)
                {
                    s_hookFunction.erase(e->id);
                    removed++;
                }
                else
                    chain->entries.push_back(e);
            }
            if (chain->entries.size() != current->entries.size())
                Publish(*slot, std::move(chain), retired);
        }
        s_hookCount -= removed;
    }
    Retire(retired);
    return removed;
}

void CallNext(HookCall& call)
{
    const HookChain& chain = *call.chain;
    const uint32_t i = call.next;
    if (i < chain.entries.size())
    {
        // The hook's own CallNext goes on from the one after it. Afterwards the position is back, so the hook before can
        // pass the call on again.
        struct Restore
        {
            HookCall& call;
            uint32_t at;
            ~Restore() { call.next = at; }
        } restore{ call, i };
        call.next = i + 1;
        const HookEntry& e = *chain.entries[i];
        e.fn(call, e.user);
        return;
    }
    // The function itself: its gate lets this call through (and clears the bypass). Cleared here too, for a gate that
    // found its slot already emptied (the last hook removed meanwhile) and didn't look. A guest exception unwinding
    // through (ExTerminateThread) clears it as well.
    struct Clear
    {
        ~Clear() { t_bypass = nullptr; }
    } clear;
    t_bypass = call.slot;
    call.slot->body(call.ctx, call.base);
}

size_t HookedFunctionCount()
{
    std::lock_guard lock(s_mutex);
    size_t n = 0;
    for (auto& [function, slot] : s_slots)
        if (slot->chain.load(std::memory_order_acquire))
            n++;
    return n;
}

size_t HookCount()
{
    std::lock_guard lock(s_mutex);
    return s_hookCount;
}

size_t HookCountOf(std::string_view owner)
{
    std::lock_guard lock(s_mutex);
    size_t n = 0;
    for (auto& [function, slot] : s_slots)
        if (const HookChain* chain = slot->chain.load(std::memory_order_acquire))
            for (const auto& e : chain->entries)
                if (e->owner == owner)
                    n++;
    return n;
}

} // namespace ac6::mods

// The gate of a hooked function (PPC_HOOK_GATE): true when the hooks ran the call, false to run the function as usual
// (the hooks' own call of it, or no hooks left).
extern "C" bool PPCHookEnter(PPCHookSlot* slot, PPCContext& ctx, uint8_t* base)
{
    using namespace ac6::mods;
    if (t_bypass == slot)
    {
        t_bypass = nullptr;
        return false;
    }
    HookChain* chain = Acquire(*slot);
    if (!chain)
        return false;
    // (Released however the call ends: a guest thread ending unwinds through here.)
    struct Held
    {
        HookChain* chain;
        ~Held() { Release(chain); }
    } held{ chain };
    HookCall call{ ctx, base, slot->guest, chain, slot, 0 };
    CallNext(call);
    return true;
}
