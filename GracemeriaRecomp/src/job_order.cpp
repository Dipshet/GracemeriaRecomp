// Ordering of game jobs whose correctness relied on the console's prompt worker threads.
//
// 1. The effect managers' async jobs.
//
// CX360EffectManager / CAce6EffectManager run their per-frame update and draw as jobs on the game's second
// worker pool, both through one entry point (vtbl[+4], sub_822B3248), which picks the operation from a
// mode word the kick (sub_822B3148) stores in the manager ([this + 0xA1E84]: 0 update, 1 update variant,
// 2 draw) just before submitting. Nothing orders a job's read of that word before the next kick's store:
// the main thread kicks the draw 120-380 us after the update, and the worker (pinned to hardware thread
// 3) has to have started the update by then. A console hardware thread does. A host thread under CPU
// pressure may wait milliseconds for a time slice, and then both queued jobs read "draw" - the effects and
// clouds drawn twice (thicker, flickering) and the update skipped.
//
// A kick first waits for the manager's previously submitted jobs to finish, so a job always runs with
// the mode it was kicked with, and the draw follows the update as on the console. Without host pressure
// the update has long finished and nothing waits. AC6_EFFECT_JOB_ORDER=0 turns the wait off (comparisons).
// Either way, a job that reads a different mode than it was submitted with is logged.
// AC6_EFFECT_JOB_STALL=<n>: every n-th job sleeps 2 ms before it starts, as a host preemption of the
// worker would (reproduces the race on demand, testing only).
//
// 2. The HUD update job. The frame update (sub_82099370 / sub_8219DFB8) submits the HUD's update
// (job entry sub_822357D0 -> sub_8223DDB0) to worker pool B thread 2 and joins it only at the end of the
// frame, running the rest of the frame update meanwhile. On the console the job (about 20 us) is done long
// before that work reaches the state it shares with it. A host thread that isn't scheduled promptly runs
// it late, and it builds the HUD from that state mid-change: for one frame the first-person HUD (pitch
// ladder) pops up and the readouts jump back. The submitting thread waits for the job to finish right
// after submitting it, as the console's timing has it. AC6_HUD_JOB_ORDER=0 turns that off.
//
// 3. Storage jobs' op codes. Twelve submitters of the async save/storage layer (save.dat,
// replay.dat, dlcolorname.dat, reads from the game's DPL data packages) store the job's operation code to [job+4] only after Submit
// returns, while the pool-B worker reads it as soon as it dequeues the job. The job objects are reused, so
// a prompt worker can run the previous operation. The title update moves those stores before the call.
// Mid-asm hooks at the twelve calls (config/AC6.toml: StorageSubmitOp*) store the code first. The game's own
// store after the call writes the same value again. AC6_STORAGE_SUBMIT_ORDER=0 turns it off.
#include "endian.h"
#include "kernel/function.h"
#include "log.h"
#include "memory.h"
#include "platform/platform.h"
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <thread>
#include <unordered_map>

extern "C" PPC_FUNC(__imp__sub_822B3148);   // effect manager Kick(this, mode, f1 = dt, async)
extern "C" PPC_FUNC(__imp__sub_822B3248);   // effect manager job entry (vtbl[+4]): dispatches on the mode word
extern "C" PPC_FUNC(__imp__sub_821D4A78);   // worker pool B Submit(pool, thread, job, arg) -> ticket
extern "C" PPC_FUNC(__imp__sub_821D4988);   // worker pool A Submit(job, arg) (shared queue, fork/join)
extern "C" PPC_FUNC(__imp__sub_822357D0);   // HUD update job entry (thunk to sub_8223DDB0)

namespace {

constexpr uint32_t kJobEntry = 0x822B3248;
constexpr uint32_t kHudJobEntry = 0x822357D0;

struct Jobs
{
    std::atomic<uint32_t> submitted{ 0 };
    std::atomic<uint32_t> finished{ 0 };
    std::deque<uint32_t> modes;   // the mode word at each submit, in queue order (one worker runs them)
};

constexpr uint32_t kModeOffset = 0xA1E84;

uint32_t Mode(uint32_t manager)
{
    return ac6::g_memory.As<ac6::be<uint32_t>>(manager + kModeOffset)->get();
}

bool OrderHudJob()
{
    static const bool enabled = [] {
        const char* v = std::getenv("AC6_HUD_JOB_ORDER");
        return !v || *v != '0';
    }();
    return enabled;
}

bool OrderStorageSubmits()
{
    static const bool enabled = [] {
        const char* v = std::getenv("AC6_STORAGE_SUBMIT_ORDER");
        return !v || *v != '0';
    }();
    return enabled;
}

bool OrderJobs()
{
    static const bool enabled = [] {
        const char* v = std::getenv("AC6_EFFECT_JOB_ORDER");
        return !v || *v != '0';
    }();
    return enabled;
}

std::mutex s_mutex;
std::unordered_map<uint32_t, Jobs> s_managers;   // by manager address (the game has two)

Jobs& For(uint32_t manager)
{
    std::lock_guard lock(s_mutex);
    return s_managers[manager];   // unordered_map nodes are stable
}

// A pool job's entry (its vtable's +4), or 0.
uint32_t JobEntry(uint32_t job)
{
    if (job < 0x40000000u || job >= 0xC0000000u)
        return 0;
    const uint32_t vtable = ac6::g_memory.As<ac6::be<uint32_t>>(job)->get();
    if (vtable < 0x82000000u || vtable >= 0x83000000u)
        return 0;
    return ac6::g_memory.As<ac6::be<uint32_t>>(vtable + 4)->get();
}

// A pool job is an effect manager when its entry is the effect job entry.
bool IsEffectManager(uint32_t job)
{
    return JobEntry(job) == kJobEntry;
}

// Waits at most 100 ms until `jobs` has finished `target` jobs. False on a timeout.
bool WaitFinished(Jobs& jobs, uint32_t target)
{
    uint32_t finished = jobs.finished.load();
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(100);
    while (int32_t(finished - target) < 0 && std::chrono::steady_clock::now() < deadline)
    {
        ac6::platform::WaitOnAddress(&jobs.finished, finished, 2);
        finished = jobs.finished.load();
    }
    return int32_t(finished - target) >= 0;
}

// --- AC6_JOB_CENSUS=1: every pool job object, and whether one is ever submitted again before its previous
// submission started (the precondition of the race above). Job entries are wrapped through the
// indirect-call table, which the pool workers call them through. Logged every 30 s.
struct CensusEntry
{
    uint32_t entry = 0, pool = 0, submits = 0, overlaps = 0, maxQueued = 0, threads = 0;
    uint32_t lastThread = 0;
    int32_t queued = 0;   // submitted, not yet started
    double minGapUs = 1e9;
    std::chrono::steady_clock::time_point last{};
};

bool CensusOn()
{
    static const bool on = [] { const char* v = std::getenv("AC6_JOB_CENSUS"); return v && *v == '1'; }();
    return on;
}

std::mutex s_censusMutex;
std::unordered_map<uint32_t, CensusEntry> s_census;        // by job object
std::unordered_map<uint32_t, PPCFunc*> s_wrappedEntries;   // job entry -> original

void CensusLog()
{
    static auto next = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    if (std::chrono::steady_clock::now() < next)
        return;
    next += std::chrono::seconds(30);
    for (const auto& [job, e] : s_census)
        ac6::log::Info("[CENSUS] pool {} job {:08X} entry {:08X} submits {} overlaps {} maxQueued {} minGap {:.0f} us threads {}",
            e.pool == 0 ? "A" : "B", job, e.entry, e.submits, e.overlaps, e.maxQueued, e.minGapUs, e.threads);
}

void CensusTrampoline(PPCContext& __restrict ctx, uint8_t* base)
{
    // AC6_JOB_STALL=<n>[:<ms>]: every n-th pool job sleeps before it starts (a stalled worker, testing).
    static const auto stall = [] {
        const char* v = std::getenv("AC6_JOB_STALL");
        std::pair<uint32_t, uint32_t> r{ 0, 20 };
        if (v && *v)
        {
            r.first = uint32_t(std::atoi(v));
            if (const char* c = std::strchr(v, ':'))
                r.second = uint32_t(std::atoi(c + 1));
        }
        return r;
    }();
    const uint32_t job = ctx.r3.u32;
    bool poolB = false;
    {
        std::lock_guard lock(s_censusMutex);
        if (auto it = s_census.find(job); it != s_census.end())
            poolB = it->second.pool == 1;
    }
    // AC6_JOB_STALL_ENTRY=<hex>: only that job entry.
    static const uint32_t onlyEntry = [] { const char* v = std::getenv("AC6_JOB_STALL_ENTRY"); return v ? uint32_t(std::strtoul(v, nullptr, 16)) : 0u; }();
    const uint32_t jobEntry = ac6::g_memory.As<ac6::be<uint32_t>>(ac6::g_memory.As<ac6::be<uint32_t>>(job)->get() + 4)->get();
    static std::atomic<uint32_t> count{ 0 };
    // AC6_JOB_STALL_POOL=A: stall pool A jobs instead.
    static const bool poolA = [] { const char* v = std::getenv("AC6_JOB_STALL_POOL"); return v && (*v == 'A' || *v == 'a'); }();
    if (stall.first && poolB != poolA && (!onlyEntry || onlyEntry == jobEntry) && count++ % stall.first == 0)
        std::this_thread::sleep_for(std::chrono::milliseconds(stall.second));
    const uint32_t entry = ac6::g_memory.As<ac6::be<uint32_t>>(ac6::g_memory.As<ac6::be<uint32_t>>(job)->get() + 4)->get();
    PPCFunc* original = nullptr;
    {
        std::lock_guard lock(s_censusMutex);
        if (auto it = s_census.find(job); it != s_census.end())
            it->second.queued--;
        if (auto it = s_wrappedEntries.find(entry); it != s_wrappedEntries.end())
            original = it->second;
    }
    if (original)
        original(ctx, base);
}

void CensusSubmit(uint32_t pool, uint32_t job)
{
    if (!CensusOn() || job < 0x40000000u || job >= 0xC0000000u)
        return;
    const uint32_t vtable = ac6::g_memory.As<ac6::be<uint32_t>>(job)->get();
    if (vtable < 0x82000000u || vtable >= 0x83000000u)
        return;
    const uint32_t entry = ac6::g_memory.As<ac6::be<uint32_t>>(vtable + 4)->get();
    const auto now = std::chrono::steady_clock::now();
    std::lock_guard lock(s_censusMutex);
    if (!s_wrappedEntries.contains(entry))
    {
        PPCFunc*& slot = ac6::g_memory.Function(entry);
        s_wrappedEntries[entry] = slot;
        slot = CensusTrampoline;
    }
    CensusEntry& e = s_census[job];
    e.entry = entry;
    e.pool = pool;
    if (e.submits)
        e.minGapUs = std::min(e.minGapUs, std::chrono::duration<double, std::micro>(now - e.last).count());
    if (e.queued > 0)
        e.overlaps++;
    e.queued++;
    e.maxQueued = std::max<uint32_t>(e.maxQueued, uint32_t(e.queued));
    e.submits++;
    e.last = now;
    const uint32_t thread = ac6::platform::CurrentThreadId();
    if (thread != e.lastThread)
        e.threads++;
    e.lastThread = thread;
    CensusLog();
}

} // namespace

// Mid-asm hooks at the storage submits (before each bl JobPoolB::Submit): r5 is the job.
namespace {
void StoreStorageOpCode(uint32_t job, uint32_t op, uint32_t site)
{
    if (!OrderStorageSubmits() || job < 0x40000000u || job >= 0xC0000000u)
        return;
    *ac6::g_memory.As<ac6::be<uint32_t>>(job + 4) = op;
    static std::atomic<uint32_t> logged{ 0 };
    if (logged++ < 4)
        ac6::log::Info("Storage job {:08X}: op code {} stored before its submit (site {:08X})", job, op, site);
}
} // namespace

void StorageSubmitOpInR29(PPCRegister& r5, PPCRegister& r29) { StoreStorageOpCode(r5.u32, r29.u32, 0x821C5A78); }
#define AC6_STORAGE_SITE(op, site)     void StorageSubmitOp##op(PPCRegister& r5) { StoreStorageOpCode(r5.u32, op, site); }
AC6_STORAGE_SITE(13, 0x821C7538)
AC6_STORAGE_SITE(8, 0x821C8EC4)
AC6_STORAGE_SITE(9, 0x821C8F78)
AC6_STORAGE_SITE(16, 0x821C9298)
AC6_STORAGE_SITE(2, 0x821D3C8C)
AC6_STORAGE_SITE(3, 0x821D3D0C)
AC6_STORAGE_SITE(6, 0x821D3DC8)
AC6_STORAGE_SITE(7, 0x821D3E58)
AC6_STORAGE_SITE(10, 0x821D3F70)
AC6_STORAGE_SITE(11, 0x821D4030)
AC6_STORAGE_SITE(12, 0x821D40F0)
#undef AC6_STORAGE_SITE

PPC_FUNC(sub_821D4988)
{
    CensusSubmit(0, ctx.r4.u32);
    __imp__sub_821D4988(ctx, base);
}

PPC_FUNC(sub_821D4A78)
{
    CensusSubmit(1, ctx.r5.u32);
    // Counted before the job is queued, so the worker can't finish it first.
    if (IsEffectManager(ctx.r5.u32))
    {
        Jobs& jobs = For(ctx.r5.u32);
        {
            std::lock_guard lock(s_mutex);
            jobs.modes.push_back(Mode(ctx.r5.u32));
        }
        jobs.submitted.fetch_add(1);
    }
    const uint32_t job = ctx.r5.u32;
    Jobs* hud = OrderHudJob() && JobEntry(job) == kHudJobEntry ? &For(job) : nullptr;
    const uint32_t hudTarget = hud ? hud->submitted.fetch_add(1) + 1 : 0;
    __imp__sub_821D4A78(ctx, base);
    if (hud && !WaitFinished(*hud, hudTarget))
    {
        static std::atomic<uint32_t> timeouts{ 0 };
        if (timeouts++ < 8)
            ac6::log::Warn("HUD job {:08X} still pending 100 ms after its submit, going on", job);
    }
}

// The HUD job's entry: counts it finished.
PPC_FUNC(sub_822357D0)
{
    Jobs& jobs = For(ctx.r3.u32);
    __imp__sub_822357D0(ctx, base);
    jobs.finished.fetch_add(1);
    ac6::platform::WakeAll(&jobs.finished);
}

PPC_FUNC(sub_822B3248)
{
    static const uint32_t stallEvery = [] { const char* v = std::getenv("AC6_EFFECT_JOB_STALL"); return v ? uint32_t(std::atoi(v)) : 0u; }();
    static std::atomic<uint32_t> jobCount{ 0 };
    if (stallEvery && jobCount++ % stallEvery == 0)
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    Jobs& jobs = For(ctx.r3.u32);
    {
        const uint32_t now = Mode(ctx.r3.u32);
        std::lock_guard lock(s_mutex);
        if (!jobs.modes.empty())
        {
            const uint32_t submitted = jobs.modes.front();
            jobs.modes.pop_front();
            static uint32_t mismatches = 0;
            if (now != submitted && ++mismatches <= 32)
                ac6::log::Warn("Effect manager {:08X}: job submitted as mode {} runs as mode {} ({} so far)", ctx.r3.u32, submitted, now,
                    mismatches);
        }
    }
    __imp__sub_822B3248(ctx, base);
    jobs.finished.fetch_add(1);
    ac6::platform::WakeAll(&jobs.finished);
}

PPC_FUNC(sub_822B3148)
{
    Jobs& jobs = For(ctx.r3.u32);
    const uint32_t target = jobs.submitted.load();
    uint32_t finished = jobs.finished.load();
    if (finished != target && OrderJobs())
    {
        // Bounded, in case a job ever waits on this thread: then the kick goes ahead without the wait.
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(100);
        while (finished != target && std::chrono::steady_clock::now() < deadline)
        {
            ac6::platform::WaitOnAddress(&jobs.finished, finished, 2);
            finished = jobs.finished.load();
        }
        static std::atomic<uint32_t> waits{ 0 }, timeouts{ 0 };
        if (finished != target)
        {
            if (timeouts++ < 8)
                ac6::log::Warn("Effect manager {:08X}: previous job still pending after 100 ms, kicking anyway", ctx.r3.u32);
        }
        else if (const uint32_t n = ++waits; n <= 8 || (n & 1023) == 0)
        {
            ac6::log::Info("Effect manager {:08X}: kick waited for the previous job ({} so far)", ctx.r3.u32, n);
        }
    }
    __imp__sub_822B3148(ctx, base);
}
