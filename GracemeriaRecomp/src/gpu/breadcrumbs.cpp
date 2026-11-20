#include "gpu/breadcrumbs.h"

#include "log.h"
#include "plume_vulkan.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <format>
#include <memory>
#include <string>
#include <string_view>
#include <thread>

namespace ac6::gpu::breadcrumbs {

namespace {

// A ring: breadcrumb n is slot n % kSlots, in the GPU buffer (reached: the first kSlots words, finished: the next
// kSlots) and in the CPU's records. 4096 is a few frames of passes, more than the frames in flight.
constexpr uint32_t kSlots = 4096;

struct Record
{
    uint32_t number = 0;
    uint32_t draws = 0;
    uint64_t frame = 0;
    const char* file = nullptr;
    const char* function = nullptr;
    uint32_t line = 0;
    uint32_t width = 0, height = 0;
    uint64_t vs = 0, ps = 0;   // the last draw's guest shaders
};

std::unique_ptr<plume::RenderBuffer> s_buffer;
const volatile uint32_t* s_gpu = nullptr;
bool s_markers = false;
Record s_records[kSlots];
std::atomic<uint32_t> s_number{ 0 };
Record* s_current = nullptr;
std::atomic<uint32_t> s_submitted{ 0 };   // the breadcrumbs submitted to the GPU (the last frame's end)

std::string_view FileName(const char* path)
{
    std::string_view p = path ? path : "?";
    const size_t slash = p.find_last_of("/\\");
    return slash == std::string_view::npos ? p : p.substr(slash + 1);
}

// The function's own name from its signature: "bool ac6::gpu::(anonymous namespace)::BindTargets(RenderCommandList &, ...)"
// (clang) or "bool __cdecl ac6::gpu::`anonymous-namespace'::BindTargets(...)" (MSVC) -> "BindTargets". In a lambda,
// "...::Draw(...)::(lambda at ...)::operator()(...)", the name before the first parameter list: "Draw".
std::string FunctionName(const char* signature)
{
    std::string s = signature ? signature : "?";
    for (const char* anonymous : { "(anonymous namespace)::", "`anonymous namespace'::", "`anonymous-namespace'::" })
        for (size_t at; (at = s.find(anonymous)) != std::string::npos;)
            s.erase(at, std::strlen(anonymous));
    if (const size_t paren = s.find('('); paren != std::string::npos)
        s.resize(paren);
    if (const size_t colon = s.rfind("::"); colon != std::string::npos)
        s.erase(0, colon + 2);
    if (const size_t space = s.rfind(' '); space != std::string::npos)
        s.erase(0, space + 1);
    return s.empty() ? "?" : s;
}

std::string Describe(const Record& r)
{
    std::string s = std::format("#{} frame {}: {}:{} {}", r.number, r.frame, FileName(r.file), r.line, FunctionName(r.function));
    if (r.width)
        s += std::format(" ({}x{})", r.width, r.height);
    s += std::format(", {} draw(s) after it", r.draws);
    if (r.draws)
        s += std::format(", the last vs {:016X} ps {:016X}", r.vs, r.ps);
    return s;
}

// The newest numbers the GPU wrote (numbers only grow; the ring holds the last kSlots): reached, and finished (with
// markers; 0 without).
std::pair<uint32_t, uint32_t> Progress()
{
    uint32_t reached = 0, finished = 0;
    for (uint32_t i = 0; i < kSlots; i++)
    {
        reached = std::max(reached, uint32_t(s_gpu[i]));
        finished = std::max(finished, uint32_t(s_gpu[kSlots + i]));
    }
    return { reached, finished };
}

// The breadcrumbs from a few before `from` to a few after `to`, oldest first, marked.
void List(uint32_t from, uint32_t to, uint32_t reached, uint32_t finished)
{
    const uint32_t made = s_number.load(std::memory_order_relaxed);
    const uint32_t first = std::max(from > 6 ? from - 6 : 1u, made > kSlots ? made - kSlots + 1 : 1u);
    for (uint32_t n = first, last = std::min(made, to + 6); n <= last; n++)
    {
        const Record& r = s_records[n % kSlots];
        if (r.number != n)
            continue;
        const char* mark = n == reached ? "  <- the GPU's last" : n == finished && s_markers ? "  <- finished up to here" : "";
        log::Error("  {}{}{}", n > reached ? "(not reached) " : "", Describe(r), mark);
    }
}

// The watch (markers only). A stall: submitted breadcrumbs whose work before them hasn't finished, and the GPU's
// finished one not moving. An idle GPU has finished what was submitted (whatever its front end read ahead: in a hang it
// may stop before the next breadcrumb, so reached can equal finished); work recorded but not yet submitted isn't counted.
void Watch(double seconds)
{
    uint32_t lastFinished = 0;
    auto since = std::chrono::steady_clock::now();
    bool reported = false;
    // A quarter of the time looked for, 0.5 to 100 ms (100: a second's stall is seen within a tenth of it).
    const auto interval = std::chrono::microseconds(int64_t(std::clamp(seconds * 250000.0, 500.0, 100000.0)));
    for (;;)
    {
        std::this_thread::sleep_for(interval);
        const auto [reached, finished] = Progress();
        const uint32_t submitted = s_submitted.load(std::memory_order_acquire);
        const bool pending = finished < submitted;
        const auto now = std::chrono::steady_clock::now();
        if (finished != lastFinished || !pending)
        {
            if (reported)
                log::Info("GPU watch: the GPU moved on after {:.1f} s (finished up to #{})",
                    std::chrono::duration<double>(now - since).count(), finished);
            lastFinished = finished;
            since = now;
            reported = false;
            continue;
        }
        if (reported || now - since < std::chrono::duration<double>(seconds))
            continue;
        reported = true;
        log::Error("=== GPU STALL: the GPU has not finished the work after breadcrumb #{} for {:.3f} s ===", finished,
            std::chrono::duration<double>(now - since).count());
        log::Error("GPU breadcrumbs: {} made, {} submitted; the GPU reached #{}; the work it is stuck on comes after #{}:",
            s_number.load(std::memory_order_relaxed), submitted, reached, finished);
        List(finished, reached, reached, finished);
        log::FlushToDisk();
    }
}

} // namespace

void Init(plume::RenderDevice& device)
{
    const char* v = std::getenv("AC6_GPU_BREADCRUMBS");
    if (v && *v == '0')
    {
        log::Info("GPU breadcrumbs: off (AC6_GPU_BREADCRUMBS=0)");
        return;
    }
    s_buffer = device.createBuffer(plume::RenderBufferDesc::ReadbackBuffer(uint64_t(kSlots) * 2 * 4));
    void* memory = s_buffer ? s_buffer->map() : nullptr;
    if (!memory)
    {
        log::Warn("GPU breadcrumbs: no buffer (off)");
        s_buffer.reset();
        return;
    }
    std::memset(memory, 0, uint64_t(kSlots) * 2 * 4);
    s_gpu = static_cast<const volatile uint32_t*>(memory);
    s_markers = plume::VulkanBufferMarkers(&device);
    log::Info("GPU breadcrumbs: on ({})", s_markers ? "buffer markers" : "transfer fills");
    const char* w = std::getenv("AC6_GPU_WATCH_MS");
    const double seconds = (w && *w ? std::atof(w) : 1000.0) / 1000.0;
    if (s_markers && seconds > 0)
    {
        std::thread(Watch, seconds).detach();
        log::Info("GPU watch: on (the GPU stuck for {:.1f} s is logged)", seconds);
    }
}

void Mark(plume::RenderCommandList& cl, uint64_t frame, uint32_t width, uint32_t height, std::source_location where)
{
    if (!s_buffer)
        return;
    const uint32_t number = s_number.fetch_add(1, std::memory_order_relaxed) + 1;
    const uint32_t slot = number % kSlots;
    Record& r = s_records[slot];
    r = Record{ number, 0, frame, where.file_name(), where.function_name(), where.line(), width, height, 0, 0 };
    s_current = &r;
    cl.writeBreadcrumb(s_buffer.get(), uint64_t(slot) * 4, uint64_t(kSlots + slot) * 4, number);
}

void NoteDraw(uint64_t vs, uint64_t ps)
{
    if (Record* r = s_current)
    {
        r->draws++;
        r->vs = vs;
        r->ps = ps;
    }
}

uint32_t Made()
{
    return s_number.load(std::memory_order_relaxed);
}

void Submitted(uint32_t made)
{
    s_submitted.store(made, std::memory_order_release);
}

void Report(plume::RenderDevice* device)
{
    for (const std::string& line : plume::VulkanDeviceFaultReport(device))
        log::Error("Driver fault report: {}", line);
    if (!s_gpu)
        return;
    const auto [reached, finished] = Progress();
    const uint32_t made = s_number.load(std::memory_order_relaxed);
    if (s_markers)
        log::Error("GPU breadcrumbs: {} made; the GPU reached #{} and finished the work before #{}", made, reached, finished);
    else
        log::Error("GPU breadcrumbs: {} made; the last the GPU wrote is #{} (written where its render pass ended)", made, reached);
    if (!reached)
    {
        log::Error("GPU breadcrumbs: the GPU wrote none");
        return;
    }
    // A few before the GPU's last and the ones it never got to, oldest first.
    List(reached, reached, reached, finished);
    log::FlushToDisk();
}

} // namespace ac6::gpu::breadcrumbs
