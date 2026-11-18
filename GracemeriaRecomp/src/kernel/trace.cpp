#include "kernel/trace.h"

#include "log.h"
#include "ppc.h"

#include <atomic>
#include <cstdlib>
#include <string>
#include <string_view>
#include <vector>

namespace ac6::kernel {

namespace {

struct Counter
{
    std::atomic<uint64_t> calls{ 0 };
    std::atomic<uint32_t> firstCaller{ 0 };
};

std::vector<Counter>& Counters()
{
    static std::vector<Counter> counters(kImportCount);
    return counters;
}

bool EnvFlag(const char* name)
{
    const char* v = std::getenv(name);
    return v && *v && *v != '0';
}

} // namespace

void TraceUnimplemented(size_t index, PPCContext& ctx)
{
    auto& c = Counters()[index];
    uint64_t n = c.calls.fetch_add(1);
    uint32_t lr = GuestLinkRegister(ctx);

    if (n == 0)
    {
        c.firstCaller = lr;
        log::Warn("UNIMPLEMENTED {}(r3={:08X} r4={:08X} r5={:08X} r6={:08X} r7={:08X} r8={:08X}) from {:08X}",
            kImportNames[index], ctx.r3.u32, ctx.r4.u32, ctx.r5.u32, ctx.r6.u32, ctx.r7.u32, ctx.r8.u32, lr);

        static const bool stop = EnvFlag("AC6_STOP_ON_UNIMPLEMENTED");
        if (stop)
        {
            log::Error("Stopping on first unimplemented import (AC6_STOP_ON_UNIMPLEMENTED=1)");
            WriteImportSummary();
            std::_Exit(3);
        }
    }
    else
    {
        log::Trace("unimplemented {}(r3={:08X} r4={:08X} r5={:08X}) from {:08X}",
            kImportNames[index], ctx.r3.u32, ctx.r4.u32, ctx.r5.u32, lr);
    }
}

// AC6_TRACE_IMPORTS=1 traces every implemented import. Any other value is a comma-separated list of
// name prefixes (e.g. "Xam,XNotify") so that the hot kernel calls don't slow the game to a crawl.
bool TraceEnabled(const char* name)
{
    static const std::vector<std::string> prefixes = [] {
        std::vector<std::string> out;
        const char* v = std::getenv("AC6_TRACE_IMPORTS");
        if (!v || !*v || std::string_view(v) == "0")
            return out;
        if (std::string_view(v) == "1")
            return std::vector<std::string>{ "" };
        std::string_view list(v);
        while (!list.empty())
        {
            size_t comma = list.find(',');
            if (std::string_view item = list.substr(0, comma); !item.empty())
                out.emplace_back(item);
            list = comma == std::string_view::npos ? std::string_view() : list.substr(comma + 1);
        }
        return out;
    }();
    for (const std::string& p : prefixes)
        if (std::string_view(name).starts_with(p))
            return true;
    return false;
}

void TraceCall(const char* name, PPCContext& ctx)
{
    if (TraceEnabled(name))
    {
        log::Trace("{}(r3={:08X} r4={:08X} r5={:08X} r6={:08X}) from {:08X}",
            name, ctx.r3.u32, ctx.r4.u32, ctx.r5.u32, ctx.r6.u32, GuestLinkRegister(ctx));
    }
}

void WriteImportSummary()
{
    log::Info("--- unimplemented import summary ---");
    for (size_t i = 0; i < kImportCount; i++)
    {
        auto& c = Counters()[i];
        if (c.calls)
            log::Info("  {:40} calls={:8} first from {:08X}", kImportNames[i], c.calls.load(), c.firstCaller.load());
    }
}

} // namespace ac6::kernel
