#pragma once
// Import tracing: generated weak stubs report here. Real implementations can report too.

#include "ppc.h"

#include <cstddef>

namespace ac6::kernel {

extern const char* const kImportNames[];
extern const size_t kImportCount;

// Called by every unimplemented (stub) import. Logs, counts, optionally stops.
void TraceUnimplemented(size_t index, PPCContext& ctx);

// Optional tracing for implemented imports: AC6_TRACE_IMPORTS=1 logs every call.
void TraceCall(const char* name, PPCContext& ctx);

// Call counts and first callers of the unimplemented imports, written to the log at exit.
void WriteImportSummary();

} // namespace ac6::kernel

#define AC6_IMPORT_STUB(name, index)                                        \
    __attribute__((weak)) PPC_FUNC(__imp__##name)                           \
    {                                                                       \
        (void)base;                                                         \
        ac6::kernel::TraceUnimplemented(index, ctx);                        \
        ctx.r3.u64 = 0;                                                     \
    }
