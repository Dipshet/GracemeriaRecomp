#pragma once
// Shared pieces of the XAM imports (kernel/imports_xam*.cpp).

#include "kernel/dispatcher.h"
#include "kernel/object.h"
#include "kernel/xtypes.h"
#include "log.h"

#include <cstdint>

namespace ac6::kernel {

constexpr uint32_t ERROR_SUCCESS_ = 0;
constexpr uint32_t ERROR_FILE_NOT_FOUND_ = 2;
constexpr uint32_t ERROR_PATH_NOT_FOUND_ = 3;
constexpr uint32_t ERROR_ACCESS_DENIED_ = 5;
constexpr uint32_t ERROR_NO_MORE_FILES_ = 18;
constexpr uint32_t ERROR_INVALID_PARAMETER_ = 87;
constexpr uint32_t ERROR_INSUFFICIENT_BUFFER_ = 122;
constexpr uint32_t ERROR_ALREADY_EXISTS_ = 183;
constexpr uint32_t ERROR_IO_PENDING_ = 997;
constexpr uint32_t ERROR_DEVICE_NOT_CONNECTED_ = 1167;
constexpr uint32_t ERROR_NO_SUCH_USER_ = 1317;
constexpr uint32_t ERROR_FILE_CORRUPT_ = 1392;

struct XXOverlapped
{
    be<uint32_t> error;
    be<uint32_t> length;
    be<uint32_t> internalContext;
    be<uint32_t> event;
    be<uint32_t> completionRoutine;
    be<uint32_t> completionContext;
    be<uint32_t> extendedError;
};
static_assert(sizeof(XXOverlapped) == 0x1C);

// XAM calls that take an XOVERLAPPED complete it immediately and report ERROR_IO_PENDING.
inline uint32_t CompleteXam(XXOverlapped* overlapped, uint32_t result, uint32_t length = 0)
{
    if (!overlapped)
        return result;
    overlapped->error = result;
    overlapped->length = length;
    overlapped->extendedError = result;
    if (overlapped->event)
        if (KernelObject* e = LookupHandle(overlapped->event); e && e->DispatcherHeader())
            SetEvent(e->DispatcherHeader());
    if (overlapped->completionRoutine)
        log::Warn("XAM overlapped completion routine {:08X} not called (no APCs yet)", uint32_t(overlapped->completionRoutine));
    return ERROR_IO_PENDING_;
}

// Queues a notification (XN_*) for every listener whose area mask includes it.
void XamNotify(uint32_t id, uint32_t param);

} // namespace ac6::kernel
