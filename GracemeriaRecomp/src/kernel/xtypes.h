#pragma once
// Guest-visible Xbox 360 kernel structures and constants (big-endian fields, as the game sees them).
// Layouts are the documented NT-family ones. Offsets the game itself touches were checked against the
// recompiled code.

#include "endian.h"

#include <cstdint>

namespace ac6::kernel {

// --- NTSTATUS ---------------------------------------------------------------------------------
constexpr uint32_t X_STATUS_SUCCESS                 = 0x00000000;
constexpr uint32_t X_STATUS_WAIT_0                  = 0x00000000;
constexpr uint32_t X_STATUS_ABANDONED_WAIT_0        = 0x00000080;
constexpr uint32_t X_STATUS_USER_APC                = 0x000000C0;
constexpr uint32_t X_STATUS_TIMEOUT                 = 0x00000102;
constexpr uint32_t X_STATUS_PENDING                 = 0x00000103;
constexpr uint32_t X_STATUS_OBJECT_NAME_EXISTS      = 0x40000000;
constexpr uint32_t X_STATUS_UNSUCCESSFUL            = 0xC0000001;
constexpr uint32_t X_STATUS_NOT_IMPLEMENTED         = 0xC0000002;
constexpr uint32_t X_STATUS_INVALID_HANDLE          = 0xC0000008;
constexpr uint32_t X_STATUS_INVALID_PARAMETER       = 0xC000000D;
constexpr uint32_t X_STATUS_NO_MEMORY               = 0xC0000017;
constexpr uint32_t X_STATUS_CONFLICTING_ADDRESSES   = 0xC0000018;
constexpr uint32_t X_STATUS_ACCESS_DENIED           = 0xC0000022;
constexpr uint32_t X_STATUS_BUFFER_TOO_SMALL        = 0xC0000023;
constexpr uint32_t X_STATUS_OBJECT_TYPE_MISMATCH    = 0xC0000024;
constexpr uint32_t X_STATUS_MUTANT_NOT_OWNED        = 0xC0000046;
constexpr uint32_t X_STATUS_SEMAPHORE_LIMIT_EXCEEDED= 0xC0000047;
constexpr uint32_t X_STATUS_MEMORY_NOT_ALLOCATED    = 0xC00000A0;
constexpr uint32_t X_STATUS_FREE_VM_NOT_AT_BASE     = 0xC000009F;
constexpr uint32_t X_STATUS_NOT_COMMITTED           = 0xC000002D;

constexpr uint32_t X_INFINITE              = 0xFFFFFFFF;
constexpr uint32_t X_CURRENT_THREAD_HANDLE = 0xFFFFFFFE;
constexpr uint32_t X_INVALID_HANDLE_VALUE  = 0xFFFFFFFF;

// --- Dispatcher objects -------------------------------------------------------------------------
// XDispatcherHeader::type
enum : uint8_t
{
    kObjEventNotification = 0,
    kObjEventSynchronization = 1,
    kObjMutant = 2,
    kObjSemaphore = 5,
    kObjThread = 6,
    kObjTimerNotification = 8,
    kObjTimerSynchronization = 9,
};

struct XDispatcherHeader
{
    uint8_t type;
    uint8_t absolute;
    uint8_t size;
    uint8_t inserted;
    be<int32_t> signalState;
    be<uint32_t> waitListFlink;
    be<uint32_t> waitListBlink;
};
static_assert(sizeof(XDispatcherHeader) == 0x10);

struct XKSemaphore
{
    XDispatcherHeader header;
    be<int32_t> limit;
};
static_assert(sizeof(XKSemaphore) == 0x14);

struct XKMutant
{
    XDispatcherHeader header;
    be<uint32_t> mutantListFlink;
    be<uint32_t> mutantListBlink;
    be<uint32_t> ownerThread;   // KTHREAD guest address
    uint8_t abandoned;
    uint8_t pad[3];
};
static_assert(sizeof(XKMutant) == 0x20);

struct XKTimer
{
    XDispatcherHeader header;
    be<uint64_t> dueTime;
    be<uint32_t> timerListFlink;
    be<uint32_t> timerListBlink;
    be<uint32_t> dpc;
    be<int32_t> period;
};
static_assert(sizeof(XKTimer) == 0x28);

// The guest's critical section: the embedded header is a synchronization event (the classic NT
// algorithm). OwningThread holds the owner's KTHREAD address. Fields are big-endian so guest code that
// peeks at them sees real values.
struct XRtlCriticalSection
{
    XDispatcherHeader header;
    be<int32_t> lockCount;       // -1 = free
    be<int32_t> recursionCount;
    be<uint32_t> owningThread;
};
static_assert(sizeof(XRtlCriticalSection) == 0x1C);

// KTHREAD: only the fields the game or the runtime touch are named (offsets checked in the recompiled code).
struct XKThread
{
    XDispatcherHeader header;       // 0x000: signalled when the thread terminates
    uint8_t pad0[0x14C - 0x10];
    be<uint32_t> threadId;          // 0x14C
    uint8_t pad1[0x160 - 0x150];
    be<uint32_t> lastError;         // 0x160: Get/SetLastError (game code, guarded by PCR+0x150 == 0)
    uint8_t pad2[0x300 - 0x164];
};
static_assert(sizeof(XKThread) == 0x300);

// --- Misc -----------------------------------------------------------------------------------------
struct XAnsiString
{
    be<uint16_t> length;
    be<uint16_t> maximumLength;
    be<uint32_t> buffer;
};

struct XUnicodeString
{
    be<uint16_t> length;          // bytes
    be<uint16_t> maximumLength;   // bytes
    be<uint32_t> buffer;          // big-endian UTF-16
};

struct XObjectAttributes
{
    be<uint32_t> rootDirectory;
    be<uint32_t> name;            // XAnsiString*
    be<uint32_t> attributes;
};

struct XMemoryBasicInformation
{
    be<uint32_t> baseAddress;
    be<uint32_t> allocationBase;
    be<uint32_t> allocationProtect;
    be<uint32_t> regionSize;
    be<uint32_t> state;
    be<uint32_t> protect;
    be<uint32_t> type;
};
static_assert(sizeof(XMemoryBasicInformation) == 0x1C);

} // namespace ac6::kernel
