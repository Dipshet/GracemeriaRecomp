// Run-time library and time imports: system time, time fields, strings, status codes, debug output.
#include "heap.h"
#include "kernel/clock.h"
#include "kernel/dispatcher.h"
#include "kernel/function.h"
#include "log.h"
#include "memory.h"

#include <chrono>
#include <cstring>
#include <string>

namespace ac6::kernel {

namespace {

struct XTimeFields
{
    be<uint16_t> year, month, day, hour, minute, second, milliseconds, weekday;
};

size_t WideLength(const be<uint16_t>* s)
{
    size_t n = 0;
    while (s[n] != 0)
        n++;
    return n;
}

// The console's ANSI conversions are Latin-1-ish. Characters above 0xFF become '?'.
char NarrowChar(uint16_t c) { return c < 0x100 ? static_cast<char>(c) : '?'; }

} // namespace

// --- Time ------------------------------------------------------------------------------------------

void KeQuerySystemTime(be<uint64_t>* time)
{
    *time = SystemTime100ns();
}

uint64_t KeQueryPerformanceFrequency()
{
    return TimebaseFrequency();
}

// NT times (100 ns since 1601-01-01) and calendar fields, the proleptic Gregorian calendar as Windows' FileTimeToSystemTime /
// SystemTimeToFileTime have it.
namespace {
constexpr uint64_t kTicksPerMs = 10000, kTicksPerDay = 864000000000ull;
constexpr std::chrono::sys_days kNtEpoch = std::chrono::sys_days(std::chrono::year(1601) / 1 / 1);
} // namespace

void RtlTimeToTimeFields(const be<int64_t>* time, XTimeFields* fields)
{
    const uint64_t t = static_cast<uint64_t>(int64_t(*time));
    const auto day = kNtEpoch + std::chrono::days(int64_t(t / kTicksPerDay));
    const std::chrono::year_month_day date(day);
    const uint64_t ms = (t % kTicksPerDay) / kTicksPerMs;
    fields->year = uint16_t(int(date.year()));
    fields->month = uint16_t(unsigned(date.month()));
    fields->day = uint16_t(unsigned(date.day()));
    fields->hour = uint16_t(ms / 3600000);
    fields->minute = uint16_t(ms / 60000 % 60);
    fields->second = uint16_t(ms / 1000 % 60);
    fields->milliseconds = uint16_t(ms % 1000);
    fields->weekday = uint16_t(std::chrono::weekday(day).c_encoding());
}

uint32_t RtlTimeFieldsToTime(const XTimeFields* fields, be<int64_t>* time)
{
    const std::chrono::year_month_day date(std::chrono::year(int(fields->year)), std::chrono::month(unsigned(fields->month)),
        std::chrono::day(unsigned(fields->day)));
    if (fields->year < 1601 || fields->year > 30827 || !date.ok() || fields->hour > 23 || fields->minute > 59 || fields->second > 59 ||
        fields->milliseconds > 999)
        return 0;
    const uint64_t days = uint64_t((std::chrono::sys_days(date) - kNtEpoch).count());
    const uint64_t ms = ((uint64_t(fields->hour) * 60 + fields->minute) * 60 + fields->second) * 1000 + fields->milliseconds;
    *time = static_cast<int64_t>(days * kTicksPerDay + ms * kTicksPerMs);
    return 1;
}

// --- Strings ---------------------------------------------------------------------------------------

void RtlInitAnsiString(XAnsiString* string, const char* source)
{
    uint16_t length = source ? static_cast<uint16_t>(std::strlen(source)) : 0;
    string->length = length;
    string->maximumLength = source ? length + 1 : 0;
    string->buffer = g_memory.MapVirtual(source);
}

void RtlInitUnicodeString(XUnicodeString* string, const be<uint16_t>* source)
{
    uint16_t bytes = source ? static_cast<uint16_t>(WideLength(source) * 2) : 0;
    string->length = bytes;
    string->maximumLength = source ? bytes + 2 : 0;
    string->buffer = g_memory.MapVirtual(source);
}

uint32_t RtlUnicodeStringToAnsiString(XAnsiString* destination, const XUnicodeString* source, uint32_t allocate)
{
    const uint32_t chars = source->length / 2;
    const auto* src = g_memory.As<be<uint16_t>>(source->buffer);
    if (allocate)
    {
        void* buffer = g_heap.Alloc(chars + 1);
        if (!buffer)
            return X_STATUS_NO_MEMORY;
        destination->buffer = g_memory.MapVirtual(buffer);
        destination->maximumLength = static_cast<uint16_t>(chars + 1);
    }
    else if (destination->maximumLength < chars + 1)
    {
        return X_STATUS_BUFFER_TOO_SMALL;
    }
    char* dst = g_memory.As<char>(destination->buffer);
    for (uint32_t i = 0; i < chars; i++)
        dst[i] = NarrowChar(src[i]);
    dst[chars] = 0;
    destination->length = static_cast<uint16_t>(chars);
    return X_STATUS_SUCCESS;
}

void RtlFreeAnsiString(XAnsiString* string)
{
    if (string->buffer)
        g_heap.Free(g_memory.Translate(string->buffer));
    string->buffer = 0;
    string->length = 0;
    string->maximumLength = 0;
}

uint32_t RtlUnicodeToMultiByteN(char* destination, uint32_t maxBytes, be<uint32_t>* bytesWritten,
    const be<uint16_t>* source, uint32_t sourceBytes)
{
    uint32_t n = std::min(maxBytes, sourceBytes / 2);
    for (uint32_t i = 0; i < n; i++)
        destination[i] = NarrowChar(source[i]);
    if (bytesWritten)
        *bytesWritten = n;
    return X_STATUS_SUCCESS;
}

// --- Status codes and debug output -------------------------------------------------------------------

uint32_t RtlNtStatusToDosError(uint32_t status)
{
    if ((status & 0xFFFF0000) == 0x80070000)   // HRESULT_FROM_WIN32
        return status & 0xFFFF;
    switch (status)
    {
    case X_STATUS_SUCCESS: return 0;
    case X_STATUS_TIMEOUT: return 1460;              // ERROR_TIMEOUT
    case X_STATUS_PENDING: return 997;               // ERROR_IO_PENDING
    case X_STATUS_UNSUCCESSFUL: return 31;           // ERROR_GEN_FAILURE
    case X_STATUS_NOT_IMPLEMENTED: return 1;         // ERROR_INVALID_FUNCTION
    case X_STATUS_INVALID_HANDLE: return 6;          // ERROR_INVALID_HANDLE
    case X_STATUS_INVALID_PARAMETER: return 87;      // ERROR_INVALID_PARAMETER
    case 0xC000000F: return 2;                       // STATUS_NO_SUCH_FILE -> ERROR_FILE_NOT_FOUND
    case 0xC0000011: return 38;                      // STATUS_END_OF_FILE -> ERROR_HANDLE_EOF
    case X_STATUS_NO_MEMORY: return 8;               // ERROR_NOT_ENOUGH_MEMORY
    case X_STATUS_ACCESS_DENIED: return 5;           // ERROR_ACCESS_DENIED
    case X_STATUS_BUFFER_TOO_SMALL: return 122;      // ERROR_INSUFFICIENT_BUFFER
    case 0xC0000034: return 2;                       // STATUS_OBJECT_NAME_NOT_FOUND -> ERROR_FILE_NOT_FOUND
    case 0xC0000035: return 183;                     // STATUS_OBJECT_NAME_COLLISION -> ERROR_ALREADY_EXISTS
    case 0xC000003A: return 3;                       // STATUS_OBJECT_PATH_NOT_FOUND -> ERROR_PATH_NOT_FOUND
    case 0xC0000043: return 32;                      // STATUS_SHARING_VIOLATION
    case 0xC000007F: return 112;                     // STATUS_DISK_FULL
    case 0xC000009C: return 23;                      // STATUS_DEVICE_DATA_ERROR -> ERROR_CRC
    case 0x80000006: return 18;                      // STATUS_NO_MORE_FILES
    default:
        log::Warn("RtlNtStatusToDosError: unmapped status {:08X}", status);
        return 317;                                  // ERROR_MR_MID_NOT_FOUND, as NT does
    }
}

void DbgPrint(const char* format)
{
    // Arguments are not formatted (guest varargs). The format string alone is usually telling.
    if (format)
    {
        std::string text(format);
        while (!text.empty() && (text.back() == '\n' || text.back() == '\r'))
            text.pop_back();
        log::Info("[DbgPrint] {}", text);
    }
}

} // namespace ac6::kernel

AC6_IMPORT(KeQuerySystemTime)
AC6_IMPORT(KeQueryPerformanceFrequency)
AC6_IMPORT(RtlTimeToTimeFields)
AC6_IMPORT(RtlTimeFieldsToTime)
AC6_IMPORT(RtlInitAnsiString)
AC6_IMPORT(RtlInitUnicodeString)
AC6_IMPORT(RtlUnicodeStringToAnsiString)
AC6_IMPORT(RtlFreeAnsiString)
AC6_IMPORT(RtlUnicodeToMultiByteN)
AC6_IMPORT(RtlNtStatusToDosError)
AC6_IMPORT(DbgPrint)
