#include "kernel/object.h"

#include "log.h"
#include "memory.h"

#include <mutex>
#include <unordered_map>

namespace ac6::kernel {

namespace {

// Handle values: 0xF8000000 + 4n. Never 0, -1 or the current-thread pseudo-handle.
constexpr uint32_t kHandleBase = 0xF8000000;

std::mutex s_tableMutex;
std::unordered_map<uint32_t, KernelObject*> s_handles;
std::unordered_map<uint32_t, KernelObject*> s_headers;
uint32_t s_nextHandle = kHandleBase + 4;

void Destroy(KernelObject* obj)
{
    {
        std::lock_guard lock(s_tableMutex);
        auto it = s_headers.find(ObjectAddress(obj));
        if (it != s_headers.end() && it->second == obj)
            s_headers.erase(it);
    }
    obj->~KernelObject();
    g_heap.Free(obj);
}

} // namespace

void RetainObject(KernelObject* obj)
{
    obj->refCount.fetch_add(1, std::memory_order_relaxed);
}

void ReleaseObject(KernelObject* obj)
{
    if (obj->refCount.fetch_sub(1, std::memory_order_acq_rel) == 1)
        Destroy(obj);
}

uint32_t ObjectAddress(KernelObject* obj)
{
    uint32_t header = obj->DispatcherHeader();
    return header ? header : g_memory.MapVirtual(obj);
}

void RegisterObjectHeader(KernelObject* obj)
{
    std::lock_guard lock(s_tableMutex);
    s_headers[ObjectAddress(obj)] = obj;
}

KernelObject* ObjectFromHeader(uint32_t header)
{
    std::lock_guard lock(s_tableMutex);
    auto it = s_headers.find(header);
    return it != s_headers.end() ? it->second : nullptr;
}

uint32_t CreateHandle(KernelObject* obj)
{
    RetainObject(obj);
    RegisterObjectHeader(obj);
    std::lock_guard lock(s_tableMutex);
    while (s_handles.contains(s_nextHandle) || s_nextHandle == 0 || s_nextHandle >= X_CURRENT_THREAD_HANDLE)
        s_nextHandle = s_nextHandle >= 0xFFFFFFF0 ? kHandleBase + 4 : s_nextHandle + 4;
    uint32_t handle = s_nextHandle;
    s_nextHandle += 4;
    s_handles[handle] = obj;
    return handle;
}

KernelObject* LookupHandle(uint32_t handle)
{
    if (handle == X_CURRENT_THREAD_HANDLE)
        return CurrentThreadObject();
    std::lock_guard lock(s_tableMutex);
    auto it = s_handles.find(handle);
    return it != s_handles.end() ? it->second : nullptr;
}

uint32_t CloseHandle(uint32_t handle)
{
    KernelObject* obj = nullptr;
    {
        std::lock_guard lock(s_tableMutex);
        auto it = s_handles.find(handle);
        if (it == s_handles.end())
            return X_STATUS_INVALID_HANDLE;
        obj = it->second;
        s_handles.erase(it);
    }
    ReleaseObject(obj);
    return X_STATUS_SUCCESS;
}

uint32_t DuplicateHandle(uint32_t handle, uint32_t& out)
{
    KernelObject* obj = LookupHandle(handle);
    if (!obj)
        return X_STATUS_INVALID_HANDLE;
    out = CreateHandle(obj);
    return X_STATUS_SUCCESS;
}

} // namespace ac6::kernel
