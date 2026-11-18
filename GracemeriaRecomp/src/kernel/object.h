#pragma once
// Kernel objects and the handle table.
//
// Objects live in the physical heap, so they are guest-addressable: a waitable object embeds its
// guest dispatcher header (event, semaphore, KTHREAD, ...), and that header's guest address is the
// "object pointer" the game gets from ObReferenceObjectByHandle and passes to Ke* functions. The
// dispatcher (dispatcher.h) works on headers only, so game-owned headers (critical sections, events
// the game initialises inline) and runtime-owned ones behave the same.
//
// Handles are table entries (not addresses) so stale or garbage handles are detected. Objects are
// reference counted: one reference per handle, plus ObReferenceObjectByHandle references, plus any
// the runtime holds (e.g. a running thread holds its own).

#include "heap.h"
#include "kernel/xtypes.h"

#include <atomic>
#include <cstdint>
#include <type_traits>
#include <utility>

namespace ac6::kernel {

struct KernelObject
{
    std::atomic<int32_t> refCount{ 0 };

    virtual ~KernelObject() = default;
    // Guest address of the embedded dispatcher header, or 0 if the object isn't waitable.
    virtual uint32_t DispatcherHeader() { return 0; }
    virtual const char* TypeName() const { return "object"; }
};

template<typename T, typename... Args>
T* CreateKernelObject(Args&&... args)
{
    static_assert(std::is_base_of_v<KernelObject, T>);
    void* mem = g_heap.AllocPhysical(sizeof(T), alignof(T) < 16 ? 16 : alignof(T));
    return mem ? new (mem) T(std::forward<Args>(args)...) : nullptr;
}

void RetainObject(KernelObject* obj);
void ReleaseObject(KernelObject* obj);   // destroys at zero

// Handle table. CreateHandle adds a reference, CloseHandle drops it.
uint32_t CreateHandle(KernelObject* obj);
KernelObject* LookupHandle(uint32_t handle);     // no reference added, resolves X_CURRENT_THREAD_HANDLE
uint32_t CloseHandle(uint32_t handle);           // NTSTATUS
uint32_t DuplicateHandle(uint32_t handle, uint32_t& out);

template<typename T>
T* LookupHandleAs(uint32_t handle) { return dynamic_cast<T*>(LookupHandle(handle)); }

// The guest "object pointer": the dispatcher header's address, or the object's own address if it has
// none. ObjectFromHeader maps it back (for Ob/Ke calls that take object pointers). Objects are
// registered when they get a handle, or explicitly.
uint32_t ObjectAddress(KernelObject* obj);
void RegisterObjectHeader(KernelObject* obj);
KernelObject* ObjectFromHeader(uint32_t header);

// Implemented by the thread module: the current guest thread's object (for X_CURRENT_THREAD_HANDLE).
KernelObject* CurrentThreadObject();

} // namespace ac6::kernel
