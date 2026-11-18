#pragma once
// Host <-> guest calling-convention glue: write imports and hooks as ordinary typed C++ functions.
// Adapted from UnleashedRecomp (UnleashedRecomp/kernel/function.h, GPL-3.0). The difference is the ac6
// namespace and memory API. As upstream, arguments beyond r10 are read from the stack for integers only, and
// calls into guest code (GuestCall) can't pass stack arguments.
//
//   uint32_t NtClose(uint32_t handle);                 // integer/pointer args from r3.., floats from f1..
//   AC6_IMPORT(NtClose)                                 // defines __imp__NtClose, marshalling + tracing
//   AC6_HOOK(sub_82123456, MyReplacement)              // replaces a recompiled guest function

#include "kernel/trace.h"
#include "kernel/xtypes.h"
#include "memory.h"
#include "ppc.h"

#include <array>
#include <cstddef>
#include <tuple>
#include <type_traits>

namespace ac6 {

template<typename R, typename... T>
constexpr std::tuple<T...> function_args(R (*)(T...)) noexcept { return std::tuple<T...>(); }

template<typename T>
constexpr bool is_precise_v = std::is_same_v<T, float> || std::is_same_v<T, double>;

struct ArgTranslator
{
    static uint64_t GetInteger(const PPCContext& ctx, uint8_t* base, size_t arg) noexcept
    {
        switch (arg)
        {
        case 0: return ctx.r3.u32;
        case 1: return ctx.r4.u32;
        case 2: return ctx.r5.u32;
        case 3: return ctx.r6.u32;
        case 4: return ctx.r7.u32;
        case 5: return ctx.r8.u32;
        case 6: return ctx.r9.u32;
        case 7: return ctx.r10.u32;
        default:
            // Xbox ABI: further integer arguments live at r1 + 0x54 + 8 * (n - 8).
            return *reinterpret_cast<be<uint32_t>*>(base + ctx.r1.u32 + 0x54 + ((arg - 8) * 8));
        }
    }

    static double GetPrecise(const PPCContext& ctx, size_t arg) noexcept
    {
        switch (arg)
        {
        case 0: return ctx.f1.f64;
        case 1: return ctx.f2.f64;
        case 2: return ctx.f3.f64;
        case 3: return ctx.f4.f64;
        case 4: return ctx.f5.f64;
        case 5: return ctx.f6.f64;
        case 6: return ctx.f7.f64;
        case 7: return ctx.f8.f64;
        case 8: return ctx.f9.f64;
        case 9: return ctx.f10.f64;
        case 10: return ctx.f11.f64;
        case 11: return ctx.f12.f64;
        case 12: return ctx.f13.f64;
        default: return 0.0;
        }
    }

    static void SetInteger(PPCContext& ctx, size_t arg, uint64_t value) noexcept
    {
        switch (arg)
        {
        case 0: ctx.r3.u64 = value; return;
        case 1: ctx.r4.u64 = value; return;
        case 2: ctx.r5.u64 = value; return;
        case 3: ctx.r6.u64 = value; return;
        case 4: ctx.r7.u64 = value; return;
        case 5: ctx.r8.u64 = value; return;
        case 6: ctx.r9.u64 = value; return;
        case 7: ctx.r10.u64 = value; return;
        default: return; // stack arguments to guest calls are not supported (as upstream)
        }
    }

    static void SetPrecise(PPCContext& ctx, size_t arg, double value) noexcept
    {
        switch (arg)
        {
        case 0: ctx.f1.f64 = value; return;
        case 1: ctx.f2.f64 = value; return;
        case 2: ctx.f3.f64 = value; return;
        case 3: ctx.f4.f64 = value; return;
        case 4: ctx.f5.f64 = value; return;
        case 5: ctx.f6.f64 = value; return;
        case 6: ctx.f7.f64 = value; return;
        case 7: ctx.f8.f64 = value; return;
        default: return;
        }
    }

    template<typename T>
    static T Get(PPCContext& ctx, uint8_t* base, size_t idx) noexcept
    {
        if constexpr (std::is_pointer_v<T>)
        {
            const auto v = static_cast<uint32_t>(GetInteger(ctx, base, idx));
            return v ? reinterpret_cast<T>(base + v) : nullptr;
        }
        else if constexpr (is_precise_v<T>)
            return static_cast<T>(GetPrecise(ctx, idx));
        else
            return static_cast<T>(GetInteger(ctx, base, idx));
    }

    template<typename T>
    static void Set(PPCContext& ctx, size_t idx, T value) noexcept
    {
        if constexpr (std::is_pointer_v<T>)
            SetInteger(ctx, idx, g_memory.MapVirtual(value));
        else if constexpr (is_precise_v<T>)
            SetPrecise(ctx, idx, value);
        else
            SetInteger(ctx, idx, static_cast<uint64_t>(value));
    }
};

// Integer arguments use the integer register sequence by position. Float arguments use their own sequence.
template<typename Tuple>
constexpr auto ArgumentOrdinals()
{
    std::array<size_t, std::tuple_size_v<Tuple>> ordinals{};
    size_t floatIndex = 0;
    [&]<size_t... I>(std::index_sequence<I...>) {
        ((ordinals[I] = is_precise_v<std::tuple_element_t<I, Tuple>> ? floatIndex++ : I), ...);
    }(std::make_index_sequence<std::tuple_size_v<Tuple>>{});
    return ordinals;
}

template<auto Func>
void HostToGuestFunction(PPCContext& ctx, uint8_t* base)
{
    using Args = decltype(function_args(Func));
    constexpr auto ordinals = ArgumentOrdinals<Args>();
    Args args;
    [&]<size_t... I>(std::index_sequence<I...>) {
        ((std::get<I>(args) = ArgTranslator::Get<std::tuple_element_t<I, Args>>(ctx, base, ordinals[I])), ...);
    }(std::make_index_sequence<std::tuple_size_v<Args>>{});

    using Ret = decltype(std::apply(Func, args));
    if constexpr (std::is_void_v<Ret>)
        std::apply(Func, args);
    else
    {
        auto v = std::apply(Func, args);
        if constexpr (std::is_pointer_v<Ret>)
            ctx.r3.u64 = g_memory.MapVirtual(v);
        else if constexpr (is_precise_v<Ret>)
            ctx.f1.f64 = v;
        else
            ctx.r3.u64 = static_cast<uint64_t>(v);
    }
}

// Call a guest function from host code (e.g. thread entry points, callbacks).
template<typename Ret = void, typename... Args>
Ret GuestCall(uint32_t guestFunction, Args... argv)
{
    PPCContext& current = *GetPPCContext();
    PPCContext call = current; // shares stack pointer, r13 and FPSCR with the caller
    constexpr auto ordinals = ArgumentOrdinals<std::tuple<Args...>>();
    [&]<size_t... I>(std::index_sequence<I...>) {
        (ArgTranslator::Set(call, ordinals[I], argv), ...);
    }(std::index_sequence_for<Args...>{});

    SetPPCContext(call);
    g_memory.Function(guestFunction)(call, g_memory.base);
    current.fpscr = call.fpscr;
    SetPPCContext(current);

    if constexpr (std::is_pointer_v<Ret>)
        return reinterpret_cast<Ret>(g_memory.Translate(call.r3.u32));
    else if constexpr (is_precise_v<Ret>)
        return static_cast<Ret>(call.f1.f64);
    else if constexpr (!std::is_void_v<Ret>)
        return static_cast<Ret>(call.r3.u64);
}

} // namespace ac6

// Defines the import __imp__<name> from a host function ac6::kernel::<name> with the same name.
#define AC6_IMPORT(name)                                                           \
    PPC_FUNC(__imp__##name)                                                        \
    {                                                                              \
        ac6::kernel::TraceCall(#name, ctx);                                        \
        ac6::HostToGuestFunction<&ac6::kernel::name>(ctx, base);                   \
    }

// Replaces a recompiled guest function with a host function.
#define AC6_HOOK(guestFunction, hostFunction)                                      \
    PPC_FUNC(guestFunction) { ac6::HostToGuestFunction<&hostFunction>(ctx, base); }
