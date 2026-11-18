# CMake toolchain: the Linux build against the Steam Runtime 3 "sniper" SDK sysroot, so the
# executables run on any distribution with glibc 2.31 or newer (SteamOS / the Steam Deck, Debian 11+, Ubuntu 20.04+,
# Fedora, Arch, ...). The system clang compiles against the sysroot's headers and libraries (glibc 2.31, GCC 14's
# libstdc++, X11 / ALSA / PulseAudio / Vulkan headers); libstdc++ is linked statically, as the runtime's own is
# GCC 10's. build_linux.sh passes AC6_SNIPER_SYSROOT (tools/linux/fetch_sniper.sh downloads the sysroot).

set(AC6_SNIPER_SYSROOT "$ENV{AC6_SNIPER_SYSROOT}" CACHE PATH "Steam Runtime sniper SDK sysroot")
if (NOT EXISTS "${AC6_SNIPER_SYSROOT}/usr/include/stdio.h")
    message(FATAL_ERROR "AC6_SNIPER_SYSROOT is not a sniper SDK sysroot: '${AC6_SNIPER_SYSROOT}'")
endif()

set(CMAKE_SYSROOT "${AC6_SNIPER_SYSROOT}")
set(CMAKE_C_COMPILER clang)
set(CMAKE_CXX_COMPILER clang++)
set(CMAKE_C_COMPILER_TARGET x86_64-linux-gnu)
set(CMAKE_CXX_COMPILER_TARGET x86_64-linux-gnu)
# GCC 14 lives in its own prefix in the SDK (/usr/lib/gcc-14): its libstdc++ headers and libraries, not GCC 10's.
set(AC6_SNIPER_GCC "--gcc-install-dir=${AC6_SNIPER_SYSROOT}/usr/lib/gcc-14/lib/gcc/x86_64-linux-gnu/14")
set(CMAKE_C_FLAGS_INIT "${AC6_SNIPER_GCC}")
set(CMAKE_CXX_FLAGS_INIT "${AC6_SNIPER_GCC}")
# libstdc++ static (the runtime's is GCC 10's); libgcc_s (the unwinder, ABI-stable, in every distribution) shared. The
# GCC 14 prefix has no libgcc_s of its own: GCC 10's, through a directory holding only it (GCC 10's own directory would
# also offer its older libstdc++ ahead of 14's).
set(AC6_SNIPER_LIBGCC "${CMAKE_BINARY_DIR}/sniper-libgcc")
file(MAKE_DIRECTORY "${AC6_SNIPER_LIBGCC}")
file(CREATE_LINK "${AC6_SNIPER_SYSROOT}/usr/lib/gcc/x86_64-linux-gnu/10/libgcc_s.so" "${AC6_SNIPER_LIBGCC}/libgcc_s.so" SYMBOLIC)
set(AC6_SNIPER_LINK "${AC6_SNIPER_GCC} -L${AC6_SNIPER_LIBGCC} -fuse-ld=lld -static-libstdc++")
set(CMAKE_EXE_LINKER_FLAGS_INIT "${AC6_SNIPER_LINK}")
set(CMAKE_SHARED_LINKER_FLAGS_INIT "${AC6_SNIPER_LINK}")

# Libraries, headers and packages from the sysroot only; programs (python, the code generators) from the host.
set(CMAKE_FIND_ROOT_PATH "${AC6_SNIPER_SYSROOT}")
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
set(ENV{PKG_CONFIG_SYSROOT_DIR} "${AC6_SNIPER_SYSROOT}")
set(ENV{PKG_CONFIG_LIBDIR} "${AC6_SNIPER_SYSROOT}/usr/lib/x86_64-linux-gnu/pkgconfig:${AC6_SNIPER_SYSROOT}/usr/lib/pkgconfig:${AC6_SNIPER_SYSROOT}/usr/share/pkgconfig")
set(ENV{PKG_CONFIG_PATH} "")

# SDL without its Wayland backend: the host's wayland-scanner writes code for a newer libwayland than the SDK's 1.18, and
# the game's window is an X11 one anyway (XWayland on Wayland desktops; plume's Vulkan surfaces are Xlib ones).
set(SDL_WAYLAND OFF CACHE BOOL "" FORCE)
