#!/bin/sh
# Builds the AC6 runtime for Linux with the system clang: build_linux.sh [build dir] [cmake --build options]. The
# build dir defaults to build-linux here; on WSL put it on the Linux disk (e.g. ~/ac6-build): building over /mnt is
# several times slower. Needs clang, lld, cmake, ninja, curl and SDL3's X11 / Wayland / audio headers. DXC, the
# compiler for the renderer's own shaders: AC6_DXC, dxc on the PATH, or Microsoft's Linux release, downloaded into
# toolchain/dxc-linux (13 MB, checked against its SHA-256). FSR 3.1's passes come compiled (thirdparty/fsr3-shaders).
# With AC6_SNIPER_SYSROOT set (a Steam Runtime sniper SDK sysroot: tools/linux/fetch_sniper.sh) the build compiles
# against it instead of the system's libraries: executables that run on any distribution with glibc 2.31 or newer, the
# Steam Deck included (tools/linux/steamrt-sniper.cmake). Without it they need the building system's glibc or newer.
set -e
here=$(cd "$(dirname "$0")" && pwd)
build=${1:-$here/build-linux}
[ $# -gt 0 ] && shift
dxc=${AC6_DXC:-$(command -v dxc || true)}
if [ -z "$dxc" ]; then
    dxc="$here/toolchain/dxc-linux/bin/dxc"
    if [ ! -x "$dxc" ]; then
        url=https://github.com/microsoft/DirectXShaderCompiler/releases/download/v1.9.2609/linux_dxc_2026_09_28.x86_x64.tar.gz
        sum=96faadc7f5c282d2ffda49804beb4c3ee38127bc252b723234e3c5cdf7aa39a1
        archive="$here/toolchain/dxc-linux.tar.gz"
        echo "Downloading DXC 1.9.2609 ($url)"
        mkdir -p "$here/toolchain/dxc-linux"
        curl -fL -o "$archive" "$url"
        echo "$sum  $archive" | sha256sum -c -
        tar -xzf "$archive" -C "$here/toolchain/dxc-linux"
        rm "$archive"
    fi
fi
# AC6_RELEASE=ON: a release build, which updates itself from the GitHub releases (off unless set).
release=${AC6_RELEASE:-OFF}
if [ -n "$AC6_SNIPER_SYSROOT" ]; then
    sysroot=$(cd "$AC6_SNIPER_SYSROOT" && pwd)
    AC6_SNIPER_SYSROOT="$sysroot" cmake -S "$here" -B "$build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
        "-DCMAKE_TOOLCHAIN_FILE=$here/tools/linux/steamrt-sniper.cmake" "-DAC6_SNIPER_SYSROOT=$sysroot" "-DAC6_DXC=$dxc" "-DAC6_RELEASE=$release"
else
    cmake -S "$here" -B "$build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
        -DCMAKE_EXE_LINKER_FLAGS=-fuse-ld=lld "-DAC6_DXC=$dxc" "-DAC6_RELEASE=$release"
fi
cmake --build "$build" "$@"
