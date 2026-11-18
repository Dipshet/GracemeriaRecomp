#!/bin/sh
# Downloads the Steam Runtime 3 "sniper" SDK sysroot that portable Linux builds compile against (build_linux.sh with
# AC6_SNIPER_SYSROOT, and the releases): Debian 11's glibc 2.31 and GCC 14's libstdc++, so the executables run on
# SteamOS / the Steam Deck, Debian 11+, Ubuntu 20.04+, Fedora, Arch and the rest. fetch_sniper.sh [folder], default
# toolchain/sniper. Valve's snapshot 3.0.20260928.262393 (1.2 GB, checked against its SHA-256), unpacked into
# <folder>/sysroot with its absolute symbolic links made relative (they would point into the host otherwise).
# Prints the sysroot's path. Needs curl, tar and realpath.
set -e
here=$(cd "$(dirname "$0")/../.." && pwd)
dir=${1:-$here/toolchain/sniper}
version=3.0.20260928.262393
name='com.valvesoftware.SteamRuntime.Sdk-amd64,i386-sniper-sysroot.tar.gz'
sum=adb13b9e124e73e0b2cf25e0022703978ec8e917b9a192dcec61bebd72479521
sysroot="$dir/sysroot"
if [ -f "$sysroot/usr/include/stdio.h" ] && [ "$(cat "$dir/VERSION.txt" 2>/dev/null)" = "$version" ]; then
    echo "$sysroot"
    exit 0
fi
mkdir -p "$dir"
archive="$dir/$name"
if ! echo "$sum  $archive" | sha256sum -c - >/dev/null 2>&1; then
    echo "Downloading the Steam Runtime sniper SDK sysroot $version (1.2 GB)" >&2
    curl -fL -o "$archive" "https://repo.steampowered.com/steamrt-images-sniper/snapshots/$version/$name"
    echo "$sum  $archive" | sha256sum -c - >&2
fi
rm -rf "$sysroot"
mkdir -p "$sysroot"
echo "Unpacking it into $sysroot" >&2
tar -xzf "$archive" -C "$sysroot"
# Absolute links (/usr/lib/...) made relative to the sysroot.
find "$sysroot" -type l -lname '/*' | while IFS= read -r link; do
    target=$(readlink "$link")
    ln -sfn "$(realpath -m --relative-to="$(dirname "$link")" "$sysroot$target")" "$link"
done
echo "$version" > "$dir/VERSION.txt"
rm -f "$archive"
echo "$sysroot"
