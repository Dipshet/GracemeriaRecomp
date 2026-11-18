# Third-party code

What each library is, where it comes from, how it is included and what this port changed. Every library keeps its
own licence file in its folder.

| Folder | Library | Version | Licence | Included as | Local changes |
|-|-|-|-|-|-|
| `SDL` | [SDL](https://github.com/libsdl-org/SDL) | 3.4.18 (`release-3.4.18`) | zlib | git submodule | none |
| `imgui` | [Dear ImGui](https://github.com/ocornut/imgui) | 1.93.0 (`v1.93.0`) | MIT | git submodule | none |
| `FidelityFX-SDK` | [AMD FidelityFX SDK](https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK) | 1.1.4 (`v1.1.4`) | MIT | cloned by the build (`tools/cmake/fetch_thirdparty.cmake`) | `tools/patches/fidelityfx-ac6.patch`: the Vulkan backend for FSR 3.1 |
| `fsr3-shaders` | FSR 3.1's compute passes from that SDK, compiled to SPIR-V | 1.1.4 | MIT | files | none: compiled by the SDK's FidelityFX_SC (Windows only) for the builds on other platforms (its README) |
| `ffmpeg-src` | [FFmpeg](https://ffmpeg.org) | n8.1.3 | LGPL 2.1 or later | files, trimmed | `tools/patches/ffmpeg-xma.patch` (XMA packet boundary, single-stream hold-back, first frame). Only the files the WMA Pro / XMA2 decoders build from are kept; regenerating `ffmpeg-config` needs the full n8.1.3 source with the patch |
| `ffmpeg-config` | FFmpeg's configure output for that build | | LGPL 2.1 or later | files | made for the port (only the two decoders, no assembly) |
| `plume` | [plume](https://github.com/renderbag/plume) | from `d723793` | MIT | files | adapted for the port: the Vulkan backend only, with the features the renderer uses (bindless textures, dynamic rendering, HDR swap chains and more); volk, the VulkanMemoryAllocator header and the Vulkan C headers in `contrib` |
| `o1heap` | [o1heap](https://github.com/pavel-kirienko/o1heap) | 2.2.0 | MIT | files | none |
| `renodx` | [RenoDX](https://github.com/clshortfuse/renodx) `src/shaders` | `a99ff41c` | MIT | files, a subset | none: only RenoDRT, PsychoV-30, the inverse tone mapping and what they include |
| `smaa` | [SMAA](https://github.com/iryoku/smaa) | master | MIT | files, a subset | none: `SMAA.hlsl`, `AreaTex.h`, `SearchTex.h` |
| `XeGTAO` | [XeGTAO](https://github.com/GameTechDev/XeGTAO) | | MIT | files, a subset | none: `XeGTAO.h`, `XeGTAO.hlsli` |
| `xxhash` | [xxHash](https://github.com/Cyan4973/xxHash) | 0.8.3 | BSD-2-Clause | files, a subset | none |
| `spirv-headers` | [SPIRV-Headers](https://github.com/KhronosGroup/SPIRV-Headers) | | MIT | files, a subset | none: `spirv.hpp`, `GLSL.std.450.h` |
| `fonts` | [Share Tech, Share Tech Mono](https://fonts.google.com/specimen/Share+Tech) | | SIL OFL 1.1 | files | none |

XenonRecomp (MIT, from `ddd128b`, adapted for the port) and its own libraries are in `tools/XenonRecomp`.
XenosRecomp's shader translator (MIT, from `990d03b`, adapted for the port: the Xenos microcode decoding and the HLSL
backend) is in `GracemeriaRecomp/src/gpu/xenos`, with its licence. The SPIR-V backend there, the one in use, is the
port's own, written after it.
