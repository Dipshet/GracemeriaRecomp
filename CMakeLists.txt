cmake_minimum_required(VERSION 3.20)
project(GracemeriaRecomp LANGUAGES C CXX)

# AC6 native runtime. Builds with llvm-mingw on Windows (toolchain/llvm-mingw.cmake) and with
# the system clang on Linux (build_linux.sh).

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)

if (NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
    message(FATAL_ERROR "The AC6 runtime requires GNU-style Clang (llvm-mingw on Windows, clang on Linux).")
endif()

# Source file names in the executables (asserts, log locations, the toolchain's headers) relative to the checkout
# instead of the building machine's folders: the same binary from any checkout.
add_compile_options("-ffile-prefix-map=${CMAKE_CURRENT_SOURCE_DIR}/=")

# The CPU level: v3 (AVX2, FMA, BMI2: the main build) or v2 (SSE4.2, POPCNT: the compatibility version, built by
# build_compat.bat into build-compat, its executables named *_compat). The v3 executables start the compat ones on CPUs
# without v3 (GracemeriaRecomp/src/cpu_level.cpp).
set(AC6_X86_LEVEL v3 CACHE STRING "x86-64 level: v3 or v2")
set_property(CACHE AC6_X86_LEVEL PROPERTY STRINGS v3 v2)

set(AC6_LIB_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/GracemeriaRecompLib")   # the recompiled code: config/, ppc/
set(AC6_XENON "${CMAKE_CURRENT_SOURCE_DIR}/tools/XenonRecomp")

# --- The third-party code fetched at configure time, and the game's code recompiled -------
include(tools/cmake/fetch_thirdparty.cmake)
include(tools/cmake/recompile.cmake)
# LuaJIT, for the mods' Lua scripts (GracemeriaRecomp/src/mods): lua51.dll / libluajit-5.1.so next to the executable.
include(tools/cmake/luajit.cmake)

# --- Recompiled game code (GracemeriaRecompLib/CMakeLists.txt) -----------------------------------------
add_subdirectory(GracemeriaRecompLib)

# --- XenonUtils XEX loader (MIT) -------------------------------------------------------------
add_library(ac6_xenonutils STATIC
    "${AC6_XENON}/XenonUtils/xex.cpp"
    "${AC6_XENON}/XenonUtils/image.cpp"
    "${AC6_XENON}/XenonUtils/xex_patcher.cpp"
    "${AC6_XENON}/XenonUtils/memory_mapped_file.cpp"
    "${AC6_XENON}/thirdparty/libmspack/libmspack/mspack/lzxd.c"
    "${AC6_XENON}/thirdparty/tiny-AES-c/aes.c"
)
target_include_directories(ac6_xenonutils
    PUBLIC
        "${AC6_XENON}/XenonUtils"
        "${AC6_XENON}/thirdparty/simde"
    PRIVATE
        "${AC6_XENON}/thirdparty/libmspack/libmspack/mspack"
        "${AC6_XENON}/thirdparty/tiny-AES-c"
        "${AC6_XENON}/thirdparty/TinySHA1"
)
target_compile_definitions(ac6_xenonutils PRIVATE NOMINMAX)
target_compile_options(ac6_xenonutils PRIVATE -w)

# --- o1heap (MIT) ----------------------------------------------------------------------------
# Pinned to 2.2.0: UnleashedRecomp's heap code relies on v2's fragment header (explicit size field).
add_library(ac6_o1heap STATIC thirdparty/o1heap/o1heap/o1heap.c)
target_include_directories(ac6_o1heap PUBLIC thirdparty/o1heap/o1heap)
# Windows is LLP64 (long is 32-bit): o1heap's default __builtin_clzl miscounts 64-bit sizes and
# corrupts bin indices. UnleashedRecomp patches the same line in its vendored copy.
target_compile_definitions(ac6_o1heap PRIVATE O1HEAP_CLZ=__builtin_clzll)

# --- SDL3 (zlib): window, events, input --------------------------------------
set(SDL_SHARED OFF CACHE BOOL "" FORCE)
set(SDL_STATIC ON CACHE BOOL "" FORCE)
set(SDL_TEST_LIBRARY OFF CACHE BOOL "" FORCE)
set(SDL_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(thirdparty/SDL EXCLUDE_FROM_ALL)
# SDL's own warnings (newer clang flags parts of it, e.g. its Vulkan renderer, which the port doesn't use) not shown.
target_compile_options(SDL3-static PRIVATE -w)

# --- plume (MIT): rendering interface, Vulkan backend only --------------------
# Built here rather than through plume's CMakeLists, which always adds the D3D12 backend on Windows.
add_library(ac6_plume STATIC thirdparty/plume/plume_vulkan.cpp)
target_include_directories(ac6_plume PUBLIC
    thirdparty/plume
    thirdparty/plume/contrib/volk
    thirdparty/plume/contrib/Vulkan-Headers/include
    thirdparty/plume/contrib/VulkanMemoryAllocator/include
)
target_compile_definitions(ac6_plume PUBLIC NOMINMAX)
target_compile_options(ac6_plume PRIVATE -w)
if (NOT WIN32)
    find_package(X11 REQUIRED)   # its Vulkan surfaces are Xlib ones on Linux
    target_link_libraries(ac6_plume PUBLIC X11::X11)
endif()

# --- AMD FidelityFX SDK v1.1.4 (MIT): FSR 3.1 ------------------------------------------
# The FSR 3.1 upscaler component and AMD's Vulkan backend, built from source against plume's volk (the backend's direct
# Vulkan calls go to volk's loaded entry points). Its compute passes are compiled with the SDK's FidelityFX_SC
# (Windows only, tools/cmake/compile_ffx_shaders.cmake). Elsewhere they come from thirdparty/fsr3-shaders, the same
# passes compiled (SPIR-V, the same on every platform), or from AC6_FFX_SHADERS_DIR.
set(AC6_FFX_SDK "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/FidelityFX-SDK/sdk")
set(AC6_FFX_SHADERS "${CMAKE_CURRENT_BINARY_DIR}/generated/ffx")
set(AC6_FFX_PREBUILT "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/fsr3-shaders")
set(AC6_FFX_SHADERS_DIR "${AC6_FFX_PREBUILT}" CACHE PATH "Prebuilt FSR 3.1 shader headers, for builds without FidelityFX_SC")
file(GLOB AC6_FFX_SHADER_SOURCES "${AC6_FFX_SDK}/src/backends/vk/shaders/fsr3upscaler/*.glsl" "${AC6_FFX_SDK}/include/FidelityFX/gpu/fsr3upscaler/*.h")
if (WIN32)
    add_custom_command(
        OUTPUT "${AC6_FFX_SHADERS}/stamp"
        COMMAND "${CMAKE_COMMAND}" -DSDK=${AC6_FFX_SDK} -DOUTPUT=${AC6_FFX_SHADERS} -DSTAMP=${AC6_FFX_SHADERS}/stamp
            -DPREBUILT=${AC6_FFX_PREBUILT} -P ${CMAKE_CURRENT_SOURCE_DIR}/tools/cmake/compile_ffx_shaders.cmake
        DEPENDS ${AC6_FFX_SHADER_SOURCES} "${CMAKE_CURRENT_SOURCE_DIR}/tools/cmake/compile_ffx_shaders.cmake"
        COMMENT "Compiling FSR 3.1's compute passes (FidelityFX_SC)"
        VERBATIM
    )
else()
    if (NOT AC6_FFX_SHADERS_DIR)
        set(AC6_FFX_SHADERS_DIR "${AC6_FFX_PREBUILT}")   # (also when the build folder's cache holds it empty)
    endif()
    file(GLOB AC6_FFX_PREBUILT_FILES "${AC6_FFX_SHADERS_DIR}/*.h")
    if (NOT AC6_FFX_PREBUILT_FILES)
        message(FATAL_ERROR "No FSR 3.1 shader headers in ${AC6_FFX_SHADERS_DIR} (AC6_FFX_SHADERS_DIR)")
    endif()
    add_custom_command(
        OUTPUT "${AC6_FFX_SHADERS}/stamp"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${AC6_FFX_SHADERS}"
        COMMAND "${CMAKE_COMMAND}" -E copy ${AC6_FFX_PREBUILT_FILES} "${AC6_FFX_SHADERS}"
        COMMAND "${CMAKE_COMMAND}" -E touch "${AC6_FFX_SHADERS}/stamp"
        DEPENDS ${AC6_FFX_PREBUILT_FILES}
        COMMENT "Using the prebuilt FSR 3.1 passes (${AC6_FFX_SHADERS_DIR})"
        VERBATIM
    )
endif()
add_custom_target(ac6_ffx_shaders DEPENDS "${AC6_FFX_SHADERS}/stamp")
add_library(ac6_ffx STATIC
    ${AC6_FFX_SDK}/src/components/fsr3upscaler/ffx_fsr3upscaler.cpp
    ${AC6_FFX_SDK}/src/backends/vk/ffx_vk.cpp
    ${AC6_FFX_SDK}/src/backends/shared/ffx_shader_blobs.cpp
    ${AC6_FFX_SDK}/src/backends/shared/blob_accessors/ffx_fsr3upscaler_shaderblobs.cpp
    ${AC6_FFX_SDK}/src/shared/ffx_assert.cpp
    ${AC6_FFX_SDK}/src/shared/ffx_message.cpp
    ${AC6_FFX_SDK}/src/shared/ffx_breadcrumbs_list.cpp
    ${AC6_FFX_SDK}/src/shared/ffx_object_management.cpp
)
add_dependencies(ac6_ffx ac6_ffx_shaders)
target_include_directories(ac6_ffx SYSTEM PUBLIC "${AC6_FFX_SDK}/include")   # (SYSTEM: its headers' warnings not shown)
target_include_directories(ac6_ffx PRIVATE "${AC6_FFX_SDK}/src/shared" "${AC6_FFX_SDK}/src/components" "${AC6_FFX_SDK}/src/backends/shared"
    "${AC6_FFX_SHADERS}")
target_compile_definitions(ac6_ffx PRIVATE FFX_FSR3UPSCALER VK_NO_PROTOTYPES _UNICODE UNICODE)
target_compile_options(ac6_ffx PRIVATE -w "SHELL:-include volk.h" "SHELL:-include ${CMAKE_CURRENT_SOURCE_DIR}/GracemeriaRecomp/src/gpu/ffx_compat.h")
target_link_libraries(ac6_ffx PUBLIC ac6_plume)

# --- FFmpeg (LGPL 2.1+, n8.1.3 source): XMA decoding -------------------------
# Static libavutil + libavcodec with only the WMA Pro / XMA2 decoders, no asm: linked into the
# executables, nothing to ship. FFmpeg's configure output (config headers, codec lists, source list) is in
# thirdparty/ffmpeg-config, a folder per platform.
set(AC6_FFMPEG_SRC "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/ffmpeg-src")
if (WIN32)
    set(AC6_FFMPEG_CONFIG "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/ffmpeg-config/windows")
else()
    set(AC6_FFMPEG_CONFIG "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/ffmpeg-config/linux")
endif()
include("${AC6_FFMPEG_CONFIG}/sources.cmake")
list(TRANSFORM AC6_FFMPEG_SOURCES PREPEND "${AC6_FFMPEG_SRC}/")
add_library(ac6_ffmpeg STATIC ${AC6_FFMPEG_SOURCES})
target_include_directories(ac6_ffmpeg PRIVATE "${AC6_FFMPEG_CONFIG}" "${AC6_FFMPEG_SRC}" "${AC6_FFMPEG_SRC}/compat/stdbit")
target_include_directories(ac6_ffmpeg SYSTEM INTERFACE "${AC6_FFMPEG_CONFIG}" "${AC6_FFMPEG_SRC}")
target_compile_definitions(ac6_ffmpeg PRIVATE HAVE_AV_CONFIG_H _ISOC11_SOURCE _FILE_OFFSET_BITS=64 _LARGEFILE_SOURCE
    WIN32_LEAN_AND_MEAN _POSIX_C_SOURCE=200112 _XOPEN_SOURCE=600)
set_target_properties(ac6_ffmpeg PROPERTIES C_STANDARD 17)
target_compile_options(ac6_ffmpeg PRIVATE -O3 -U__STRICT_ANSI__ -fno-math-errno -fno-signed-zeros -w)
if (WIN32)
    target_link_libraries(ac6_ffmpeg INTERFACE bcrypt ole32 user32)
else()
    target_link_libraries(ac6_ffmpeg INTERFACE m)
endif()

# --- Import stubs (generated from ppc_recomp_shared.h, tools/cmake/gen_imports.cmake) ---------------
set(AC6_IMPORTS_GEN "${CMAKE_CURRENT_BINARY_DIR}/generated/imports_gen.cpp")
add_custom_command(
    OUTPUT "${AC6_IMPORTS_GEN}"
    COMMAND "${CMAKE_COMMAND}" "-DSHARED=${AC6_LIB_ROOT}/ppc/ppc_recomp_shared.h" "-DOUTPUT=${AC6_IMPORTS_GEN}"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/tools/cmake/gen_imports.cmake"
    DEPENDS "${AC6_LIB_ROOT}/ppc/ppc_recomp_shared.h" "${CMAKE_CURRENT_SOURCE_DIR}/tools/cmake/gen_imports.cmake"
    COMMENT "Generating import stubs"
    VERBATIM
)

# --- The Lua mods' API script (GracemeriaRecomp/src/mods/lua/ac6.lua), embedded ---------------------
set(AC6_MODS_LUA_GEN "${CMAKE_CURRENT_BINARY_DIR}/generated/mods_lua_api.cpp")
add_custom_command(
    OUTPUT "${AC6_MODS_LUA_GEN}"
    COMMAND "${CMAKE_COMMAND}" -DINPUT=${CMAKE_CURRENT_SOURCE_DIR}/GracemeriaRecomp/src/mods/lua/ac6.lua -DOUTPUT=${AC6_MODS_LUA_GEN} -DSYMBOL=kModsLuaApi -P ${CMAKE_CURRENT_SOURCE_DIR}/tools/cmake/embed_text.cmake
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/GracemeriaRecomp/src/mods/lua/ac6.lua" "${CMAKE_CURRENT_SOURCE_DIR}/tools/cmake/embed_text.cmake"
    COMMENT "Embedding the Lua mods' API script"
    VERBATIM
)
set(AC6_MODS_SYMBOLS_GEN "${CMAKE_CURRENT_BINARY_DIR}/generated/mods_symbols.cpp")
add_custom_command(
    OUTPUT "${AC6_MODS_SYMBOLS_GEN}"
    COMMAND "${CMAKE_COMMAND}" -DINPUT=${CMAKE_CURRENT_SOURCE_DIR}/GracemeriaRecomp/src/mods/symbols.toml -DOUTPUT=${AC6_MODS_SYMBOLS_GEN} -DSYMBOL=kModsSymbols -P ${CMAKE_CURRENT_SOURCE_DIR}/tools/cmake/embed_text.cmake
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/GracemeriaRecomp/src/mods/symbols.toml" "${CMAKE_CURRENT_SOURCE_DIR}/tools/cmake/embed_text.cmake"
    COMMENT "Embedding the Lua mods' built-in symbols"
    VERBATIM
)

# --- Shader translation: XenosRecomp (vendored, MIT) + DXC at runtime ---------
set(AC6_XENOS_COMMON_GEN "${CMAKE_CURRENT_BINARY_DIR}/generated/xenos_shader_common.cpp")
add_custom_command(
    OUTPUT "${AC6_XENOS_COMMON_GEN}"
    COMMAND "${CMAKE_COMMAND}" -DINPUT=${CMAKE_CURRENT_SOURCE_DIR}/GracemeriaRecomp/src/gpu/xenos/shader_common.h -DOUTPUT=${AC6_XENOS_COMMON_GEN} -DSYMBOL=kXenosShaderCommon -P ${CMAKE_CURRENT_SOURCE_DIR}/tools/cmake/embed_text.cmake
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/GracemeriaRecomp/src/gpu/xenos/shader_common.h" "${CMAKE_CURRENT_SOURCE_DIR}/tools/cmake/embed_text.cmake"
    COMMENT "Embedding the Xenos HLSL common header"
    VERBATIM
)
# The shader cache's key: a hash of everything that shapes a translated shader.
set(AC6_SHADER_KEY_GEN "${CMAKE_CURRENT_BINARY_DIR}/generated/shader_cache_key.cpp")
set(AC6_SHADER_KEY_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/GracemeriaRecomp/src/gpu/xenos/shader_recompiler.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/GracemeriaRecomp/src/gpu/xenos/shader_recompiler.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/GracemeriaRecomp/src/gpu/xenos/spirv_recompiler.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/GracemeriaRecomp/src/gpu/xenos/spirv_recompiler.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/GracemeriaRecomp/src/gpu/xenos/spirv_builder.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/GracemeriaRecomp/src/gpu/xenos/shader_common.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/GracemeriaRecomp/src/gpu/xenos/shader_code.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/GracemeriaRecomp/src/gpu/xenos/shader.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/GracemeriaRecomp/src/gpu/xenos/constant_table.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/GracemeriaRecomp/src/gpu/xenos/xenos_prelude.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/GracemeriaRecomp/src/gpu/dxc.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/GracemeriaRecomp/src/gpu/shaders.cpp")   # (its options decide translation inputs too)
add_custom_command(
    OUTPUT "${AC6_SHADER_KEY_GEN}"
    COMMAND "${CMAKE_COMMAND}" -DOUTPUT=${AC6_SHADER_KEY_GEN} -DSYMBOL=kShaderCacheKey "-DFILES=${AC6_SHADER_KEY_SOURCES}" -P ${CMAKE_CURRENT_SOURCE_DIR}/tools/cmake/source_key.cmake
    DEPENDS ${AC6_SHADER_KEY_SOURCES} "${CMAKE_CURRENT_SOURCE_DIR}/tools/cmake/source_key.cmake"
    COMMENT "Hashing the shader translator sources"
    VERBATIM
)
# DXC: on Windows the one build.bat downloads into toolchain/dxc (through toolchain/setup.bat: no Vulkan
# SDK needed), else the Vulkan SDK's (VULKAN_SDK, or the newest in C:/VulkanSDK). Linux: AC6_DXC (build_linux.sh
# downloads Microsoft's Linux release into toolchain/dxc-linux), or dxc on the PATH. Without one the internal shaders
# can come from AC6_INTERNAL_SHADERS_CPP, the generated/internal_shaders.cpp of a build that had DXC: SPIR-V, the same
# on every platform. AC6_DXC_INCLUDE: the folder with dxcapi.h, for gpu/dxc.cpp.
set(AC6_DXC "" CACHE FILEPATH "dxc executable (empty: toolchain/dxc's, the Vulkan SDK's / the PATH's)")
set(AC6_INTERNAL_SHADERS_CPP "" CACHE FILEPATH "Prebuilt internal_shaders.cpp, for builds without DXC")
set(AC6_VULKAN_SDK_INCLUDE "")
set(AC6_DXC_INCLUDE "")
get_filename_component(AC6_TOOLCHAIN_DXC "${CMAKE_CURRENT_SOURCE_DIR}/toolchain/dxc" ABSOLUTE)
if (WIN32 AND EXISTS "${AC6_TOOLCHAIN_DXC}/bin/x64/dxc.exe")
    if (NOT AC6_DXC)
        set(AC6_DXC "${AC6_TOOLCHAIN_DXC}/bin/x64/dxc.exe")
    endif()
    set(AC6_DXC_INCLUDE "${AC6_TOOLCHAIN_DXC}/inc")
elseif (WIN32)
    if (DEFINED ENV{VULKAN_SDK})
        file(TO_CMAKE_PATH "$ENV{VULKAN_SDK}" AC6_VULKAN_SDK)
    else()
        # Newest SDK installed in the default location.
        file(GLOB AC6_VULKAN_SDKS LIST_DIRECTORIES true "C:/VulkanSDK/*")
        list(SORT AC6_VULKAN_SDKS COMPARE NATURAL ORDER DESCENDING)
        list(GET AC6_VULKAN_SDKS 0 AC6_VULKAN_SDK)
    endif()
    if (NOT EXISTS "${AC6_VULKAN_SDK}/Bin/dxc.exe")
        message(FATAL_ERROR "No DXC: run build.bat in the repository's top folder (it downloads one into toolchain/dxc), "
            "or install the Vulkan SDK")
    endif()
    if (NOT AC6_DXC)
        set(AC6_DXC "${AC6_VULKAN_SDK}/Bin/dxc.exe")
    endif()
    set(AC6_VULKAN_SDK_INCLUDE "${AC6_VULKAN_SDK}/Include")
    set(AC6_DXC_INCLUDE "${AC6_VULKAN_SDK}/Include/dxc")
elseif (NOT AC6_DXC)
    find_program(AC6_DXC_FOUND dxc)
    if (AC6_DXC_FOUND)
        set(AC6_DXC "${AC6_DXC_FOUND}")
    endif()
endif()
# The settings overlay's fonts (thirdparty/fonts, SIL OFL), the shipped pipeline list (renderer.cpp PipelineList) and the
# PlayStation face button glyphs (assets/button_glyphs, gpu/texture_mods.cpp), embedded (tools/cmake/embed_files.cmake).
set(AC6_FONTS_GEN "${CMAKE_CURRENT_BINARY_DIR}/generated/fonts.cpp")
set(AC6_SHIPPED_PIPELINES "${CMAKE_CURRENT_SOURCE_DIR}/GracemeriaRecomp/src/gpu/shipped_pipelines.bin")
set(AC6_FONTS "ShareTech=ShareTech-Regular.ttf|ShareTechMono=ShareTechMono-Regular.ttf|ShippedPipelines=${AC6_SHIPPED_PIPELINES}")
set(AC6_PS_GLYPHS "")
foreach(hash 567C603ACEC3BEF6 5815009FFD003F54 62B5B1271CAFB882 B30926707688AFA7)
    set(glyph "${CMAKE_CURRENT_SOURCE_DIR}/assets/button_glyphs/playstation/${hash}.dds")
    string(APPEND AC6_FONTS "|PlayStationGlyph_${hash}=${glyph}")
    list(APPEND AC6_PS_GLYPHS "${glyph}")
endforeach()
add_custom_command(
    OUTPUT "${AC6_FONTS_GEN}"
    COMMAND "${CMAKE_COMMAND}" "-DOUTPUT=${AC6_FONTS_GEN}" "-DFILES=${AC6_FONTS}" "-DDIR=${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/fonts" -P ${CMAKE_CURRENT_SOURCE_DIR}/tools/cmake/embed_files.cmake
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/tools/cmake/embed_files.cmake" "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/fonts/ShareTech-Regular.ttf"
        "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/fonts/ShareTechMono-Regular.ttf" "${AC6_SHIPPED_PIPELINES}" ${AC6_PS_GLYPHS}
    COMMENT "Embedding the overlay's fonts, the shipped pipeline list and the button glyphs"
    VERBATIM)
# The renderer's own shaders, compiled at build time.
set(AC6_INTERNAL_SHADERS_GEN "${CMAKE_CURRENT_BINARY_DIR}/generated/internal_shaders.cpp")
if (AC6_DXC)
    add_custom_command(
        OUTPUT "${AC6_INTERNAL_SHADERS_GEN}"
        COMMAND "${CMAKE_COMMAND}" "-DDXC=${AC6_DXC}" -DINPUT=${CMAKE_CURRENT_SOURCE_DIR}/GracemeriaRecomp/src/gpu/internal_shaders.hlsl
            -DOUTPUT=${AC6_INTERNAL_SHADERS_GEN} -DTEMP=${CMAKE_CURRENT_BINARY_DIR}/generated/internal_shaders -P ${CMAKE_CURRENT_SOURCE_DIR}/tools/cmake/compile_internal_shaders.cmake
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/GracemeriaRecomp/src/gpu/internal_shaders.hlsl" "${CMAKE_CURRENT_SOURCE_DIR}/tools/cmake/compile_internal_shaders.cmake"
            "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/smaa/SMAA.hlsl" "${CMAKE_CURRENT_SOURCE_DIR}/GracemeriaRecomp/src/gpu/volumetrics.hlsl"
            "${CMAKE_CURRENT_SOURCE_DIR}/GracemeriaRecomp/src/gpu/ao.hlsl" "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/XeGTAO/XeGTAO.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/GracemeriaRecomp/src/gpu/lens_fx.hlsl" "${CMAKE_CURRENT_SOURCE_DIR}/GracemeriaRecomp/src/gpu/motion_blur.hlsl" "${CMAKE_CURRENT_SOURCE_DIR}/GracemeriaRecomp/src/gpu/emitter_light.hlsl" "${CMAKE_CURRENT_SOURCE_DIR}/GracemeriaRecomp/src/gpu/bloom.hlsl"
            "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/renodx/src/shaders/tonemap/reno_drt.hlsl"
        COMMENT "Compiling the renderer's internal shaders"
        VERBATIM
    )
elseif (AC6_INTERNAL_SHADERS_CPP)
    add_custom_command(
        OUTPUT "${AC6_INTERNAL_SHADERS_GEN}"
        COMMAND "${CMAKE_COMMAND}" -E copy "${AC6_INTERNAL_SHADERS_CPP}" "${AC6_INTERNAL_SHADERS_GEN}"
        DEPENDS "${AC6_INTERNAL_SHADERS_CPP}"
        COMMENT "Using the prebuilt internal shaders (${AC6_INTERNAL_SHADERS_CPP})"
        VERBATIM
    )
else()
    message(FATAL_ERROR "No DXC: build with build_linux.sh (it downloads one), set AC6_DXC, or set AC6_INTERNAL_SHADERS_CPP to "
        "a Windows build's generated/internal_shaders.cpp")
endif()

# --- Runtime ---------------------------------------------------------------------------------
# Everything but main.cpp, shared by the two executables below.
# Dear ImGui (MIT, thirdparty/imgui: 1.93.0 with its SDL3 platform backend): the settings overlay (GracemeriaRecomp/src/ui/overlay.cpp).
add_library(ac6_imgui STATIC
    thirdparty/imgui/imgui.cpp
    thirdparty/imgui/imgui_draw.cpp
    thirdparty/imgui/imgui_tables.cpp
    thirdparty/imgui/imgui_widgets.cpp
    thirdparty/imgui/backends/imgui_impl_sdl3.cpp)
target_include_directories(ac6_imgui PUBLIC thirdparty/imgui thirdparty/imgui/backends)
target_link_libraries(ac6_imgui PUBLIC SDL3::SDL3-static)

add_library(ac6_runtime OBJECT
    GracemeriaRecomp/src/log.cpp
    GracemeriaRecomp/src/settings.cpp
    GracemeriaRecomp/src/system_info.cpp
    GracemeriaRecomp/src/update/updater.cpp
    GracemeriaRecomp/src/memory.cpp
    GracemeriaRecomp/src/heap.cpp
    GracemeriaRecomp/src/loader.cpp
    GracemeriaRecomp/src/crash.cpp
    GracemeriaRecomp/src/profiler.cpp
    $<IF:$<BOOL:${WIN32}>,GracemeriaRecomp/src/platform/platform_win32.cpp,GracemeriaRecomp/src/platform/platform_linux.cpp>
    $<IF:$<BOOL:${WIN32}>,GracemeriaRecomp/src/crash_win32.cpp,GracemeriaRecomp/src/crash_linux.cpp>
    GracemeriaRecomp/src/simd.cpp
    GracemeriaRecomp/src/cpu_level.cpp
    GracemeriaRecomp/src/apu/audio_driver.cpp
    GracemeriaRecomp/src/apu/cutscene_sync.cpp
    GracemeriaRecomp/src/job_order.cpp
    GracemeriaRecomp/src/apu/xma.cpp
    GracemeriaRecomp/src/fs/dlc.cpp
    GracemeriaRecomp/src/fs/iso_device.cpp
    GracemeriaRecomp/src/fs/save_store.cpp
    GracemeriaRecomp/src/fs/stfs.cpp
    GracemeriaRecomp/src/fs/vfs.cpp
    GracemeriaRecomp/src/gpu/capture.cpp
    GracemeriaRecomp/src/gpu/d3d.cpp
    GracemeriaRecomp/src/gpu/d3d_census.cpp
    GracemeriaRecomp/src/gpu/deferred_commands.cpp
    GracemeriaRecomp/src/gpu/render_queue.cpp
    GracemeriaRecomp/src/gpu/dxc.cpp
    GracemeriaRecomp/src/gpu/no_gpu.cpp
    GracemeriaRecomp/src/gpu/renderer.cpp
    GracemeriaRecomp/src/gpu/breadcrumbs.cpp
    GracemeriaRecomp/src/gpu/shaders.cpp
    GracemeriaRecomp/src/gpu/vblank.cpp
    GracemeriaRecomp/src/gpu/frame_rate.cpp
    GracemeriaRecomp/src/gpu/fsr.cpp
    GracemeriaRecomp/src/gpu/texture_mods.cpp
    GracemeriaRecomp/src/game/frame_rate_fixes.cpp
    GracemeriaRecomp/src/game/packages.cpp
    GracemeriaRecomp/src/game/inflate.cpp
    GracemeriaRecomp/src/game/cutscene_unlock.cpp
    GracemeriaRecomp/src/game/damage_report_guard.cpp
    GracemeriaRecomp/src/game/tu_mission_check.cpp
    GracemeriaRecomp/src/game/cockpit.cpp
    GracemeriaRecomp/src/game/hud_colour.cpp
    GracemeriaRecomp/src/game/game_modes.cpp
    GracemeriaRecomp/src/game/frame_handoff.cpp
    GracemeriaRecomp/src/game/ui_layers.cpp
    GracemeriaRecomp/src/game/missile_trace.cpp
    GracemeriaRecomp/src/game/draw_distance.cpp
    GracemeriaRecomp/src/game/language.cpp
    GracemeriaRecomp/src/game/localization.cpp
    GracemeriaRecomp/src/game/widescreen.cpp
    GracemeriaRecomp/src/gpu/video.cpp
    GracemeriaRecomp/src/gpu/xenos.cpp
    GracemeriaRecomp/src/gpu/xenos/shader_recompiler.cpp
    GracemeriaRecomp/src/gpu/xenos/spirv_recompiler.cpp
    "${AC6_XENOS_COMMON_GEN}"
    "${AC6_SHADER_KEY_GEN}"
    "${AC6_INTERNAL_SHADERS_GEN}"
    "${AC6_FONTS_GEN}"
    GracemeriaRecomp/src/cpu/guest_thread.cpp
    GracemeriaRecomp/src/input/input.cpp
    GracemeriaRecomp/src/input/kbm.cpp
    GracemeriaRecomp/src/input/dualsense.cpp
    GracemeriaRecomp/src/input/advanced.cpp
    GracemeriaRecomp/src/ui/overlay.cpp
    GracemeriaRecomp/src/game/kbm_heartbeat.cpp
    GracemeriaRecomp/src/game/flight_input.cpp
    GracemeriaRecomp/src/plugin/plugins.cpp
    GracemeriaRecomp/src/mods/hooks.cpp
    GracemeriaRecomp/src/mods/mods.cpp
    GracemeriaRecomp/src/mods/rtti.cpp
    "${AC6_MODS_LUA_GEN}"
    "${AC6_MODS_SYMBOLS_GEN}"
    GracemeriaRecomp/src/kernel/clock.cpp
    GracemeriaRecomp/src/kernel/dispatcher.cpp
    GracemeriaRecomp/src/kernel/imports_file.cpp
    GracemeriaRecomp/src/kernel/imports_memory.cpp
    GracemeriaRecomp/src/kernel/imports_rtl.cpp
    GracemeriaRecomp/src/kernel/imports_sync.cpp
    GracemeriaRecomp/src/kernel/imports_timer.cpp
    GracemeriaRecomp/src/kernel/imports_system.cpp
    GracemeriaRecomp/src/kernel/imports_thread.cpp
    GracemeriaRecomp/src/kernel/imports_video.cpp
    GracemeriaRecomp/src/kernel/imports_xam.cpp
    GracemeriaRecomp/src/kernel/imports_xam_content.cpp
    GracemeriaRecomp/src/kernel/object.cpp
    GracemeriaRecomp/src/kernel/trace.cpp
    GracemeriaRecomp/src/kernel/vm.cpp
    "${AC6_IMPORTS_GEN}"
)
# GracemeriaRecomp/src/ for quoted includes only: its endian.h would shadow the system's <endian.h> (Linux).
target_compile_options(ac6_runtime PUBLIC "-iquote${CMAKE_CURRENT_SOURCE_DIR}/GracemeriaRecomp/src")
target_include_directories(ac6_runtime PUBLIC thirdparty/spirv-headers thirdparty/xxhash ${AC6_VULKAN_SDK_INCLUDE} ${AC6_DXC_INCLUDE})
target_include_directories(ac6_runtime SYSTEM PRIVATE tools/XenonRecomp/thirdparty/tomlplusplus/include)
target_compile_definitions(ac6_runtime PUBLIC NOMINMAX WIN32_LEAN_AND_MEAN)
target_compile_options(ac6_runtime PUBLIC -march=x86-64-${AC6_X86_LEVEL} -fno-strict-aliasing -gline-tables-only -Wall -Wno-unused-parameter)
# The CPU check runs before anything else and must not itself need v3.
set_source_files_properties(GracemeriaRecomp/src/cpu_level.cpp PROPERTIES COMPILE_OPTIONS "-march=x86-64")
if (AC6_X86_LEVEL STREQUAL "v2")
    target_compile_definitions(ac6_runtime PUBLIC AC6_X86_V2=1)
endif()
target_compile_options(ac6_xenonutils PRIVATE -gline-tables-only)
# The version, for the log and the update check (update/updater.cpp), and whether this is a release build (AC6_RELEASE:
# one that updates itself from the GitHub releases; the build scripts pass the AC6_RELEASE environment variable, off
# unless set): on that file alone, so a new version or the switch rebuilds one file. The repository holds no version:
# a release is built with AC6_VERSION set (-DAC6_VERSION or the environment, e.g. 1.0.1, the release's tag without its
# v), any other build is "dev".
option(AC6_RELEASE "A release build: it updates itself from the project's GitHub releases" OFF)
if (NOT DEFINED AC6_VERSION OR AC6_VERSION STREQUAL "")
    set(AC6_VERSION "$ENV{AC6_VERSION}")
endif()
if (AC6_VERSION STREQUAL "")
    if (AC6_RELEASE)
        message(FATAL_ERROR "A release build needs its version: set AC6_VERSION (e.g. 1.0.1)")
    endif()
    set(AC6_VERSION "dev")
endif()
set(AC6_UPDATER_DEFINITIONS "AC6_VERSION=\"${AC6_VERSION}\"")
if (AC6_RELEASE)
    list(APPEND AC6_UPDATER_DEFINITIONS "AC6_RELEASE=1")
endif()
set_source_files_properties(GracemeriaRecomp/src/update/updater.cpp PROPERTIES COMPILE_DEFINITIONS "${AC6_UPDATER_DEFINITIONS}")
target_link_libraries(ac6_runtime PUBLIC AC6RecompLib ac6_xenonutils ac6_o1heap ac6_plume ac6_ffx SDL3::SDL3-static ac6_ffmpeg ac6_imgui luajit)
add_dependencies(ac6_runtime luajit_jit_modules)
if (WIN32)
    target_link_libraries(ac6_runtime PUBLIC winmm synchronization bcrypt dxgi winhttp)   # dxgi: the log's GPU list (system_info.cpp), winhttp: updates
else()
    target_link_libraries(ac6_runtime PUBLIC pthread dl)
endif()

# Functions start on 64-byte lines: without it a change in one source file moves the recompiled
# code's hot loops across fetch and cache-line boundaries, and the game runs up to 5% faster or slower from build to
# build with no change to what it runs.
foreach(target AC6RecompLib ac6_runtime)
    target_compile_options(${target} PRIVATE -falign-functions=64)
endforeach()

# Profile-guided optimisation of the recompiled code and the runtime. AC6_PGO=generate: an
# instrumented build (it writes its profile at shutdown to LLVM_PROFILE_FILE, default default.profraw),
# AC6_PGO=use: optimised with AC6_PGO_PROFILE (merged with llvm-profdata). Off by default: it gives
# no measurable gain, even CPU-bound on 4 threads (+0.8%, within noise).
set(AC6_PGO_PROFILE "${CMAKE_CURRENT_SOURCE_DIR}/GracemeriaRecomp/pgo/ac6.profdata" CACHE FILEPATH "Merged profile for AC6_PGO=use")
if(NOT DEFINED AC6_PGO)
    set(AC6_PGO off)
endif()
if(AC6_PGO STREQUAL "generate")
    foreach(target AC6RecompLib ac6_runtime)
        target_compile_options(${target} PRIVATE -fprofile-instr-generate)
    endforeach()
    target_compile_definitions(ac6_runtime PRIVATE AC6_PGO_GENERATE=1)
    target_link_options(ac6_runtime PUBLIC -fprofile-instr-generate)
elseif(AC6_PGO STREQUAL "use")
    foreach(target AC6RecompLib ac6_runtime)
        target_compile_options(${target} PRIVATE "-fprofile-instr-use=${AC6_PGO_PROFILE}" -Wno-profile-instr-unprofiled
            -Wno-profile-instr-out-of-date -Wno-profile-instr-missing)
    endforeach()
endif()
message(STATUS "AC6 profile-guided optimisation: ${AC6_PGO}")
set_source_files_properties(GracemeriaRecomp/src/gpu/dxc.cpp PROPERTIES COMPILE_OPTIONS "-fms-extensions")
set_source_files_properties(GracemeriaRecomp/src/gpu/xenos/shader_recompiler.cpp PROPERTIES COMPILE_OPTIONS "-Wno-switch;-Wno-unused-variable;-Wno-unused-but-set-variable;-Wno-null-arithmetic")

# ac6.exe: development build with a console. AceCombat6.exe: the same runtime as a windowed app (no
# console, the log still goes to ac6_runtime.log, and fatal errors show a message box).
add_executable(ac6 GracemeriaRecomp/src/main.cpp)
add_executable(ac6_release WIN32 GracemeriaRecomp/src/main.cpp)   # Linux: the same program (no console subsystem there)
set_target_properties(ac6_release PROPERTIES OUTPUT_NAME AceCombat6)
if (AC6_X86_LEVEL STREQUAL "v2")
    set_target_properties(ac6 PROPERTIES OUTPUT_NAME ac6_compat)
    set_target_properties(ac6_release PROPERTIES OUTPUT_NAME AceCombat6_compat)
endif()
target_compile_definitions(ac6_release PRIVATE AC6_WINDOWED_APP=1)
foreach(exe ac6 ac6_release)
    target_link_libraries(${exe} PRIVATE ac6_runtime)
    if (WIN32)
        # Recompiled code nests deeply on the host stack: 16 MB reserve for every thread (main + CreateThread default).
        target_link_options(${exe} PRIVATE -Wl,--stack,16777216)
    else()
        # libluajit-5.1.so sits next to the executable (tools/cmake/luajit.cmake): found there wherever the game is put.
        # The RUNPATH is $ORIGIN alone: a BUILD_RPATH would be added to CMake's own, the build folder's absolute path,
        # which then went out with the executable.
        set_target_properties(${exe} PROPERTIES BUILD_WITH_INSTALL_RPATH ON INSTALL_RPATH "\$ORIGIN")
    endif()
    # (Linux: the threads' 16 MB are set at startup, main.cpp and platform/platform_linux.cpp.)
endforeach()

# --- Plugins: native mods, mods/<mod>/<name>.dll, built on their own against GracemeriaRecomp/src/plugin/ac6_plugin.h
# only. PSM (post-stall manoeuvres) has its own repository.

# --- The example Lua mods (examples/mods): each copied to build/example_mods/<mod>, ready to copy into mods/ next to the
# game (they aren't installed there: the build doesn't change the game's mods).
set(AC6_EXAMPLE_MODS "${CMAKE_BINARY_DIR}/example_mods")
set(AC6_EXAMPLE_MOD_NAMES aircraft_list input_display)
set(AC6_EXAMPLE_MOD_COPIES "")
foreach (mod ${AC6_EXAMPLE_MOD_NAMES})
    file(GLOB_RECURSE files CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/examples/mods/${mod}/*")
    set(stamp "${AC6_EXAMPLE_MODS}/${mod}/.copied")
    add_custom_command(OUTPUT "${stamp}"
        COMMAND "${CMAKE_COMMAND}" -E copy_directory "${CMAKE_CURRENT_SOURCE_DIR}/examples/mods/${mod}" "${AC6_EXAMPLE_MODS}/${mod}"
        COMMAND "${CMAKE_COMMAND}" -E touch "${stamp}"
        DEPENDS ${files} COMMENT "Example mod ${mod}" VERBATIM)
    list(APPEND AC6_EXAMPLE_MOD_COPIES "${stamp}")
endforeach()
add_custom_target(ac6_example_mods ALL DEPENDS ${AC6_EXAMPLE_MOD_COPIES})

# --- shadertest: differential test of the shader translator (tools/shadertest) -------------------
# Windows only: it compiles through DXC's library (gpu/dxc.cpp). Built only when tools/shadertest is present.
if (WIN32 AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tools/shadertest/main.cpp")
add_executable(shadertest
    tools/shadertest/main.cpp
    tools/shadertest/xenos_interp.cpp
    GracemeriaRecomp/src/log.cpp
    GracemeriaRecomp/src/platform/platform_win32.cpp
    GracemeriaRecomp/src/gpu/dxc.cpp
    GracemeriaRecomp/src/gpu/xenos/shader_recompiler.cpp
    GracemeriaRecomp/src/gpu/xenos/spirv_recompiler.cpp
    "${AC6_XENOS_COMMON_GEN}"
)
target_compile_options(shadertest PRIVATE "-iquote${CMAKE_CURRENT_SOURCE_DIR}/GracemeriaRecomp/src")
target_include_directories(shadertest PRIVATE thirdparty/spirv-headers ${AC6_VULKAN_SDK_INCLUDE} ${AC6_DXC_INCLUDE})
target_compile_definitions(shadertest PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
target_compile_options(shadertest PRIVATE -march=x86-64-${AC6_X86_LEVEL} -Wall -Wno-unused-parameter)
target_link_libraries(shadertest PRIVATE ac6_plume synchronization winmm)   # (platform_win32.cpp)
endif()
