# FSR 3.1's compute passes for AMD's Vulkan backend: the FidelityFX SDK's shader compiler driver
# (FidelityFX_SC, prebuilt in the SDK, Windows only) compiles each pass's GLSL with glslang into SPIR-V, every permutation,
# and writes the headers the backend's blob accessors include: the same arguments as the SDK's
# CMakeCompileFSR3UpscalerShaders.txt and backends/vk/CMakeShadersFSR3Upscaler.txt, in the four variants (wave 32 / 64,
# 32 / 16 bit) of its CMakeCompileShaders.txt.
# PREBUILT: thirdparty/fsr3-shaders, the copy other platforms build with: a warning when it differs from these.
#   cmake -DSDK=<FidelityFX-SDK/sdk> -DOUTPUT=<directory> -DSTAMP=<file> [-DPREBUILT=<directory>] -P compile_ffx_shaders.cmake
set(SC "${SDK}/tools/binary_store/FidelityFX_SC.exe")
set(GPU "${SDK}/include/FidelityFX/gpu")
file(GLOB PASSES "${SDK}/src/backends/vk/shaders/fsr3upscaler/*.glsl")
file(MAKE_DIRECTORY "${OUTPUT}")
set(BASE_ARGS -reflection -deps=gcc -DFFX_GPU=1
    -DFFX_FSR3UPSCALER_OPTION_UPSAMPLE_SAMPLERS_USE_DATA_HALF=0
    -DFFX_FSR3UPSCALER_OPTION_ACCUMULATE_SAMPLERS_USE_DATA_HALF=0
    -DFFX_FSR3UPSCALER_OPTION_REPROJECT_SAMPLERS_USE_DATA_HALF=1
    -DFFX_FSR3UPSCALER_OPTION_POSTPROCESSLOCKSTATUS_SAMPLERS_USE_DATA_HALF=0
    -DFFX_FSR3UPSCALER_OPTION_UPSAMPLE_USE_LANCZOS_TYPE=2
    -compiler=glslang -e CS --target-env vulkan1.2 -S comp -Os -DFFX_GLSL=1
    "-DFFX_FSR3UPSCALER_OPTION_REPROJECT_USE_LANCZOS_TYPE={0,1}"
    "-DFFX_FSR3UPSCALER_OPTION_HDR_COLOR_INPUT={0,1}"
    "-DFFX_FSR3UPSCALER_OPTION_LOW_RESOLUTION_MOTION_VECTORS={0,1}"
    "-DFFX_FSR3UPSCALER_OPTION_JITTERED_MOTION_VECTORS={0,1}"
    "-DFFX_FSR3UPSCALER_OPTION_INVERTED_DEPTH={0,1}"
    "-DFFX_FSR3UPSCALER_OPTION_APPLY_SHARPENING={0,1}"
    "-I${GPU}" "-I${GPU}/fsr3upscaler" "-output=${OUTPUT}")
foreach (PASS ${PASSES})
    get_filename_component(NAME "${PASS}" NAME_WE)
    foreach (VARIANT "|0" "_wave64|0" "_16bit|1" "_wave64_16bit|1")
        string(REPLACE "|" ";" PARTS "${VARIANT}")
        list(GET PARTS 0 SUFFIX)
        list(GET PARTS 1 HALF)
        # (glslang: the wave size is the driver's, the wave64 variants differ by name only, as in the SDK's Vulkan build.)
        execute_process(COMMAND "${SC}" ${BASE_ARGS} -name=${NAME}${SUFFIX} -DFFX_HALF=${HALF} "${PASS}"
            WORKING_DIRECTORY "${OUTPUT}" RESULT_VARIABLE RESULT OUTPUT_VARIABLE LOG ERROR_VARIABLE LOG)
        if (NOT RESULT EQUAL 0)
            message(FATAL_ERROR "FidelityFX_SC failed on ${NAME}${SUFFIX}:\n${LOG}")
        endif()
    endforeach()
endforeach()
if (PREBUILT)
    # The compiled passes alone, the headers named by their content's hash. The *_permutations.h tables are not
    # compared: FidelityFX_SC compiles in parallel, and where permutations share one compiled pass their numbering
    # changes from run to run (the same mapping either way).
    file(GLOB MADE RELATIVE "${OUTPUT}" "${OUTPUT}/*.h")
    file(GLOB KEPT RELATIVE "${PREBUILT}" "${PREBUILT}/*.h")
    list(FILTER MADE EXCLUDE REGEX "_permutations\\.h$")
    list(FILTER KEPT EXCLUDE REGEX "_permutations\\.h$")
    set(DIFFERENT "")
    if (NOT "${MADE}" STREQUAL "${KEPT}")
        set(DIFFERENT "a different set of files")
    else()
        foreach (F ${MADE})
            file(SHA256 "${OUTPUT}/${F}" A)
            file(SHA256 "${PREBUILT}/${F}" B)
            if (NOT A STREQUAL B)
                set(DIFFERENT "${F}")
                break()
            endif()
        endforeach()
    endif()
    if (DIFFERENT)
        message(WARNING "FSR 3.1's compiled passes differ from thirdparty/fsr3-shaders (${DIFFERENT}): other platforms "
            "build with that copy. Copy ${OUTPUT}/*.h there.")
    endif()
endif()
file(TOUCH "${STAMP}")
