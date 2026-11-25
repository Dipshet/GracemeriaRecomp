# Compiles the renderer's own shaders (GracemeriaRecomp/src/gpu/internal_shaders.hlsl) to SPIR-V with DXC and embeds
# them: the executable doesn't need a shader compiler.
#   cmake -DDXC=<dxc.exe> -DINPUT=<hlsl> -DOUTPUT=<cpp> -DTEMP=<dir> -P compile_internal_shaders.cmake
# Variants: name|stage|defines (comma-separated). The flags match the runtime's DXC path (gpu/dxc.cpp).
set(VARIANTS
    "copyVs|vs|VERTEX"
    "copyPs|ps|PIXEL"
    "hdrComposePs|ps|PIXEL,HDR_COMPOSE"
    "glowMergePs|ps|PIXEL,GLOW_MERGE"
    "gammaPs|ps|PIXEL,GAMMA_RAMP"
    "pillarFrostPs|ps|PIXEL,PILLAR_FROST"
    "gammaHdrPs|ps|PIXEL,GAMMA_RAMP,GAMMA_RAMP_HDR"
    "gammaLutPs|ps|PIXEL,GAMMA_RAMP,GAMMA_RAMP_LUT"
    "depthCopyPs|ps|PIXEL,DEPTH_COPY"
    "depthCopyMsPs|ps|PIXEL,DEPTH_COPY_MS"
    "depthCopySsPs|ps|PIXEL,DEPTH_COPY_SS"
    "depthResamplePs|ps|PIXEL,DEPTH_RESAMPLE"
    "depthSwizzlePs|ps|PIXEL,DEPTH_SWIZZLE"
    "colorSwizzlePs|ps|PIXEL,COLOR_SWIZZLE"
    "depthToColorPs|ps|PIXEL,DEPTH_TO_COLOR"
    "depthLoadPs|ps|PIXEL,DEPTH_LOAD"
    "colorLoadPs|ps|PIXEL,COLOR_LOAD"
    "texelCopyPs|ps|PIXEL,TEXEL_COPY"
    "stencilBitPs|ps|PIXEL,STENCIL_BIT"
    "taaPs|ps|PIXEL,TAA"
    "sharpenPs|ps|PIXEL,SHARPEN"
    "waterSsrPs|ps|PIXEL,WATER_SSR"
    "planarBlurPs|ps|PIXEL,PLANAR_BLUR"
    "aoDepthPs|ps|AO_DEPTH"
    "aoDepthMipPs|ps|AO_DEPTH_MIP"
    "aoMainPs|ps|AO_MAIN"
    "aoDenoisePs|ps|AO_DENOISE"
    "aoApplyPs|ps|AO_APPLY"
    "aoSubjectPs|ps|AO_SUBJECT"
    "lensFxPs|ps|LENS_FX"
    "motionBlurPs|ps|MOTION_BLUR"
    "bloomBrightPs|ps|BLOOM_BRIGHT"
    "bloomDownPs|ps|BLOOM_DOWN"
    "bloomUpPs|ps|BLOOM_UP"
    "bloomApplyPs|ps|BLOOM_APPLY"
    "lightDown0Ps|ps|LIGHT_DOWN0"
    "lightDown0bPs|ps|LIGHT_DOWN0B"
    "lightDownPs|ps|LIGHT_DOWN"
    "lightApplyPs|ps|LIGHT_APPLY"
    "motionCompletePs|ps|PIXEL,MOTION_COMPLETE"
    "outputPs|ps|PIXEL,OUTPUT"
    "smaaEdgePs|ps|PIXEL,SMAA_EDGE"
    "smaaWeightPs|ps|PIXEL,SMAA_WEIGHT"
    "smaaBlendPs|ps|PIXEL,SMAA_BLEND"
    "volHarvestGs|gs|VOL_HARVEST"
    "lateBoundsGs|gs|LATE_BOUNDS"
    "line2dGs|gs|LINE_2D"
    "volVs|vs|VOL_VS"
    "volPs|ps|VOL_PS"
    "volComposePs|ps|VOL_COMPOSE"
    "volCloudVs|vs|VOL_CLOUDSCAPE,VOL_CLOUD_VS"
    "volCloudMapPs|ps|VOL_CLOUDSCAPE,VOL_CLOUD_MAP"
    "volCloudBoundsPs|ps|VOL_CLOUDSCAPE,VOL_CLOUD_BOUNDS"
    "volCloudMarchPs|ps|VOL_CLOUDSCAPE,VOL_CLOUD_MARCH"
    "volCloudTemporalPs|ps|VOL_CLOUDSCAPE,VOL_CLOUD_TEMPORAL"
    "volCloudOverPs|ps|VOL_CLOUDSCAPE,VOL_CLOUD_OVER"
    "volCloudScenePs|ps|VOL_CLOUDSCAPE,VOL_CLOUD_SCENE"
    "volCloudDepthPs|ps|VOL_CLOUDSCAPE,VOL_CLOUD_DEPTH"
    "volCloudDepthOutPs|ps|VOL_CLOUDSCAPE,VOL_CLOUD_DEPTH,VOL_CLOUD_DEPTH_OUT"
    "volCloudGatePs|ps|VOL_CLOUDSCAPE,VOL_CLOUD_GATE"
    "imguiVs|vs|IMGUI_VS"
    "imguiPs|ps|IMGUI_PS"
    "nullPs|ps|PIXEL,NULL_PIXEL")

file(MAKE_DIRECTORY "${TEMP}")
set(BLOBS "")
set(TABLE "")
set(COUNT 0)
foreach(variant ${VARIANTS})
    string(REPLACE "|" ";" parts "${variant}")
    list(GET parts 0 name)
    list(GET parts 1 stage)
    list(GET parts 2 defines)
    string(REPLACE "," ";" defines "${defines}")
    set(args -T ${stage}_6_0 -HV 2021 -all-resources-bound -spirv -fvk-use-dx-layout -Qstrip_debug -E main)
    if (stage STREQUAL "vs")
        list(APPEND args -fvk-invert-y)
    endif()
    foreach(define ${defines})
        list(APPEND args -D ${define}=1)
    endforeach()
    execute_process(COMMAND "${DXC}" ${args} -Fo "${TEMP}/${name}.spv" "${INPUT}"
        RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE errors)
    if (NOT result EQUAL 0)
        message(FATAL_ERROR "dxc failed on internal shader ${name}: ${errors}")
    endif()
    file(READ "${TEMP}/${name}.spv" hex HEX)
    string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," bytes "${hex}")
    string(APPEND BLOBS "alignas(4) const unsigned char k_${name}[] = { ${bytes} };\n")
    string(APPEND TABLE "    { \"${name}\", k_${name}, sizeof(k_${name}) },\n")
    math(EXPR COUNT "${COUNT} + 1")
endforeach()

file(WRITE "${OUTPUT}.tmp" "// Generated from ${INPUT} by tools/cmake/compile_internal_shaders.cmake. Do not edit.\n"
    "#include \"gpu/internal_shaders.h\"\nnamespace ac6::gpu {\nnamespace {\n${BLOBS}}\n"
    "extern const InternalShaderBlob kInternalShaders[] = {\n${TABLE}};\n"
    "extern const unsigned kInternalShaderCount = ${COUNT};\n}\n")
file(COPY_FILE "${OUTPUT}.tmp" "${OUTPUT}" ONLY_IF_DIFFERENT)
