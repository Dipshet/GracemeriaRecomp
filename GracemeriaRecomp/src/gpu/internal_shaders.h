#pragma once
// The renderer's own shaders, compiled at build time from internal_shaders.hlsl.

namespace ac6::gpu {

struct InternalShaderBlob
{
    const char* name;
    const unsigned char* data;   // SPIR-V, 4-byte aligned
    unsigned size;               // bytes
};

extern const InternalShaderBlob kInternalShaders[];
extern const unsigned kInternalShaderCount;

} // namespace ac6::gpu
