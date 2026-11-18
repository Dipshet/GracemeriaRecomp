#pragma once
// Xenos (Xbox 360 GPU) data layouts the renderer decodes from guest memory. Field positions come from
// the game's own D3D/XGraphics code (texture header setter 0x821E0830, SetTexture 0x821E10C8,
// SetStreamSource 0x821DD0A8, DrawVertices 0x821DEF18) and its format table at 0x82067E48.

#include "endian.h"

#include <bit>
#include <cstdint>

namespace ac6::gpu::xenos {

// --- Texture fetch constant (6 dwords: texture object +28..+48, device +0x480 + sampler * 24) ------
struct TextureFetch
{
    uint32_t dw[6];

    uint32_t Type() const { return dw[0] & 3; }                        // 2 = texture
    uint32_t Pitch() const { return ((dw[0] >> 22) & 0x1FF) * 32; }     // texels
    bool Tiled() const { return (dw[0] >> 31) != 0; }
    uint32_t Format() const { return dw[1] & 0x3F; }
    uint32_t Endian() const { return (dw[1] >> 6) & 3; }                // 0 none, 1 8in16, 2 8in32, 3 16in32
    uint32_t BaseAddress() const { return dw[1] & 0xFFFFF000; }        // physical
    uint32_t Width() const { return (dw[2] & 0x1FFF) + 1; }             // 2D
    uint32_t Height() const { return ((dw[2] >> 13) & 0x1FFF) + 1; }    // 2D
    uint32_t MaxMip() const { return (dw[4] >> 6) & 0xF; }
    uint32_t Dimension() const { return (dw[5] >> 9) & 3; }             // 0 1D, 1 2D, 2 3D, 3 cube
    uint32_t MipAddress() const { return dw[5] & 0xFFFFF000; }
};

inline TextureFetch ReadTextureFetch(const uint8_t* guest)
{
    TextureFetch f;
    for (int i = 0; i < 6; i++)
        f.dw[i] = reinterpret_cast<const be<uint32_t>*>(guest)[i];
    return f;
}

// GPU texture formats (bits per pixel from the game's table at 0x82067E48, block sizes from 0x821DF758).
constexpr uint32_t kFormat8 = 2, kFormat8888 = 6, kFormatDXT1 = 18, kFormatDXT2_3 = 19, kFormatDXT4_5 = 20;

uint32_t BitsPerPixel(uint32_t format);
uint32_t BlockWidth(uint32_t format);    // 4 for DXT-style 4x4 blocks, else 1 (2 for packed YUV, ...)
uint32_t BlockHeight(uint32_t format);

// Element index of block (x, y) in a 2D tiled surface whose pitch is `pitchBlocks` blocks of
// `bytesPerBlock` (1, 2, 4, 8 or 16) bytes. The XDK's documented XGAddress2DTiledOffset algorithm.
// Inline: called once per block of every texture upload.
inline uint32_t TiledOffset2D(uint32_t x, uint32_t y, uint32_t pitchBlocks, uint32_t bytesPerBlock)
{
    const uint32_t logBpp = std::countr_zero(bytesPerBlock);
    const uint32_t alignedWidth = (pitchBlocks + 31) & ~31u;
    const uint32_t macro = ((x >> 5) + (y >> 5) * (alignedWidth >> 5)) << (logBpp + 7);
    const uint32_t micro = ((x & 7) + ((y & 0xE) << 2)) << logBpp;
    const uint32_t offset = macro + ((micro & ~0xFu) << 1) + (micro & 0xF) + ((y & 1) << 4);
    const uint32_t address = ((offset & ~0x1FFu) << 3) + ((y & 16) << 7) + ((offset & 0x1C0) << 2) +
        (((((y & 8) >> 2) + (x >> 3)) & 3) << 6) + (offset & 0x3F);
    return address >> logBpp;
}

// In-place endian conversion of guest texture data (mode = TextureFetch::Endian()).
void SwapEndian(uint8_t* data, size_t size, uint32_t mode);

// --- Vertex fetch constant (2 dwords: device +0x6F0 + (17 - stream) * 8) ---------------------------
struct VertexFetch
{
    uint32_t dw[2];
    uint32_t Address() const { return dw[0] & 0xFFFFFFFC; }   // physical
    uint32_t Endian() const { return dw[1] & 3; }
    uint32_t SizeDwords() const { return (dw[1] >> 2) & 0xFFFFFF; }
};

// --- Primitive types (DrawVertices / DrawIndexedVertices) -------------------------------------------
enum : uint32_t
{
    kPrimPointList = 1,
    kPrimLineList = 2,
    kPrimLineStrip = 3,
    kPrimTriangleList = 4,
    kPrimTriangleFan = 5,
    kPrimTriangleStrip = 6,
    kPrimRectList = 8,
    kPrimQuadList = 13,
};

} // namespace ac6::gpu::xenos
