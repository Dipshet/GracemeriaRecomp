// Video-driver (Vd*) imports. The console GPU doesn't exist here: queries describe a 1280x720 60 Hz
// HD display (the fixed guest resolution), engine/ring-buffer/EDRAM setup calls are
// accepted and ignored (the native renderer hooks the game's D3D calls, and nothing reads the GPU command
// stream), and the interrupt callback is driven by the VBlank source (gpu/vblank.h).
#include "gpu/vblank.h"
#include "kernel/function.h"
#include "log.h"
#include "memory.h"

#include <cstring>

namespace ac6::kernel {

namespace {

struct XVideoMode
{
    be<uint32_t> displayWidth;
    be<uint32_t> displayHeight;
    be<uint32_t> isInterlaced;
    be<uint32_t> isWidescreen;
    be<uint32_t> isHighDefinition;
    be<float> refreshRate;
    be<uint32_t> videoStandard;
    be<uint32_t> unknown4A;
    be<uint32_t> unknown01;
    be<uint32_t> reserved[3];
};
static_assert(sizeof(XVideoMode) == 0x30);

} // namespace

void VdQueryVideoMode(XVideoMode* mode)
{
    std::memset(mode, 0, sizeof(*mode));
    mode->displayWidth = 1280;
    mode->displayHeight = 720;
    mode->isInterlaced = 0;
    mode->isWidescreen = 1;
    mode->isHighDefinition = 1;
    mode->refreshRate = 60.0f;
    mode->videoStandard = 1;   // NTSC-M
    mode->unknown4A = 0x4A;
    mode->unknown01 = 0x01;
}

void XGetVideoMode(XVideoMode* mode) { VdQueryVideoMode(mode); }

uint32_t VdQueryVideoFlags()
{
    return 0x3;   // widescreen | high definition, matching VdQueryVideoMode
}

void VdSetGraphicsInterruptCallback(uint32_t callback, uint32_t data)
{
    gpu::SetInterruptCallback(callback, data);
}

uint32_t VdInitializeEngines(uint32_t unknown, uint32_t callback, uint32_t data, uint32_t pfpCode, uint32_t meCode)
{
    return 1;
}

void VdShutdownEngines() {}
uint32_t VdIsHSIOTrainingSucceeded() { return 1; }
uint32_t VdRetrainEDRAM(uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t) { return 0; }
void VdRetrainEDRAMWorker(uint32_t) {}
void VdEnableDisableClockGating(uint32_t) {}
void VdInitializeRingBuffer(uint32_t, uint32_t) {}
void VdEnableRingBufferRPtrWriteBack(uint32_t, uint32_t) {}
void VdSetSystemCommandBufferGpuIdentifierAddress(uint32_t) {}
void VdInitializeScalerCommandBuffer() {}
void VdSetDisplayMode(uint32_t) {}
void VdCallGraphicsNotificationRoutines(uint32_t, uint32_t, uint32_t) {}
void VdGetCurrentDisplayGamma(be<uint32_t>* type, be<float>* power)
{
    // The console's HDTV output: BT.709 (D3D corrects the game's gamma ramp for it).
    // The power is only read for type 3.
    if (type)
        *type = 2;
    if (power)
        *power = 2.22222233f;
}

uint32_t VdPersistDisplay(uint32_t, be<uint32_t>* result)
{
    if (result)
        *result = 0;
    return 0;
}

// Presentation happens in the D3D device's swap path, which the native renderer hooks (gpu/d3d.cpp).
// The console's swap command is not needed.
void VdSwap() {}

// The system command buffer belongs to the console's GPU front end. Nothing consumes it here.
void VdGetSystemCommandBuffer(be<uint32_t>* buffer, be<uint32_t>* size)
{
    if (buffer)
        *buffer = 0;
    if (size)
        *size = 0;
}

// Display information: the front buffer and display of VdQueryVideoMode's mode. D3DDevice::Swap stores the
// display width, height and actual width (+0x48, +0x4A, +0x56) in the device when it re-sets the display
// mode, and its scaler setup divides by them: left unfilled, they hold whatever was on the stack, and a zero
// there makes the division trap.
struct XDisplayInfo
{
    be<uint16_t> frontBufferWidth;        // 0x00
    be<uint16_t> frontBufferHeight;       // 0x02
    uint8_t frontBufferColorFormat;       // 0x04
    uint8_t frontBufferPixelFormat;       // 0x05
    uint8_t scalerParameters[0x38];       // 0x08
    be<uint16_t> overscanLeft;            // 0x40
    be<uint16_t> overscanTop;             // 0x42
    be<uint16_t> overscanRight;           // 0x44
    be<uint16_t> overscanBottom;          // 0x46
    be<uint16_t> displayWidth;            // 0x48
    be<uint16_t> displayHeight;           // 0x4A
    be<float> displayRefreshRate;         // 0x4C
    be<uint32_t> displayInterlaced;       // 0x50
    uint8_t displayColorFormat;           // 0x54
    be<uint16_t> actualDisplayWidth;      // 0x56
};
static_assert(sizeof(XDisplayInfo) == 0x58);

void VdGetCurrentDisplayInformation(XDisplayInfo* info)
{
    XVideoMode mode;
    VdQueryVideoMode(&mode);
    std::memset(info, 0, sizeof(*info));
    info->frontBufferWidth = uint16_t(uint32_t(mode.displayWidth));
    info->frontBufferHeight = uint16_t(uint32_t(mode.displayHeight));
    info->displayWidth = uint16_t(uint32_t(mode.displayWidth));
    info->displayHeight = uint16_t(uint32_t(mode.displayHeight));
    info->displayRefreshRate = float(mode.refreshRate);
    info->actualDisplayWidth = uint16_t(uint32_t(mode.displayWidth));
}

} // namespace ac6::kernel

AC6_IMPORT(VdQueryVideoMode)
AC6_IMPORT(XGetVideoMode)
AC6_IMPORT(VdQueryVideoFlags)
AC6_IMPORT(VdSetGraphicsInterruptCallback)
AC6_IMPORT(VdInitializeEngines)
AC6_IMPORT(VdShutdownEngines)
AC6_IMPORT(VdIsHSIOTrainingSucceeded)
AC6_IMPORT(VdRetrainEDRAM)
AC6_IMPORT(VdRetrainEDRAMWorker)
AC6_IMPORT(VdEnableDisableClockGating)
AC6_IMPORT(VdInitializeRingBuffer)
AC6_IMPORT(VdEnableRingBufferRPtrWriteBack)
AC6_IMPORT(VdSetSystemCommandBufferGpuIdentifierAddress)
AC6_IMPORT(VdInitializeScalerCommandBuffer)
AC6_IMPORT(VdSetDisplayMode)
AC6_IMPORT(VdCallGraphicsNotificationRoutines)
AC6_IMPORT(VdGetCurrentDisplayGamma)
AC6_IMPORT(VdPersistDisplay)
AC6_IMPORT(VdSwap)
AC6_IMPORT(VdGetSystemCommandBuffer)
AC6_IMPORT(VdGetCurrentDisplayInformation)
