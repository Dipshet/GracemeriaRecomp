// XAM input (controllers from input/input.h, SDL3) and network imports. The network is down: the
// console is offline. Users, profile, notifications and content are in imports_xam_content.cpp.
#include "input/input.h"
#include "kernel/xam.h"
#include "kernel/function.h"
#include "kernel/object.h"
#include "log.h"
#include "memory.h"

#include <cstring>

namespace ac6::kernel {

namespace {

constexpr int32_t kSocketError = -1;
constexpr int32_t WSAENETDOWN_ = 10050;
thread_local int32_t t_wsaLastError = 0;

} // namespace

// --- Input -------------------------------------------------------------------------------------------

namespace {

struct XInputGamepad
{
    be<uint16_t> buttons;
    uint8_t leftTrigger, rightTrigger;
    be<int16_t> thumbLX, thumbLY, thumbRX, thumbRY;
};
static_assert(sizeof(XInputGamepad) == 12);

struct XInputState
{
    be<uint32_t> packetNumber;
    XInputGamepad gamepad;
};
static_assert(sizeof(XInputState) == 16);

struct XInputVibration
{
    be<uint16_t> leftMotorSpeed, rightMotorSpeed;
};

struct XInputCapabilities
{
    uint8_t type, subType;
    be<uint16_t> flags;
    XInputGamepad gamepad;
    XInputVibration vibration;
};
static_assert(sizeof(XInputCapabilities) == 20);

constexpr uint32_t ERROR_EMPTY_ = 4306;

} // namespace

uint32_t XamInputGetCapabilities(uint32_t userIndex, uint32_t flags, XInputCapabilities* caps)
{
    if (!input::GetState(userIndex & 0xFF).connected)
        return ERROR_DEVICE_NOT_CONNECTED_;
    if (caps)
    {
        std::memset(caps, 0, sizeof(*caps));
        caps->type = 1;      // XINPUT_DEVTYPE_GAMEPAD
        caps->subType = 1;   // XINPUT_DEVSUBTYPE_GAMEPAD
        caps->gamepad.buttons = 0xF7FF;
        caps->gamepad.leftTrigger = caps->gamepad.rightTrigger = 0xFF;
        caps->gamepad.thumbLX = caps->gamepad.thumbLY = caps->gamepad.thumbRX = caps->gamepad.thumbRY = int16_t(0x7FC0);
        caps->vibration.leftMotorSpeed = caps->vibration.rightMotorSpeed = 0xFFFF;
    }
    return 0;
}

uint32_t XamInputGetState(uint32_t userIndex, uint32_t flags, XInputState* state)
{
    const input::PadState s = input::GetState(userIndex & 0xFF);
    if (!s.connected)
        return ERROR_DEVICE_NOT_CONNECTED_;
    if (state)
    {
        state->packetNumber = s.packet;
        state->gamepad.buttons = s.buttons;
        state->gamepad.leftTrigger = s.leftTrigger;
        state->gamepad.rightTrigger = s.rightTrigger;
        state->gamepad.thumbLX = s.thumbLX;
        state->gamepad.thumbLY = s.thumbLY;
        state->gamepad.thumbRX = s.thumbRX;
        state->gamepad.thumbRY = s.thumbRY;
    }
    return 0;
}

uint32_t XamInputSetState(uint32_t userIndex, uint32_t flags, const XInputVibration* vibration)
{
    if (!input::GetState(userIndex & 0xFF).connected)
        return ERROR_DEVICE_NOT_CONNECTED_;
    if (vibration)
        input::SetVibration(userIndex & 0xFF, vibration->leftMotorSpeed, vibration->rightMotorSpeed);
    return 0;
}

uint32_t XamInputGetKeystrokeEx(be<uint32_t>* userIndex, uint32_t flags, uint8_t* keystroke)
{
    // Keystroke (text/virtual key) events aren't generated. Report "no keystroke pending".
    return input::GetState(0).connected ? ERROR_EMPTY_ : ERROR_DEVICE_NOT_CONNECTED_;
}

// --- Network: the console is offline -------------------------------------------------------------------

uint32_t NetDll_XNetStartup(uint32_t caller, uint32_t params) { return 0; }
uint32_t NetDll_XNetCleanup(uint32_t caller) { return 0; }
uint32_t NetDll_WSAStartup(uint32_t caller, uint32_t version, uint8_t* data)
{
    if (data)
    {
        std::memset(data, 0, 0x190);
        *reinterpret_cast<be<uint16_t>*>(data) = static_cast<uint16_t>(version);       // wVersion
        *reinterpret_cast<be<uint16_t>*>(data + 2) = static_cast<uint16_t>(version);   // wHighVersion
    }
    return 0;
}
uint32_t NetDll_WSACleanup(uint32_t caller) { return 0; }
int32_t NetDll_WSAGetLastError() { return t_wsaLastError; }

int32_t NetDll_socket(uint32_t caller, uint32_t af, uint32_t type, uint32_t protocol)
{
    t_wsaLastError = WSAENETDOWN_;
    return kSocketError;   // INVALID_SOCKET
}

#define AC6_NET_FAIL(name, ...)                         \
    int32_t name(__VA_ARGS__)                           \
    {                                                   \
        t_wsaLastError = WSAENETDOWN_;                  \
        return kSocketError;                            \
    }
AC6_NET_FAIL(NetDll_closesocket, uint32_t caller, uint32_t s)
AC6_NET_FAIL(NetDll_shutdown, uint32_t caller, uint32_t s, uint32_t how)
AC6_NET_FAIL(NetDll_ioctlsocket, uint32_t caller, uint32_t s, uint32_t cmd, uint32_t arg)
AC6_NET_FAIL(NetDll_setsockopt, uint32_t caller, uint32_t s, uint32_t level, uint32_t name, uint32_t value, uint32_t length)
AC6_NET_FAIL(NetDll_getsockopt, uint32_t caller, uint32_t s, uint32_t level, uint32_t name, uint32_t value, uint32_t length)
AC6_NET_FAIL(NetDll_getsockname, uint32_t caller, uint32_t s, uint32_t address, uint32_t length)
AC6_NET_FAIL(NetDll_bind, uint32_t caller, uint32_t s, uint32_t address, uint32_t length)
AC6_NET_FAIL(NetDll_connect, uint32_t caller, uint32_t s, uint32_t address, uint32_t length)
AC6_NET_FAIL(NetDll_listen, uint32_t caller, uint32_t s, uint32_t backlog)
AC6_NET_FAIL(NetDll_accept, uint32_t caller, uint32_t s, uint32_t address, uint32_t length)
AC6_NET_FAIL(NetDll_select, uint32_t caller, uint32_t count, uint32_t read, uint32_t write, uint32_t except, uint32_t timeout)
AC6_NET_FAIL(NetDll_recv, uint32_t caller, uint32_t s, uint32_t buffer, uint32_t length, uint32_t flags)
AC6_NET_FAIL(NetDll_recvfrom, uint32_t caller, uint32_t s, uint32_t buffer, uint32_t length, uint32_t flags, uint32_t from, uint32_t fromLength)
AC6_NET_FAIL(NetDll_send, uint32_t caller, uint32_t s, uint32_t buffer, uint32_t length, uint32_t flags)
AC6_NET_FAIL(NetDll_sendto, uint32_t caller, uint32_t s, uint32_t buffer, uint32_t length, uint32_t flags, uint32_t to, uint32_t toLength)
#undef AC6_NET_FAIL

uint32_t NetDll___WSAFDIsSet(uint32_t s, uint32_t set) { return 0; }

uint32_t NetDll_XNetGetTitleXnAddr(uint32_t caller, uint8_t* address)
{
    if (address)
        std::memset(address, 0, 0x24);
    return 0x1;   // XNET_GET_XNADDR_NONE
}

} // namespace ac6::kernel

AC6_IMPORT(XamInputGetCapabilities)
AC6_IMPORT(XamInputGetState)
AC6_IMPORT(XamInputSetState)
AC6_IMPORT(XamInputGetKeystrokeEx)
AC6_IMPORT(NetDll_XNetStartup)
AC6_IMPORT(NetDll_XNetCleanup)
AC6_IMPORT(NetDll_WSAStartup)
AC6_IMPORT(NetDll_WSACleanup)
AC6_IMPORT(NetDll_WSAGetLastError)
AC6_IMPORT(NetDll_socket)
AC6_IMPORT(NetDll_closesocket)
AC6_IMPORT(NetDll_shutdown)
AC6_IMPORT(NetDll_ioctlsocket)
AC6_IMPORT(NetDll_setsockopt)
AC6_IMPORT(NetDll_getsockopt)
AC6_IMPORT(NetDll_getsockname)
AC6_IMPORT(NetDll_bind)
AC6_IMPORT(NetDll_connect)
AC6_IMPORT(NetDll_listen)
AC6_IMPORT(NetDll_accept)
AC6_IMPORT(NetDll_select)
AC6_IMPORT(NetDll_recv)
AC6_IMPORT(NetDll_recvfrom)
AC6_IMPORT(NetDll_send)
AC6_IMPORT(NetDll_sendto)
AC6_IMPORT(NetDll___WSAFDIsSet)
AC6_IMPORT(NetDll_XNetGetTitleXnAddr)
