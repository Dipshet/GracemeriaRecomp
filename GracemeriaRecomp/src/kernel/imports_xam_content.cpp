// XAM users, profile settings, notifications, system UI, messages and content (save containers, DLC).
//
// Player 1 is a signed-in local profile from boot (AC6_PROFILE_NAME, default "Player", with a fixed
// offline XUID), so the game can save. Save containers live on one HDD device and are mounted as
// writable host folders while open (fs/save_store.h). Marketplace content (DLC) comes
// from the dlc folder and is mounted read-only (fs/dlc.h).
#include "fs/dlc.h"
#include "fs/save_store.h"
#include "fs/vfs.h"
#include "heap.h"
#include "kernel/function.h"
#include "kernel/xam.h"
#include "loader.h"
#include "memory.h"
#include "settings.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <string>
#include <unordered_set>
#include <vector>

namespace ac6::kernel {

namespace {

constexpr uint64_t kProfileXuid = 0xE0000AC600000001ull;   // offline (local) XUIDs start with 0xE
constexpr uint32_t kHddDeviceId = 1;
constexpr uint32_t XCONTENTDEVICETYPE_HDD = 1;
constexpr uint32_t XCONTENT_NEW = 1, XCONTENT_EXISTING = 2;
constexpr uint32_t XCONTENTTYPE_SAVEDATA = 1;
constexpr uint32_t XCONTENTTYPE_MARKETPLACE = 2;

// Notifications.
constexpr uint32_t XN_SYS_UI = 0x00000009, XN_SYS_SIGNINCHANGED = 0x0000000A;

// The signed-in profile's name: AC6_PROFILE_NAME cleaned to a gamertag's characters and length (ASCII letters,
// digits and spaces, 15 at most: any other character is dropped whole), else Player.
const std::string& ProfileName()
{
    static const std::string name = [] {
        const char* v = std::getenv("AC6_PROFILE_NAME");
        const std::string raw = v ? v : "";
        std::string n = settings::CleanProfileName(raw);
        if (n != raw && !raw.empty())
            log::Warn("AC6_PROFILE_NAME \"{}\": only letters, digits and spaces, 15 at most: \"{}\"", raw, n.empty() ? "Player" : n);
        return n.empty() ? std::string("Player") : n;
    }();
    return name;
}

bool SignedIn(uint32_t user) { return user == 0; }

// --- Notification listeners --------------------------------------------------------------------------

struct Listener : KernelObject
{
    XDispatcherHeader header{};
    uint64_t areas = 0;
    std::deque<std::pair<uint32_t, uint32_t>> queue;
    Listener() { InitializeHeader(header, kObjEventNotification, 0); }
    ~Listener() override;
    uint32_t DispatcherHeader() override { return g_memory.MapVirtual(&header); }
    const char* TypeName() const override { return "notification listener"; }
};

std::mutex s_listenerMutex;
std::unordered_set<Listener*> s_listeners;

Listener::~Listener()
{
    std::lock_guard lock(s_listenerMutex);
    s_listeners.erase(this);
}

// --- Content ----------------------------------------------------------------------------------------

struct XContentData
{
    be<uint32_t> deviceId;
    be<uint32_t> contentType;
    be<uint16_t> displayName[128];
    char fileName[42];
    uint8_t pad[2];
};
static_assert(sizeof(XContentData) == 0x134);

struct XDeviceData
{
    be<uint32_t> deviceId;
    be<uint32_t> deviceType;
    be<uint64_t> totalBytes;
    be<uint64_t> freeBytes;
    be<uint16_t> name[27];
    uint8_t pad[2];
};
static_assert(sizeof(XDeviceData) == 0x50);

std::string FileNameOf(const XContentData& d)
{
    return std::string(d.fileName, strnlen(d.fileName, sizeof(d.fileName)));
}

std::string DisplayNameOf(const XContentData& d)
{
    // UTF-16 -> UTF-8.
    std::string out;
    for (size_t i = 0; i < 128 && d.displayName[i]; i++)
    {
        uint32_t c = d.displayName[i];
        if (c >= 0xD800 && c < 0xDC00 && i + 1 < 128)
        {
            uint32_t low = d.displayName[i + 1];
            c = 0x10000 + ((c - 0xD800) << 10) + (low - 0xDC00);
            i++;
        }
        if (c < 0x80) out += char(c);
        else if (c < 0x800) { out += char(0xC0 | (c >> 6)); out += char(0x80 | (c & 0x3F)); }
        else if (c < 0x10000) { out += char(0xE0 | (c >> 12)); out += char(0x80 | ((c >> 6) & 0x3F)); out += char(0x80 | (c & 0x3F)); }
        else { out += char(0xF0 | (c >> 18)); out += char(0x80 | ((c >> 12) & 0x3F)); out += char(0x80 | ((c >> 6) & 0x3F)); out += char(0x80 | (c & 0x3F)); }
    }
    return out;
}

void SetDisplayName(XContentData& d, const std::string& utf8)
{
    std::memset(d.displayName, 0, sizeof(d.displayName));
    size_t o = 0;
    for (size_t i = 0; i < utf8.size() && o < 127;)
    {
        uint32_t c = uint8_t(utf8[i]);
        int extra = c >= 0xF0 ? 3 : c >= 0xE0 ? 2 : c >= 0xC0 ? 1 : 0;
        c &= extra == 3 ? 0x07 : extra == 2 ? 0x0F : extra == 1 ? 0x1F : 0x7F;
        for (int k = 1; k <= extra && i + k < utf8.size(); k++)
            c = (c << 6) | (uint8_t(utf8[i + k]) & 0x3F);
        i += extra + 1;
        if (c >= 0x10000 && o + 1 < 127)
        {
            c -= 0x10000;
            d.displayName[o++] = uint16_t(0xD800 + (c >> 10));
            d.displayName[o++] = uint16_t(0xDC00 + (c & 0x3FF));
        }
        else
        {
            d.displayName[o++] = uint16_t(c);
        }
    }
}

uint64_t OwnerOf(uint32_t contentType) { return contentType == XCONTENTTYPE_SAVEDATA ? kProfileXuid : 0; }

struct Enumerator : KernelObject
{
    std::vector<XContentData> items;   // empty for the achievement enumerator
    size_t cursor = 0;
    uint32_t perCall = 1;
    const char* TypeName() const override { return "enumerator"; }
};

std::mutex s_contentMutex;

} // namespace

void XamNotify(uint32_t id, uint32_t param)
{
    const uint32_t area = (id >> 25) & 0x3F;
    std::lock_guard lock(s_listenerMutex);
    for (Listener* l : s_listeners)
        if (l->areas & (1ull << area))
            l->queue.emplace_back(id, param);
}

// --- Users -------------------------------------------------------------------------------------------

uint32_t XamUserGetSigninState(uint32_t userIndex)
{
    return SignedIn(userIndex) ? 1 : 0;   // eXUserSigninState_SignedInLocally
}

uint32_t XamUserGetSigninInfo(uint32_t userIndex, uint32_t flags, uint8_t* info)
{
    if (info)
        std::memset(info, 0, 0x28);
    if (!SignedIn(userIndex))
        return ERROR_NO_SUCH_USER_;
    // XUSER_SIGNIN_INFO: xuid, info flags, sign-in state, guest number, sponsor, name[16].
    *reinterpret_cast<be<uint64_t>*>(info) = kProfileXuid;
    *reinterpret_cast<be<uint32_t>*>(info + 12) = 1;
    std::memcpy(info + 24, ProfileName().c_str(), ProfileName().size() + 1);
    return ERROR_SUCCESS_;
}

uint32_t XamUserGetXUID(uint32_t userIndex, uint32_t flags, be<uint64_t>* xuid)
{
    if (!SignedIn(userIndex))
    {
        if (xuid)
            *xuid = 0;
        return ERROR_NO_SUCH_USER_;
    }
    if (xuid)
        *xuid = kProfileXuid;
    return ERROR_SUCCESS_;
}

uint32_t XamUserGetName(uint32_t userIndex, char* buffer, uint32_t size)
{
    if (!buffer || !size)
        return ERROR_INVALID_PARAMETER_;
    buffer[0] = 0;
    if (!SignedIn(userIndex))
        return ERROR_NO_SUCH_USER_;
    std::strncpy(buffer, ProfileName().c_str(), size - 1);
    buffer[size - 1] = 0;
    return ERROR_SUCCESS_;
}

uint32_t XamUserCheckPrivilege(uint32_t userIndex, uint32_t privilege, be<uint32_t>* result)
{
    // An offline profile: no Live privileges.
    if (result)
        *result = 0;
    return SignedIn(userIndex & 0xFF) || (userIndex & 0xFF) == 0xFF ? ERROR_SUCCESS_ : ERROR_NO_SUCH_USER_;
}

// XUSER_READ_PROFILE_SETTING_RESULT { count, settings* } followed by XUSER_PROFILE_SETTING[count]
// (40 bytes: source, user/xuid, id, XUSER_DATA{type, value}) and any binary/string payloads. The data
// type and maximum size are encoded in the setting id (bits 28-31, 16-27).
uint32_t XamUserReadProfileSettings(uint32_t titleId, uint32_t userIndex, uint32_t xuidCount, uint32_t xuids,
    uint32_t settingCount, const be<uint32_t>* settingIds, be<uint32_t>* bufferSize, uint8_t* buffer, XXOverlapped* overlapped)
{
    constexpr uint32_t kSettingSize = 40;
    const uint32_t count = std::max<uint32_t>(xuidCount, 1) * settingCount;
    uint32_t payload = 0;
    for (uint32_t i = 0; i < settingCount; i++)
    {
        const uint32_t id = settingIds[i];
        const uint32_t type = id >> 28;
        if (type == 4 || type == 6)   // wide string / binary
            payload += (((id >> 16) & 0xFFF) + 3) & ~3u;
    }
    const uint32_t required = 8 + count * kSettingSize + payload * std::max<uint32_t>(xuidCount, 1);
    if (!buffer || !bufferSize || *bufferSize < required)
    {
        if (bufferSize)
            *bufferSize = required;
        return CompleteXam(overlapped, ERROR_INSUFFICIENT_BUFFER_);
    }

    std::memset(buffer, 0, required);
    const uint32_t guestBuffer = g_memory.MapVirtual(buffer);
    *reinterpret_cast<be<uint32_t>*>(buffer) = count;
    *reinterpret_cast<be<uint32_t>*>(buffer + 4) = guestBuffer + 8;
    uint32_t payloadAt = 8 + count * kSettingSize;
    for (uint32_t x = 0; x < std::max<uint32_t>(xuidCount, 1); x++)
    {
        for (uint32_t i = 0; i < settingCount; i++)
        {
            uint8_t* s = buffer + 8 + (x * settingCount + i) * kSettingSize;
            const uint32_t id = settingIds[i];
            const uint32_t type = id >> 28;
            *reinterpret_cast<be<uint32_t>*>(s + 16) = id;
            if (xuidCount)
                *reinterpret_cast<be<uint64_t>*>(s + 8) = *g_memory.As<be<uint64_t>>(xuids + x * 8);
            else
                *reinterpret_cast<be<uint32_t>*>(s + 8) = userIndex;
            s[24] = uint8_t(type);
            if (type == 4 || type == 6)
            {
                // No stored value: an empty buffer, "no value" source, the game uses its defaults.
                *reinterpret_cast<be<uint32_t>*>(s) = 0;   // XSOURCE_NO_VALUE
                *reinterpret_cast<be<uint32_t>*>(s + 32) = 0;
                *reinterpret_cast<be<uint32_t>*>(s + 36) = guestBuffer + payloadAt;
                payloadAt += (((id >> 16) & 0xFFF) + 3) & ~3u;
            }
            else
            {
                *reinterpret_cast<be<uint32_t>*>(s) = 1;   // XSOURCE_DEFAULT
                int32_t value = 0;
                if (id == 0x10040003)   // XPROFILE_OPTION_CONTROLLER_VIBRATION: on
                    value = 3;
                *reinterpret_cast<be<int32_t>*>(s + 32) = value;
            }
        }
    }
    return CompleteXam(overlapped, ERROR_SUCCESS_, required);
}

// --- Notifications, messages, system UI ---------------------------------------------------------------

uint32_t XamNotifyCreateListener(uint64_t areas, uint32_t version)
{
    auto* l = CreateKernelObject<Listener>();
    l->areas = areas ? areas : 0xFFFFFFFFull;
    {
        std::lock_guard lock(s_listenerMutex);
        s_listeners.insert(l);
        // Tell the game who is signed in, as the console does for a new listener.
        if (l->areas & 1)
            l->queue.emplace_back(XN_SYS_SIGNINCHANGED, 1);
    }
    return CreateHandle(l);
}

uint32_t XNotifyGetNext(uint32_t handle, uint32_t matchId, be<uint32_t>* id, be<uint32_t>* param)
{
    auto* l = LookupHandleAs<Listener>(handle);
    if (!l)
        return 0;
    std::lock_guard lock(s_listenerMutex);
    for (auto it = l->queue.begin(); it != l->queue.end(); ++it)
    {
        if (matchId && it->first != matchId)
            continue;
        if (id)
            *id = it->first;
        if (param)
            *param = it->second;
        l->queue.erase(it);
        return 1;
    }
    return 0;
}

void XNotifyPositionUI(uint32_t position) {}

uint32_t XamShowSigninUI(uint32_t panes, uint32_t flags)
{
    // Player 1 is already signed in. Behave like a sign-in UI that opened and closed.
    XamNotify(XN_SYS_UI, 1);
    XamNotify(XN_SYS_SIGNINCHANGED, 1);
    XamNotify(XN_SYS_UI, 0);
    return ERROR_SUCCESS_;
}

uint32_t XamShowDeviceSelectorUI(uint32_t userIndex, uint32_t contentType, uint32_t contentFlags, uint64_t bytesRequested,
    be<uint32_t>* deviceId, XXOverlapped* overlapped)
{
    XamNotify(XN_SYS_UI, 1);
    if (deviceId)
        *deviceId = kHddDeviceId;
    XamNotify(XN_SYS_UI, 0);
    return CompleteXam(overlapped, ERROR_SUCCESS_);
}

uint32_t XamShowMessageBoxUIEx(uint32_t userIndex, const be<uint16_t>* title, const be<uint16_t>* text, uint32_t buttonCount,
    uint32_t buttons, uint32_t focusButton, uint32_t flags, be<uint32_t>* result, XXOverlapped* overlapped)
{
    std::string message;
    for (size_t i = 0; text && text[i] && i < 512; i++)
        message += text[i] < 0x80 ? char(uint16_t(text[i])) : '?';
    log::Warn("System message box (answered with button {}): {}", focusButton, message);
    if (result)
        *result = focusButton;
    XamNotify(XN_SYS_UI, 1);
    XamNotify(XN_SYS_UI, 0);
    return CompleteXam(overlapped, ERROR_SUCCESS_);
}

// XMsg requests to system apps: 0xFB = XGI (rich presence, contexts, properties), 0xFA = music player.
// Nothing to do offline, they succeed.
// The return value is an HRESULT (the XAPI wrappers fail on negative values). Completion is reported
// through the overlapped block.
uint32_t XMsgStartIORequest(uint32_t app, uint32_t message, XXOverlapped* overlapped, uint32_t buffer, uint32_t size)
{
    CompleteXam(overlapped, ERROR_SUCCESS_);
    return 0;
}

uint32_t XMsgStartIORequestEx(uint32_t app, uint32_t message, XXOverlapped* overlapped, uint32_t buffer, uint32_t size, uint32_t extra)
{
    CompleteXam(overlapped, ERROR_SUCCESS_);
    return 0;
}

uint32_t XMsgInProcessCall(uint32_t app, uint32_t message, uint32_t param1, uint32_t param2) { return ERROR_SUCCESS_; }

uint32_t XMsgCancelIORequest(XXOverlapped* overlapped, uint32_t wait) { return 0; }

uint32_t XamGetExecutionId(be<uint32_t>* out)
{
    // XEX_EXECUTION_ID from the XEX header (optional header 0x40006), copied once into guest memory.
    static uint32_t guest = [] {
        const auto& h = g_module.xexHeader;
        if (h.size() < 0x18)
            return 0u;
        auto u32 = [&](size_t o) { return o + 4 <= h.size() ? uint32_t(*reinterpret_cast<const be<uint32_t>*>(h.data() + o)) : 0u; };
        const uint32_t count = u32(0x14);
        for (uint32_t i = 0; i < count; i++)
        {
            if (u32(0x18 + i * 8) == 0x00040006)
            {
                const uint32_t offset = u32(0x18 + i * 8 + 4);
                void* p = g_heap.AllocPhysical(0x18, 16);
                std::memcpy(p, h.data() + offset, 0x18);
                return g_memory.MapVirtual(p);
            }
        }
        return 0u;
    }();
    if (!guest)
        return X_STATUS_UNSUCCESSFUL;
    *out = guest;
    return X_STATUS_SUCCESS;
}

// --- Content: devices -----------------------------------------------------------------------------------

uint32_t XamContentGetDeviceState(uint32_t deviceId, XXOverlapped* overlapped)
{
    return CompleteXam(overlapped, deviceId == kHddDeviceId ? ERROR_SUCCESS_ : ERROR_DEVICE_NOT_CONNECTED_);
}

uint32_t XamContentGetDeviceData(uint32_t deviceId, XDeviceData* data)
{
    if (deviceId != kHddDeviceId)
        return ERROR_DEVICE_NOT_CONNECTED_;
    std::memset(data, 0, sizeof(*data));
    data->deviceId = kHddDeviceId;
    data->deviceType = XCONTENTDEVICETYPE_HDD;
    data->totalBytes = 20ull << 30;
    data->freeBytes = 10ull << 30;
    // The save drive's real free space, capped at the 10 GB set above (so the game's own "not enough space" check
    // sees a nearly full drive).
    std::error_code ec;
    if (const std::filesystem::space_info space = std::filesystem::space(fs::SaveRoot(), ec); !ec)
        data->freeBytes = std::min<uint64_t>(10ull << 30, space.available);
    const char* name = "Hard Drive";
    for (size_t i = 0; name[i]; i++)
        data->name[i] = uint16_t(name[i]);
    return ERROR_SUCCESS_;
}

// --- Content: containers ----------------------------------------------------------------------------------

uint32_t XamContentCreateEx(uint32_t userIndex, const char* rootName, const XContentData* content, uint32_t flags,
    be<uint32_t>* disposition, be<uint32_t>* licenseMask, uint32_t fileCacheSize, uint64_t contentSize, XXOverlapped* overlapped)
{
    if (!rootName || !content)
        return CompleteXam(overlapped, ERROR_INVALID_PARAMETER_);
    const uint32_t type = content->contentType;
    const std::string fileName = FileNameOf(*content);
    if (type == XCONTENTTYPE_MARKETPLACE)
    {
        // DLC: read-only, never created. Every licence bit is held (the game has no licence check of its
        // own, partially licensed packs would hide some of their colours).
        const fs::DlcPackage* p = fs::FindDlc(fileName);
        uint32_t result = ERROR_SUCCESS_;
        if (!p)
            result = ERROR_PATH_NOT_FOUND_;
        else if (!p->Usable())
            result = ERROR_FILE_CORRUPT_;
        else if ((flags & 0xF) != 3 && (flags & 0xF) != 4)
            result = ERROR_ACCESS_DENIED_;
        if (result == ERROR_SUCCESS_)
        {
            if (disposition)
                *disposition = XCONTENT_EXISTING;
            if (licenseMask)
                *licenseMask = 0xFFFFFFFF;
            fs::Mount(rootName, p->device);
            log::Info("XamContentCreateEx('{}', DLC '{}'{}{}, flags {:#x}): mounted {} ({} files, licence mask at {:08X})", rootName, fileName,
                p->label.empty() ? "" : " ", p->label, flags, reinterpret_cast<const char*>(p->path.u8string().c_str()), p->files,
                licenseMask ? g_memory.MapVirtual(licenseMask) : 0);
        }
        else
            log::Error("XamContentCreateEx('{}', DLC '{}', flags {:#x}) -> {}: {}", rootName, fileName, flags, result,
                !p ? "no such package in the dlc folder"
                   : !p->Usable() ? "not mounted (" + (p->errors.empty() ? std::string("see the DLC report") : p->errors.front()) + ")"
                                  : "DLC is read-only");
        return CompleteXam(overlapped, result);
    }
    const auto path = fs::ContentPath(OwnerOf(type), type, fileName);

    std::lock_guard lock(s_contentMutex);
    std::error_code ec;
    const bool exists = std::filesystem::is_directory(path, ec);
    uint32_t result = ERROR_SUCCESS_;
    bool create = false;
    switch (flags & 0xF)
    {
    case 1:   // CREATE_NEW
        if (exists)
            result = ERROR_ALREADY_EXISTS_;
        create = true;
        break;
    case 2:   // CREATE_ALWAYS
        create = true;
        break;
    case 3:   // OPEN_EXISTING
        if (!exists)
            result = ERROR_PATH_NOT_FOUND_;
        break;
    case 4:   // OPEN_ALWAYS
        create = true;
        break;
    case 5:   // TRUNCATE_EXISTING: the old contents go to the trash
        if (!exists)
            result = ERROR_PATH_NOT_FOUND_;
        else if (!fs::TrashContent(path))
            result = ERROR_ACCESS_DENIED_;   // the old contents couldn't go: not truncated
        else
            create = true;
        break;
    default:
        result = ERROR_INVALID_PARAMETER_;
        break;
    }
    if (result == ERROR_SUCCESS_)
    {
        const bool isNew = !std::filesystem::is_directory(path, ec);
        if (isNew && (!std::filesystem::create_directories(path, ec) || ec || !fs::WriteContentMeta(path, DisplayNameOf(*content))))
        {
            // Failing the create lets the game report the failed save itself, rather than every save of the session
            // failing unseen.
            log::Error("XamContentCreateEx('{}'): can't create the save folder {} ({}): is the game's folder writable?", rootName,
                reinterpret_cast<const char*>(path.u8string().c_str()), ec ? ec.message() : std::string("the .meta file"));
            return CompleteXam(overlapped, ERROR_ACCESS_DENIED_);
        }
        if (disposition)
            *disposition = isNew ? XCONTENT_NEW : XCONTENT_EXISTING;
        if (licenseMask)
            *licenseMask = 0;
        fs::Mount(rootName, std::make_shared<fs::HostFolderDevice>(path));
    }
    log::Info("XamContentCreateEx('{}', type {}, '{}', flags {:#x}) -> {} ({})", rootName, type, fileName, flags, result,
        reinterpret_cast<const char*>(path.u8string().c_str()));
    (void)create;
    return CompleteXam(overlapped, result);
}

uint32_t XamContentClose(const char* rootName, XXOverlapped* overlapped)
{
    // Files commit as their handles close. Closing the container unmounts it and tidies its staging folder.
    if (rootName)
    {
        const fs::Resolution r = fs::Resolve(std::string(rootName) + ":\\");
        if (auto* folder = dynamic_cast<fs::HostFolderDevice*>(r.device.get()))
            fs::CleanStaging(folder->rootPath);
        fs::Unmount(rootName);
    }
    return CompleteXam(overlapped, ERROR_SUCCESS_);
}

uint32_t XamContentDelete(uint32_t userIndex, const XContentData* content, XXOverlapped* overlapped)
{
    if (!content)
        return CompleteXam(overlapped, ERROR_INVALID_PARAMETER_);
    if (content->contentType == XCONTENTTYPE_MARKETPLACE)
    {
        log::Warn("XamContentDelete(DLC '{}'): DLC isn't deleted by the runtime", FileNameOf(*content));
        return CompleteXam(overlapped, ERROR_ACCESS_DENIED_);
    }
    std::lock_guard lock(s_contentMutex);
    const auto path = fs::ContentPath(OwnerOf(content->contentType), content->contentType, FileNameOf(*content));
    std::error_code ec;
    const bool existed = std::filesystem::exists(path, ec);
    const bool ok = fs::TrashContent(path);
    return CompleteXam(overlapped, ok ? ERROR_SUCCESS_ : existed ? ERROR_ACCESS_DENIED_ : ERROR_FILE_NOT_FOUND_);
}

uint32_t XamContentSetThumbnail(uint32_t userIndex, const XContentData* content, const uint8_t* image, uint32_t size, XXOverlapped* overlapped)
{
    if (!content)
        return CompleteXam(overlapped, ERROR_INVALID_PARAMETER_);
    if (content->contentType == XCONTENTTYPE_MARKETPLACE)
        return CompleteXam(overlapped, ERROR_ACCESS_DENIED_);
    const auto path = fs::ContentPath(OwnerOf(content->contentType), content->contentType, FileNameOf(*content));
    if (image && size)
        fs::WriteThumbnail(path, image, size);
    return CompleteXam(overlapped, ERROR_SUCCESS_);
}

uint32_t XamContentCreateEnumerator(uint32_t userIndex, uint32_t deviceId, uint32_t contentType, uint32_t flags,
    uint32_t itemsPerCall, be<uint32_t>* bufferSize, be<uint32_t>* handle)
{
    auto* e = CreateKernelObject<Enumerator>();
    if (!e)
        return ERROR_INVALID_PARAMETER_;
    e->perCall = std::max<uint32_t>(itemsPerCall, 1);
    if ((deviceId == 0 || deviceId == kHddDeviceId) && contentType == XCONTENTTYPE_MARKETPLACE)
    {
        for (const fs::DlcPackage& p : fs::DlcPackages())
        {
            if (!p.Usable())
                continue;   // listed in the DLC report with the reason
            XContentData d{};
            d.deviceId = kHddDeviceId;
            d.contentType = contentType;
            std::strncpy(d.fileName, p.name.c_str(), sizeof(d.fileName));
            SetDisplayName(d, !p.displayName.empty() ? p.displayName : !p.label.empty() ? p.label : p.name);
            e->items.push_back(d);
        }
    }
    else if (deviceId == 0 || deviceId == kHddDeviceId)
    {
        for (auto& c : fs::ListContent(OwnerOf(contentType), contentType))
        {
            XContentData d{};
            d.deviceId = kHddDeviceId;
            d.contentType = contentType;
            std::strncpy(d.fileName, c.fileName.c_str(), sizeof(d.fileName));
            SetDisplayName(d, c.displayName);
            e->items.push_back(d);
        }
    }
    log::Info("XamContentCreateEnumerator(type {}, device {}): {} item(s)", contentType, deviceId, e->items.size());
    if (bufferSize)
        *bufferSize = uint32_t(sizeof(XContentData) * e->perCall);
    if (handle)
        *handle = CreateHandle(e);
    return ERROR_SUCCESS_;
}

// Achievements aren't tracked: the enumerator is valid but empty. Each XACHIEVEMENT_DETAILS is 0x24
// bytes plus its three strings (label 33, description 101, unachieved text 101 WCHARs).
uint32_t XamUserCreateAchievementEnumerator(uint32_t titleId, uint32_t userIndex, uint64_t xuid, uint32_t detailFlags,
    uint32_t startingIndex, uint32_t count, be<uint32_t>* bufferSize, be<uint32_t>* handle)
{
    auto* e = CreateKernelObject<Enumerator>();
    if (!e)
        return ERROR_INVALID_PARAMETER_;
    e->perCall = std::max<uint32_t>(count, 1);
    if (bufferSize)
        *bufferSize = e->perCall * (0x24 + (33 + 101 + 101) * 2);
    if (handle)
        *handle = CreateHandle(e);
    return ERROR_SUCCESS_;
}

uint32_t XamEnumerate(uint32_t handle, uint32_t flags, uint8_t* buffer, uint32_t bufferSize, be<uint32_t>* itemsReturned, XXOverlapped* overlapped)
{
    auto* e = LookupHandleAs<Enumerator>(handle);
    if (!e)
        return CompleteXam(overlapped, ERROR_INVALID_PARAMETER_);
    const size_t fit = std::min<size_t>(e->perCall, bufferSize / sizeof(XContentData));
    size_t n = 0;
    while (n < fit && e->cursor < e->items.size())
        std::memcpy(buffer + n++ * sizeof(XContentData), &e->items[e->cursor++], sizeof(XContentData));
    if (itemsReturned)
        *itemsReturned = uint32_t(n);
    return CompleteXam(overlapped, n ? ERROR_SUCCESS_ : ERROR_NO_MORE_FILES_, uint32_t(n));
}

} // namespace ac6::kernel

AC6_IMPORT(XamUserGetSigninState)
AC6_IMPORT(XamUserGetSigninInfo)
AC6_IMPORT(XamUserGetXUID)
AC6_IMPORT(XamUserGetName)
AC6_IMPORT(XamUserCheckPrivilege)
AC6_IMPORT(XamUserReadProfileSettings)
AC6_IMPORT(XamNotifyCreateListener)
AC6_IMPORT(XNotifyGetNext)
AC6_IMPORT(XNotifyPositionUI)
AC6_IMPORT(XamShowSigninUI)
AC6_IMPORT(XamShowDeviceSelectorUI)
AC6_IMPORT(XamShowMessageBoxUIEx)
AC6_IMPORT(XMsgStartIORequest)
AC6_IMPORT(XMsgStartIORequestEx)
AC6_IMPORT(XMsgInProcessCall)
AC6_IMPORT(XMsgCancelIORequest)
AC6_IMPORT(XamGetExecutionId)
AC6_IMPORT(XamContentGetDeviceState)
AC6_IMPORT(XamContentGetDeviceData)
AC6_IMPORT(XamContentCreateEx)
AC6_IMPORT(XamContentClose)
AC6_IMPORT(XamContentDelete)
AC6_IMPORT(XamContentSetThumbnail)
AC6_IMPORT(XamContentCreateEnumerator)
AC6_IMPORT(XamEnumerate)
AC6_IMPORT(XamUserCreateAchievementEnumerator)
