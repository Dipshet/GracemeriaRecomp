// System/configuration imports: console settings, privileges, versions, title notifications.
#include "crash.h"
#include "game/localization.h"
#include "kernel/function.h"
#include "log.h"
#include "memory.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <format>
#include <string_view>

namespace ac6::kernel {

namespace {

// XCONFIG categories/settings the game asks for.
constexpr uint16_t kConfigSecured = 0x0002;
constexpr uint16_t kConfigUser = 0x0003;

constexpr uint32_t kLanguageEnglish = 1, kLanguageJapanese = 2;

// The console language: AC6_LANGUAGE=english (default) | japanese | german | french | spanish | italian (or a
// two-letter code) | <XLanguage number>, or the base language of a language pack. The game localises off
// XGetLanguage (text, and the voice / cutscene packs: voicepack_jpn.bin, demopack_jpn.bin), but its locale routine
// (sub_821BABB8) honours the language only when the region is one that sells Japanese ({0x101, 0x102, 0x1FC,
// 0x1FF}). So Japanese also reports this disc's real region, NTSC-U + NTSC-J (0x1FF). The region stays NTSC-U
// otherwise (an NTSC-J region switches region-keyed content on its own). German, French, Spanish and Italian (3 to 6)
// are text only, the disc's European text bundles (game/language.cpp): the game itself only knows English and Japanese
// (its locale index), so their voices and cutscenes are the English ones.
uint32_t Language()
{
    static const uint32_t language = [] {
        // (A language pack's value gives the language it is translated from: game/localization.h.)
        const std::string value = localization::GameLanguage();
        const char* v = value.c_str();
        uint32_t l = kLanguageEnglish;
        if (v && *v)
        {
            const std::string_view name(v);
            if (name == "japanese" || name == "ja" || name == "jp")
                l = kLanguageJapanese;
            else if (name == "english" || name == "en")
                l = kLanguageEnglish;
            else if (name == "german" || name == "de")
                l = 3;
            else if (name == "french" || name == "fr")
                l = 4;
            else if (name == "spanish" || name == "es")
                l = 5;
            else if (name == "italian" || name == "it")
                l = 6;
            else if (const int n = std::atoi(v); n > 0 && n < 32)
                l = uint32_t(n);
            else
                log::Warn("AC6_LANGUAGE={}: expected english, japanese, german, french, spanish, italian, a language number or a "
                          "language pack in the localization folder", v);
        }
        if (l != kLanguageEnglish)
            log::Info("Console language {}{}", l, l == kLanguageJapanese ? " (Japanese; region NTSC-U + NTSC-J)" : "");
        return l;
    }();
    return language;
}

} // namespace

uint32_t ExGetXConfigSetting(uint16_t category, uint16_t setting, uint8_t* buffer, uint16_t bufferSize, be<uint32_t>* requiredSize)
{
    be<uint32_t> value = 0;
    bool known = true;
    if (category == kConfigSecured)
    {
        switch (setting)
        {
        case 0x0002: value = 0x00001000; break;   // AV region: NTSC/USA
        default: known = false; break;
        }
    }
    else if (category == kConfigUser)
    {
        switch (setting)
        {
        case 0x0001: case 0x0002: case 0x0003: case 0x0004:
        case 0x0005: case 0x0006: case 0x0007:          // time zone: UTC
            value = 0;
            break;
        case 0x0009: value = Language(); break;          // language (AC6_LANGUAGE)
        case 0x000A: value = 0x00040000; break;          // video flags: widescreen
        case 0x000C: value = 1; break;                   // retail flags
        case 0x000E: value = 103; break;                 // country: United States
        default: known = false; break;
        }
    }
    else
    {
        known = false;
    }

    if (!known)
    {
        log::Warn("ExGetXConfigSetting: unknown setting {}/{}", category, setting);
        return X_STATUS_INVALID_PARAMETER;
    }
    // Settings smaller than 4 bytes (the country, 1 byte) take the value's low bytes, which are the last bytes of
    // the big-endian word.
    const size_t size = setting == 0x000E && category == kConfigUser ? 1 : sizeof(value);
    if (requiredSize)
        *requiredSize = uint32_t(size);
    if (buffer)
    {
        const size_t n = std::min<size_t>(bufferSize, size);
        std::memcpy(buffer, reinterpret_cast<const uint8_t*>(&value) + (sizeof(value) - n), n);
    }
    return X_STATUS_SUCCESS;
}

uint32_t XexCheckExecutablePrivilege(uint32_t privilege)
{
    return 0;
}

uint32_t XamGetSystemVersion()
{
    // 0 selects the oldest-dashboard code paths, the conservative choice.
    return 0;
}

uint32_t XGetAVPack()
{
    return 6;   // HDMI
}

uint32_t XGetGameRegion()
{
    return Language() == kLanguageJapanese ? 0x01FF : 0x00FF;   // NTSC-U/C, plus NTSC-J Japan for Japanese
}

uint32_t XGetLanguage()
{
    return Language();
}

void ExRegisterTitleTerminateNotification(uint32_t registration, uint32_t create)
{
    // The title never gets a terminate notification: the process just exits.
}

void KiApcNormalRoutineNop() {}

// The game's way out when its data can't be read: the console shows the disc-error screen, then goes back to the
// dashboard, and neither returns. Here the game stops with a message.
void XamShowDirtyDiscErrorUI(uint32_t userIndex)
{
    (void)userIndex;
    log::Error("The game reported that its data can't be read (its disc-error screen)");
    crash::Shutdown(1, "the game's files can't be read: the game image is damaged or incomplete");
}

// Leaving the game, for another title or the dashboard: the game ends, as it does on the console.
void XamLoaderLaunchTitle(uint32_t name, uint32_t flags)
{
    (void)name;
    (void)flags;
    crash::Shutdown(0, "the game left to the dashboard (XamLoaderLaunchTitle)");
}

void XamLoaderTerminateTitle()
{
    crash::Shutdown(0, "the game ended itself (XamLoaderTerminateTitle)");
}

void HalReturnToFirmware(uint32_t routine)
{
    crash::Shutdown(0, std::format("the game returned to the firmware ({}): HalReturnToFirmware", routine).c_str());
}

} // namespace ac6::kernel

AC6_IMPORT(ExGetXConfigSetting)
AC6_IMPORT(XexCheckExecutablePrivilege)
AC6_IMPORT(XamGetSystemVersion)
AC6_IMPORT(XGetAVPack)
AC6_IMPORT(XGetGameRegion)
AC6_IMPORT(XGetLanguage)
AC6_IMPORT(ExRegisterTitleTerminateNotification)
AC6_IMPORT(KiApcNormalRoutineNop)
AC6_IMPORT(XamShowDirtyDiscErrorUI)
AC6_IMPORT(XamLoaderLaunchTitle)
AC6_IMPORT(XamLoaderTerminateTitle)
AC6_IMPORT(HalReturnToFirmware)
