// Languages. The console language itself is AC6_LANGUAGE (kernel/imports_system.cpp). This is the game side of the
// text-only European languages.
#include "endian.h"
#include "game/localization.h"
#include "log.h"
#include "memory.h"
#include "ppc.h"

#include <cstdlib>
#include <string_view>

namespace {

// The European text bundles on this (North American / Japanese) disc, found by loading each: 4 English (the European
// edition's), 5 German, 6 Italian, 7 French, 8 Spanish. AC6_LANGUAGE names one of them (text only: the voices and
// cutscenes stay English, the only other packs on the disc being Japanese).
int LanguageBundle()
{
    static const std::string value = ac6::localization::GameLanguage();   // (a language pack's: its base language)
    if (value.empty())
        return -1;
    const std::string_view name(value);
    if (name == "german" || name == "de" || name == "3") return 5;
    if (name == "french" || name == "fr" || name == "4") return 7;
    if (name == "spanish" || name == "es" || name == "5") return 8;
    if (name == "italian" || name == "it" || name == "6") return 6;
    return -1;
}

} // namespace

// The locale routine (0x821BABB8, once at boot) leaves the game's language index at root + 0x24474 (root = [0x826E4E54]):
// 0 Japanese, 1 English on this disc. The rest of the game switches on 0 to 6 (jump tables at 0x82184500, 0x821857B0,
// 0x821949C8, 0x821AF5B8, 0x821AF8B0, 0x821B02C0, 0x821B79A8, 0x8214E738: the European edition's code), the text bundle
// being 2 + index: a European language sets its index (bundle - 2) after it, so everything keyed by the language follows.
extern "C" PPC_FUNC(__imp__sub_821BABB8);
PPC_FUNC(sub_821BABB8)
{
    __imp__sub_821BABB8(ctx, base);
    const int bundle = LanguageBundle();
    if (bundle < 4)
        return;
    const uint32_t root = ac6::g_memory.As<ac6::be<uint32_t>>(0x826E4E54)->get();
    if (!root)
        return;
    *ac6::g_memory.As<ac6::be<uint32_t>>(root + 0x24474) = uint32_t(bundle - 2);
    ac6::log::Info("Language index {} (text bundle {})", bundle - 2, bundle);
}

// Mid-asm hook at 0x821BB1E0 (the text bundle loader 0x821BB180, after `addi r28,r11,2`): r28 = the text bundle the
// game loads, 2 + (language index != 0): 2 Japanese, 3 English (the locale routine 0x821BABB8 only produces those two).
// A European language (AC6_LANGUAGE) loads its bundle instead, AC6_TEXT_BUNDLE=<n> loads bundle n (diagnostics).
void TextBundleOverride(PPCRegister& r28)
{
    static const int bundle = [] {
        const char* v = std::getenv("AC6_TEXT_BUNDLE");
        return v && *v ? std::atoi(v) : LanguageBundle();
    }();
    if (bundle < 0)
        return;
    static bool logged = false;
    if (!logged)
        ac6::log::Info("Text bundle {} instead of {}", bundle, r28.u32);
    logged = true;
    r28.u64 = uint64_t(bundle);
}
