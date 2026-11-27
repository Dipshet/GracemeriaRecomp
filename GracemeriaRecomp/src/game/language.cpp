// Languages. The console language itself is AC6_LANGUAGE (kernel/imports_system.cpp). This is the game side of the
// text-only European languages.
#include "endian.h"
#include "game/language.h"
#include "fs/vfs.h"
#include "game/localization.h"
#include "settings.h"
#include "log.h"
#include "memory.h"
#include "ppc.h"

#include <cstdlib>
#include <string>
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

// The game's language index for a European text bundle: bundle - 2, 2 English, 3 German, 4 Italian, 5 French, 6 Spanish.
// The European edition's locale routine gives these (German 3, French 5, Italian 4, Spanish 6) and loads bundle
// index + 2. The cutscene subtitles (packages 466 + index) and the rest of the index's tables follow it.
int LanguageIndex(int bundle)
{
    return bundle >= 4 && bundle <= 8 ? bundle - 2 : -1;
}

int GameLanguageIndex()
{
    const uint32_t root = ac6::g_memory.As<ac6::be<uint32_t>>(0x826E4E54)->get();
    return root ? int(ac6::g_memory.As<ac6::be<uint32_t>>(root + 0x24474)->get()) : -1;
}

} // namespace

// The locale routine (0x821BABB8, once at boot) leaves the game's language index at root + 0x24474 (root = [0x826E4E54]):
// 0 Japanese, 1 English on this disc. The rest of the game switches on 0 to 6 (jump tables at 0x82184500, 0x821857B0,
// 0x821949C8, 0x821AF5B8, 0x821AF8B0, 0x821B02C0, 0x821B79A8, 0x8214E738: the European edition's code): a European
// language sets its index (LanguageIndex) after it, so everything keyed by the language follows. Its text bundle is
// loaded by TextBundleOverride below.
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
    const int index = LanguageIndex(bundle);
    if (index < 0)
        return;
    *ac6::g_memory.As<ac6::be<uint32_t>>(root + 0x24474) = uint32_t(index);
    ac6::log::Info("Language index {} (text bundle {})", index, bundle);
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

// Mid-asm hooks after the index is read (r11) by three switches that take a package item per language
// (0x8214E908, 0x821AF6E4, 0x821AFB14: the hangar's aircraft descriptions among them). Their items are in EFIGS order,
// 3 French, 4 Italian, 5 German, while this executable picks item = index, so German got French's and French German's.
// The European edition's executable swaps 3 and 5 there: so does this.
// Also after the HUD's set-up (0x822353D8) reads it (r30 at 0x822354D0) to take its labels' block from the core
// package's label table (package 1, item 51: a block of 195 labels per language). This disc's blocks are in EFIGS
// order too (VITESSE in block 3, GESCHW. in 5); the European disc reordered them to the index instead.
void LanguageItemSwap(PPCRegister& r)
{
    if (r.u32 == 3)
        r.u64 = 5;
    else if (r.u32 == 5)
        r.u64 = 3;
}

namespace {

// A European language's own package of a group in the European edition's order: English, then German, French,
// Italian, Spanish (the order its executable picks them in).
uint32_t EuropeanPackage(uint32_t english)
{
    switch (GameLanguageIndex())
    {
    case 2: return english;
    case 3: return english + 1;   // German
    case 5: return english + 2;   // French
    case 4: return english + 3;   // Italian
    case 6: return english + 4;   // Spanish
    default: return 0;
    }
}

} // namespace

// Mid-asm hooks where the boot logo's (0x821B5520: 170 Japanese, 171 English) and the title screen's (0x821B56E4:
// 177, 178) package has been chosen (r4). This executable has only those two each; a European language gets its own,
// as the European edition's executable picks them: boot logo 172-176, title screen 179-183.
void BootLogoPackage(PPCRegister& r4)
{
    if (const uint32_t package = EuropeanPackage(172))
        r4.u64 = package;
}

void TitleScreenPackage(PPCRegister& r4)
{
    if (const uint32_t package = EuropeanPackage(179))
        r4.u64 = package;
}

// Force Japanese Audio (AC6_FORCE_JAPANESE_AUDIO): the voice is Japanese regardless what the game's option says. The game's
// sound object loads its voice pack and its cutscene pack together by one flag, 0 Japanese, else English: at boot
// (CNuSound vtable slot 1, 0x82121308, the flag in r6) and again whenever they're switched (slot 92, 0x82124968, the flag
// in r4: at a mission's start, always English outside Japan, and by the Japanese edition's Language settings menu). The
// subtitles follow the game's own voice and subtitle option (the save's, read by 0x82273B70), which outside Japan is
// always a subtitled one in the game's language.
// The mission's voice script goes with the voices: its mission package holds one per voice pack (item 11 Japanese, 12
// English: the clips its lines play, numbered differently in the two packs), and the mission manager's set-up (its
// option switch, 0x82098510 and its twins in the other two managers) takes 12 for the English voices. With the English
// script on the Japanese pack the lines played the wrong clips, cut off: VoiceScriptItem takes 11 there.
// The cutscenes' subtitles are timed to the voices: game/captions.cpp re-times them to the Japanese ones.
bool ac6::game::ForceJapaneseAudio()
{
    static const bool forced = [] {
        const std::string v = ac6::settings::Get("AC6_FORCE_JAPANESE_AUDIO");
        return v == "1" || v == "true";
    }();
    return forced;
}

namespace {

bool JapaneseVoices()
{
    return ac6::game::ForceJapaneseAudio();
}

} // namespace

void VoiceScriptItem(PPCRegister& r4)
{
    if (JapaneseVoices() && r4.u32 == 12)
    {
        r4.u64 = 11;
        static bool logged = false;
        if (!logged)
        {
            logged = true;
            ac6::log::Info("Force Japanese Audio: the mission's Japanese voice script (item 11)");
        }
    }
}

extern "C" PPC_FUNC(__imp__sub_82121308);
PPC_FUNC(sub_82121308)
{
    if (JapaneseVoices())
    {
        ctx.r6.u64 = 0;
        ac6::log::Info("Force Japanese Audio: the Japanese voice and cutscene packs");
    }
    __imp__sub_82121308(ctx, base);
}

extern "C" PPC_FUNC(__imp__sub_82124968);
PPC_FUNC(sub_82124968)
{
    if (JapaneseVoices())
        ctx.r4.u64 = 0;
    __imp__sub_82124968(ctx, base);
}

// The movies (moviepack.bin: ASF-like files back to back, the North American disc's four at 0, 164657152, 329302016
// and 512849920): the title movie's Japanese and English versions, then the opening story's (the movie in scene 2,
// "Realization", after its in-engine part). The game opens one by its language. The story movie's two are the same
// picture with its staff credits, no text drawn in (the subtitles are the game's, packages 466-472), the sound
// Japanese or English: with Force Japanese Audio the English one is read from the Japanese one's place. (The title
// movies differ: the Japanese one has its subtitles drawn in and is partly another edit, so it is left alone.)
uint64_t ac6::game::MovieReadOffset(const std::string& devicePath, uint64_t fileSize, uint64_t offset)
{
    constexpr uint64_t kPackSize = 698417152, kJapanese = 329302016, kEnglish = 512849920;
    if (offset < kEnglish || fileSize != kPackSize || !ForceJapaneseAudio() || devicePath.size() != 13 ||
        ac6::fs::ToLower(devicePath) != "moviepack.bin")
        return offset;
    static bool logged = false;
    if (!logged)
    {
        logged = true;
        ac6::log::Info("Force Japanese Audio: the story movie's Japanese version");
    }
    return offset - kEnglish + kJapanese;
}

// The game's voice-language query (0x820F7DA0: *r4 = 0 for Japanese voices, only in the Japanese edition with a
// Japanese-voice option, else 1). The briefing times its lines by it (a Japanese and an English schedule for its own
// clips, which follow the voice packs): with Force Japanese Audio it answers Japanese.
extern "C" PPC_FUNC(__imp__sub_820F7DA0);
PPC_FUNC(sub_820F7DA0)
{
    const uint32_t out = ctx.r4.u32;
    __imp__sub_820F7DA0(ctx, base);
    if (JapaneseVoices() && out)
        *ac6::g_memory.As<ac6::be<uint32_t>>(out) = 0u;
}
