#pragma once
// Languages: see game/language.cpp.

#include <cstdint>
#include <string>

namespace ac6::game {

// Force Japanese Audio (AC6_FORCE_JAPANESE_AUDIO): Japanese voices whatever the game's own voice and subtitle option.
bool ForceJapaneseAudio();

// A disc read's offset: with Force Japanese Audio, the English cutscene movie in moviepack.bin is instead read from the
// Japanese version. Anything else as it is.
uint64_t MovieReadOffset(const std::string& devicePath, uint64_t fileSize, uint64_t offset);

} // namespace ac6::game
