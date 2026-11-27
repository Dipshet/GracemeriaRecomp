#pragma once
// Cutscene subtitles (captions) re-timed for Force Japanese Audio.
//
// A cutscene's subtitles are a caption group: an FHM of 7 slots, one per language index (0 Japanese, 1 English, 2 the
// European edition's English, 3 German, 4 Italian, 5 French, 6 Spanish), in the cutscene's package (item 0/2) or the
// mission's (items 18 to 21, each <item>/2). Packages 466 to 472 hold one slot each, 466 + the language index. A slot
// is an FHM: 0 "CAPT", 1 per-line flags, 2 an FHM of the lines: 1 the line records, 3 their times (32-byte records,
// record 0 empty, the start and end in seconds as big-endian floats at +4 and +8), 5 the text as glyph numbers, 6 and
// 7 the font. The times go with the text: the English slot is timed to the English voices. The Japanese slot holds its
// lines twice, timed to the Japanese voices and, a fixed offset later (the float at 0x82069DA4), to the English ones
// for the Japanese edition's "English voices, Japanese subtitles" option.
//
// With Force Japanese Audio the voices are Japanese but the subtitles stay the game language's, timed to the English
// voices: every other slot's lines are re-timed here from the Japanese lines they overlap, as the game unpacks the
// package (same sizes, the times written in place).

#include <cstddef>
#include <cstdint>

namespace ac6::game::captions {

// A package the game finished unpacking: `data` its bytes in guest memory (an FHM).
void OnUnpacked(uint32_t index, uint8_t* data, size_t size);

} // namespace ac6::game::captions
