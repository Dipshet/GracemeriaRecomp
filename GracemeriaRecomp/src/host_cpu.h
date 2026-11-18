#pragma once
// Processors this process may run on (its affinity mask), for sizing thread pools: on a machine with more
// processors than the process may use (or a test run restricted with `start /affinity`), pools sized from the
// machine's count oversubscribe the ones it has.

#include "platform/platform.h"

#include <cstdint>

namespace ac6 {

inline uint32_t HostProcessors() { return platform::ProcessorCount(); }

} // namespace ac6
