#pragma once
// Sampling profiler for performance work (AC6_PROFILE=<report interval in seconds>, see profiler.cpp).

namespace ac6::profiler {

void Start();
// Ends sampling (at shutdown: a thread suspended by a sample in progress would block the exit).
void Stop();

} // namespace ac6::profiler
