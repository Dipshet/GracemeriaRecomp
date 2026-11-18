@echo off
rem Builds the x86-64-v2 compatibility version (SSE4.2, POPCNT) into build-compat:
rem ac6_compat.exe and AceCombat6_compat.exe. Next to the main executables, they start them on CPUs without AVX2.
rem Windows only: Linux builds need AVX2.
call "%~dp0toolchain\env.bat" || exit /b 1
cd /d %~dp0
rem AC6_RELEASE=ON: a release build, which updates itself from the GitHub releases (off unless set).
if not defined AC6_RELEASE set AC6_RELEASE=OFF
cmake -S . -B build-compat -G Ninja -DCMAKE_BUILD_TYPE=Release -DAC6_X86_LEVEL=v2 "-DCMAKE_TOOLCHAIN_FILE=%AC6_CMAKE_TOOLCHAIN%" "-DAC6_RELEASE=%AC6_RELEASE%" || exit /b 1
cmake --build build-compat --target ac6 ac6_release %* || exit /b 1
