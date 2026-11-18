@echo off
rem Builds the game. Needs Git and CMake. Downloads the compiler, Ninja and DXC into toolchain\
rem (toolchain\setup.bat). CMake then fetches the third-party code, recompiles the game's code from default.xex and
rem builds build\AceCombat6.exe. Arguments go to the build (e.g. -j 4).
setlocal
cd /d "%~dp0"
where git >nul 2>nul || (echo Git is needed: https://git-scm.com/& exit /b 1)
where cmake >nul 2>nul || (echo CMake 3.20 or newer is needed: https://cmake.org/& exit /b 1)
if not exist default.xex if not exist GracemeriaRecompLib\ppc\ppc_recomp_shared.h (
    echo Put the game's default.xex, from your disc image, in %~dp0
    exit /b 1
)
call toolchain\setup.bat || exit /b 1
rem SDL and Dear ImGui are git submodules (a clone without --recursive hasn't checked them out yet).
if exist .gitmodules git submodule update --init || exit /b 1
rem CMake and Git where they are installed (toolchain\env.bat keeps the PATH to the project's own tools otherwise).
for /f "delims=" %%p in ('where cmake') do (set "AC6_EXTRA_PATH=%%~dpp" & goto :cmake_found)
:cmake_found
for /f "delims=" %%p in ('where git') do (set "AC6_EXTRA_PATH=%AC6_EXTRA_PATH%;%%~dpp" & goto :git_found)
:git_found
call toolchain\env.bat || exit /b 1
rem AC6_RELEASE=ON: a release build, which updates itself from the GitHub releases (off unless set).
if not defined AC6_RELEASE set AC6_RELEASE=OFF
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release "-DCMAKE_TOOLCHAIN_FILE=%AC6_CMAKE_TOOLCHAIN%" "-DAC6_RELEASE=%AC6_RELEASE%" || exit /b 1
cmake --build build %* || exit /b 1
