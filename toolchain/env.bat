@echo off
rem Puts the project's non-Microsoft toolchain first on PATH: llvm-mingw, standalone Ninja, CMake.
rem Never call vcvars*.bat in this project.
set "AC6_TOOLCHAIN=%~dp0"
rem AC6_EXTRA_PATH: CMake's and Git's folders where build.bat found them.
set "PATH=%~dp0llvm-mingw\bin;%~dp0bin;%AC6_EXTRA_PATH%;C:\Program Files\CMake\bin;%SystemRoot%\system32;%SystemRoot%"
set "AC6_CMAKE_TOOLCHAIN=%~dp0llvm-mingw.cmake"
