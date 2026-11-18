# CMake toolchain for Windows builds without any Microsoft toolchain:
# llvm-mingw (clang + lld + libc++ + mingw-w64 headers/UCRT import libs), standalone Ninja.
# Usage: cmake -G Ninja -DCMAKE_TOOLCHAIN_FILE=<repo>/toolchain/llvm-mingw.cmake ...

set(AC6_TOOLCHAIN_DIR "${CMAKE_CURRENT_LIST_DIR}")
set(AC6_LLVM_MINGW "${AC6_TOOLCHAIN_DIR}/llvm-mingw")

set(CMAKE_SYSTEM_NAME Windows)
set(CMAKE_SYSTEM_PROCESSOR x86_64)

set(CMAKE_C_COMPILER   "${AC6_LLVM_MINGW}/bin/x86_64-w64-mingw32-clang.exe")
set(CMAKE_CXX_COMPILER "${AC6_LLVM_MINGW}/bin/x86_64-w64-mingw32-clang++.exe")
set(CMAKE_RC_COMPILER  "${AC6_LLVM_MINGW}/bin/x86_64-w64-mingw32-windres.exe")
set(CMAKE_AR           "${AC6_LLVM_MINGW}/bin/llvm-ar.exe")
set(CMAKE_RANLIB       "${AC6_LLVM_MINGW}/bin/llvm-ranlib.exe")
set(CMAKE_MAKE_PROGRAM "${AC6_TOOLCHAIN_DIR}/bin/ninja.exe" CACHE FILEPATH "Ninja" FORCE)

set(CMAKE_FIND_ROOT_PATH "${AC6_LLVM_MINGW}/x86_64-w64-mingw32")
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE BOTH)

# Self-contained executables: no libc++/libunwind DLLs from the toolchain needed at runtime.
set(CMAKE_EXE_LINKER_FLAGS_INIT "-static")
