# LuaJIT 2.1 (thirdparty/LuaJIT, fetched by fetch_thirdparty.cmake) built with this build's compiler, in place of its
# Makefile: the same steps as its src/msvcbuild.bat (and the Makefile's for Linux). The mods' Lua runtime
# (GracemeriaRecomp/src/mods) links it.
#
# It is a shared library next to the executable, lua51.dll (libluajit-5.1.so on Linux), so a mod's native Lua modules
# (luaopen_*, mods/<mod>/bin) link against the same VM the game runs. The JIT and the FFI are on; the jit.* Lua modules
# (jit.v, jit.p, jit.dump, ...) are copied to lua/jit next to the executable.
#
#   minilua         host Lua, runs DynASM and genversion.lua
#   buildvm_arch.h  DynASM's output for vm_x64.dasc
#   luajit.h        luajit_rolling.h with the rolling release's version (the commit's time)
#   buildvm         makes the interpreter (lj_vm.obj / lj_vm.S) and the generated headers from the library sources
set(AC6_LUAJIT "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/LuaJIT")
set(LJ_SRC "${AC6_LUAJIT}/src")
set(LJ_GEN "${CMAKE_BINARY_DIR}/luajit")
file(MAKE_DIRECTORY "${LJ_GEN}" "${LJ_GEN}/jit")

if (NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
    message(FATAL_ERROR "LuaJIT: only x86-64 is set up (${CMAKE_SYSTEM_PROCESSOR})")
endif()

set(LJ_ALL_LIB lib_base.c lib_math.c lib_bit.c lib_string.c lib_table.c lib_io.c lib_os.c lib_package.c lib_debug.c lib_jit.c
    lib_ffi.c lib_buffer.c)
set(LJ_DASM_FLAGS -D ENDIAN_LE -D P64 -D JIT -D FFI -D FPU -D HFABI "-D" "VER=")
if (WIN32)
    list(APPEND LJ_DASM_FLAGS -D WIN)
endif()

# The rolling release's version: the commit's time (as the Makefile and msvcbuild.bat take it).
execute_process(COMMAND "${GIT_EXECUTABLE}" show -s --format=%ct WORKING_DIRECTORY "${AC6_LUAJIT}" OUTPUT_VARIABLE lj_relver
    OUTPUT_STRIP_TRAILING_WHITESPACE RESULT_VARIABLE lj_relver_result)
if (NOT lj_relver_result EQUAL 0 OR lj_relver STREQUAL "")
    file(READ "${AC6_LUAJIT}/.relver" lj_relver)
endif()
file(CONFIGURE OUTPUT "${LJ_GEN}/luajit_relver.txt" CONTENT "${lj_relver}\n" @ONLY)

# --- Host tools -------------------------------------------------------------------------------------------------------
add_executable(luajit_minilua "${LJ_SRC}/host/minilua.c")
target_compile_options(luajit_minilua PRIVATE -w)
if (NOT WIN32)
    target_link_libraries(luajit_minilua PRIVATE m)
endif()

add_custom_command(OUTPUT "${LJ_GEN}/buildvm_arch.h"
    COMMAND "$<TARGET_FILE:luajit_minilua>" "${AC6_LUAJIT}/dynasm/dynasm.lua" -LN ${LJ_DASM_FLAGS} -o "${LJ_GEN}/buildvm_arch.h" "${LJ_SRC}/vm_x64.dasc"
    DEPENDS luajit_minilua "${LJ_SRC}/vm_x64.dasc" "${AC6_LUAJIT}/dynasm/dynasm.lua" "${AC6_LUAJIT}/dynasm/dasm_x86.lua"
    COMMENT "LuaJIT: DynASM (vm_x64.dasc)" VERBATIM)
add_custom_command(OUTPUT "${LJ_GEN}/luajit.h"
    COMMAND "$<TARGET_FILE:luajit_minilua>" "${LJ_SRC}/host/genversion.lua" "${LJ_SRC}/luajit_rolling.h" "${LJ_GEN}/luajit_relver.txt" "${LJ_GEN}/luajit.h"
    DEPENDS luajit_minilua "${LJ_SRC}/host/genversion.lua" "${LJ_SRC}/luajit_rolling.h" "${LJ_GEN}/luajit_relver.txt"
    COMMENT "LuaJIT: luajit.h" VERBATIM)
add_custom_target(luajit_headers DEPENDS "${LJ_GEN}/buildvm_arch.h" "${LJ_GEN}/luajit.h")

file(GLOB LJ_BUILDVM_SOURCES "${LJ_SRC}/host/buildvm*.c")
add_executable(luajit_buildvm ${LJ_BUILDVM_SOURCES})
add_dependencies(luajit_buildvm luajit_headers)
target_include_directories(luajit_buildvm PRIVATE "${LJ_GEN}" "${LJ_SRC}" "${AC6_LUAJIT}/dynasm")
target_compile_options(luajit_buildvm PRIVATE -w)

# --- Generated sources ------------------------------------------------------------------------------------------------
if (WIN32)
    set(LJ_VM "${LJ_GEN}/lj_vm.obj")
    set(LJ_VM_MODE peobj)
else()
    set(LJ_VM "${LJ_GEN}/lj_vm.S")
    set(LJ_VM_MODE elfasm)
endif()
set(LJ_GENERATED "${LJ_GEN}/lj_bcdef.h" "${LJ_GEN}/lj_ffdef.h" "${LJ_GEN}/lj_libdef.h" "${LJ_GEN}/lj_recdef.h"
    "${LJ_GEN}/lj_folddef.h" "${LJ_GEN}/jit/vmdef.lua")
set(LJ_LIB_PATHS "")
foreach (f ${LJ_ALL_LIB})
    list(APPEND LJ_LIB_PATHS "${LJ_SRC}/${f}")
endforeach()
# (Run from src/: buildvm names each library after its file, lib_<name>.c.)
add_custom_command(OUTPUT "${LJ_VM}" ${LJ_GENERATED}
    COMMAND "$<TARGET_FILE:luajit_buildvm>" -m ${LJ_VM_MODE} -o "${LJ_VM}"
    COMMAND "$<TARGET_FILE:luajit_buildvm>" -m bcdef -o "${LJ_GEN}/lj_bcdef.h" ${LJ_ALL_LIB}
    COMMAND "$<TARGET_FILE:luajit_buildvm>" -m ffdef -o "${LJ_GEN}/lj_ffdef.h" ${LJ_ALL_LIB}
    COMMAND "$<TARGET_FILE:luajit_buildvm>" -m libdef -o "${LJ_GEN}/lj_libdef.h" ${LJ_ALL_LIB}
    COMMAND "$<TARGET_FILE:luajit_buildvm>" -m recdef -o "${LJ_GEN}/lj_recdef.h" ${LJ_ALL_LIB}
    COMMAND "$<TARGET_FILE:luajit_buildvm>" -m vmdef -o "${LJ_GEN}/jit/vmdef.lua" ${LJ_ALL_LIB}
    COMMAND "$<TARGET_FILE:luajit_buildvm>" -m folddef -o "${LJ_GEN}/lj_folddef.h" lj_opt_fold.c
    WORKING_DIRECTORY "${LJ_SRC}"
    DEPENDS luajit_buildvm ${LJ_LIB_PATHS} "${LJ_SRC}/lj_opt_fold.c"
    COMMENT "LuaJIT: buildvm (interpreter and generated headers)" VERBATIM)
add_custom_target(luajit_generated DEPENDS "${LJ_VM}" ${LJ_GENERATED})

# --- The library ------------------------------------------------------------------------------------------------------
# (Every lib_*.c: the libraries buildvm reads, and lib_aux.c / lib_init.c, luaL_* and luaL_openlibs.)
file(GLOB LJ_CORE_SOURCES "${LJ_SRC}/lj_*.c" "${LJ_SRC}/lib_*.c")
add_library(luajit SHARED ${LJ_CORE_SOURCES} "${LJ_VM}")
add_dependencies(luajit luajit_generated)
if (WIN32)
    set_source_files_properties("${LJ_VM}" PROPERTIES EXTERNAL_OBJECT ON GENERATED ON)
    set_target_properties(luajit PROPERTIES OUTPUT_NAME lua51 PREFIX "" RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
    target_compile_definitions(luajit PUBLIC LUA_BUILD_AS_DLL)
    # Self-contained: the C runtime's static parts in the DLL, as the executables have them.
    target_link_options(luajit PRIVATE -static)
else()
    enable_language(ASM)
    set_source_files_properties("${LJ_VM}" PROPERTIES LANGUAGE ASM GENERATED ON)
    set_target_properties(luajit PROPERTIES OUTPUT_NAME luajit-5.1 LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
    target_compile_definitions(luajit PRIVATE LUAJIT_UNWIND_EXTERNAL _FILE_OFFSET_BITS=64 _LARGEFILE_SOURCE)
    target_link_libraries(luajit PRIVATE m dl)
endif()
target_compile_options(luajit PRIVATE -O2 -fomit-frame-pointer -fno-stack-protector -U_FORTIFY_SOURCE -w)
target_include_directories(luajit PRIVATE "${LJ_GEN}")
# Users: lua.h, lauxlib.h, lualib.h, luajit.h (generated).
target_include_directories(luajit PUBLIC "${LJ_SRC}" "${LJ_GEN}")
# (The headers aren't in src/ until generated: the library's own sources include them by name.)

# The jit.* modules (jit.v, jit.p, jit.dump, the disassemblers, vmdef.lua) next to the executable, under lua/jit.
file(GLOB LJ_JIT_LUA "${LJ_SRC}/jit/*.lua")
add_custom_command(OUTPUT "${CMAKE_BINARY_DIR}/lua/jit/vmdef.lua"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_BINARY_DIR}/lua/jit"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different ${LJ_JIT_LUA} "${LJ_GEN}/jit/vmdef.lua" "${CMAKE_BINARY_DIR}/lua/jit"
    DEPENDS ${LJ_JIT_LUA} "${LJ_GEN}/jit/vmdef.lua"
    COMMENT "LuaJIT: jit.* modules" VERBATIM)
add_custom_target(luajit_jit_modules ALL DEPENDS "${CMAKE_BINARY_DIR}/lua/jit/vmdef.lua")
