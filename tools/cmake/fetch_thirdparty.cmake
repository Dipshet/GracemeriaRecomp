# The third-party code that isn't files of the repository, at the commits this port was made with: the FidelityFX SDK
# (cloned here, with the port's change from tools/patches applied), and SDL and Dear ImGui (git submodules, cloned here
# only when they weren't checked out, e.g. from a ZIP download of the repository). A library already there is left as
# it is. A patch already applied is skipped. Included by CMakeLists.txt at configure time: Git is all it needs.
find_package(Git REQUIRED)
# Git for Windows' mingw64/bin/git.exe runs its shell-script commands (git submodule) only with Git's own folders on the
# PATH. Its cmd/git.exe sets them up.
if (GIT_EXECUTABLE MATCHES "/mingw64/bin/git\\.exe$")
    string(REGEX REPLACE "/mingw64/bin/git\\.exe$" "/cmd/git.exe" AC6_GIT_WRAPPER "${GIT_EXECUTABLE}")
    if (EXISTS "${AC6_GIT_WRAPPER}")
        set(GIT_EXECUTABLE "${AC6_GIT_WRAPPER}")
    endif()
endif()

function(ac6_git dir)
    execute_process(COMMAND "${GIT_EXECUTABLE}" ${ARGN} WORKING_DIRECTORY "${dir}" RESULT_VARIABLE result)
    if (NOT result EQUAL 0)
        message(FATAL_ERROR "git ${ARGN} failed in ${dir}")
    endif()
endfunction()

# ac6_fetch(<folder> <url> <commit> [TAG <tag>: a shallow clone of the tag, checked to be <commit>]
#           [SUBMODULES <path>... | SUBMODULES_RECURSIVE] [PATCH <file>])
function(ac6_fetch dir url commit)
    cmake_parse_arguments(F "SUBMODULES_RECURSIVE" "TAG;PATCH" "SUBMODULES" ${ARGN})
    get_filename_component(name "${dir}" NAME)
    get_filename_component(parent "${dir}" DIRECTORY)
    if (NOT EXISTS "${dir}/.git")
        message(STATUS "Fetching ${name} (${url})")
        file(MAKE_DIRECTORY "${parent}")
        if (F_TAG)
            ac6_git("${parent}" clone --depth 1 --branch "${F_TAG}" "${url}" "${dir}")
            execute_process(COMMAND "${GIT_EXECUTABLE}" rev-parse HEAD WORKING_DIRECTORY "${dir}" OUTPUT_VARIABLE head
                OUTPUT_STRIP_TRAILING_WHITESPACE)
            if (NOT head STREQUAL commit)
                message(FATAL_ERROR "${name}: tag ${F_TAG} is ${head}, not ${commit}")
            endif()
        else()
            ac6_git("${parent}" clone --no-checkout "${url}" "${dir}")
            ac6_git("${dir}" -c advice.detachedHead=false checkout --detach "${commit}")
        endif()
    endif()
    # The submodules, until a fetch got them all (.git/ac6_fetched: a fetch cut short goes on where it stopped).
    if ((F_SUBMODULES OR F_SUBMODULES_RECURSIVE) AND IS_DIRECTORY "${dir}/.git" AND NOT EXISTS "${dir}/.git/ac6_fetched")
        if (F_SUBMODULES_RECURSIVE)
            ac6_git("${dir}" submodule update --init --recursive)
        elseif (F_SUBMODULES)
            ac6_git("${dir}" submodule update --init -- ${F_SUBMODULES})
        endif()
        file(WRITE "${dir}/.git/ac6_fetched" "")
    endif()
    # The patch only on a checkout at the pinned commit: one elsewhere (a developer's branch with the changes committed,
    # for example) is left as it is.
    if (F_PATCH)
        execute_process(COMMAND "${GIT_EXECUTABLE}" rev-parse HEAD WORKING_DIRECTORY "${dir}" OUTPUT_VARIABLE head
            OUTPUT_STRIP_TRAILING_WHITESPACE)
        if (NOT head STREQUAL commit)
            set(F_PATCH "")
        endif()
    endif()
    if (F_PATCH)
        execute_process(COMMAND "${GIT_EXECUTABLE}" apply --reverse --check "${F_PATCH}" WORKING_DIRECTORY "${dir}"
            RESULT_VARIABLE applied OUTPUT_QUIET ERROR_QUIET)
        if (NOT applied EQUAL 0)
            execute_process(COMMAND "${GIT_EXECUTABLE}" apply --check "${F_PATCH}" WORKING_DIRECTORY "${dir}"
                RESULT_VARIABLE applies OUTPUT_QUIET ERROR_QUIET)
            if (NOT applies EQUAL 0)
                message(FATAL_ERROR "${name}: ${F_PATCH} does not apply (a changed checkout? delete ${dir} and configure again)")
            endif()
            message(STATUS "Applying ${F_PATCH}")
            ac6_git("${dir}" apply "${F_PATCH}")
        endif()
    endif()
endfunction()

set(AC6_TP "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty")
ac6_fetch("${AC6_TP}/FidelityFX-SDK" https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK.git
    c6efa6bf7f2027b3ec94f28578bb5965eabb9e55 TAG v1.1.4 PATCH "${CMAKE_CURRENT_SOURCE_DIR}/tools/patches/fidelityfx-ac6.patch")
ac6_fetch("${AC6_TP}/SDL" https://github.com/libsdl-org/SDL.git 829a65d769d935c4852f8159e964312c0957260a TAG release-3.4.18)
ac6_fetch("${AC6_TP}/imgui" https://github.com/ocornut/imgui.git c1aaac65e896a45f81f47a4d68f9959e0f86c859 TAG v1.93.0)
