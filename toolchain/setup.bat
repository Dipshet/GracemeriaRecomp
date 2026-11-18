@echo off
rem Downloads the build tools into this folder when they are missing: llvm-mingw (the C++ compiler:
rem clang, no Microsoft compiler), Ninja, and DXC (the shader compiler for the renderer's own shaders, Microsoft's release,
rem as UnleashedRecomp's build gets it). Windows' own curl, tar and certutil (Windows 10 and 11; by full path, so another
rem tar on the PATH, Git's, isn't used). Each download is checked against its SHA-256 before it is unpacked; one already
rem here with the right checksum isn't downloaded again.
setlocal
set "TC=%~dp0"
set "SYS=%SystemRoot%\System32"
if not exist "%TC%llvm-mingw\bin\clang.exe" (
    call :fetch "https://github.com/mstorsjo/llvm-mingw/releases/download/20261006/llvm-mingw-20261006-ucrt-x86_64.zip" llvm-mingw.zip 317492c456aa27ee607a5919f1d2d38dcdc1112516a24d0bf4b00d078f52d17a || exit /b 1
    echo Unpacking llvm-mingw
    "%SYS%\tar.exe" -xf "%TC%llvm-mingw.zip" -C "%TC%." || exit /b 1
    move "%TC%llvm-mingw-20261006-ucrt-x86_64" "%TC%llvm-mingw" >nul || exit /b 1
    del "%TC%llvm-mingw.zip"
)
if not exist "%TC%bin\ninja.exe" (
    call :fetch "https://github.com/ninja-build/ninja/releases/download/v1.13.2/ninja-win.zip" ninja-win.zip 07fc8261b42b20e71d1720b39068c2e14ffcee6396b76fb7a795fb460b78dc65 || exit /b 1
    if not exist "%TC%bin" mkdir "%TC%bin"
    "%SYS%\tar.exe" -xf "%TC%ninja-win.zip" -C "%TC%bin" || exit /b 1
    del "%TC%ninja-win.zip"
)
if not exist "%TC%dxc\bin\x64\dxc.exe" (
    call :fetch "https://github.com/microsoft/DirectXShaderCompiler/releases/download/v1.9.2609/dxc_2026_09_29.zip" dxc.zip ad31b1fc8443175d204f77a611fdb3ef2ec42759bdc2f1167368de24a4a7e7f1 || exit /b 1
    if not exist "%TC%dxc" mkdir "%TC%dxc"
    "%SYS%\tar.exe" -xf "%TC%dxc.zip" -C "%TC%dxc" || exit /b 1
    del "%TC%dxc.zip"
)
exit /b 0

rem :fetch <url> <file> <sha256>: the file in this folder, checked.
:fetch
if exist "%TC%%~2" (
    call :check "%~2" "%~3" && exit /b 0
    del "%TC%%~2"
)
echo Downloading %~2
"%SYS%\curl.exe" -L --fail --progress-bar -o "%TC%%~2" "%~1" || (echo Download failed: %~1& exit /b 1)
call :check "%~2" "%~3" && exit /b 0
echo Checksum mismatch for %~2: got %HASH%, expected %~3
del "%TC%%~2"
exit /b 1

rem :check <file> <sha256>: errorlevel 0 when the file's SHA-256 is the one given (HASH: the file's).
:check
set "HASH="
for /f "skip=1 delims=" %%h in ('%SYS%\certutil.exe -hashfile "%TC%%~1" SHA256') do if not defined HASH set "HASH=%%h"
set "HASH=%HASH: =%"
if /i "%HASH%"=="%~2" exit /b 0
exit /b 1
