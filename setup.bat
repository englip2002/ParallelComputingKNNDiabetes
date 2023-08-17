@echo off

set "projectDir=E:\DSPC_Assignment\Assignment"
set "vcpkgDir=%projectDir%\vcpkg"

REM Clone vcpkg repository if not already done
if not exist "%vcpkgDir%" (
    git clone https://github.com/microsoft/vcpkg.git "%vcpkgDir%"
)

REM Update and bootstrap vcpkg
cd /d "%vcpkgDir%"
.\bootstrap-vcpkg.bat

REM Install dependencies using vcpkg
.\vcpkg.exe install --triplet x86-windows

REM Compile and run your C++ program
.\vcpkg.exe integrate install
.\vcpkg.exe integrate project --out=%projectDir%\build --triplet x86-windows

REM Navigate to the build directory
cd /d "%projectDir%\build"

REM Run your compiled executable
your-app.exe  REM Replace "your-app.exe" with the actual executable name
