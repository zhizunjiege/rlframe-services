@REM Build the DLL for the simulation termination function.

@REM Turn off echo
@echo off

@REM Set the VS and Windows version
set VS_VER=2019
set MSVC_VER=14.29.30133
set WIN_VER=10
set KITS_VER=10.0.19041.0

@REM Set the environment variables
set PATH=C:\Program Files (x86)\Microsoft Visual Studio\%VS_VER%\Community\VC\Tools\MSVC\%MSVC_VER%\bin\Hostx64\x64;%PATH%
set INCLUDE=C:\Program Files (x86)\Microsoft Visual Studio\%VS_VER%\Community\VC\Tools\MSVC\%MSVC_VER%\include;^
C:\Program Files (x86)\Windows Kits\%WIN_VER%\Include\%KITS_VER%\shared;^
C:\Program Files (x86)\Windows Kits\%WIN_VER%\Include\%KITS_VER%\ucrt;^
C:\Program Files (x86)\Windows Kits\%WIN_VER%\Include\%KITS_VER%\um;
set LIB=C:\Program Files (x86)\Microsoft Visual Studio\%VS_VER%\Community\VC\Tools\MSVC\%MSVC_VER%\lib\x64;^
C:\Program Files (x86)\Windows Kits\%WIN_VER%\Lib\%KITS_VER%\ucrt\x64;^
C:\Program Files (x86)\Windows Kits\%WIN_VER%\Lib\%KITS_VER%\um\x64;

@REM Call cl.exe
cl /LD /Fe:./sim_term_func.dll /std:c++17 sim_term_func.cc
