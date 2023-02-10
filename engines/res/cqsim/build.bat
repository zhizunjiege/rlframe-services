@REM Build the DLL for the simulation termination function.

@REM Turn off echo
@echo off

@REM Set the path to the Visual Studio 2019 Community Edition
set PATH=%PATH%;C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.29.30133\bin\Hostx64\x64
set INCLUDE=C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.29.30133\include;^
C:\Program Files (x86)\Windows Kits\10\Include\10.0.19041.0\shared;^
C:\Program Files (x86)\Windows Kits\10\Include\10.0.19041.0\ucrt;^
C:\Program Files (x86)\Windows Kits\10\Include\10.0.19041.0\um;
set LIB=C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.29.30133\lib\x64;^
C:\Program Files (x86)\Windows Kits\10\Lib\10.0.19041.0\ucrt\x64;^
C:\Program Files (x86)\Windows Kits\10\Lib\10.0.19041.0\um\x64;

@REM Call cl.exe
cl /LD /Fe:./sim_term_func.dll /std:c++17 sim_term_func.cc
