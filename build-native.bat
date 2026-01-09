@echo off
setlocal

set JAVA_HOME=C:\Program Files\Java\jdk-22
set SRC=jnn\nativo\jnn_native.c

set OUT_DIR=bin\native\win64
set DLL=jnn_native.dll

if not exist "%OUT_DIR%" mkdir "%OUT_DIR%"

gcc ^
 -O2 ^
 -shared ^
 -march=x86-64 ^
 -mtune=generic ^
 -fno-fast-math ^
 -fopenmp ^
 -I"%JAVA_HOME%\include" ^
 -I"%JAVA_HOME%\include\win32" ^
 "%SRC%" ^
 -o "%OUT_DIR%\%DLL%"

if errorlevel 1 (
    echo ERRO na compilacao JNI
    exit /b 1
)

copy C:\mingw64\bin\libgomp-1.dll "%OUT_DIR%"
copy C:\mingw64\bin\libgcc_s_seh-1.dll "%OUT_DIR%"
copy C:\mingw64\bin\libwinpthread-1.dll "%OUT_DIR%"

echo Build nativo OK