@echo off

@REM utf-8
chcp 65001 > nul 

setlocal enabledelayedexpansion

set SRC_DIR=jnn
set BIN_DIR=bin\classes
set JNI_HEADERS=jnn/nativo
set TMP_FILE=%TEMP%\java_sources.txt

if exist bin rmdir /S /Q bin
mkdir "%BIN_DIR%"

if exist "%TMP_FILE%" del "%TMP_FILE%"

for /R "%SRC_DIR%" %%f in (*.java) do (
    set "FILE=%%f"
    set "FILE=!FILE:\=/!"
    echo "!FILE!" >> "%TMP_FILE%"
)

javac ^
 -Xdiags:verbose ^
 -g ^
 -parameters ^
 -h "%JNI_HEADERS%" ^
 -d "%BIN_DIR%" ^
 -sourcepath "%SRC_DIR%" ^
 @"%TMP_FILE%"

if errorlevel 1 (
    echo Erro na compilacao Java
    exit /b 1
)

del "%TMP_FILE%"

echo Build java OK
