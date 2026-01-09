@echo off
setlocal enabledelayedexpansion

set SRC_DIR=jnn
set BIN_DIR=bin\classes
set JAVA_FILES=

if exist bin rmdir /S /Q bin
mkdir "%BIN_DIR%"

for /R "%SRC_DIR%" %%f in (*.java) do (
    set JAVA_FILES=!JAVA_FILES! "%%f"
)

javac ^
 -g ^
 -parameters ^
 -d "%BIN_DIR%" ^
 -sourcepath "%SRC_DIR%" ^
 %JAVA_FILES%

if errorlevel 1 (
    echo Erro na compilacao Java
    exit /b 1
)

echo Compilacao Java OK
