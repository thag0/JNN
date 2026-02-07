@echo off

@REM utf-8
chcp 65001 > nul

setlocal enabledelayedexpansion

set SRC_DIR=jnn
set DOC_DIR=bin\javadoc
set TMP_FILE=%TEMP%\java_sources.txt

if exist "%DOC_DIR%" rmdir /S /Q "%DOC_DIR%"
mkdir "%DOC_DIR%"

if exist "%TMP_FILE%" del "%TMP_FILE%"

for /R "%SRC_DIR%" %%f in (*.java) do (
    set "FILE=%%f"
    set "FILE=!FILE:\=/!"
    echo "!FILE!" >> "%TMP_FILE%"
)

javadoc ^
 -encoding UTF-8 ^
 -charset UTF-8 ^
 -docencoding UTF-8 ^
 -d "%DOC_DIR%" ^
 -windowtitle "JNN - Java Neural Network Library" ^
 -doctitle "JNN<br>Java Neural Network Library" ^
 -bottom "2026 Thiago Barroso" ^
 -sourcepath "%SRC_DIR%" ^
 @"%TMP_FILE%"

del "%TMP_FILE%"

echo Javadoc gerado com sucesso em %DOC_DIR%
