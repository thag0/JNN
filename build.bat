@echo off
setlocal enabledelayedexpansion

set SRC_DIR=jnn
set BIN_DIR=bin
set JAVA_FILES=

if exist "%BIN_DIR%" rmdir /S /Q "%BIN_DIR%"
mkdir "%BIN_DIR%"

echo Coletando arquivos
for /R "%SRC_DIR%" %%f in (*.java) do (
    set JAVA_FILES=!JAVA_FILES! "%%f"
)

javac -g -parameters -d "%BIN_DIR%" -sourcepath "%SRC_DIR%" %JAVA_FILES%

if %errorlevel% neq 0 (
    echo Erro durante a compilação.
    exit /b %errorlevel%
)

echo.
if exist "%BIN_DIR%\jnn.jar" del "%BIN_DIR%\jnn.jar"
jar cvf "%BIN_DIR%\jnn.jar" -C "%BIN_DIR%" .

@REM echo Gerando Doc 
@REM javadoc -d "%DOC_DIR%" -sourcepath "." -subpackages jnn

echo Build finalizado.