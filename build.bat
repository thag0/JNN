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

echo. Gerando Jar
jar cvf "%BIN_DIR%\jnn.jar" -C "%BIN_DIR%" .

@REM echo Gerando Doc 
@REM set DOC_DIR=bin/doc
@REM if exist "%DOC_DIR%" rmdir /S /Q "%DOC_DIR%"
@REM mkdir "%DOC_DIR%"
@REM javadoc -d "%DOC_DIR%" %JAVA_FILES%


echo Build finalizado.