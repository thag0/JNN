@echo off
setlocal enabledelayedexpansion

set SRC_DIR=jnn
set DOC_DIR=bin/doc

@rem removendo resíduo de arquivos
if exist "%DOC_DIR%" rmdir /S /Q "%DOC_DIR%"
mkdir "%DOC_DIR%"

echo Coletando arquivos
set JAVA_FILES=
for /R "%SRC_DIR%" %%f in (*.java) do (
    set JAVA_FILES=!JAVA_FILES! "%%f"
)

echo Gerando Javadoc
javadoc -d "%DOC_DIR%" -sourcepath "%SRC_DIR%" %JAVA_FILES%

if %errorlevel% neq 0 (
    echo Erro durante a geração do Javadoc.
    exit /b %errorlevel%
)

echo.
echo Javadoc gerado em "%DOC_DIR%".