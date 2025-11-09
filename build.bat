@echo off
setlocal enabledelayedexpansion

set SRC_DIR=jnn
set BIN_DIR=bin
set JAVA_FILES=

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
echo Gerando JAR
jar cvf "%BIN_DIR%\jnn.jar" -C "%BIN_DIR%" .

echo.
echo Jar gerado

echo Build finalizado.