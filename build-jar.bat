@echo off
setlocal

set BIN_CLASSES=bin\classes
set OUT_JAR=bin\jnn.jar

jar cf "%OUT_JAR%" ^
 -C "%BIN_CLASSES%" . ^
 -C bin nativo

if errorlevel 1 (
    echo ERRO ao gerar JAR
    exit /b 1
)

echo Build jar OK