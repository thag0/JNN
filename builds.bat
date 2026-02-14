@echo off
setlocal

rem dir base
set OUT_DIR=testes\bin

rem class path
set CP_JNN_GED=bin\jnn.jar;lib\ged.jar
set CP_JNN_GED_GEIM=bin\jnn.jar;lib\ged.jar;lib\geim.jar

if not exist %OUT_DIR% mkdir %OUT_DIR%

@REM javac -cp "%CP_JNN_GED%" -d %OUT_DIR% MainConv.java
@REM javac -cp "%CP_JNN_GED%" -d %OUT_DIR% Benchmark.java
javac -cp "%CP_JNN_GED_GEIM%" -d %OUT_DIR% Conv.java
@REM javac -cp "%CP_JNN_GED%" -d %OUT_DIR% Playground.java
@REM javac -cp "%CP_JNN_GED%" -d %OUT_DIR% TesteConv.java
@REM javac -cp "%CP_JNN_GED_GEIM%" -d %OUT_DIR% TesteJNI.java

@rem Exemplos
@REM javac -cp "%CP_JNN_GED%" -d %OUT_DIR% exemplos\Iris.java
@REM javac -cp "%CP_JNN_GED%" -d %OUT_DIR% exemplos\ModelIO.java
@REM javac -cp "%CP_JNN_GED_GEIM%" -d %OUT_DIR% exemplos\UpscaleImg.java
@REM javac -cp "%CP_JNN_GED%" -d %OUT_DIR% exemplos\Xor.java