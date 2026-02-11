@echo off
setlocal

rem jvm
set ENABLE_NATIVE=--enable-native-access=ALL-UNNAMED

rem deixar esse jar da jnn sempre primeiro
set CP_JNN_GED=bin\jnn.jar;lib\ged.jar;testes\bin
set CP_JNN_GED_GEIM=bin\jnn.jar;lib\ged.jar;lib\geim.jar;testes\bin


java %ENABLE_NATIVE% -cp "%CP_JNN_GED%" MainConv
@REM java %ENABLE_NATIVE% -cp "%CP_JNN_GED_GEIM%" Conv
@REM java %ENABLE_NATIVE% -cp "%CP_JNN_GED%" Benchmark
@REM java %ENABLE_NATIVE% -cp "%CP_JNN_GED" Playground
@REM java %ENABLE_NATIVE% -cp "%CP_JNN_GED%" TesteConv 1 6
@REM java %ENABLE_NATIVE% -cp "%CP_JNN_GED_GEIM%" TesteJNI
@REM java %ENABLE_NATIVE% -cp "%CP_JNN_GED%" TesteTempos 1 6

@REM java %ENABLE_NATIVE% -cp "%CP_JNN_GED%" exemplos.Iris
@REM java %ENABLE_NATIVE% -cp "%CP_JNN_GED%" exemplos.Xor