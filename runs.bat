@echo off
setlocal

rem jvm
set ENABLE_NATIVE=--enable-native-access=ALL-UNNAMED

rem deixar esse jar da jnn sempre primeiro
set CP_JNN_GED=bin\jnn.jar;lib\ged.jar;testes\bin
set CP_JNN_GED_GEIM=bin\jnn.jar;lib\ged.jar;lib\geim.jar;testes\bin

rem java %ENABLE_NATIVE% -cp "%CP_JNN_GED%" MainConv
java %ENABLE_NATIVE% -cp "%CP_JNN_GED_GEIM%" Conv
rem java %ENABLE_NATIVE% -cp "%CP_JNN_GED_GEIM%" Benchmark
rem java %ENABLE_NATIVE% -cp "%CP_JNN_GED%" TesteConv 1
rem java %ENABLE_NATIVE% -cp "%CP_JNN_GED_GEIM%" TesteJNI
@REM java %ENABLE_NATIVE% -cp "%CP_JNN_GED_GEIM%" exemplos.Iris