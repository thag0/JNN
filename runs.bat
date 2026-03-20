@echo off
setlocal

rem jvm
set ENABLE_NATIVE=--enable-native-access=ALL-UNNAMED
set MEM_OPS=-Xmx6g -Xms6g

rem deixar esse jar da jnn sempre primeiro
set CP_JNN_GED=bin\jnn.jar;lib\ged.jar;testes\bin
set CP_JNN_GED_GEIM=bin\jnn.jar;lib\ged.jar;lib\geim.jar;testes\bin
set CP_JNN_GED_GEIM_VIEW=bin\jnn.jar;lib\ged.jar;lib\geim.jar;lib\jnnview.jar;testes\bin


@REM java %ENABLE_NATIVE% -cp "%CP_JNN_GED%" MainConv "¨%MEM_OPS%"
@REM java %ENABLE_NATIVE% -cp "%CP_JNN_GED_GEIM_VIEW%" MainImg
@REM java %ENABLE_NATIVE% -cp "%CP_JNN_GED%" Benchmark
@REM java %ENABLE_NATIVE% -cp "%CP_JNN_GED_GEIM%" Conv
@REM java %ENABLE_NATIVE% -cp "%CP_JNN_GED_GEIM_VIEW%" Lab
java %ENABLE_NATIVE% -cp "%CP_JNN_GED%" TesteConv 1 4
@REM java %ENABLE_NATIVE% -cp "%CP_JNN_GED_GEIM%" TesteJNI

@rem exemplos
@REM java %ENABLE_NATIVE% -cp "%CP_JNN_GED%" exemplos.Iris
@REM java %ENABLE_NATIVE% -cp "%CP_JNN_GED%" exemplos.MNIST
@REM java %ENABLE_NATIVE% -cp "%CP_JNN_GED%" exemplos.ModelIO
@REM java %ENABLE_NATIVE% -cp "%CP_JNN_GED_GEIM%" exemplos.UpscaleImg
@REM java %ENABLE_NATIVE% -cp "%CP_JNN_GED%" exemplos.Xor