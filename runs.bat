@echo off
set F_NATIVE=--enable-native-access=ALL-UNNAMED

@REM java --enable-native-access=ALL-UNNAMED -cp "testes\bin;bin\jnn.jar;lib\ged.jar" MainConv
@REM java --enable-native-access=ALL-UNNAMED -cp "testes\bin;bin\jnn.jar;lib\ged.jar;lib\geim.jar" Conv
@rem java --enable-native-access=ALL-UNNAMED -cp "testes\bin;bin\jnn.jar;lib\ged.jar;lib\geim.jar" Benchmark
@REM java --enable-native-access=ALL-UNNAMED -cp "testes\bin;bin\jnn.jar;lib\ged.jar;" TesteConv
java --enable-native-access=ALL-UNNAMED -cp "testes\bin;bin\jnn.jar;lib\ged.jar" TesteJNI