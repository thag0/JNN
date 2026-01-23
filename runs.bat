@echo off

@REM java --enable-native-access=ALL-UNNAMED -cp "testes\bin;bin\jnn.jar;lib\ged.jar" MainConv
@REM java --enable-native-access=ALL-UNNAMED -cp "testes\bin;bin\jnn.jar;lib\ged.jar;lib\geim.jar" Conv
@rem java --enable-native-access=ALL-UNNAMED -cp "testes\bin;bin\jnn.jar;lib\ged.jar;lib\geim.jar" Benchmark
java --enable-native-access=ALL-UNNAMED -cp "testes\bin;bin\jnn.jar;lib\ged.jar;" TesteConv 1
@REM java --enable-native-access=ALL-UNNAMED -cp "testes\bin;bin\jnn.jar;lib\ged.jar" TesteJNI