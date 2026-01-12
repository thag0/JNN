@echo off

@rem - arquivos principais
@REM javac -cp "bin\jnn.jar;lib\ged.jar" -d testes/bin MainConv.java

@rem -arquivos de testes
@REM javac -cp "bin\jnn.jar;lib\ged.jar;lib\geim.jar" -d testes/bin Benchmark.java
javac -cp "bin\jnn.jar;lib\ged.jar;lib\geim.jar" -d testes/bin Conv.java
@REM javac -cp "bin\jnn.jar;lib\ged.jar;lib\geim.jar" -d testes/bin TesteJNI.java
@REM javac -cp "bin\jnn.jar;lib\ged.jar;" -d testes/bin TesteConv.java