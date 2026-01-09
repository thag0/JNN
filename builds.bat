@echo off

@rem -arquivos de testes
javac -cp "bin\jnn.jar;lib\ged.jar;lib\geim.jar" -d testes/bin Benchmark.java
javac -cp "bin\jnn.jar;lib\ged.jar;lib\geim.jar" -d testes/bin Conv.java

javac -cp "bin\jnn.jar;lib\ged.jar" -d testes/bin MainConv.java