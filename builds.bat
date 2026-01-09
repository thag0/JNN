@echo off

javac -cp "bin\jnn.jar;lib\ged.jar" -d testes/bin MainConv.java
javac -cp "bin\jnn.jar;lib\ged.jar;lib\geim.jar" -d testes/bin Conv.java