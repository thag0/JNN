@REM gerar bytecode em /bin 
javac -g -parameters -d bin jnn/*.java

@REM exportar .jar
jar cvf bin/jnn.jar -C bin .