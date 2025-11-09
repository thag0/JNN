@REM gerar bytecode em /bin 
@REM depois revisar uma forma mais rápida de compilar tudo
for /R jnn %%f in (*.java) do (
    javac -g -parameters -d bin "%%f"
)

@REM exportar .jar
jar cvf bin/jnn.jar -C bin .