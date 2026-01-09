package jnn.nativo;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;

public final class JNNNative {
    
    static {
        try {
            carregarDoJar();
        
        } catch (Exception e) {
            throw new RuntimeException("Erro ao carregar JNI", e);
        }
    }

    private static void carregarDoJar() throws IOException {
        String nomeDLL = "jnn_native.dll";

        InputStream in = JNNNative.class.getResourceAsStream("/native/win64/" + nomeDLL);

        if (in == null) {
            throw new FileNotFoundException("DLL não encontrada dentro do JAR");
        }

        Path tempDir = Files.createTempDirectory("jnn_native");
        tempDir.toFile().deleteOnExit();

        Path dll = tempDir.resolve(nomeDLL);
        Files.copy(in, dll, StandardCopyOption.REPLACE_EXISTING);
        dll.toFile().deleteOnExit();

        System.load(dll.toAbsolutePath().toString());
    }

    private JNNNative() {}

    /**
     * Experimental
     */
    public static native void matmul(
        double[] A, int offA, int s0A, int s1A,
        double[] B, int offB, int s0B, int s1B,
        double[] C, int offC, int s0C, int s1C,
        int M, int K, int N
    );

    /**
     * Experimental
     */
    public static native void conv2dForward(
        double[] X, int offX,
        double[] K, int offK,
        double[] B, int offB, boolean hasBias,
        double[] Y, int offY,
        int BATCH, int CIN, int COUT,
        int H, int W,
        int kH, int kW
    );
    /**
     * Experimental
     */
    public static native void conv2dBackward(
        double[] X, int offX,
        double[] K, int offK,
        double[] GS, int offGS,
        double[] GK, int offGK,
        double[] GB, int offGB, boolean temBias,
        double[] GE, int offGE,
        int lotes, int canais, int filtros,
        int altX, int largX,
        int altK, int largK    
    );

}
