package jnn.nativo;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;

public final class JNNNative {

    /**
     * Nome da biblioteca dinâmica (windows).
     */
    static final String nomeDLL = "jnn_native.dll";
    
    /**
     * Caminho dentro do arquivo final .jar
     */
    static final String caminhoDLL = "/native/win64/";

    static {
        try {
            carregarDoJar();
        
        } catch (Exception e) {
            throw new RuntimeException("\nErro ao carregar JNI", e);
        }
    }

    /**
     * Tenta carregar o arquivo dll.
     * @throws IOException caso ocora algum erro.
     */
    private static void carregarDoJar() throws IOException {
        InputStream is = JNNNative.class.getResourceAsStream(caminhoDLL + nomeDLL);

        if (is == null) {
            throw new FileNotFoundException("\nDLL não encontrada: " + nomeDLL);
        }

        Path tmp = Files.createTempDirectory("jnn_native");
        tmp.toFile().deleteOnExit();

        Path dll = tmp.resolve(nomeDLL);
        Files.copy(is, dll, StandardCopyOption.REPLACE_EXISTING);
        dll.toFile().deleteOnExit();

        System.load(dll.toAbsolutePath().toString());
    }

    private JNNNative() {}

    /**
     * Altera a quantidade de threads usadas pelo código nativo.
     * @param t quantidade desejada de threads.
    */
    public static native void setThreads(int t);

    /**
     * Realiza a multiplicação matricial entre A e B.
     * <p>
     *      Essa função já assume que A e B são compatíveis.
     * </p>
     * @param A conjunto de elementos de A.
     * @param offA offset de A.
     * @param s0A stride de linhas de A.
     * @param s1A stride de colunas de A.
     * @param B conjunto de elementos de B.
     * @param offB offset de B.
     * @param s0B stride de linhas de B.
     * @param s1B stride de colunas de B.
     * @param C conjunto de elementos do destino.
     * @param offC offset do Destino.
     * @param s0C stride de linhas do destino.
     * @param s1C stride de colunas do destino.
     * @param linA linhas de A.
     * @param colA colunas de A.
     * @param colB colunas de B.
     */
    public static native void matmul(
        double[] A, int offA, int s0A, int s1A,
        double[] B, int offB, int s0B, int s1B,
        double[] C, int offC, int s0C, int s1C,
        int linA, int colA, int colB
    );

    /**
     * Experimental
     */
    public static native void conv2dForward(
        double[] X, int offX,
        double[] K, int offK,
        double[] B, int offB, boolean hasBias,
        double[] Y, int offY,
        int lotes, int canais, int filtros,
        int atlX, int largX,
        int altK, int largK
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
