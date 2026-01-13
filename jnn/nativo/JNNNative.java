package jnn.nativo;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;

/**
 * Interface para implementação nativa em C.
 */
public final class JNNNative {

    /**
     * Nome da biblioteca dinâmica (windows).
     */
    static final String nomeDLL = "jnn_native.dll";
    
    /**
     * Caminho dentro do arquivo final .jar
     */
    static final String caminhoDLL = "/nativo/cpu/win64/";

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
     * Realiza a progração direta através da camada Conv2D.
     * @param X entrada.
     * @param offX offset de entrada.
     * @param K kernel.
     * @param offK offset do kernel.
     * @param B bias (se houver).
     * @param offB offset do bias (se houver).
     * @param hasBias verificador do bias.
     * @param Y saída.
     * @param offY offset da saída.
     * @param lotes quantidade de lotes de entrada.
     * @param canais quantidade de canais de entrada.
     * @param filtros quantidade de kernels.
     * @param altX altura da entrada.
     * @param largX largura da entrada.
     * @param altK altura do kernel.
     * @param largK largura do kernel.
     */
    public static native void conv2dForward(
        double[] X, int offX,
        double[] K, int offK,
        double[] B, int offB, boolean hasBias,
        double[] Y, int offY,
        int lotes, int canais, int filtros,
        int altX, int largX,
        int altK, int largK
    );

    /**
     * Realiza a progração reversa através da camada Conv2D.
     * @param X entrada.
     * @param offX offset de entrada.
     * @param K kernel.
     * @param offK offset do kernel.
     * @param GS gradiente de saída.
     * @param offGS offset do gradiente de saída.
     * @param GK gradiente do kernel.
     * @param offGK offset do gradiente do kernel.
     * @param GB gradiente do bias (se houver).
     * @param offGB offset do gradiente do bias (se houver).
     * @param temBias verificador do bias.
     * @param GE gradiente de entrada.
     * @param offGE offset do gradiente de entrada.
     * @param lotes quantidade de lotes de entrada.
     * @param canais quantidade de canais de entrada.
     * @param filtros quantidade de kernels.
     * @param altX altura da entrada.
     * @param largX largura da entrada.
     * @param altK altura do kernel.
     * @param largK largura do kernel.
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

    /**
     * Realiza a propagação direta através da camada de MaxPooling2D 
     * com lotes de dados de entrada.
     * @param X entrada.
     * @param offX offset de entrada.
     * @param Y destino.
     * @param offY offset do destino.
     * @param lotes quantidade de lotes de entrada.
     * @param canais quantidade de canais de entrada.
     * @param altX altura da entrada.
     * @param largX largura da entrada.
     * @param altFiltro altura do filtro de pooling.
     * @param largFiltro largura do filtro de pooling.
     * @param altStride altura do stride de pooling.
     * @param largStride largura do stride de pooling.
     */
    public static native void maxPool2dForward(
        double[] X, int offX,
        double[] Y, int offY,
        int lotes, int canais,
        int altX, int largX,
        int altFiltro, int largFiltro,
        int altStride, int largStride
    );

    /**
     * Realiza a progração reversa pela camada MaxPool2D com
     * lotes de dados.
     * @param X entrada.
     * @param G gradiente de saída da camada.
     * @param GE gradiente de entrada da camada.
     * @param lotes quantidade de lotes.
     * @param canais quantidade de canais de entrada.
     * @param altX altura da entrada.
     * @param largX largura da entrada.
     * @param altG altura do gradiente de saída.
     * @param largG largura do gradiente de saída.
     * @param altFiltro altura do filtro de pooling.
     * @param largFiltro largura do filtro de pooling.
     * @param altStride altura do stride de pooling.
     * @param largStride largura do stride de pooling.
     */
    public static native void maxPool2dBackward(
        double[] X, int offX,
        double[] G, int offG,
        double[] GE, int offGE,
        int lotes, int canais,
        int altX, int largX,
        int altG, int largG,
        int altFiltro, int largFiltro,
        int altStride, int largStride
    );

}
