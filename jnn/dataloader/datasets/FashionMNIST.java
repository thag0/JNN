import java.io.IOException;
import java.io.InputStream;
import java.net.ConnectException;
import java.net.SocketTimeoutException;
import java.net.URI;
import java.net.UnknownHostException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.HashMap;
import java.util.Map;
import java.util.zip.GZIPInputStream;

import javax.net.ssl.SSLException;

import jnn.core.tensor.Tensor;
import jnn.dataloader.DataLoader;

/**
 * Conjunto de dados do dataset {@code Fashion MNIST}
 */
public class FashionMNIST {
    
    /**
     * Fonte dos dados.
     */
    private static final String BASE = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/";

    /**
     * Mapeamento dos dados.
     */
    private static final Map<String, String> ARQUIVOS = new HashMap<>(); 

    /**
     * Caminho dos arquivos do dataset
     */
    private static Path cacheDir = Paths.get(System.getProperty("user.home"), ".jnn/datasets", "fashion-mnist");

    static {
        try { 
            Files.createDirectories(cacheDir);
        } catch (IOException e) {}

        ARQUIVOS.put("treino-x", "train-images-idx3-ubyte.gz");
        ARQUIVOS.put("treino-y", "train-labels-idx1-ubyte.gz");
        ARQUIVOS.put("teste-x", "t10k-images-idx3-ubyte.gz");
        ARQUIVOS.put("teste-y", "t10k-labels-idx1-ubyte.gz");
    }

    /**
     * Carrega todo o conjunto de dados de treino do dataset {@code Fashion MNIST}.
     * @return {@code DataLoader} com dados lidos.
     */
    public static DataLoader treino() {
        return carregar("treino-x", "treino-y");
    }
    
    /**
     * Carrega todo o conjunto de dados de teste do dataset {@code Fashion MNIST}.
     * @return {@code DataLoader} com dados lidos.
     */
    public static DataLoader teste() {
        return carregar("teste-x", "teste-y");
    }

    /**
     * Carregamento genérnico.
     * @param caminhoX dados de treino.
     * @param caminhoY dados de teste.
     * @return {@code DataLoader} com dados carregados.
     */
    private static DataLoader carregar(String caminhoX, String caminhoY) {
        DataLoader loader = null;

        try {
            Path imgsPath = cacheDir.resolve(ARQUIVOS.get(caminhoX));
            Path labelsPath = cacheDir.resolve(ARQUIVOS.get(caminhoY));
            
            talvezBaixar(imgsPath);
            talvezBaixar(labelsPath);
            
            byte[] imgBytes = descompactarGzip(imgsPath);
            byte[] labelBytes = descompactarGzip(labelsPath);
            
            Tensor[] imgs = lerIDXImagens(imgBytes);
            Tensor[] labels = lerIDXLabels(labelBytes);
            
            loader = new DataLoader(imgs, labels);
        
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }

        return loader;
    }

    /**
     * Baixa o conteúdo do dataset caso não exista o diretório.
     * @param destino diretório do dataset.
     * @throws IOException caso ocorra algum erro.
     */
    private static void talvezBaixar(Path destino) throws IOException {
        if (!Files.exists(destino)) {
            String url = BASE + destino.getFileName().toString();
            System.out.println("Baixando " + url);

            try {
                baixar(url, destino);
            
            } catch (UnknownHostException e) {
                throw new IOException("\nFalha de DNS ao resolver o host em: " + url, e);
            
            } catch (ConnectException e) {
                throw new IOException("\nFalha de conexão ao tentar acessar: " + url, e);
            
            } catch (SSLException e) {
                throw new IOException("\nErro SSL ao conectar em: " + url, e);
            
            } catch (SocketTimeoutException e) {
                throw new IOException("\nTimeout ao tentar baixar: " + url, e);
            
            } catch (IOException e) {
                throw new IOException("\nFalha ao baixar arquivo: " + url, e);
            }
        }
    }

    /**
     * Descompacta os dados do diretório do dataset.
     * @param arquivo caminho do arquivo.
     * @return conjunto de bytes lidos.
     * @throws IOException caso ocorra algum erro.
     */
    private static byte[] descompactarGzip(Path arquivo) throws IOException {
        try (GZIPInputStream gis = new GZIPInputStream(Files.newInputStream(arquivo))) {
            return gis.readAllBytes();
        }
    }

    /**
     * Baixa de fato os arquivos do dataset.
     * @param url url ou link do dataset.
     * @param destino diretório de destino dos arquivos.
     * @throws IOException caso ocorra algum erro.
     */
    private static void baixar(String url, Path destino) throws IOException {
        try (InputStream in = URI.create(url).toURL().openStream()) {
            Files.copy(in, destino, StandardCopyOption.REPLACE_EXISTING);
        }
    }

    /**
     * Converte os bytes dos arquivos de dados.
     * @param dados conjuntos de dados.
     * @return array de {@code Tensor} com dados convertidos.
     */
    @SuppressWarnings("unused")
    private static Tensor[] lerIDXImagens(byte[] dados) {
        ByteBuffer buffer = ByteBuffer.wrap(dados).order(ByteOrder.BIG_ENDIAN);

        int magic = buffer.getInt();// tem que ser lido
        int numImagens = buffer.getInt();
        int linhas = buffer.getInt();
        int colunas = buffer.getInt();

        Tensor[] imgs = new Tensor[numImagens];

        for (int i = 0; i < numImagens; i++) {
            double[] arr = new double[linhas * colunas];

            for (int p = 0; p < arr.length; p++) {
                arr[p] = (buffer.get() & 0xFF) / 255.0f;
            }

            imgs[i] = new Tensor(arr).reshape(1, linhas, colunas);
        }

        return imgs;
    }

    /**
     * Converte os bytes dos arquivos de dados.
     * @param dados conjuntos de dados.
     * @return array de {@code Tensor} com dados convertidos.
     */
    @SuppressWarnings("unused")
    private static Tensor[] lerIDXLabels(byte[] dados) {
        ByteBuffer buffer = ByteBuffer.wrap(dados).order(ByteOrder.BIG_ENDIAN);

        int magic = buffer.getInt();// tem que ser lido
        int total = buffer.getInt();
        Tensor[] labels = new Tensor[total];

        for (int i = 0; i < total; i++) {
            int label = buffer.get() & 0xFF;
            double[] data = new double[10];
            data[label] = 1.0;

            labels[i] = new Tensor(data).nome("Label-" + label);
        }

        return labels;
    }

}
