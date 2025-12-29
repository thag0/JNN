package jnn.dataloader.dataset;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.zip.GZIPInputStream;

import jnn.core.tensor.Tensor;
import jnn.dataloader.DataLoader;

/**
 * Conjunto de dados do dataset {@code CIFAR-10}.
 */
public class CIFAR10 {

    /**
     * Fonte dos dados.
     */
    static private final String BASE = "https://www.cs.toronto.edu/~kriz/";

    /**
     * Caminho dos arquivos do dataset
     */
    static private Path cacheDir = Paths.get(System.getProperty("user.home"), ".jnn/datasets", "cifar-10");

    // constantes auxiliares
    static final int IMG_TAM = 32;
    static final int IMG_CANAIS = 3;
    static final int IMG_BYTES = 3 * 32 * 32;
    static final int RECORD_BYTES = 1 + IMG_BYTES;
    static final int NUM_CLASSES = 10;
    static final String ARQUIVO_TAR = "cifar-10-binary.tar.gz";
    static final String DIR_INTERNO = "cifar-10-batches-bin";

    static final String[] BATCHES = {
        "data_batch_1.bin",
        "data_batch_2.bin",
        "data_batch_3.bin",
        "data_batch_4.bin",
        "data_batch_5.bin",
        "test_batch.bin"
    };

    static {
        try { 
            Files.createDirectories(cacheDir); 
        } catch (IOException e) {}
    }

    /**
     * Carrega todo o conjunto de dados de treino do dataset {@code CIFAR-10}.
     * @return {@code DataLoader} com dados lidos.
     */
    public static DataLoader treino() {
        return carregarTreino();
    }

    /**
     * Carrega todo o conjunto de dados de teste do dataset {@code CIFAR-10}.
     * @return {@code DataLoader} com dados lidos.
     */
    public static DataLoader teste() {
        return carregarTeste();
    }

    /**
     * Retorna as classes correspondentes aos dados de saída do dataset.
     * <p>
     *      Exemplo
     * </p>
     * <pre>
     *String labels = CIFAR10.labels();
     *int label = (int) y.argmax().item();
     *labels.get(label);//retorna a classe correspondente
     * </pre>
     * @return {@code HashMap} que liga os dítigos aos labels correspondentes.
     */
    public static Map<Integer, String> labels() {
        Map<Integer, String> labels = new HashMap<>();

        labels.put(0, "airplane");
        labels.put(1, "automobile");
        labels.put(2, "bird");
        labels.put(3, "cat");
        labels.put(4, "deer");
        labels.put(5, "dog");
        labels.put(6, "frog");
        labels.put(7, "horse");
        labels.put(8, "ship");
        labels.put(9, "truck");

        return labels;
    }

    /**
     * Carrega os dados de treino.
     * @return {@code DataLoader} contendo as amostras de dados.
     */
    private static DataLoader carregarTreino() {
        try {
            talvezBaixar();

            List<Tensor> xs = new ArrayList<>();
            List<Tensor> ys = new ArrayList<>();

            for (int i = 1; i <= 5; i++) {
                lerBatch(
                    cacheDir.resolve("data_batch_" + i + ".bin"),
                    xs, ys
                );
            }

            Tensor[] t = {};
            return new DataLoader(
                xs.toArray(t),
                ys.toArray(t)
            );

        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Carrega os dados de teste.
     * @return {@code DataLoader} contendo as amostras de dados.
     */
    private static DataLoader carregarTeste() {
        try {
            talvezBaixar();

            List<Tensor> xs = new ArrayList<>();
            List<Tensor> ys = new ArrayList<>();

            lerBatch(
                cacheDir.resolve("test_batch.bin"),
                xs, ys
            );

            Tensor[] t = {};
            return new DataLoader(
                xs.toArray(t),
                ys.toArray(t)
            );

        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Lê um lote de dados.
     * @param arquivo caminho do arquivo base. 
     * @param imagens lista para armazenar os dados das imagens.
     * @param labels lista para armazenar os rótulos das imagens.
     * @throws IOException caso ocorra algum erro.
     */
    private static void lerBatch(Path arquivo, List<Tensor> imagens, List<Tensor> labels) throws IOException {
        byte[] dados = Files.readAllBytes(arquivo);
        int total = dados.length / RECORD_BYTES;

        for (int i = 0; i < total; i++) {
            int offset = i * RECORD_BYTES;

            int label = dados[offset] & 0xFF;

            double[] img = new double[IMG_BYTES];

            for (int p = 0; p < IMG_BYTES; p++) {
                img[p] = (dados[offset + 1 + p] & 0xFF) / 255.0;
            }

            // CIFAR vem em RRR.. GGG.. BBB..
            Tensor x = new Tensor(img).reshape(3, IMG_TAM, IMG_TAM);

            double[] y = new double[NUM_CLASSES];
            y[label] = 1.0;

            imagens.add(x);
            labels.add(new Tensor(y));
        }
    }

    /**
     * Baixa o conteúdo do dataset caso não exista o diretório.
     * @throws IOException caso ocorra algum erro.
     */
    private static void talvezBaixar() throws IOException {
        boolean ok = true;
        for (String b : BATCHES) {
            if (!Files.exists(cacheDir.resolve(b))) {
                ok = false;
                break;
            }
        }

        if (ok) return;

        Path tarGz = cacheDir.resolve(ARQUIVO_TAR);

        if (!Files.exists(tarGz)) {
            String url = BASE + ARQUIVO_TAR;
            System.out.println("Baixando " + url);
            baixar(url, tarGz);
        }

        extrairTarGz(tarGz, cacheDir);

        Path interno = cacheDir.resolve(DIR_INTERNO);
        for (String b : BATCHES) {
            Files.move(
                interno.resolve(b),
                cacheDir.resolve(b),
                StandardCopyOption.REPLACE_EXISTING
            );
        }

        Files.walk(interno)
            .sorted(Comparator.reverseOrder())
            .forEach(p -> {
                try { Files.delete(p); } catch (IOException e) {}
            });
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
     * Extrai o conteúdo do arquivo.
     * @param arquivo caminho do arquivo base.
     * @param destino caminho para o arquivo de destino.
     * @throws IOException caso ocorra algum erro.
     */
    private static void extrairTarGz(Path arquivo, Path destino) throws IOException {
        try (InputStream fis = Files.newInputStream(arquivo);
            GZIPInputStream gis = new GZIPInputStream(fis)) {

            byte[] header = new byte[512];

            while (true) {
                int lidos = gis.readNBytes(header, 0, 512);
                if (lidos < 512) break;

                boolean vazio = true;
                for (byte b : header) {
                    if (b != 0) { vazio = false; break; }
                }

                if (vazio) break;

                String nome = new String(header, 0, 100).trim();
                if (nome.isEmpty()) break;

                char flag = (char) header[156];

                String tamOct = new String(header, 124, 12).trim();
                int tam = tamOct.isEmpty() ? 0 : Integer.parseInt(tamOct, 8);

                Path saida = destino.resolve(nome);

                if (flag == '5') {
                    Files.createDirectories(saida);
                    continue;
                }

                Files.createDirectories(saida.getParent());

                try (OutputStream os = Files.newOutputStream(saida, StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING)) {
                    os.write(gis.readNBytes(tam));
                }

                int pad = (512 - (tam % 512)) % 512;
                if (pad > 0) {
                    gis.skip(pad);
                }
            }
        }
    }

}
