package jnn.dataloader;

import java.text.DecimalFormat;
import java.util.Iterator;
import java.util.Random;

import jnn.core.Utils;
import jnn.core.tensor.Tensor;
import jnn.dataloader.Transform.Transform;

/**
 * <h2>
 *      JNN DataLoader
 * </h2>
 *      O DataLoader é um conteiner criado para encapsular amostras
 *      de um conjunto de dados, cada amostra possuindo um {@code Tensor}
 *      para dados de entrada (X) e outro {@code Tensor} para dados de 
 *      saída (Y).
 * @see {@code Tensor} {@link jnn.core.tensor.Tensor}
 * @see {@code Amostra} {@link jnn.dataloader.Amostra}
 */
public class DataLoader implements Iterable<Amostra> {

    /**
     * Conjunto de elementos.
     */
    Amostra[] dados = {};

    /**
     * Utilitário.
     */
    Utils utils = new Utils();

    /**
     * Inicializa um DataLoader vazio.
     */
    public DataLoader() {}

    /**
     * Inicializa um DataLoader com uma amostra inicial.
     * @param a {@code Amostra} desejada.
     * @see {@link jnn.dataloader.Amostra}
     */
    public DataLoader(Amostra a) {
        add(a);
    }

    /**
     * Inicializa um DataLoader com uma amostra inicial.
     * @param a {@code Amostra} desejada.
     * @see {@link jnn.dataloader.Amostra}
     */
    public DataLoader(Amostra[] as) {
        add(as);
    }

    /**
     * Inicializa um DataLoader a partir de um conjunto de {@code Tensor}
     * para {@code X} e {@code Y}.
     * @param x {@code array} de {@code Tensor} para dados de entrada.
     * @param y {@code array} de {@code Tensor} para dados de saída.
     */
    public DataLoader(Tensor[] x, Tensor[] y) {
        if (x.length != y.length) {
            throw new IllegalArgumentException(
                "\nX e Y devem ter o mesmo tamanho, mas X = " + x.length +
                " e Y = " + y.length + "."
            );
        }

        int n = x.length;
        for (int i = 0; i < n; i++) {
            add(new Amostra(x[i], y[i]));
        }
    }

    /**
     * Adiciona um conjunto de amostras.
     * @param as conjunto de {@code Amostra} desejada.
     * @see {@link jnn.dataloader.Amostra}
     */
    public void add(Amostra[] as) {
       for (Amostra a : as) {
            add(a);
        }
    }

    /**
     * Adiciona uma nova amostra.
     * @param a {@code Amostra} desejada.
     * @see {@link jnn.dataloader.Amostra}
     */
    public void add(Amostra a) {
        if (numel() > 1) {
            int[] shapeX = a.x().shape();
            int[] shapeY = a.y().shape();

            if (!utils.comparar(shapeX, dados[0].x().shape())) {
                throw new IllegalArgumentException(
                    "\nFormato de X da amostra " + a.x().shapeStr() + 
                    " deve ser igual ao das amostras de X do DataLoader " + 
                    dados[0].x().shapeStr() 
                );
            }
            if (!utils.comparar(shapeY, dados[0].y().shape())) {
                throw new IllegalArgumentException(
                    "\nFormato de Y da amostra " + a.y().shapeStr() + 
                    " deve ser igual ao das amostras de Y do DataLoader " + 
                    dados[0].x().shapeStr() 
                );
            }
        }

        dados = utils.addEmArray(dados, a);
    }

    /**
     * Adiciona uma nova amostra a partir de um conjunto de entrada e saída.
     * @param x {@code Tensor} contendo dados de entrada.
     * @param y {@code Tensor} contendo dados de saída.
     * @see {@link jnn.core.tensor.Tensor}
     */
    public void add(Tensor x, Tensor y) {
        add(new Amostra(x, y));
    }

    /**
     * Embaralha os dados do DataLoader usando o algoritmo Fisher-Yates
     * @see {@link jnn.core.Utils} 
     */
    public void embaralhar() {
        utils.embaralhar(dados, null);
    }

    /**
     * Embaralha os dados do DataLoader usando o algoritmo Fisher-Yates.
     * @param rng gerador de números aleatórios desejado.
     * @see {@link jnn.core.Utils} 
     */
    public void embaralhar(Random rng) {
        utils.embaralhar(dados, rng);
    }

    /**
     * Aplica uma transformação nos dados de {@code X} do DataLoader.
     * <p> 
     *      Exemplo de uso
     * </p> 
     * <pre> 
     * //normaliza todos os dados de X.
     *dl.transformX(t -> t.norm(0, 1))
     * </pre> 
     * @param tf função {@code Transform} para aplicação.
     * @return {@code DataLoader} alterado.
     */
    public DataLoader transformX(Transform tf) {
        for (Amostra a : dados) {
            a.setX(tf.apply(a.x()));
        }

        return this;
    }

    /**
     * Aplica uma transformação nos dados de {@code Y} do DataLoader.
     * <p> 
     *      Exemplo de uso
     * </p> 
     * <pre> 
     * //normaliza todos os dados de Y.
     *dl.transformY(t -> t.norm(0, 1))
     * </pre> 
     * @param tf função {@code Transform} para aplicação.
     * @return {@code DataLoader} alterado.
     */
    public DataLoader transformY(Transform tf) {
        for (Amostra a : dados) {
            a.setY(tf.apply(a.y()));
        }
        
        return this;
    }
    
    /**
     * Aplica uma transformação nos dados de {@code X} e {@code Y} 
     * do DataLoader.
     * @param tx função {@code Transform} para aplicação em {@code X}.
     * @param ty função {@code Transform} para aplicação em {@code Y}.
     * @return {@code DataLoader} alterado.
     */
    public DataLoader transformXY(Transform tx, Transform ty) {
        for (Amostra a : dados) {
            a.setX(tx.apply(a.x()));
            a.setY(ty.apply(a.y()));
        }
        
        return this;
    }

    /**
     * Divie o conjunto de amostras do DataLoader em duas partes.
     * @param p1 quantidade reservada para parte 1.
     * @param p2 quantidade reservada para parte 2.
     * @return {@code array} de {@code DataLoader} onde:
     * <pre>
     *dl[0] -> Parte 1
     *dl[1] -> Parte 2
     * </pre>
     */
    public DataLoader[] separar(double p1, double p2) {
        if (p1 + p2 > 1.0) {
            throw new IllegalArgumentException(
                "\nAs partes p1 e p2 devem somar 1."
            );
        }

        int tam = numel();
        int n1 = (int) (tam * p1);
    
        DataLoader dl1 = new DataLoader(utils.subArray(dados, 0, n1));
        DataLoader dl2 = new DataLoader(utils.subArray(dados, n1, tam));

        return new DataLoader[] {
            dl1, dl2
        };
    }

    /**
     * Retorna o número de amostras contidas no Dataloader.
     * @return número total de amostras.
     */
    public int numel() {
        return dados.length;
    }

    /**
     * Retorna uma amostra a partir do índice informado.
     * @param id índice desejado.
     * @return {@code Amostra} baseada no índice.
     */
    public Amostra get(int id) {
        if (numel() < 1) {
            throw new IllegalArgumentException(
                "\nDataLoader vaizo."
            );
        }

        if (id < 0 || id >= numel()) {
            throw new IllegalArgumentException(
                "\nÍndice " + id + " inválido para total de elementos = " + numel() + "."
            );
        }

        return dados[id];
    }

    /**
     * Retorna um conjunto de amostras do DataLoader
     * @param in índice de início (inclusivo).
     * @param tam tamanho do subconjunto.
     * @return lote de dados.
     */
    public Amostra[] getLote(int in, int tam) {
        if (in < 0) {
            throw new IllegalArgumentException(
                "\nInicio deve ser maior ou igual a zero, mas recebido " + in
            );
        }

        int fim = Math.min(in + tam, numel());
        Amostra[] lote = utils.subArray(dados, in, fim);

        return lote;
    }

    /**
     * Retorna todos os elementos de {@code X}.
     * @return {@code array} de {@code Tensor} contendo X.
     */
    public Tensor[] getX() {
        int n = numel();
        Tensor[] xs = new Tensor[n];

        for (int i = 0; i < n; i++) {
            xs[i] = dados[i].x();
        }

        return xs;
    }

    /**
     * Retorna todos os elementos de {@code Y}.
     * @return {@code array} de {@code Tensor} contendo Y.
     */
    public Tensor[] getY() {
        int n = numel();
        Tensor[] ys = new Tensor[n];

        for (int i = 0; i < n; i++) {
            ys[i] = dados[i].y();
        }

        return ys;
    }

    /**
     * Retorna um {@code Iterator} do DataLoader
     * @return iterator por amostra.
     */
    public Iterator<Amostra> iterator() {
        return new DLIterator();
    }

    /**
     * Iterator do datalaoder
     */
    class DLIterator implements Iterator<Amostra> {
        int id = 0;

        @Override
        public boolean hasNext() {
            return id < numel();
        }

        @Override
        public Amostra next() {
            return dados[id++];
        }
        
    }

    @Override
    public DataLoader clone() {
        DataLoader clone = new DataLoader(dados);
        return clone;
    }

    /**
     * Retorna o tamanho estimado do DataLoader em bytes na memória.
     * @return quantidade de bytes estimada.
     */
    public long tamBytes() {
        String jvmBits = System.getProperty("sun.arch.data.model");
        long bits = Long.valueOf(jvmBits);

        long tamObj;
        if (bits == 32) tamObj = 8;
        else if (bits == 64) tamObj = 16;
        else throw new IllegalStateException(
            "\nSem suporte para plataforma de " + bits + " bits."
        );

        long refSize = (bits == 32) ? 4 : 8;
        long tamArrayRefs = refSize * dados.length;

        long somaAmostras = 0;
        for (Amostra a : dados) {
            somaAmostras += a.tamBytes();    
        }

        return tamObj + tamArrayRefs + somaAmostras;
    }

    /**
     * Monta uma {@code String} que descreve o DataLoader.
     * @return resumo do DataLoader. 
     */
    private String info() {
        StringBuilder sb = new StringBuilder();
        String pad = " ".repeat(4);

        String n = new DecimalFormat("#,###").format(numel()).replaceAll(",", ".");

        sb.append("DataLoader = [\n");
        
        sb.append(pad).append("Amostras: ").append(n).append("\n");
        sb.append(pad).append("Tamanho: ").append(formatarTamanho(tamBytes())).append("\n");

        if (numel() > 1) {
            sb.append(pad).append("Shape X: ").append(dados[0].x().shapeStr()).append("\n");
            sb.append(pad).append("Shape Y: ").append(dados[0].y().shapeStr()).append("\n");
        }

        sb.append("]\n");

        return sb.toString();
    }

    /**
     * Formata o tamanho do DataLoader.
     * @param bytes quantidade total em bytes.
     * @return valor formatado.
     */
    private static String formatarTamanho(long bytes) {
        if (bytes < 1024) return bytes + " B";
        int exp = (int) (Math.log(bytes) / Math.log(1024));
        char prefixo = "KMGTPE".charAt(exp - 1); // K, M, G, T, P, E
        
        return String
        .format("%.2f %sB", bytes / Math.pow(1024, exp), prefixo)
        .replaceAll(",", ".");
    }

    @Override
    public String toString() {
        return info();
    }

    /**
     * Exibe, via terminal, as informações do DataLoader.
     */
    public void print() {
        System.out.println(info());
    }

}
