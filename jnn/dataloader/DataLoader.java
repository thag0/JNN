package jnn.dataloader;

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
}
