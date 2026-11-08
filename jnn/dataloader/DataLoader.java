package jnn.dataloader;

import java.text.DecimalFormat;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.Random;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.ForkJoinTask;
import java.util.function.BiConsumer;
import java.util.function.Function;
import java.util.function.Predicate;

import jnn.core.JNNutils;
import jnn.core.parallel.JNNparallel;
import jnn.core.tensor.Tensor;
import jnn.dataloader.transform.Transform;

/**
 * <h2>
 *      JNN DataLoader
 * </h2>
 *      O DataLoader é um conteiner criado para encapsular amostras
 *      de um conjunto de dados, cada amostra possuindo um {@code Tensor}
 *      para dados de entrada (X) e outro {@code Tensor} para dados de 
 *      saída (Y).
 * @see jnn.core.tensor.Tensor Tensor
 * @see jnn.dataloader.Sample Sample
 */
public class DataLoader implements Iterable<Sample> {

    /**
     * Conjunto de elementos.
     */
    private Sample[] data = {};

    /**
     * Transformações para aplicar em X de cada amostra.
     */
    private Transform transformX = null;
    
    /**
     * Transformações para aplicar em Y de cada amostra.
     */
    private Transform transformY = null;

    /**
     * Inicializa um DataLoader vazio.
     */
    public DataLoader() {}

    /**
     * Inicializa um DataLoader com uma amostra inicial.
     * @param samples {@code Sample} desejada.
     * @see jnn.dataloader.Sample Sample
     */
    public DataLoader(Sample[] samples) {
        add(samples);
    }

    /**
     * Inicializa um DataLoader com uma amostra inicial.
     * @param s {@code Sample} desejada.
     * @see jnn.dataloader.Sample Sample
     */
    public DataLoader(Sample s) {
        add(s);
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
            add(new Sample(x[i], y[i]));
        }
    }

    /**
     * Adiciona um conjunto de amostras.
     * @param samples conjunto de {@code Sample} desejada.
     * @see jnn.dataloader.Sample Sample
     */
    public void add(Sample[] samples) {
        for (Sample a : samples) {
            add(a);
        }
    }

    /**
     * Adiciona uma nova amostra.
     * @param s {@code Sample} desejada.
     * @see jnn.dataloader.Sample Sample
     */
    public void add(Sample s) {
        if (len() > 0) {
            int[] shapeX = s.x().shape();
            int[] shapeY = s.y().shape();

            if (!JNNutils.arrayComp(shapeX, data[0].x().shape())) {
                throw new IllegalArgumentException(
                    "\nFormato de X da amostra " + s.x().shapeStr() + 
                    " deve ser igual ao das amostras de X do DataLoader " + 
                    data[0].x().shapeStr() 
                );
            }
            if (!JNNutils.arrayComp(shapeY, data[0].y().shape())) {
                throw new IllegalArgumentException(
                    "\nFormato de Y da amostra " + s.y().shapeStr() + 
                    " deve ser igual ao das amostras de Y do DataLoader " + 
                    data[0].x().shapeStr() 
                );
            }
        }

        data = JNNutils.addEmArray(data, s);
    }

    /**
     * Adiciona uma nova amostra a partir de um conjunto de entrada e saída.
     * @param x {@code Tensor} contendo dados de entrada.
     * @param y {@code Tensor} contendo dados de saída.
     * @see jnn.core.tensor.Tensor Tensor
     */
    public void add(Tensor x, Tensor y) {
        add(new Sample(x, y));
    }

    /**
     * Aplica a transformação no tensor de entrada.
     * @param tf transformação.
     * @param x tensor base.
     * @return tensor com transformação aplicada.
     */
    private Tensor applyTransform(Transform tf, Tensor x) {
        Tensor out = new Tensor(x.shape());
        tf.apply(x, out);
        return out;
    }

    /**
     * Embaralha os dados do DataLoader usando o algoritmo Fisher-Yates
     * @see jnn.core.JNNutils JNNutils. 
     */
    public void suffle() {
        JNNutils.embaralhar(data, null);
    }

    /**
     * Embaralha os dados do DataLoader usando o algoritmo Fisher-Yates.
     * @param rng gerador de números aleatórios desejado.
     * @see jnn.core.JNNutils JNNutils. 
     */
    public void shuffle(Random rng) {
        JNNutils.embaralhar(data, rng);
    }

    /**
     * Aplica uma transformação no conjunto de dados definido pelo getter e setter.
     * @param tf Transformação desejada.
     * @param getter função que recebe um Sample e retorna o tensor correspondente.
     * @param setter função que recebe um Sample e aplica no tensor a transformação.
     */
    private void applyAll(Transform tf, Function<Sample, Tensor> getter, BiConsumer<Sample, Tensor> setter) {
        int n = len();

        ForkJoinPool pool = JNNparallel.common();

        int threads = pool.getParallelism();
        int chunkSize = (n + threads - 1) / threads;

        var tasks = new ArrayList<ForkJoinTask<?>>();

        for (int i = 0; i < n; i += chunkSize) {
            int start = i;
            int end = Math.min(i + chunkSize, n);

            tasks.add(
                pool.submit(() -> {
                    for (int j = start; j < end; j++) {
                        Sample sample = data[j];

                        setter.accept(
                            sample,
                            applyTransform(tf, getter.apply(sample))
                        );
                    }
                })
            );
        }

        for (var task : tasks) task.join();
    }

    /**
     * Aplica uma transformação nos dados de {@code X} do DataLoader.
     * @param tf função transformadora.
     * @return {@code DataLoader} alterado.
     */
    public DataLoader applyX(Transform tf) {
        applyAll(tf, Sample::x, Sample::setX);

        return this;
    }

    /**
     * Aplica uma transformação nos dados de {@code Y} do DataLoader.
     * @param tf função transformadora.
     * @return {@code DataLoader} alterado.
     */
    public DataLoader applyY(Transform tf) {
        applyAll(tf, Sample::y, Sample::setY);

        return this;
    }

    /**
     * configura uma transformação para aplicar aos valores de X
     * das amostras.
     * @param tf transformação.
     * @return {@code DataLoader} alterado.
     */
    public DataLoader setTransformX(Transform tf) {
        this.transformX = JNNutils.validarNaoNulo(tf, "tf == null");;
        return this;
    }

    /**
     * configura uma transformação para aplicar aos valores de Y
     * das amostras.
     * @param tf transformação.
     * @return {@code DataLoader} alterado.
     */
    public DataLoader setTransformY(Transform tf) {
        if (tf != null) this.transformY = tf;
        return this;
    }

    /**
     * Aplica um filtro a nos dados de {@code X} do DataLoader, retornando um
     * novo subconjunto de dados com seu resultado.
     * @param p filtro que será aplicado em cada amostra de X.
     * @return {@code DataLoader} contendo o subconjunto filtrado.
     */
    public DataLoader filterX(Predicate<Tensor> p) {
        List<Tensor> xs = new ArrayList<>(); 
        List<Tensor> ys = new ArrayList<>();

        final int n = len();
        for (int i = 0; i < n; i++) {
            Sample a = data[i];

            if (p.test(a.x())) {
                xs.add(a.x());
                ys.add(a.y());
            }
        }
    
        Tensor[] t = {};

        return new DataLoader(
            xs.toArray(t),
            ys.toArray(t)
        );
    }

    /**
     * Aplica um filtro a nos dados de Y do DataLoader, retornando um
     * novo subconjunto de dados com seu resultado.
     * @param p filtro que será aplicado em cada amostra de Y.
     * @return {@code DataLoader} contendo o subconjunto filtrado.
     */
    public DataLoader filterY(Predicate<Tensor> p) {
        List<Tensor> xs = new ArrayList<>(); 
        List<Tensor> ys = new ArrayList<>();

        final int n = len();
        for (int i = 0; i < n; i++) {
            Sample a = data[i];

            if (p.test(a.y())) {
                xs.add(a.x());
                ys.add(a.y());
            }
        }
    
        Tensor[] t = {};

        return new DataLoader(
            xs.toArray(t),
            ys.toArray(t)
        );
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
    public DataLoader[] split(double p1, double p2) {
        if (p1 + p2 > 1.0) {
            throw new IllegalArgumentException(
                "\nAs partes p1 e p2 devem somar 1."
            );
        }

        int tam = len();
        int n1 = (int) (tam * p1);
    
        DataLoader dl1 = new DataLoader(JNNutils.subArray(data, 0, n1));
        DataLoader dl2 = new DataLoader(JNNutils.subArray(data, n1, tam));

        return new DataLoader[] {
            dl1, dl2
        };
    }

    /**
     * Retorna o tamanho do Dataloader, equivalente a quantidade de amostras.
     * @return número total de amostras.
     */
    public int len() {
        return data.length;
    }

    /**
     * Retorna uma amostra a partir do índice informado.
     * <p>
     *      Caso exista uma transformação configurada para
     *      a amostra, tanto em X como em Y, ela será aplicada.
     * </p>
     * @param i índice desejado.
     * @return {@code Sample} baseada no índice.
     */
    public Sample get(int i) {
        if (len() < 1) {
            throw new IllegalArgumentException(
                "\nDataLoader vazio."
            );
        }

        if (i < 0 || i >= len()) {
            throw new IllegalArgumentException(
                "\nÍndice " + i + " inválido para total de elementos = " + len() + "."
            );
        }

        if (transformX == null && transformY == null) {
            return data[i];
        }

        Sample a = data[i];
        Tensor x = a.x();
        Tensor y = a.y();

        if (transformX != null) x = applyTransform(transformX, x);
        if (transformY != null) y = applyTransform(transformY, y);

        return new Sample(x, y);
    }

    /**
     * Retorna um conjunto de amostras do DataLoader
     * <p>
     *      Caso exista uma transformação configurada no DataLoader, 
     *      tanto em X como em Y, ela será aplicada em todas as amostras
     *      do lote.
     * </p>
     * @param start índice de início (inclusivo).
     * @param len tamanho do subconjunto.
     * @return lote de dados.
     */
    public Batch batch(int start, int len) {
        if (start < 0) {
            throw new IllegalArgumentException(
                "\nInicio deve ser maior ou igual a zero, mas recebido " + start
            );
        }

        int fim = Math.min(start + len, len());
        int tamLote = fim - start;
        
        Tensor[] xs = new Tensor[tamLote];
        Tensor[] ys = new Tensor[tamLote];

        for (int i = 0; i < tamLote; i++) {
            xs[i] = get(start + i).x();
            ys[i] = get(start + i).y();
        }

        return new Batch(xs, ys);
    }

    /**
     * Retorna um novo {@code DataLoader} a partir de um subconjunto
     * de amostras.
     * @param start índice inicial.
     * @param len tamanho final.
     * @return {@code DataLoader} com subamostras.
     */
    public DataLoader subLoader(int start, int len) {
        Batch lote = batch(start ,len);
        return new DataLoader(lote.arrX(), lote.arrY());
    }

    /**
     * Retorna um novo {@code DataLoader} sem as transformações configuradas.
     * @return {@code DataLoader} sem transformações.
     */
    public DataLoader noTransform() {
        DataLoader clone = clone();
        clone.transformX = null;
        clone.transformY = null;

        return clone;
    }

    /**
     * Retorna todos os elementos de {@code X}, sem aplicar transformações.
     * @return {@code array} de {@code Tensor} contendo X.
     */
    public Tensor[] getX() {
        int n = len();
        Tensor[] xs = new Tensor[n];

        for (int i = 0; i < n; i++) {
            Tensor x = data[i].x();
            xs[i] = (transformX == null) ? x : applyTransform(transformX, x);
        }

        return xs;
    }

    /**
     * Retorna todos os elementos de {@code Y}.
     * @return {@code array} de {@code Tensor} contendo Y.
     */
    public Tensor[] getY() {
        int n = len();
        Tensor[] ys = new Tensor[n];

        for (int i = 0; i < n; i++) {
            Tensor y = data[i].y();
            ys[i] = (transformY == null) ? y : applyTransform(transformY, y);
        }

        return ys;
    }

    /**
     * Retorna um {@code Iterator} do DataLoader
     * @return iterator por amostra.
     */
    public Iterator<Sample> iterator() {
        return new DLIterator();
    }

    /**
     * Iterator do datalaoder
     */
    class DLIterator implements Iterator<Sample> {
        int index = 0;

        @Override
        public boolean hasNext() {
            return index < len();
        }

        @Override
        public Sample next() {
            if (!hasNext()) {
                throw new NoSuchElementException();
            }

            return get(index++);
        }
        
    }

    @Override
    public DataLoader clone() {
        DataLoader clone = new DataLoader(data);
        return clone;
    }

    /**
     * Retorna o tamanho estimado do DataLoader em bytes na memória.
     * @return quantidade de bytes estimada.
     */
    public long sizeBytes() {
        String jvmBits = System.getProperty("sun.arch.data.model");
        long bits = Long.valueOf(jvmBits);

        long tamObj;
        if (bits == 32) tamObj = 8;
        else if (bits == 64) tamObj = 16;
        else throw new IllegalStateException(
            "\nSem suporte para plataforma de " + bits + " bits."
        );

        long refSize = (bits == 32) ? 4 : 8;
        long tamArrayRefs = refSize * data.length;

        long somaAmostras = 0;
        for (Sample a : data) {
            somaAmostras += a.sizeBytes();    
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

        String n = new DecimalFormat("#,###").format(len()).replaceAll(",", ".");

        sb.append("DataLoader = [\n");
        sb.append(pad).append("Amostras: ").append(n).append("\n");
        
        if (len() > 0) {
            sb.append(pad).append("Shape X: ").append(data[0].x().shapeStr()).append("\n");
            sb.append(pad).append("Shape Y: ").append(data[0].y().shapeStr()).append("\n");
        }

        sb.append(pad).append("Tamanho: ").append(JNNutils.formatarTamBytes(sizeBytes())).append("\n");

        sb.append(pad).append("Tx: ").append(transformX == null ? "-" : transformX.nome()).append("\n");
        sb.append(pad).append("Ty: ").append(transformY == null ? "-" : transformY.nome()).append("\n");

        sb.append("]\n");

        return sb.toString();
    }

    @Override
    public String toString() {
        return info();
    }

    /**
     * Exibe, via terminal, as informações do DataLoader.
     */
    public void print() {
        System.out.print(info());
    }

    /**
     * Gera um {@code DataLoader} baseado no problema da porta lógica XOR.
     * @return {@code DataLoader}.
     */
    public static DataLoader Xor() {
        DataLoader loader = new DataLoader();

        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                loader.add(
                    new Tensor(new float[]{ i,j }),
                    new Tensor(new float[]{ i^j })
                );
            }
        }

        return loader;
    }

    /**
     * Gera um {@code DataLoader} baseado no problema da porta lógica AND.
     * @return {@code DataLoader}.
     */
    public static DataLoader And() {
        DataLoader loader = new DataLoader();

        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                loader.add(
                    new Tensor(new float[]{ i,j }),
                    new Tensor(new float[]{ i&j })
                );
            }
        }

        return loader;
    }

}
