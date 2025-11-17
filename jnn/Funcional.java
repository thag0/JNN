package jnn;

import jnn.acts.Argmax;
import jnn.acts.Ativacao;
import jnn.acts.Softmax;
import jnn.core.Dicionario;
import jnn.core.OpTensor;
import jnn.core.Utils;
import jnn.core.tensor.Tensor;
import jnn.core.tensor.TensorConverter;
import jnn.dataloader.Amostra;
import jnn.dataloader.DataLoader;
import jnn.inicializadores.Identidade;
import jnn.inicializadores.Inicializador;
import jnn.metrica.metrica.Acuracia;
import jnn.metrica.metrica.F1Score;
import jnn.metrica.metrica.MatrizConfusao;
import jnn.metrica.perda.EntropiaCruzada;
import jnn.metrica.perda.EntropiaCruzadaBinaria;
import jnn.metrica.perda.MAE;
import jnn.metrica.perda.MSE;
import jnn.metrica.perda.MSLE;
import jnn.metrica.perda.Perda;
import jnn.metrica.perda.RMSE;
import jnn.otm.Otimizador;

import java.util.random.RandomGenerator;

/**
 * Interface funcional.
 */
public final class Funcional {

    /**
     * Dicionário de recursos.
     */
    Dicionario dicionario = new Dicionario();
    
    /**
     * Operador para tensores.
     */
    OpTensor opt = new OpTensor();

    /**
     * Utilitário.
     */
    Utils utils = new Utils();

    /**
     * Interface para algumas funcionalidades da biblioteca.
     */
    public Funcional() {}

    // tensor

    /**
     * Inicializa um Tensor a partir de um objeto informado.
     * @param obj objeto base.
     * @return {@code Tensor} vazio.
     */
    public Tensor tensor(Object obj) {
        return TensorConverter.tensor(obj);
    }

    /**
     * Inicializa um tensor vazio com o formato especificado.
     * @param shape formato desejado do tensor.
     * @return {@code Tensor} vazio.
     */
    public Tensor zeros(int... shape) {
        return new Tensor(shape);
    }

    /**
     * Inicializa um tensor preenchido com valor 1, baseado no formato especificado.
     * @param shape formato desejado do tensor.
     * @return {@code Tensor} preenchido.
     */
    public Tensor ones(int... shape) {
        return zeros(shape).preencher(1.0);
    }

    /**
     * Inicializa um tensor com valores da diagonal principal iguais a 1 e
     * o restante igual a zero.
     * @param n tamanho do tensor (linhas e colunas).
     * @return {@code Tensor} identidade.
     */
    public Tensor identidade(int n) {
        if (n < 1) {
            throw new IllegalArgumentException(
                "\nTamanho do tensor deve ser maior que 1, recebido " + n
            );
        }

        Tensor t = new Tensor(n, n);
        new Identidade().forward(t);

        return t;
    }

    /**
     * Inicializa um tensor com valores aleatórios entre -1 e 1.
     * @param shape formato desejado do tensor.
     * @return {@code Tensor} aleatório.
     */
    public Tensor random(int... shape) {
        Tensor t = new Tensor(shape);
        t.aplicar(_ -> Math.random()*2-1);

        return t;
    }

    /**
     * Inicializa um tensor com valores aleatórios usando um gerador.
     * @param gen gerador de números pseudo-aleatórios.
     * @param shape formato desejado do tensor.
     * @return {@code Tensor} aleatório.
     */
    public Tensor random(RandomGenerator gen, int... shape) {
        Tensor t = new Tensor(shape);
        t.aplicar(_ -> gen.nextDouble(-1.0, 1.0));

        return t;
    }

    // operações

    /**
     * Realiza a operação {@code A + B}.
     * @param a {@code Tensor} A.
     * @param b {@code Tensor} B.
     * @return {@code Tensor} resultado.
     */
    public Tensor add(Tensor a, Tensor b) {
        try {
            return new Tensor(a).add(b);
        } catch(Exception e) {
            //
        }
    
        // ultimo caso (mais lento)
        return a.broadcast(b, (_a, _b) -> _a + _b);
    }

    /**
     * Realiza a operação {@code A - B}.
     * @param a {@code Tensor} A.
     * @param b {@code Tensor} B.
     * @return {@code Tensor} resultado.
     */
    public Tensor sub(Tensor a, Tensor b) {
        try {
            return new Tensor(a).sub(b);
        } catch(Exception e) {
            //
        }
    
        // ultimo caso (mais lento)
        return a.broadcast(b, (_a, _b) -> _a - _b);
    }

    /**
     * Realiza a operação {@code A ⊙ B} (produto Hadamard).
     * @param a {@code Tensor} A.
     * @param b {@code Tensor} B.
     * @return {@code Tensor} resultado.
     */
    public Tensor mul(Tensor a, Tensor b) {
        try {
            return new Tensor(a).mul(b);
        } catch(Exception e) {
            //
        }
    
        // ultimo caso (mais lento)
        return a.broadcast(b, (_a, _b) -> _a * _b);
    }

    /**
     * Realiza a operação {@code A * B} (Produto Matricial).
     * @param a {@code Tensor} A.
     * @param b {@code Tensor} B.
     * @return {@code Tensor} resultado.
     */
    public Tensor matmul(Tensor a, Tensor b) {
        return opt.matmul(a, b);
    }

    /**
     * Realiza a operação {@code A / B}.
     * @param a {@code Tensor} A.
     * @param b {@code Tensor} B.
     * @return {@code Tensor} resultado.
     */
    public Tensor div(Tensor a, Tensor b) {
        try {
            return new Tensor(a).mul(b);
        } catch(Exception e) {
            //
        }
    
        // ultimo caso (mais lento)
        return a.broadcast(b, (_a, _b) -> _a / _b);
    }

    /**
     * Calcula o valor exponencial os elementos do tensor.
     * @param t {@code Tensor} desejado usado como base.
     * @param exp expoente.
     * @return {@code Tensor} resultado.
     */
    public Tensor pow(Tensor t, Number exp) {
        double e = exp.doubleValue();
        return new Tensor(t).aplicar(x -> Math.pow(x, e));
    }

    /**
     * Retorna o valor mínimo contido no tensor.
     * @param t {@code Tensor} desejado.
     * @return {@code Tensor} resultado.
     */
    public Tensor min(Tensor t) {
        return new Tensor(t).min();
    }

    /**
     * Retorna o valor máximo contido no tensor.
     * @param t {@code Tensor} desejado.
     * @return {@code Tensor} resultado.
     */
    public Tensor max(Tensor t) {
        return new Tensor(t).max();
    }

    /**
     * Retorna o valor da média de todos os elementos do tensor.
     * @param t {@code Tensor} desejado.
     * @return {@code Tensor} resultado.
     */
    public Tensor media(Tensor t) {
        return new Tensor(t).media();
    }

    /**
     * Retorna o valor do desvio padrão de todos os elementos do tensor.
     * @param t {@code Tensor} desejado.
     * @return {@code Tensor} resultado.
     */
    public Tensor desvp(Tensor t) {
        return new Tensor(t).desvp();
    }

    /**
     * Realiza a operação {@code correlação cruzada} entre os tensores
     * A e B.
     * @param a {@code Tensor} usado como entrada.
     * @param b {@code Tensor} usado como kernel.
     * @return {@code Tensor} resultado.
     */
    public Tensor corr2D(Tensor a, Tensor b) {
        return opt.corr2D(a, b);
    }

    /**
     * Realiza a operação {@code convolucional} entre os tensores
     * A e B.
     * @param a {@code Tensor} usado como entrada.
     * @param b {@code Tensor} usado como kernel.
     * @return {@code Tensor} resultado.
     */
    public Tensor conv2D(Tensor a, Tensor b) {
        return opt.conv2D(a, b);
    }

    // funções

    /**
     * Calcula a ativação relu aos elementos do tensor.
     * @param t {@code Tensor} desejado.
     * @return {@code Tensor} com resultado aplicado.
     */
    public Tensor relu(Tensor t) {
        return new Tensor(t).relu();
    }

    /**
     * Calcula a ativação sigmoid aos elementos do tensor.
     * @param t {@code Tensor} desejado.
     * @return {@code Tensor} com resultado aplicado.
     */
    public Tensor sigmoid(Tensor t) {
        return new Tensor(t).sigmoid();
    }

    /**
     * Calcula a ativação Tangente Hiperbólica aos elementos do tensor.
     * @param t {@code Tensor} desejado.
     * @return {@code Tensor} com resultado aplicado.
     */
    public Tensor tanh(Tensor t) {
        return new Tensor(t).tanh();
    }

    /**
     * Calcula a ativação softmax aos elementos do tensor.
     * @param t {@code Tensor} desejado.
     * @return {@code Tensor} com resultado aplicado.
     */
    public Tensor softmax(Tensor t) {
        Tensor soft = new Tensor(t);
        new Softmax().forward(t, soft);
        return soft;
    }

    /**
     * Calcula a ativação argmax aos elementos do tensor.
     * @param t {@code Tensor} desejado.
     * @return {@code Tensor} com resultado aplicado.
     */
    public Tensor argmax(Tensor t) {
        Tensor arg = new Tensor(t);
        new Argmax().forward(t, arg);
        return arg;
    }

    // perdas

    /**
     * Calcula o valor do {@code Erro Médio Quadrático} dos dados 
     * previstos em relação aos dados reais.
     * @param prev {@code Tensor} com dados previstos.
     * @param real {@code Tensor} com dados reais.
     * @return {@code Tensor} contendo o resultado.
     */
    public Tensor mse(Tensor prev, Tensor real) {
        return new MSE().forward(prev, real);
    }

    /**
     * Calcula o valor do {@code Erro Absoluto Médio} dos dados 
     * previstos em relação aos dados reais.
     * @param prev {@code Tensor} com dados previstos.
     * @param real {@code Tensor} com dados reais.
     * @return {@code Tensor} contendo o resultado.
     */
    public Tensor mae(Tensor prev, Tensor real) {
        return new MAE().forward(prev, real);
    }

    /**
     * Calcula o valor do {@code Erro Médio Quadrado Logarítmico} dos 
     * dados previstos em relação aos dados reais.
     * @param prev {@code Tensor} com dados previstos.
     * @param real {@code Tensor} com dados reais.
     * @return {@code Tensor} contendo o resultado.
     */
    public Tensor msle(Tensor prev, Tensor real) {
        return new MSLE().forward(prev, real);
    }

    /**
     * Calcula o valor da {@code Raiz do Erro Médio Quadrático} dos 
     * dados previstos em relação aos dados reais.
     * @param prev {@code Tensor} com dados previstos.
     * @param real {@code Tensor} com dados reais.
     * @return {@code Tensor} contendo o resultado.
     */
    public Tensor rmse(Tensor prev, Tensor real) {
        return new RMSE().forward(prev, real);
    }

    /**
     * Calcula o valor da {@code Entropia Cruzada Categórica} dos 
     * dados previstos em relação aos dados reais.
     * @param prev {@code Tensor} com dados previstos.
     * @param real {@code Tensor} com dados reais.
     * @return {@code Tensor} contendo o resultado.
     */
    public Tensor entropiaCruzada(Tensor prev, Tensor real) {
        return new EntropiaCruzada().forward(prev, real);
    }

    /**
     * Calcula o valor da {@code Entropia Cruzada Binária} dos 
     * dados previstos em relação aos dados reais.
     * @param prev {@code Tensor} com dados previstos.
     * @param real {@code Tensor} com dados reais.
     * @return {@code Tensor} contendo o resultado.
     */
    public Tensor entropiaCruzadaBinaria(Tensor prev, Tensor real) {
        return new EntropiaCruzadaBinaria().forward(prev, real);
    }

    // métricas

    /**
     * Calcula o valor de acurácia dos dados previstos em relação aos
     * dados raais
     * @param prev {@code Tensor} com dados previstos.
     * @param real {@code Tensor} com dados reais.
     * @return {@code Tensor} contendo o resultado.
     */
    public Tensor acuracia(Tensor prev, Tensor real) { 
        return acuracia(
            new Tensor[]{ prev }, 
            new Tensor[]{ real }
        );
    }

    /**
     * Calcula o valor de acurácia dos dados previstos em relação aos
     * dados raais
     * @param prev {@code Tensores} com dados previstos.
     * @param real {@code Tensores} com dados reais.
     * @return {@code Tensor} contendo o resultado.
     */
    public Tensor acuracia(Tensor[] prev, Tensor[] real) { 
        return new Acuracia().forward(prev, real);
    }

    /**
     * Calcula o F1 Score dos dados previstos em relação aos
     * dados raais
     * @param prev {@code Tensor} com dados previstos.
     * @param real {@code Tensor} com dados reais.
     * @return {@code Tensor} contendo o resultado.
     */
    public Tensor f1Score(Tensor prev, Tensor real) {
        return f1Score(
            new Tensor[]{ prev }, 
            new Tensor[]{ real }
        );
    }

    /**
     * Calcula o F1 Score dos dados previstos em relação aos
     * dados raais
     * @param prev {@code Tensores} com dados previstos.
     * @param real {@code Tensores} com dados reais.
     * @return {@code Tensor} contendo o resultado.
     */
    public Tensor f1Score(Tensor[] prev, Tensor[] real) {
        return new F1Score().forward(prev, real);
    }

    /**
     * Calcula a matriz de confusão usando os dados previstos em relação aos
     * dados raais
     * @param prev {@code Tensor} com dados previstos.
     * @param real {@code Tensor} com dados reais.
     * @return {@code Tensor} contendo o resultado.
     */
    public Tensor matrizConfusao(Tensor prev, Tensor real) {
        return matrizConfusao(
            new Tensor[]{ prev }, 
            new Tensor[]{ real }
        );
    }

    /**
     * Calcula a matriz de confusão usando os dados previstos em relação aos
     * dados raais
     * @param prev {@code Tensores} com dados previstos.
     * @param real {@code Tensores} com dados reais.
     * @return {@code Tensor} contendo o resultado.
     */
    public Tensor matrizConfusao(Tensor[] prev, Tensor[] real) {
        return new MatrizConfusao().forward(prev, real);
    }

    // dicionario

    /**
     * Retorna uma ativação com base no nome informado.
     * @param act nome da ativação desejada.
     * @return {@code Ativacao} buscada.
     */
    public Ativacao getAtivacao(String act) {
        return dicionario.getAtivacao(act);
    }

    /**
     * Retorna um otimizador com base no nome informado.
     * @param otm nome do otimizador desejado.
     * @return {@code Otimizador} buscado.
     */
    public Otimizador getOtimizador(String otm) {
        return dicionario.getOtimizador(otm);
    }

    /**
     * Retorna uma função de perda com base no nome informado.
     * @param perda nome da função de perda desejado.
     * @return {@code Perda} buscada.
     */
    public Perda getPerda(String perda) {
        return dicionario.getPerda(perda);
    }

    /**
     * Retorna um inicializador com base no nome informado.
     * @param ini nome do inicializador desejado.
     * @return {@code Inicializador} buscado.
     */
    public Inicializador getInicializador(String ini) {
        return dicionario.getInicializador(ini);
    }

    // transformações de dados

    /**
     * Converte o objeto em um {@code Tensor}.
     * @param obj objeto base.
     * @return representação do objeto em um {@code Tensor}.
     */
    public Tensor paraTensor(Object obj) {
        return utils.paraTensor(obj);
    }

    /**
     * Transforma o conteúdo do array em tensores individuais.
     * @param arr array desejado.
     * @return array de {@code Tensores}
     */
    public Tensor[] arrayParaTensores(double[] arr) {
        return utils.arrayParaTensores(arr);
    }

    /**
     * Transforma o conteúdo do array 2d em tensores 1d.
     * @param arr array desejado.
     * @return array de {@code Tensores}
     */
    public Tensor[] arrayParaTensores(double[][] arr) {
        return utils.arrayParaTensores(arr);
    }

    /**
     * Transforma o conteúdo do array 3d em tensores 2d.
     * @param arr array desejado.
     * @return array de {@code Tensores}
     */
    public Tensor[] arrayParaTensores(double[][][] arr) {
        return utils.arrayParaTensores(arr);
    }

    /**
     * Transforma o conteúdo do array 4d em tensores 3d.
     * @param arr array desejado.
     * @return array de {@code Tensores}
     */
    public Tensor[] arrayParaTensores(double[][][][] arr) {
        return utils.arrayParaTensores(arr);
    }

	/**
	 * Normaliza os valores do tensor dentro do intervalo especificado.
	 * @param t {@code Tensor desejado}.
	 * @param min valor mínimo do intervalo.
	 * @param max valor máximo do intervalo.
     * @return {@code Tensor} normalizado.
     */
    public Tensor normalizar(Tensor t, double min, double max) {
        Tensor norm = new Tensor(t);
        return norm.norm(min, max);
    }
  
    // dataloader

    /**
     * Inicializa um {@code DataLoader} vazio.
     * @return {@code DataLoader}.
     * @see {@link DataLoader}
     */
    public DataLoader dataloader() {
        return new DataLoader();
    }

    /**
     * Inicializa um {@code DataLoader} a partir de um conjunto de amostras
     * de entrada (X) e de saída (Y).
     * @param X {@code array} de {@code Tensor} com dados de entrada.
     * @param Y {@code array} de {@code Tensor} com dados de saída.
     * @return {@code DataLoader}.
     * @see {@link DataLoader}
     */
    public DataLoader dataloader(Tensor[] x, Tensor[] y) {
        return new DataLoader(x, y);
    }

    /**
     * Inicializa um {@code DataLoader} a partir de um conjunto de amostras.
     * @param as {@code array} de {@code Amostra}.
     * @return {@code DataLoader}.
     * @see {@link DataLoader}
     */
    public DataLoader dataloader(Amostra[] as) {
        return new DataLoader(as);
    }

    /**
     * Inicializa um {@code DataLoader} a partir de uma amostra inicial.
     * @param a {@code Amostra} base.
     * @return {@code DataLoader}.
     * @see {@link DataLoader}
     */
    public DataLoader dataloader(Amostra a) {
        return new DataLoader(a);
    }
    
    /**
     * Inicializa um {@code DataLoader} a partir de um array de dados.
     * <h3>
     *      Nota
     * </h3>
     * <p>
     *      Esse tipo de abordagem leva em consideração que as amostras serão
     *      todas no formato de {@code arrays}, tanto em X como em Y. 
     * </p>
     * <p>
     *      Para criação de DataLoaders mais completa, use a própia 
     *      inicialização da classe DataLoader.
     * </p>
     * @param arr {@code array} base.
     * @param in quantidade de dados de entrada (X).
     * @param out quantidade de dados de saída (Y).
     * @return {@code DataLoader}.
     * @see {@link DataLoader}
     */
    public DataLoader dataloader(double[][] arr, int in, int out) {
        if (in + out > arr[0].length) {
            throw new IllegalArgumentException(
                "\nA soma de in + out deve ser igual a " + arr[0].length +
                ", mas resultou em " + (in + out)
            );
        }

        DataLoader dl = new DataLoader();

        int n = arr.length;
        for (int i = 0; i < n; i++) {
            double[] x = new double[in];
            double[] y = new double[out];

            System.arraycopy(arr[i], 0, x, 0, x.length);
            System.arraycopy(arr[i], x.length, y, 0, y.length);

            dl.add(
                new Tensor(x),
                new Tensor(y)
            );
        }

        return dl;
    }

}
