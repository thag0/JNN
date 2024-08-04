package jnn;

import jnn.ativacoes.Argmax;
import jnn.ativacoes.Ativacao;
import jnn.ativacoes.Softmax;
import jnn.avaliacao.metrica.Acuracia;
import jnn.avaliacao.metrica.F1Score;
import jnn.avaliacao.metrica.MatrizConfusao;
import jnn.avaliacao.perda.EntropiaCruzada;
import jnn.avaliacao.perda.EntropiaCruzadaBinaria;
import jnn.avaliacao.perda.MAE;
import jnn.avaliacao.perda.MSE;
import jnn.avaliacao.perda.MSLE;
import jnn.avaliacao.perda.Perda;
import jnn.avaliacao.perda.RMSE;
import jnn.core.Dicionario;
import jnn.core.Utils;
import jnn.core.tensor.OpTensor;
import jnn.core.tensor.Tensor;
import jnn.inicializadores.Identidade;
import jnn.inicializadores.Inicializador;
import jnn.otimizadores.Otimizador;

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
     * Inicializa um tensor vazio com o formato especificado.
     * @param shape formato desejado do tensor.
     * @return {@code Tensor} vazio.
     */
    public Tensor tensor(int... shape) {
        return new Tensor(shape);
    }

    /**
     * Inicializa um tensor preenchido com o valor informado.
     * @param x valor desejado.
     * @param shape formato desejado do tensor.
     * @return {@code Tensor} vazio.
     */
    public Tensor tensorConst(double x, int... shape) {
        return new Tensor(shape).preencher(x);
    }

    /**
     * Inicializa um tensor com valores da diagonal principal iguais a 1 e
     * o restante igual a zero.
     * @param n tamanho do tensor (linhas e colunas).
     * @return {@code Tensor} identidade.
     */
    public Tensor tensorId(int n) {
        if (n < 1) {
            throw new IllegalArgumentException(
                "\nTamanho do tensor deve ser maior que 1, recebido " + n
            );
        }

        Tensor t = new Tensor(n, n);
        new Identidade().inicializar(t);

        return t;
    }

    /**
     * Inicializa um tensor com valores aleatórios entre -1 e 1.
     * @param shape formato desejado do tensor.
     * @return {@code Tensor} aleatório.
     */
    public Tensor tensorRandom(int... shape) {
        Tensor t = new Tensor(shape);
        t.aplicar(x -> Math.random()*2-1);

        return t;
    }

    /**
     * Inicializa um tensor com valores aleatórios usando um gerador.
     * @param gen gerador de números pseudo-aleatórios.
     * @param shape formato desejado do tensor.
     * @return {@code Tensor} aleatório.
     */
    public Tensor tensorRandom(RandomGenerator gen, int... shape) {
        Tensor t = new Tensor(shape);
        t.aplicar(x -> gen.nextDouble(-1.0, 1.0));

        return t;
    }

    // operações

    /**
     * Realiza a operação {@code A+B}, {@code elemento a elemento}.
     * @param a {@code Tensor} A.
     * @param b {@code Tensor} B.
     * @return {@code Tensor} resultado.
     */
    public Tensor add(Tensor a, Tensor b) {
        return new Tensor(a).add(b);
    }

    /**
     * Realiza a operação {@code A-B}, {@code elemento a elemento}.
     * @param a {@code Tensor} A.
     * @param b {@code Tensor} B.
     * @return {@code Tensor} resultado.
     */
    public Tensor sub(Tensor a, Tensor b) {
        return new Tensor(a).sub(b);
    }

    /**
     * Realiza a operação {@code A*B}, {@code elemento a elemento}.
     * @param a {@code Tensor} A.
     * @param b {@code Tensor} B.
     * @return {@code Tensor} resultado.
     */
    public Tensor mul(Tensor a, Tensor b) {
        return new Tensor(a).mul(b);
    }

    /**
     * Realiza a multiplicação matricial entre A e B.
     * @param a {@code Tensor} A.
     * @param b {@code Tensor} B.
     * @return {@code Tensor} resultado.
     */
    public Tensor matmul(Tensor a, Tensor b) {
        return opt.matMul(a, b);
    }

    /**
     * Realiza a operação {@code A/B}, {@code elemento a elemento}.
     * @param a {@code Tensor} A.
     * @param b {@code Tensor} B.
     * @return {@code Tensor} resultado.
     */
    public Tensor div(Tensor a, Tensor b) {
        return new Tensor(a).div(b);
    }

    /**
     * Calcula o valor exponencial os elementos do tensor.
     * @param t {@code Tensor} desejado usado como base.
     * @param exp expoente.
     * @return {@code Tensor} resultado.
     */
    public Tensor pow(Tensor t, double exp) {
        return t.map(x -> Math.pow(x, exp));
    }

    /**
     * Retorna o valor mínimo contido no tensor.
     * @param t {@code Tensor} desejado.
     * @return {@code Tensor} resultado.
     */
    public Tensor min(Tensor t) {
        return t.min();
    }

    /**
     * Retorna o valor máximo contido no tensor.
     * @param t {@code Tensor} desejado.
     * @return {@code Tensor} resultado.
     */
    public Tensor max(Tensor t) {
        return t.max();
    }

    /**
     * Retorna o valor da média de todos os elementos do tensor.
     * @param t {@code Tensor} desejado.
     * @return {@code Tensor} resultado.
     */
    public Tensor media(Tensor t) {
        return t.media();
    }

    /**
     * Retorna o valor do desvio padrão de todos os elementos do tensor.
     * @param t {@code Tensor} desejado.
     * @return {@code Tensor} resultado.
     */
    public Tensor desvp(Tensor t) {
        return t.desvp();
    }

    /**
     * Realiza a operação {@code correlação cruzada} entre os tensores
     * A e B.
     * @param a {@code Tensor} usado como entrada.
     * @param b {@code Tensor} usado como kernel.
     * @return {@code Tensor} resultado.
     */
    public Tensor correlacao2D(Tensor a, Tensor b) {
        return opt.correlacao2D(a, b);
    }

    /**
     * Realiza a operação {@code convolucional} entre os tensores
     * A e B.
     * @param a {@code Tensor} usado como entrada.
     * @param b {@code Tensor} usado como kernel.
     * @return {@code Tensor} resultado.
     */
    public Tensor convolucao2D(Tensor a, Tensor b) {
        return opt.convolucao2D(a, b);
    }

    // funções

    /**
     * Calcula a ativação relu aos elementos do tensor.
     * @param t {@code Tensor} desejado.
     * @return {@code Tensor} com resultado aplicado.
     */
    public Tensor relu(Tensor t) {
        Tensor relu = new Tensor(t);
        return relu.relu();
    }

    /**
     * Calcula a ativação sigmoid aos elementos do tensor.
     * @param t {@code Tensor} desejado.
     * @return {@code Tensor} com resultado aplicado.
     */
    public Tensor sigmoid(Tensor t) {
        Tensor sig = new Tensor(t);
        return sig.sigmoid();
    }

    /**
     * Calcula a ativação Tangente Hiperbólica aos elementos do tensor.
     * @param t {@code Tensor} desejado.
     * @return {@code Tensor} com resultado aplicado.
     */
    public Tensor tanh(Tensor t) {
        Tensor tanh = new Tensor(t);
        return tanh.tanh();
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
        return new MSE().calcular(prev, real);
    }

    /**
     * Calcula o valor do {@code Erro Absoluto Médio} dos dados 
     * previstos em relação aos dados reais.
     * @param prev {@code Tensor} com dados previstos.
     * @param real {@code Tensor} com dados reais.
     * @return {@code Tensor} contendo o resultado.
     */
    public Tensor mae(Tensor prev, Tensor real) {
        return new MAE().calcular(prev, real);
    }

    /**
     * Calcula o valor do {@code Erro Médio Quadrado Logarítmico} dos 
     * dados previstos em relação aos dados reais.
     * @param prev {@code Tensor} com dados previstos.
     * @param real {@code Tensor} com dados reais.
     * @return {@code Tensor} contendo o resultado.
     */
    public Tensor msle(Tensor prev, Tensor real) {
        return new MSLE().calcular(prev, real);
    }

    /**
     * Calcula o valor da {@code Raiz do Erro Médio Quadrático} dos 
     * dados previstos em relação aos dados reais.
     * @param prev {@code Tensor} com dados previstos.
     * @param real {@code Tensor} com dados reais.
     * @return {@code Tensor} contendo o resultado.
     */
    public Tensor rmse(Tensor prev, Tensor real) {
        return new RMSE().calcular(prev, real);
    }

    /**
     * Calcula o valor da {@code Entropia Cruzada Categórica} dos 
     * dados previstos em relação aos dados reais.
     * @param prev {@code Tensor} com dados previstos.
     * @param real {@code Tensor} com dados reais.
     * @return {@code Tensor} contendo o resultado.
     */
    public Tensor entropiaCruzada(Tensor prev, Tensor real) {
        return new EntropiaCruzada().calcular(prev, real);
    }

    /**
     * Calcula o valor da {@code Entropia Cruzada Binária} dos 
     * dados previstos em relação aos dados reais.
     * @param prev {@code Tensor} com dados previstos.
     * @param real {@code Tensor} com dados reais.
     * @return {@code Tensor} contendo o resultado.
     */
    public Tensor entropiaCruzadaBinaria(Tensor prev, Tensor real) {
        return new EntropiaCruzadaBinaria().calcular(prev, real);
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
        return new Acuracia().calcular(prev, real);
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
        return new F1Score().calcular(prev, real);
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
        return new MatrizConfusao().calcular(prev, real);
    }

    // dicionario

    /**
     * Retorna uma ativação com base no nome informado.
     * @param atv nome da ativação desejada.
     * @return {@code Ativacao} buscada.
     */
    public Ativacao getAtivacao(String atv) {
        return dicionario.getAtivacao(atv);
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
    
}
