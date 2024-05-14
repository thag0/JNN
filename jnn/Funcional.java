package jnn;

import jnn.ativacoes.Argmax;
import jnn.ativacoes.Ativacao;
import jnn.ativacoes.Softmax;
import jnn.avaliacao.metrica.Acuracia;
import jnn.avaliacao.metrica.F1Score;
import jnn.avaliacao.metrica.MatrizConfusao;
import jnn.avaliacao.perda.Perda;
import jnn.core.Dicionario;
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
     * Interface para algumas funcionalidades da biblioteca.
     */
    public Funcional() {}

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

    /**
     * Realiza a operação {@code A+B}, {@code elemento a elemento}.
     * @param a {@code Tensor} A.
     * @param b {@code Tensor} B.
     * @return {@code Tensor} resultado.
     */
    public Tensor add(Tensor a, Tensor b) {
        Tensor r = new Tensor(a);
        r.add(b);

        return r;
    }

    /**
     * Realiza a operação {@code A-B}, {@code elemento a elemento}.
     * @param a {@code Tensor} A.
     * @param b {@code Tensor} B.
     * @return {@code Tensor} resultado.
     */
    public Tensor sub(Tensor a, Tensor b) {
        Tensor r = new Tensor(a);
        r.add(b);

        return r;
    }

    /**
     * Realiza a operação {@code A*B}, {@code elemento a elemento}.
     * @param a {@code Tensor} A.
     * @param b {@code Tensor} B.
     * @return {@code Tensor} resultado.
     */
    public Tensor mult(Tensor a, Tensor b) {
        Tensor r = new Tensor(a);
        r.add(b);

        return r;
    }

    /**
     * Realiza a multiplicação matricial entre A e B.
     * @param a {@code Tensor} A.
     * @param b {@code Tensor} B.
     * @return {@code Tensor} resultado.
     */
    public Tensor matmult(Tensor a, Tensor b) {
        return opt.matMult(a, b);
    }

    /**
     * Realiza a operação {@code A/B}, {@code elemento a elemento}.
     * @param a {@code Tensor} A.
     * @param b {@code Tensor} B.
     * @return {@code Tensor} resultado.
     */
    public Tensor div(Tensor a, Tensor b) {
        Tensor r = new Tensor(a);
        r.div(b);

        return r;
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
        return new Acuracia().calcular(prev, real).nome("acurácia");
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
        return new F1Score().calcular(prev, real).nome("f1 score");
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
        return new MatrizConfusao().calcular(prev, real).nome("matriz confusão");
    }

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

}
