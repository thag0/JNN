package jnn.treino.scheduler;

import jnn.otm.Otimizador;

/**
 * Scheduler de decaimento por gamma a cada passo de iteração.
 */
public class StepLR extends Scheduler {

    /**
     * Lr configurado no otimizador antes do treino.
     */
    private float lrBase;

    /**
     * Contador interno de passos.
     */
    private int passo;

    /**
     * Valor multiplicativo de decaimento.
     */
    private float gamma;
    
    /**
     * Inicializa um scheduler de passos.
     * @param otm otimizador alvo.
     * @param passo passo de iteração.
     * @param gamma escalar multiplicador de decaimento.
     */
    public StepLR(Otimizador otm, int passo, float gamma) {
        super(otm);
        this.lrBase = otm.getLr();
        this.passo = passo;
        this.gamma = gamma;
    }

    @Override
    protected float run(long iterecao) {
        int exp = (int) (iterecao / passo);
        return (float) (lrBase * Math.pow(gamma, exp));
    }

}
