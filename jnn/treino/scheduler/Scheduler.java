package jnn.treino.scheduler;

import jnn.core.JNNutils;
import jnn.otm.Otimizador;

/**
 * Scheduler base.
 */
public abstract class Scheduler {
    
    /**
     * Otimizador alvo.
     */
    protected Otimizador otm;

    /**
     * Contador interno de iterações.
     */
    protected long iteracao = 0;

    /**
     * Construtor base privado.
     * @param otm otimizador base.
     */
    protected Scheduler(Otimizador otm) {
        JNNutils.validarNaoNulo(otm, "otm == null.");
        this.otm = otm;
    }

    /**
     * Executa um passo de atualização do scheduler.
     */
    public void update() {
        iteracao++;
        float lr = run(iteracao);
        otm.setLr(lr);
    }

    /**
     * Calcula o novo valor de learning rate baseado no scheduler.
     * @param iteracao iteração atual.
     * @return novo valor de learning rate.
     */
    protected abstract float run(long iteracao);

}
