package jnn.otm.scheduler;

import jnn.core.JNNutils;
import jnn.otm.Optimizer;

/**
 * Scheduler base.
 */
public abstract class Scheduler {
    
    /**
     * Otimizador alvo.
     */
    protected Optimizer otm;

    /**
     * Contador interno de iterações.
     */
    protected long iteracao = 0;

    /**
     * Construtor base privado.
     * @param otm otimizador base.
     */
    protected Scheduler(Optimizer otm) {
        this.otm = JNNutils.validateNonNull(otm, "otm == null.");
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
