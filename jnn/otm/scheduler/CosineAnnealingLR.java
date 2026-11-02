package jnn.otm.scheduler;

import jnn.otm.Otimizador;

/**
 * Scheduler de decaimento da taxa de aprendizado utilizando uma função cosseno.
 */ 
public class CosineAnnealingLR extends Scheduler {

    /**
     * Taxa de aprendizado mínima.
     */
    private float lrMin;

    /**
     * Taxa de aprendizado máxima.
     */
    private float lrMax;

    /**
     * Quantidade máxima de iterações.
     */
    private long tMax;

    /**
     * Inicializa um scheduler de annealing cossenoidal.
     * @param otm otimizador alvo.
     * @param lrMin taxa de aprendizado mínima.
     * @param lrMax taxa de aprendizado máxima.
     * @param tMax número máximo de iterações realizadas durante o treino.
     * Também pode ser calculado como: tMax = épocas * (amostras / lotes).
     */
    public CosineAnnealingLR(Otimizador otm, float lrMin, float lrMax, long tMax) {
        super(otm);
        
        if (lrMax <= 0f || lrMin < 0f || lrMin >= lrMax) {
            throw new IllegalArgumentException(
                "\nLearning rates inválidos."
            );
        }
        
        if (tMax < 0) {
            throw new IllegalArgumentException(
                "\ntMax deve ser maior que zero."
            );
        }

        this.lrMin = lrMin;
        this.lrMax = lrMax;
        this.tMax = tMax;
    }

    @Override
    protected float run(long iteracao) {
        iteracao = Math.min(iteracao, tMax);

        double cos = Math.cos(Math.PI * iteracao / tMax);

        return lrMin + (lrMax - lrMin) * (float) ((1.0 + cos) * 0.5);
    }

}