package jnn.dataloader.transform;

import jnn.core.JNNutils;
import jnn.core.tensor.Tensor;

/**
 * Transformação de normalização.
 */
public class TNorm implements Transform {

    /**
     * Média para cada canal.
     */
    private float[] media;
    
    /** 
     * Desvio padrão para cada canal.
     */
    private float[] desvp;

    /**
     * 
     * @param media
     * @param desvp
     */
    public TNorm(float[] media, float[] desvp) {
        JNNutils.validarNaoNulo(media, "media == null");
        JNNutils.validarNaoNulo(desvp, "desvp == null");

        if (media.length != desvp.length) {
            throw new IllegalArgumentException(
                "\nMedia e Desvp devem ter o mesmo tamanho."
            );
        }

        if (!JNNutils.apenasMaiorZero(desvp)) {
            throw new IllegalArgumentException(
                "\nDesvp não pode conter valores menores que zero."
            );
        }

        this.media = media;
        this.desvp = desvp;
    }

    @Override
    public Tensor apply(Tensor t) {
        final int[] shape = t.shape();
        final int dims = shape.length;

        int canais;

        if (dims == 1) {
            canais = shape[0];
        
        } else if (dims == 2) {
            canais = shape[1];
    
        } else if (dims == 3) {
            canais = shape[0];
        
        } else if (dims == 4) {
            canais = shape[1];
        
        } else {
            throw new IllegalArgumentException(
                "\nSuporte até tensor 4D, recebido " + t.numDim() + "D."
            );
        }

        if (canais != media.length) {
            throw new IllegalArgumentException(
                "\nNúmero de canais do tensor (" + canais +
                ") não corresponde ao tamanho de media/desvp (" +
                media.length + "). Shape=" + t.shapeStr()
            );
        }

        Tensor norm = t.clone();
        float[] data = norm.array();

        int stdCanal;
        int stdAmostra;

        if (dims == 1) {
            stdCanal = 1;
            stdAmostra = canais;
        
        } else if (dims == 2) {
            stdCanal = 1;
            stdAmostra = canais;
        
        } else if (dims == 3) {
            stdCanal = shape[1] * shape[2];
            stdAmostra = canais * stdCanal;
        
        } else {
            stdCanal = shape[2] * shape[3];
            stdAmostra = canais * stdCanal;
        }

        final int total = data.length;

        for (int baseAmostra = 0; baseAmostra < total; baseAmostra += stdAmostra) {
            for (int c = 0; c < canais; c++) {
                final float m = media[c];
                final float sd = desvp[c];

                int inicio = baseAmostra + c * stdCanal;
                int fim = inicio + stdCanal;

                for (int i = inicio; i < fim; i++) {
                    data[i] = (data[i] - m) / sd;
                }
            }
        }

        return norm;
    }
    
}
