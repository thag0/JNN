package jnn.dataloader.transform;

import jnn.core.JNNutils;
import jnn.core.tensor.Tensor;

/**
 * Transformação de recorte.
 */
public class CutOut implements Transform {

    /**
     * Quantidade de repetições.
     */
    private final int n;

    /**
     * Tamanho do corte.
     */
    private final int tam;

    /**
     * Valor de preenchimento.
     */
    private final float valor;

    /**
     * Inicializa uma transformação de recorte.
     * @param n Quantidade de vezes que será aplicado o corte.
     * @param tam tamanho do corte
     * @param valor valor e preenchimento na região do corte.
     */
    public CutOut(int n, int tam, float valor) {
        if (n < 0 || tam <= 0) {
            throw new IllegalArgumentException(
                "\nQuantidade deve ser maior que zero."
            );
        }
        
        if (tam <= 0) {
            throw new IllegalArgumentException(
                "\nTamanho deve ser maior que zero."
            );
        }

        this.n = n;
        this.tam = tam;
        this.valor = valor;
    }

    /**
     * Inicializa uma transformação de recorte.
     * @param tam tamanho do corte.
     */
    public CutOut(int tam) {
        this(1, tam, 0f);
    }

    @Override
    public void apply(Tensor in, Tensor out) {
        validarShapes(in, out);

        if (in.numDim() != 3) {
            throw new IllegalArgumentException(
                "\nCutout espera tensores 3D (CHW), mas recebido " + in.numDim() + "D."
            );
        }

        int canais  = in.lenDim(0);
        int altura  = in.lenDim(1);
        int largura = in.lenDim(2);
    
        out.copy(in);

        float[] dst = out.array();

        int metade = tam / 2;

        for (int i = 0; i < n; i++) {
            int cx = JNNutils.randInt(altura);
            int cy = JNNutils.randInt(largura);

            int y1 = Math.max(0, cy - metade);
            int y2 = Math.min(altura, cy + tam - metade);
            
            int x1 = Math.max(0, cx - metade);
            int x2 = Math.min(largura, cx + tam - metade);

            for (int c = 0; c < canais; c++) {
                int base = c * altura * largura;

                for (int y = y1; y < y2; y++) {
                    int inicio = base + y * largura + x1;
                    int fim =    base + y * largura + x2;
                
                    for (int j = inicio; j < fim; j++) {
                        dst[j] = valor;
                    }
                }
            }
        }
    }
    
}
