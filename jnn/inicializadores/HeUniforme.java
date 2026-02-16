package jnn.inicializadores;

import jnn.core.JNNutils;
import jnn.core.tensor.Tensor;

/**
 * Inicializador He Uniforme (Kaiming Uniform).
 * 
 * <p>
 *		Indicado para modelos com funções de ativação ReLU e variações,
 *		pois mantém a variância das ativações e dos gradientes estável
 *		ao longo das camadas. Diferente da versão normal (gaussiana),
 *		os valores são amostrados de uma distribuição uniforme no
 *		intervalo [-limite, limite], onde limite = sqrt(6 / fanIn).
 * </p>
 */
public class HeUniforme extends Inicializador {

    public HeUniforme() {}

    @Override
    public void forward(Tensor t) {
		int fanIn = calcularFans(t)[0];
		float limit = (float) Math.sqrt(6.0 / fanIn);
		
		t.aplicar(_ -> randUniform(-limit, limit));
    }

	private float randUniform(float min, float max) {
		return min + (max - min) * JNNutils.randFloat();
	}
    
}
