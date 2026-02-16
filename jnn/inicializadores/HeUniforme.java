package jnn.inicializadores;

import jnn.core.JNNutils;
import jnn.core.tensor.Tensor;

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
