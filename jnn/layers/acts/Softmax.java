package jnn.layers.acts;

import jnn.core.tensor.Tensor;

/**
 * Camada que opera a função de ativação Softmax.
 */
public class Softmax extends Activation {
    
    /**
     * Inicializa uma camada de ativação Softmax.
     */
    public Softmax() {}

    /**
     * Inicializa uma camada de ativação Softmax.
     * @param input formato de entrada para a camada.
     */
    public Softmax(int... input) {
        build(input);
    }

    @Override
    public Tensor forward(Tensor x) {
        validateBuid();

        final int numDim = x.numDim();
        int numSamples = 1;

        if (numDim == baseDim) {
            validateShapes(x.shape(), shapeIn);
            if (this.batchSize != 0) fitToBatch(0);
            
        } else if (numDim == baseDim + 1) {
            validateShapes(x.shape(), shapeIn);
            int lotes = x.lenDim(0);
            if (lotes != this.batchSize) fitToBatch(lotes);
            
            numSamples = lotes;
        
        } else {
            throw new UnsupportedOperationException(
                "Esperado tensor com " + baseDim +
                " ou " + (baseDim + 1) +
                " dimensões. Recebido: " + x.numDim()
            );
        }

        _input = x.contiguous();

        int arrayLen = _input.len() / numSamples;

        softmax(
            _input.array(), 
            _output.array(),
            numSamples,
            arrayLen
        );

        return _output;
    }

    @Override
    public Tensor backward(Tensor g) {
        validateBuid();

        _gradOutput = g.contiguous();
        
        int totalElementos = _gradOutput.len();
        int tamVetor = 1;
        for (int dim : shapeIn) tamVetor *= dim;
        int samples = totalElementos / tamVetor;

        softmaxGrad(
            _gradOutput.array(), 
            _gradInput.array(), 
            _output.array(),
            samples,
            tamVetor
        );

        return _gradInput;
    }

    private void softmax(float[] input, float[] output, int samples, int arrayLen) {
        // o valor maximo é subtraido pra ter melhor estabilidade numerica
        
        for (int i = 0; i < samples; i++) {
            int start = i * arrayLen;
            
            float max = Float.NEGATIVE_INFINITY;
            for (int j = 0; j < arrayLen; j++) {
                float val = input[start + j];
                if (val > max) max = val;
            }

            float sum = 0;
            for (int j = 0; j < arrayLen; j++) {
                float exp = (float) Math.exp(input[start + j] - max);
                output[start + j] = exp;
                sum += exp;
            }

            for (int j = 0; j < arrayLen; j++) {
                output[start + j] /= sum;
            }
        }
    }

    private void softmaxGrad(float[] gOut, float[] gIn, float[] output, int samples, int arrayLen) {
        for (int i = 0; i < samples; i++) {
            int start = i * arrayLen;

            float sum = 0.f;
            for (int j = 0; j < arrayLen; j++) {
                sum += gOut[start + j] * output[start + j];
            }

            for (int j = 0; j < arrayLen; j++) {
                float out = output[start + j];
                float grad = gOut[start + j];
                
                gIn[start + j] = out * (grad - sum);
            }
        }
    }

}
