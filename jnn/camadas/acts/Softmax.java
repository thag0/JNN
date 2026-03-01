package jnn.camadas.acts;

import jnn.core.tensor.Tensor;

/**
 * Camada que opera a função de ativação Softmax.
 */
public class Softmax extends Ativacao {
    
    /**
     * Inicializa uma camada de ativação Softmax.
     */
    public Softmax() {}

    /**
     * Inicializa uma camada de ativação Softmax.
     * @param shape formato de entrada para a camada.
     */
    public Softmax(int... shape) {
        construir(shape);
    }

    @Override
    public Tensor forward(Tensor x) {
        verificarConstrucao();

        final int numDim = x.numDim();
        int numAmostras = 1;

        if (numDim == dimBase) {
            validarShapes(x.shape(), shapeIn);
            if (_tamLote != 0) ajustarParaLote(0);
            
        } else if (numDim == dimBase + 1) {
            validarShapes(x.shape(), shapeIn);
            int lotes = x.tamDim(0);
            if (lotes != this._tamLote) ajustarParaLote(lotes);
            
            numAmostras = lotes;
        
        } else {
            throw new UnsupportedOperationException(
                "Esperado tensor com " + dimBase +
                " ou " + (dimBase + 1) +
                " dimensões. Recebido: " + x.numDim()
            );
        }

        _entrada = x.contiguous();

        int tamArray = _entrada.tam() / numAmostras;

        softmax(
            _entrada.array(), _entrada.offset(),
            _saida.array(), _saida.offset(),
            numAmostras,
            tamArray
        );

        return _saida;
    }

    @Override
    public Tensor backward(Tensor g) {
        verificarConstrucao();

        _gradSaida = g.contiguous();
        
        int totalElementos = _gradSaida.tam();
        int tamVetor = 1;
        for(int s : shapeIn) tamVetor *= s;
        int samples = totalElementos / tamVetor;

        softmaxGrad(
            _gradSaida.array(), _gradSaida.offset(),
            _gradEntrada.array(), _gradEntrada.offset(),
            _saida.array(), _saida.offset(),
            samples,
            tamVetor
        );

        return _gradEntrada;
    }

    private void softmax(float[] x, int offX, float[] dest, int offDest, int samples, int tamArray) {
        // o valor maximo é subtraido pra ter melhor estabilidade numerica
        
        for (int i = 0; i < samples; i++) {
            int inicio = i * tamArray;
            
            float max = Float.NEGATIVE_INFINITY;
            for (int j = 0; j < tamArray; j++) {
                float val = x[offX + inicio + j];
                if (val > max) max = val;
            }

            float soma = 0;
            for (int j = 0; j < tamArray; j++) {
                float exp = (float) Math.exp(x[offX + inicio + j] - max);
                dest[offDest + inicio + j] = exp;
                soma += exp;
            }

            for (int j = 0; j < tamArray; j++) {
                dest[offDest + inicio + j] /= soma;
            }
        }
    }

    private void softmaxGrad(float[] g, int offG, float[] gradE, int offGE, float[] y, int offY, int amostras, int tamArray) {
        for (int i = 0; i < amostras; i++) {
            int inicio = i * tamArray;

            float soma = 0.f;
            for (int j = 0; j < tamArray; j++) {
                soma += g[offG + inicio + j] * y[offY + inicio + j];
            }

            for (int j = 0; j < tamArray; j++) {
                float vSaida = y[offY + inicio + j];
                float vGrad = g[offG + inicio + j];
                
                gradE[offGE + inicio + j] = vSaida * (vGrad - soma);
            }
        }
    }

}
