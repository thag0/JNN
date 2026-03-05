package jnn.camadas;

import jnn.core.JNNutils;
import jnn.core.tensor.Tensor;

/**
 * Experimental ainda
 */
public class BatchNorm2D extends Camada implements Cloneable {

    private float eps;
    private float momentum;
    
    public Tensor _gamma;// escala (da pra chamar de kernel)
    public Tensor _beta;// deslocamento (também da pra chamar de bias)
    public Tensor _runningMean;// media movel (para inferencia)
    public Tensor _runningVar;// variancia movel (para inferencia)
    public Tensor _entradaNorm;// x chapeu (xHat) pro backward
    public Tensor _media;// media por canal
    public Tensor _var;// variancia por canal
    public Tensor _gradGamma;
    public Tensor _gradBeta;
    public Tensor _entrada;
    public Tensor _gradEntrada;
    public Tensor _saida;
    public Tensor _gradSaida;
    
    private int tamLote;
    private int[] shapeIn = {1, 1, 1};

    public BatchNorm2D(Number momentum, Number eps) {
        float m = momentum.floatValue();
        float e = eps.floatValue();

        if (m < 0) {
            throw new IllegalArgumentException(
                "\nValor de momentum deve ser maior que zero, recebido " + m
            );
        }

        if (e < 0) {
            throw new IllegalArgumentException(
                "\nValor de epsilon deve ser maior que zero, recebido " + e
            );
        }

        this.momentum = m;
        this.eps = e;
    }

    public BatchNorm2D() {
        this(0.1, 1e-5);
    }

    @Override
    public void construir(int[] shape) {
        JNNutils.validarNaoNulo(shape, "shape == null");

        if (shape.length != 3) {
            throw new IllegalArgumentException("\nShape deve ter 3 dimensões (C, H, W)");
        }

        if (!JNNutils.apenasMaiorZero(shape)) {
            throw new IllegalArgumentException("\nShape deve conter valores maiores que zero");
        }

        shapeIn[0] = shape[0];//canais
        shapeIn[1] = shape[1];//altura
        shapeIn[2] = shape[2];//largura

        int canais = shapeIn[0];

        _gamma = addParam("Gamma", canais);
        _beta = addParam("Beta", canais);
        _gradGamma = addGrad("Grad Gamma", canais);
        _gradBeta = addGrad("Grad Beta", canais);

        _gradEntrada = addBuffer("Grad Entrada", shape);
        _saida = addBuffer("Saida", shape);
        _entradaNorm = addBuffer("Entrada Normalizada", shape);

        _runningMean = addBuffer("Media movel", canais);
        _runningVar = addBuffer("Variancia Movel", canais);

        _media = addBuffer("Media", canais);
        _var = addBuffer("Variancia", canais);

		_treinavel = true;// camada pode ser treinada.
		_construida = true;// camada pode ser usada.
    }

    @Override
    public void inicializar() {
        verificarConstrucao();
        
        _gamma.preencher(1);
        _beta.preencher(0);

        _runningMean.preencher(0);
        _runningVar.preencher(1);
    }

    @Override
    public void ajustarParaLote(int tamLote) {
        int[] shape = {};
        
        if (tamLote == 0) {
            shape = shapeIn;
            
        } else {
            shape = new int[shapeIn.length + 1];
            shape[0] = tamLote;
            for (int i = 0; i < shapeIn.length; i++) {
                shape[i+1] = shapeIn[i];
            }
        }
        
        _entradaNorm = addBuffer("Entrada Normalizada", shape);
        _saida       = addBuffer("Saida", shape);
        _gradEntrada = addBuffer("Grad Entrada", shape);

        this.tamLote = tamLote;
    }

    @Override
    public Tensor forward(Tensor x) {
        verificarConstrucao();

        boolean entrada3D = (x.numDim() == 3);
        if (entrada3D) {
            validarShapes(x.shape(), shapeIn);
            ajustarParaLote(0);

        } else if (x.numDim() == 4) {
            validarShapes(x.shape(), shapeIn);
            int lote = x.tamDim(0);
            if (this.tamLote != lote) ajustarParaLote(lote);  
            
        } else {
            throw new IllegalArgumentException(
                "\nEntrada deve ter 3 (C, H, W) ou 4 dimensões (N, C, H, W), recebido " + x.numDim()
            );
        }
        
        _entrada = x.contiguous();

        int lotes = tamLote == 0 ? 1 : tamLote;
        int canais = shapeIn[0];
        int altX = shapeIn[1];
        int largX = shapeIn[2];
        int areaX = altX * largX;
        int M = lotes * areaX;

        float[] dataX = _entrada.array();
        float[] dataY = _saida.array();
        float[] dataGamma = _gamma.array();
        float[] dataBeta  = _beta.array();
        float[] dataRM = _runningMean.array();
        float[] dataRV = _runningVar.array();

        if (treinando) {
            float[] media = _media.array();
            float[] variancia  = _var.array();
            float[] xNorm = _entradaNorm.array();

            for (int c = 0; c < canais; c++) {
                double soma = 0;
                for (int n = 0; n < lotes; n++) {
                    int base = n * canais * areaX + c * areaX;
                    for (int i = 0; i < areaX; i++) {
                        soma += dataX[base + i];
                    }
                }
                media[c] = (float) soma / M;
            }

            for (int c = 0; c < canais; c++) {
                float med = media[c];
                double soma = 0;
                
                for (int n = 0; n < lotes; n++) {
                    int base = n * canais * areaX + c * areaX;
                    for (int i = 0; i < areaX; i++) {
                        float dif = dataX[base + i] - med;
                        soma += dif * dif;
                    }
                }
                variancia[c] = (float) soma / M;
            }

            for (int c = 0; c < canais; c++) {
                float invStd = 1.0f / (float)Math.sqrt(variancia[c] + eps);
                float g = dataGamma[c];
                float b = dataBeta[c];
                float m = media[c];

                for (int n = 0; n < lotes; n++) {
                    int base = n * canais * areaX + c * areaX;
                    for (int i = 0; i < areaX; i++) {
                        int id = base + i;
                        float norm = (dataX[id] - m) * invStd;
                        xNorm[id] = norm;
                        dataY[id]  = g * norm + b;
                    }
                }

                float varDesviezada = (M > 1) ? (variancia[c] * M / (M - 1f)) : variancia[c];
                dataRM[c] = (1 - momentum) * dataRM[c] + momentum * media[c];
                dataRV[c] = (1 - momentum) * dataRV[c]  + momentum * varDesviezada;
            }

        } else {
            for (int c = 0; c < canais; c++) {
                float invStd = 1.0f / (float)Math.sqrt(dataRV[c] + eps);
                float g = dataGamma[c];
                float b = dataBeta[c];
                float m = dataRM[c];

                for (int n = 0; n < lotes; n++) {
                    int base = n*canais*areaX + c*areaX;

                    for (int i = 0; i < areaX; i++) {
                        int id = base + i;
                        dataY[id] = g * (dataX[id] - m) * invStd + b;
                    }
                }
            }
        }

        return _saida;
    }

    @Override
    public Tensor backward(Tensor g) {
        verificarConstrucao();

        if (!treinando) {
            throw new IllegalStateException("\nBackward não deve ser chamado em modo avaliação.");
        }

        _gradSaida = g.contiguous();
        _gradEntrada.zero();

        int lotes = tamLote;
        int canais = shapeIn[0];
        int altura = shapeIn[1];
        int largura = shapeIn[2];
        int areaX = altura * largura;
        int M = lotes * areaX;

        float[] dataGS = _gradSaida.array();
        float[] dataXNorm = _entradaNorm.array();
        float[] dataVar = _var.array();
        float[] dataGamma = _gamma.array();

        float[] dataGE = _gradEntrada.array();
        float[] dataGG = _gradGamma.array();
        float[] dataGB = _gradBeta.array();

        for (int c = 0; c < canais; c++) {
            double somaGamma = 0;
            double somaBeta  = 0;

            for (int n = 0; n < lotes; n++) {
                int base = n * canais * areaX + c * areaX;

                for (int i = 0; i < areaX; i++) {
                    int id = base + i;
                    float grad = dataGS[id];
                    somaGamma += grad * dataXNorm[id];
                    somaBeta  += grad;
                }
            }

            dataGG[c] += (float) somaGamma;
            dataGB[c] += (float) somaBeta;
        }

        for (int c = 0; c < canais; c++) {
            float valGamma = dataGamma[c];
            float invStd = 1.0f / (float)Math.sqrt(dataVar[c] + eps);

            double somaG = 0;
            double somaGENorm = 0;

            for (int n = 0; n < lotes; n++) {
                int base = n * canais * areaX + c * areaX;

                for (int i = 0; i < areaX; i++) {
                    int id = base + i;
                    float grad = dataGS[id];
                    somaG += grad;
                    somaGENorm += grad * dataXNorm[id];
                }
            }

            float coef = valGamma * invStd / M;

            for (int n = 0; n < lotes; n++) {
                int base = n * canais * areaX + c * areaX;

                for (int i = 0; i < areaX; i++) {
                    int idx = base + i;
                    float grad = dataGS[idx];
                    float xNorm = dataXNorm[idx];

                    dataGE[idx] = (float) (coef * (M * grad - somaG - xNorm * somaGENorm));
                }
            }
        }

        return _gradEntrada;
    }
    
    @Override
    public void gradZero() {
        verificarConstrucao();

        _gradGamma.preencher(0);
        _gradBeta.preencher(0);
    }

    @Override
    public int[] shapeIn() {
        verificarConstrucao();
        return shapeIn.clone();
    }

    @Override
    public int[] shapeOut() {
        return shapeIn();
    }

    @Override
    public int numParams() {
        return _gamma.tam() + _beta.tam();
    }

    @Override
    public Tensor saida() {
        verificarConstrucao();
        return _saida;
    }

    @Override
    public Tensor gradEntrada() {
        verificarConstrucao();
        return _gradEntrada;
    }

    public float eps() {
        return eps;
    }

    public float momentum() {
        return momentum;
    }

    @Override
    public Camada clone() {
        try {
            BatchNorm2D clone = (BatchNorm2D) super.clone();
            
            clone._gamma = _gamma.clone();
            clone._beta = _beta.clone();
            clone._runningMean = _runningMean.clone();
            clone._runningVar = _runningVar.clone();
            clone._entradaNorm = _entradaNorm.clone();
            clone._media = _media.clone();
            clone._var = _var.clone();
            clone._gradGamma = _gradGamma.clone();
            clone._gradBeta = _gradBeta.clone();
            clone._gradEntrada = _gradEntrada.clone();
            clone._saida = _saida.clone();
            
            clone.shapeIn = shapeIn.clone();
            
            return clone;

        } catch (Exception e) {
            throw e;
        }
    }

}