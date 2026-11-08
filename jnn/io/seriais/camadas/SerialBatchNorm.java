package jnn.io.seriais.camadas;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

import jnn.core.Parametro;
import jnn.io.seriais.SerialBase;
import jnn.io.seriais.SerializadorCamada;
import jnn.layers.BatchNorm2D;

/**
 * Interface de IO para camada BatchNorm2D.
 * @see jnn.layers.BatchNorm2D
 */
public class SerialBatchNorm extends SerialBase implements SerializadorCamada<BatchNorm2D> {

	/**
	 * Interface de IO para camada MaxPool2D.
	 * @see jnn.layers.pooling.AvgPool2D
	 */
    public SerialBatchNorm() {}

    @Override
    public void serializar(BatchNorm2D camada, DataOutputStream dos) throws IOException {
        escrever(dos, camada.name());

        escrever(dos, camada.shapeIn());
        escrever(dos, camada.eps());
        escrever(dos, camada.momentum());

        Parametro[] params = camada.params();
        escrever(dos, params[0].weight.data().toArray());//gamma
        escrever(dos, params[1].weight.data().toArray());//beta
    
        escrever(dos, camada._runningMean.data().toArray());
        escrever(dos, camada._runningVar.data().toArray());
    }

    @Override
    public BatchNorm2D ler(DataInputStream dis) throws IOException {
        int[] shapeIn = lerArrInt(dis);
        float eps = lerFloat(dis);
        float momentum = lerFloat(dis);

        float[] gamma = lerArrFloat(dis);
        float[] beta = lerArrFloat(dis);
        float[] runMean = lerArrFloat(dis);
        float[] runVar = lerArrFloat(dis);
    
        BatchNorm2D bn = new BatchNorm2D(momentum, eps);
        bn.build(shapeIn);

        Parametro[] params = bn.params();
        params[0].weight.copyElements(gamma);
        params[1].weight.copyElements(beta);
    
        bn._runningMean.copyElements(runMean);
        bn._runningVar.copyElements(runVar);

        return bn;
    }

    @Override
    public String nome() {
        return "batchnorm2d";
    }

    @Override
    public Class<BatchNorm2D> tipo() {
        return BatchNorm2D.class;
    }
    
}
