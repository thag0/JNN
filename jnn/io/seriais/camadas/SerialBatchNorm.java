package jnn.io.seriais.camadas;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

import jnn.core.Parameter;
import jnn.io.seriais.SerialBase;
import jnn.io.seriais.SerialLayer;
import jnn.layers.BatchNorm2D;

/**
 * Interface de IO para camada BatchNorm2D.
 * @see jnn.layers.BatchNorm2D
 */
public class SerialBatchNorm extends SerialBase implements SerialLayer<BatchNorm2D> {

	/**
	 * Interface de IO para camada MaxPool2D.
	 * @see jnn.layers.pooling.AvgPool2D
	 */
    public SerialBatchNorm() {}

    @Override
    public void serialize(BatchNorm2D layer, DataOutputStream dos) throws IOException {
        write(dos, layer.name());

        write(dos, layer.shapeIn());
        write(dos, layer.eps());
        write(dos, layer.momentum());

        Parameter[] params = layer.params();
        write(dos, params[0].weight.data().toArray());//gamma
        write(dos, params[1].weight.data().toArray());//beta
    
        write(dos, layer._runningMean.data().toArray());
        write(dos, layer._runningVar.data().toArray());
    }

    @Override
    public BatchNorm2D read(DataInputStream dis) throws IOException {
        int[] shapeIn = readIntArray(dis);
        float eps = readFloat(dis);
        float momentum = readFloat(dis);

        float[] gamma = readFloatArray(dis);
        float[] beta = readFloatArray(dis);
        float[] runMean = readFloatArray(dis);
        float[] runVar = readFloatArray(dis);
    
        BatchNorm2D layer = new BatchNorm2D(momentum, eps);
        layer.build(shapeIn);

        Parameter[] params = layer.params();
        params[0].weight.copyElements(gamma);
        params[1].weight.copyElements(beta);
    
        layer._runningMean.copyElements(runMean);
        layer._runningVar.copyElements(runVar);

        return layer;
    }

    @Override
    public String name() {
        return "batchnorm2d";
    }

    @Override
    public Class<BatchNorm2D> type() {
        return BatchNorm2D.class;
    }
    
}
