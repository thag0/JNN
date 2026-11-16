package jnn.io.seriais.camadas;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

import jnn.io.seriais.SerialBase;
import jnn.io.seriais.SerialLayer;
import jnn.layers.pooling.GlobalAvgPool2D;

/**
 * Interface de IO para camada GlobalAvgPool2D.
 * @see jnn.layers.pooling.GlobalAvgPool2D
 */
public class SerialGAP extends SerialBase implements SerialLayer<GlobalAvgPool2D> {

	/**
	 * Interface de IO para camada GlobalAvgPool2D.
	 * @see jnn.layers.pooling.GlobalAvgPool2D
	 */
    public SerialGAP() {}

    @Override
    public void serialize(GlobalAvgPool2D layer, DataOutputStream dos) throws IOException {
        write(dos, layer.name());

        int[] in = layer.shapeIn();
        write(dos, in);

        //nem precisa do formato de saida
    }

    @Override
    public GlobalAvgPool2D read(DataInputStream dis) throws IOException {
        //nome ja lido
        int[] in = readIntArray(dis);
        return new GlobalAvgPool2D(in);
    }

    @Override
    public String name() {
        return "globalavgpool2d";
    }

    @Override
    public Class<GlobalAvgPool2D> type() {
        return GlobalAvgPool2D.class;
    }
    
}
