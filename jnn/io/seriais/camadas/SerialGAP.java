package jnn.io.seriais.camadas;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

import jnn.camadas.pooling.GlobalAvgPool2D;
import jnn.io.seriais.SerialBase;
import jnn.io.seriais.SerializadorCamada;

/**
 * Interface de IO para camada GlobalAvgPool2D.
 * @see jnn.camadas.pooling.GlobalAvgPool2D
 */
public class SerialGAP extends SerialBase implements SerializadorCamada<GlobalAvgPool2D> {

	/**
	 * Interface de IO para camada GlobalAvgPool2D.
	 * @see jnn.camadas.pooling.GlobalAvgPool2D
	 */
    public SerialGAP() {}

    @Override
    public void serializar(GlobalAvgPool2D camada, DataOutputStream dos) throws IOException {
        escrever(dos, camada.nome());

        int[] in = camada.shapeIn();
        escrever(dos, in);

        //nem precisa do formato de saida
    }

    @Override
    public GlobalAvgPool2D ler(DataInputStream dis) throws IOException {
        //nome ja lido
        int[] in = lerArrInt(dis);
        return new GlobalAvgPool2D(in);
    }

    @Override
    public String nome() {
        return "globalavgpool2d";
    }

    @Override
    public Class<GlobalAvgPool2D> tipo() {
        return GlobalAvgPool2D.class;
    }
    
}
