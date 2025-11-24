package jnn.io.seriais;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

import jnn.camadas.Camada;
import jnn.camadas.Conv2D;
import jnn.camadas.Densa;
import jnn.camadas.Dropout;
import jnn.camadas.Flatten;
import jnn.camadas.pooling.AvgPool2D;
import jnn.camadas.pooling.MaxPool2D;

/**
 * Interface para gravação/leitura de camadas.
 */
public class SerialCamada {

    /**
     * Auxiliar.
     */
    SerialDensa sDensa = new SerialDensa();

    /**
     * Auxiliar.
     */
    SerialConv sConv = new SerialConv();

    /**
     * Auxiliar.
     */
    SerialMaxPool sMaxPool = new SerialMaxPool();

    /**
     * Auxiliar.
     */
    SerialAvgPool sAvgPool = new SerialAvgPool();

    /**
     * Auxiliar.
     */
    SerialFlatten sFlatten = new SerialFlatten();

    /**
     * Auxiliar.
     */
    SerialDropout sDropout = new SerialDropout();

    /**
     * Interface para gravação/leitura de camadas.
     */
    public SerialCamada() {}
    
    /**
     * Serializa uma camada.
     * @param c {@code Camada} desejada.
     * @param dos {@code DataOutputStream} gravador.
     * @throws IOException caso ocorra algum erro.
     */
    public void serializar(Camada c, DataOutputStream dos) throws IOException {
        // ta feio mas funciona

        if (c instanceof Densa) {
            sDensa.serializar((Densa) c, dos);
        
        } else if (c instanceof Conv2D) {
            sConv.serializar((Conv2D) c, dos);

        } else if (c instanceof MaxPool2D) {
            sMaxPool.serializar((MaxPool2D) c, dos);

        } else if (c instanceof AvgPool2D) {
            sAvgPool.serializar((AvgPool2D) c, dos);

        } else if (c instanceof Flatten) {
            sFlatten.serializar((Flatten) c, dos);

        } else if (c instanceof Dropout) {
            sDropout.serializar((Dropout) c, dos);

        } else {
            throw new UnsupportedOperationException(
                "\nCamada " + c.nome() + " sem suporte."
            );
        }
    }

    /**
     * Lê os dados de uma camada baseada em um nome.
     * @param dis {@code DataInputStream} leitor.
     * @param nome nome da camada.
     * @return {@code Camada} lida.
     * @throws IOException caso ocorra algum erro.
     */
    public Camada ler(DataInputStream dis, String nome) throws IOException {
        Camada c = null;

        switch (nome.toLowerCase()) {
            case "densa":       c = sDensa.ler(dis);    break;
            case "conv2d":      c = sConv.ler(dis);     break;
            case "maxpool2d":   c = sMaxPool.ler(dis);  break;
            case "avgpool2d":   c = sAvgPool.ler(dis);  break;
            case "flatten":     c = sFlatten.ler(dis);  break;
            case "dropout":     c = sDropout.ler(dis);  break;

            default:
                throw new UnsupportedOperationException(
                    "\nCamada " + nome + " não suportada."
                );
        }

        return c;
    }

}
