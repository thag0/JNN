package jnn.io.seriais.camadas;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import jnn.io.seriais.SerialLayer;
import jnn.io.seriais.acts.SerialELU;
import jnn.io.seriais.acts.SerialGELU;
import jnn.io.seriais.acts.SerialLeakyRelu;
import jnn.io.seriais.acts.SerialReLU;
import jnn.io.seriais.acts.SerialSELU;
import jnn.io.seriais.acts.SerialSigmoid;
import jnn.io.seriais.acts.SerialSoftPlus;
import jnn.io.seriais.acts.SerialSoftmax;
import jnn.io.seriais.acts.SerialSwish;
import jnn.io.seriais.acts.SerialTanh;
import jnn.layers.Layer;

/**
 * Interface para gravação/leitura de camadas.
 */
public class LayerSerializer {

    private Map<Class<? extends Layer>, SerialLayer<?>> perType = new HashMap<>();
    private Map<String, SerialLayer<?>> perName = new HashMap<>();

    /**
     * Interface para gravação/leitura de camadas.
     */
    public LayerSerializer() {
        register(
            //camadas
            new SerialAvgPool(),
            new SerialConv(),
            new SerialDense(),
            new SerialDropout(),
            new SerialFlatten(),
            new SerialMaxPool(),
            new SerialGAP(),
            new SerialBatchNorm(),

            //ativações
            new SerialELU(),
            new SerialGELU(),
            new SerialLeakyRelu(),
            new SerialReLU(),
            new SerialSELU(),
            new SerialSigmoid(),
            new SerialSoftmax(),
            new SerialSoftPlus(),
            new SerialSwish(),
            new SerialTanh()
        );
    }

    private final void register(SerialLayer<?>... serials) {
        for (var serial : serials) {
            perType.put(serial.type(), serial);
            perName.put(serial.name().toLowerCase(), serial);
        }
    }
    
    /**
     * Serializa uma camada.
     * @param layer {@code Camada} desejada.
     * @param dos {@code DataOutputStream} gravador.
     * @throws IOException caso ocorra algum erro.
     */
    @SuppressWarnings("unchecked")
    public void serialize(Layer layer, DataOutputStream dos) throws IOException {
        var serial = perType.get(layer.getClass());

        if (serial == null) {
            throw new UnsupportedOperationException(
                "\nCamada " + layer.name() + " sem suporte."
            );
        }

        ((SerialLayer<Layer>) serial).serialize(layer, dos);
    }

    /**
     * Lê os dados de uma camada baseada em um nome.
     * @param dis {@code DataInputStream} leitor.
     * @param nome nome da camada.
     * @return {@code Camada} lida.
     * @throws IOException caso ocorra algum erro.
     */
    public Layer read(DataInputStream dis, String nome) throws IOException {
        var serial = perName.get(nome.toLowerCase());

        if (serial == null) {
            throw new UnsupportedOperationException(
                "\nCamada " + nome + " não suportada."
            );
        }

        return serial.read(dis);
    }

}
