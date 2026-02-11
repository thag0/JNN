package jnn.io.seriais.camadas;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import jnn.camadas.Camada;
import jnn.io.seriais.SerializadorCamada;
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

/**
 * Interface para gravação/leitura de camadas.
 */
public class SerialCamada {

    private Map<Class<? extends Camada>, SerializadorCamada<?>> porTipo = new HashMap<>();
    private Map<String, SerializadorCamada<?>> porNome = new HashMap<>();

    /**
     * Interface para gravação/leitura de camadas.
     */
    public SerialCamada() {
        registrar(
            //camadas
            new SerialAvgPool(),
            new SerialConv(),
            new SerialDensa(),
            new SerialDropout(),
            new SerialFlatten(),
            new SerialMaxPool(),

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

    private final void registrar(SerializadorCamada<?>... seriais) {
        for (var s : seriais) {
            porTipo.put(s.tipo(), s);
            porNome.put(s.nome().toLowerCase(), s);
        }
    }
    
    /**
     * Serializa uma camada.
     * @param c {@code Camada} desejada.
     * @param dos {@code DataOutputStream} gravador.
     * @throws IOException caso ocorra algum erro.
     */
    @SuppressWarnings("unchecked")
    public void serializar(Camada c, DataOutputStream dos) throws IOException {
        var serial = porTipo.get(c.getClass());

        if (serial == null) {
            throw new UnsupportedOperationException(
                "\nCamada " + c.nome() + " sem suporte."
            );
        }

        ((SerializadorCamada<Camada>) serial).serializar(c, dos);
    }

    /**
     * Lê os dados de uma camada baseada em um nome.
     * @param dis {@code DataInputStream} leitor.
     * @param nome nome da camada.
     * @return {@code Camada} lida.
     * @throws IOException caso ocorra algum erro.
     */
    public Camada ler(DataInputStream dis, String nome) throws IOException {
        var s = porNome.get(nome.toLowerCase());

        if (s == null) {
            throw new UnsupportedOperationException(
                "\nCamada " + nome + " não suportada."
            );
        }

        return s.ler(dis);
    }

}
