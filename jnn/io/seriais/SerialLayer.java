package jnn.io.seriais;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

import jnn.layers.Layer;

/**
 * Interface base para camadas serializáveis.
 * @param <T> tipo da camada
 */
public interface SerialLayer <T extends Layer> {
    
    /**
     * Retorna o nome da camada.
     * @return nome da camamda.
     */
    String name();

    /**
     * Retorna a classe da camada.
     * @return classe da camada.
     */
    Class<T> type();

    /**
     * Serializa os dados importantes da camada.
     * @param camada camada base.
     * @param dos escritor de dados.
     * @throws IOException caso ocorra algum erro.
     */
    void serialize(T camada, DataOutputStream dos) throws IOException;

    /**
     * Cria uma camada a partir de dados serializados.
     * @param dis leitor de dados.
     * @return camada inicializada.
     * @throws IOException caso ocorra algum erro.
     */
    T read(DataInputStream dis) throws IOException;

}
