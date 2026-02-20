package jnn.io.seriais;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

import jnn.camadas.Camada;

/**
 * Interface base para camadas serializáveis.
 * @param <T> tipo da camada
 */
public interface SerializadorCamada <T extends Camada> {
    
    /**
     * Retorna o nome da camada.
     * @return nome da camamda.
     */
    String nome();

    /**
     * Retorna a classe da camada.
     * @return classe da camada.
     */
    Class<T> tipo();

    /**
     * Serializa os dados importantes da camada.
     * @param camada camada base.
     * @param dos escritor de dados.
     * @throws IOException caso ocorra algum erro.
     */
    void serializar(T camada, DataOutputStream dos) throws IOException;

    /**
     * Cria uma camada a partir de dados serializados.
     * @param dis leitor de dados.
     * @return camada inicializada.
     * @throws IOException caso ocorra algum erro.
     */
    T ler(DataInputStream dis) throws IOException;

}
