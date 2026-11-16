package jnn.io.seriais;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

/**
 * Base de serialização de elementos da biblioteca.
 */
public abstract class SerialBase {

    /**
     * Base de serialização de elementos da biblioteca.
     */
    protected SerialBase() {}

    /**
     * Grava o conteúdo de um valor primitivo {@code int}.
     * @param dos {@code DataOutputStream} gravador.
     * @param val valor desejado.
     * @throws IOException caso ocorra um erro.
     */
    static protected void write(DataOutputStream dos, int val) throws IOException {
        dos.writeInt(val);
    }

    /**
     * Grava o conteúdo de um valor primitivo {@code float}.
     * @param dos {@code DataOutputStream} gravador.
     * @param val valor desejado.
     * @throws IOException caso ocorra um erro.
     */
    static protected void write(DataOutputStream dos, float val) throws IOException {
        dos.writeFloat(val);
    }

    /**
     * Grava o conteúdo de um valor primitivo {@code boolean}.
     * @param dos {@code DataOutputStream} gravador.
     * @param val valor desejado.
     * @throws IOException caso ocorra um erro.
     */
    static protected void write(DataOutputStream dos, boolean val) throws IOException {
        dos.writeBoolean(val);
    }

    /**
     * Grava o conteúdo de um array primitivo {@code int[]}.
     * @param dos {@code DataOutputStream} gravador.
     * @param arr {@code array} desejado.
     * @throws IOException caso ocorra um erro.
     */
    static protected void write(DataOutputStream dos, int[] arr) throws IOException {
        dos.writeInt(arr.length);
        for (int val : arr) {
            dos.writeInt(val);
        }
    }

    /**
     * Grava o conteúdo de um array primitivo {@code float[]}.
     * @param dos {@code DataOutputStream} gravador.
     * @param arr {@code array} desejado.
     * @throws IOException caso ocorra um erro.
     */
    static protected void write(DataOutputStream dos, float[] arr) throws IOException {
        dos.writeInt(arr.length);
        for (float val : arr) {
            dos.writeFloat(val);
        }
    }

    /**
     * Grava o conteúdo de uma {@code String}.
     * @param dos {@code DataOutputStream} gravador.
     * @param s {@code String} desejada.
     * @throws IOException caso ocorra um erro.
     */
    static protected void write(DataOutputStream dos, String s) throws IOException {
        byte[] bytes = s.getBytes("UTF-8");
        dos.writeInt(bytes.length);
        dos.write(bytes);
    }

    /**
     * Lê o conteúdo de um valor primitivo {@code int}.
     * @param dis {@code DataInputStream} leitor.
     * @return valor lido.
     * @throws IOException caso ocorra um erro.
     */
    static protected int readInt(DataInputStream dis) throws IOException {
        return dis.readInt();
    }

    /**
     * Lê o conteúdo de um valor primitivo {@code float}.
     * @param dis {@code DataInputStream} leitor.
     * @return valor lido.
     * @throws IOException caso ocorra um erro.
     */
    static protected float readFloat(DataInputStream dis) throws IOException {
        return dis.readFloat();
    }

    /**
     * Lê o conteúdo de um valor primitivo {@code boolean}.
     * @param dis {@code DataInputStream} leitor.
     * @return valor lido.
     * @throws IOException caso ocorra um erro.
     */
    static protected boolean readBoolean(DataInputStream dis) throws IOException {
        return dis.readBoolean();
    }

    /**
     * Lê o conteúdo de um array primitivo {@code int[]}.
     * @param dis {@code DataInputStream} leitor.
     * @return array lido.
     * @throws IOException caso ocorra um erro.
     */
    static protected int[] readIntArray(DataInputStream dis) throws IOException {
        int len = dis.readInt();// considerando que já escreve o tamanho.
        
        int[] arr = new int[len];
        for (int i = 0; i < len; i++) {
            arr[i] = dis.readInt();
        }

        return arr;
    }

    /**
     * Lê o conteúdo de um array primitivo {@code float[]}.
     * @param dis {@code DataInputStream} leitor.
     * @return array lido.
     * @throws IOException caso ocorra um erro.
     */
    static protected float[] readFloatArray(DataInputStream dis) throws IOException {
        int len = dis.readInt();// considerando que já escreve o tamanho.

        float[] arr = new float[len];
        for (int i = 0; i < len; i++) {
            arr[i] = dis.readFloat();
        }

        return arr;
    }
    
    /**
     * Lê o conteúdo de um array primitivo {@code byte[]} e 
     * converte em uma String.
     * @param dis {@code DataInputStream} leitor.
     * @return String lida
     * @throws IOException caso ocorra algum erro.
     */
    static protected String readString(DataInputStream dis) throws IOException {
        int len = dis.readInt();
        byte[] bytes = new byte[len];
        dis.readFully(bytes);
        return new String(bytes, "UTF-8");
    }

}
