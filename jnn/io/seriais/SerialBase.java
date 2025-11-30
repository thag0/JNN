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
    protected void escrever(DataOutputStream dos, int val) throws IOException {
        dos.writeInt(val);
    }

    /**
     * Grava o conteúdo de um valor primitivo {@code double}.
     * @param dos {@code DataOutputStream} gravador.
     * @param val valor desejado.
     * @throws IOException caso ocorra um erro.
     */
    protected void escrever(DataOutputStream dos, double val) throws IOException {
        dos.writeDouble(val);
    }

    /**
     * Grava o conteúdo de um valor primitivo {@code boolean}.
     * @param dos {@code DataOutputStream} gravador.
     * @param val valor desejado.
     * @throws IOException caso ocorra um erro.
     */
    protected void escrever(DataOutputStream dos, boolean val) throws IOException {
        dos.writeBoolean(val);
    }

    /**
     * Grava o conteúdo de um array primitivo {@code int[]}.
     * @param dos {@code DataOutputStream} gravador.
     * @param arr {@code array} desejado.
     * @throws IOException caso ocorra um erro.
     */
    protected void escrever(DataOutputStream dos, int[] arr) throws IOException {
        dos.writeInt(arr.length);
        for (int val : arr) {
            dos.writeInt(val);
        }
    }

    /**
     * Grava o conteúdo de um array primitivo {@code double[]}.
     * @param dos {@code DataOutputStream} gravador.
     * @param arr {@code array} desejado.
     * @throws IOException caso ocorra um erro.
     */
    protected void escrever(DataOutputStream dos, double[] arr) throws IOException {
        dos.writeInt(arr.length);
        for (double val : arr) {
            dos.writeDouble(val);
        }
    }

    /**
     * Grava o conteúdo de uma {@code String}.
     * @param dos {@code DataOutputStream} gravador.
     * @param s {@code String} desejada.
     * @throws IOException caso ocorra um erro.
     */
    protected void escrever(DataOutputStream dos, String s) throws IOException {
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
    protected int lerInt(DataInputStream dis) throws IOException {
        return dis.readInt();
    }

    /**
     * Lê o conteúdo de um valor primitivo {@code double}.
     * @param dis {@code DataInputStream} leitor.
     * @return valor lido.
     * @throws IOException caso ocorra um erro.
     */
    protected double lerDouble(DataInputStream dis) throws IOException {
        return dis.readDouble();
    }

    protected boolean lerBoolean(DataInputStream dis) throws IOException {
        return dis.readBoolean();
    }

    /**
     * Lê o conteúdo de um array primitivo {@code int[]}.
     * @param dis {@code DataInputStream} leitor.
     * @param tam tamanho do array.
     * @return array lido.
     * @throws IOException caso ocorra um erro.
     */
    protected int[] lerArrInt(DataInputStream dis) throws IOException {
        int tam = dis.readInt();// considerando que já escreve o tamanho.
        
        int[] arr = new int[tam];
        for (int i = 0; i < tam; i++) {
            arr[i] = dis.readInt();
        }

        return arr;
    }

    /**
     * Lê o conteúdo de um array primitivo {@code double[]}.
     * @param dis {@code DataInputStream} leitor.
     * @param tam tamanho do array.
     * @return array lido.
     * @throws IOException caso ocorra um erro.
     */
    protected double[] lerArrDouble(DataInputStream dis) throws IOException {
        int tam = dis.readInt();// considerando que já escreve o tamanho.

        double[] arr = new double[tam];
        for (int i = 0; i < tam; i++) {
            arr[i] = dis.readDouble();
        }

        return arr;
    }
    
    protected String lerString(DataInputStream in) throws IOException {
        int tam = in.readInt();
        byte[] bytes = new byte[tam];
        in.readFully(bytes);
        return new String(bytes, "UTF-8");
    }

}
