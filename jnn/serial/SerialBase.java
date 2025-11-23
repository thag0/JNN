package jnn.serial;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

/**
 * Base de serialização de elementos da biblioteca.
 */
class SerialBase {

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
    public void escrever(DataOutputStream dos, int val) throws IOException {
        dos.writeInt(val);
    }

    /**
     * Grava o conteúdo de um array primitivo {@code int[]}.
     * @param dos {@code DataOutputStream} gravador.
     * @param arr {@code array} desejado.
     * @throws IOException caso ocorra um erro.
     */
    public void escrever(DataOutputStream dos, int[] arr) throws IOException {
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
    public void escrever(DataOutputStream dos, double[] arr) throws IOException {
        for (double val : arr) {
            dos.writeDouble(val);
        }
    }

    /**
     * Lê o conteúdo de um valor primitivo {@code int}.
     * @param dis {@code DataInputStream} leitor.
     * @return valor lido.
     * @throws IOException caso ocorra um erro.
     */
    public int lerInt(DataInputStream dis) throws IOException {
        return dis.readInt();
    }

    /**
     * Lê o conteúdo de um array primitivo {@code int[]}.
     * @param dis {@code DataInputStream} leitor.
     * @param tam tamanho do array.
     * @return array lido.
     * @throws IOException caso ocorra um erro.
     */
    public int[] lerArrInt(DataInputStream dis, int tam) throws IOException {
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
    public double[] lerArrDouble(DataInputStream dis, int tam) throws IOException {
        double[] arr = new double[tam];
        for (int i = 0; i < tam; i++) {
            arr[i] = dis.readDouble();
        }

        return arr;
    }
    
}
