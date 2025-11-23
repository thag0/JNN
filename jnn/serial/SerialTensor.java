package jnn.serial;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;

import jnn.core.tensor.Tensor;

/**
 * Interface para io de tensores.
 */
public class SerialTensor extends SerialBase {

    /**
     * Formato padrão do Tensor.
     */
    final String FORMATO = ".tensor";

    /**
     * Interface para io de tensores.
     */
    public SerialTensor() {}
    
	/**
	 * Exporta os dados do tensor num arquivo {@code .tensor}.
     * @param t {@code Tensor} base.
     * @param caminho caminho de destino, deve conter a extensão {@code .tensor}.
     */
    public void salvar(Tensor t, String caminho) {
        File arquivo = new File(caminho);
        if (!arquivo.getName().toLowerCase().endsWith(FORMATO)) {
            throw new IllegalArgumentException("O caminho deve conter a extensão " + FORMATO);
        }

        int[] shape = t.shape();
        int dims = shape.length;

        try (DataOutputStream out = new DataOutputStream(new FileOutputStream(arquivo))) {
            escrever(out, dims);
            escrever(out, shape);

            // copiar internamente o conteúdo pra tratar casos de views
            double[] data = t.data().paraArray();
            escrever(out, data);

        } catch (IOException e) {
            System.out.println("\nErro ao salvar Tensor");
            e.printStackTrace();
        }
    }

    /**
     * Carrega um {@code Tensor} a partir de um arquivo {@code .tensor}.
     * @param caminho caminho do arquivo, deve conter a extensão {@code .tensor}.
     * @return {@code Tensor} carregado.
     */
    public Tensor ler(String caminho) {
        File arquivo = new File(caminho);
        if (!arquivo.getName().toLowerCase().endsWith(FORMATO)) {
            throw new IllegalArgumentException("O caminho deve conter a extensão " + FORMATO);
        }

        Tensor t = null;

        try (DataInputStream in = new DataInputStream(new FileInputStream(arquivo))) {
            // numero de dimensoes
            int dims = lerInt(in);

            // shape
            int[] shape = new int[dims];
            int[] arrS = lerArrInt(in, dims);
            
            int tam = 1;
            for (int i = 0; i < dims; i++) {
                shape[i] = arrS[i];
                tam *= shape[i];
            }

            // dados
            double[] dados = new double[tam];
            double[] arrD = lerArrDouble(in, tam); 
            System.arraycopy(arrD, 0, dados, 0, tam);

            t = new Tensor(dados).reshape(shape);

        } catch (IOException e) {
            System.out.println("Erro ao ler tensor: " + e.getMessage());
            e.printStackTrace();
        }

        return t;

    }

}
