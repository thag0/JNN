package jnn.serial;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Iterator;

import jnn.core.tensor.Tensor;

/**
 * Interface para io de tensores.
 */
public class SerialTensor {

    final String FORMATO = ".tensor";

    /**
     * Interface para io de tensores.
     */
    public SerialTensor() {}
    
	/**
	 * Exporta os dados do tensor num arquivo externo.
     * @param t {@code Tensor} desejado.
     * @param caminho caminho de destino.
     */
    public void serializar(Tensor t, String caminho) {
		File arquivo = new File(caminho);
		if (!arquivo.getName().toLowerCase().endsWith(FORMATO)) {
			throw new IllegalArgumentException(
				"\nO caminho deve conter a extensão " + FORMATO
			);
		}

        StringBuilder sb = new StringBuilder();

        for (int dim : t.shape()) {
            sb.append(dim).append(" ");
        }

        Iterator<Double> it = t.iterator();
        while (it.hasNext()) {
            sb.append("\n").append(it.next().doubleValue());
        }

		try (BufferedWriter bw = new BufferedWriter(new FileWriter(caminho))) {
			bw.write(sb.toString());

	 	} catch (IOException e) {
			System.out.println("\nErro ao salvar tensor:");
			System.out.println(e.getMessage());
		}
    }

    /**
     * Carrega um {@code Tensor} a partir de um arquivo externo.
     * @param caminho caminho do arquivo {@code .tensor}.
     * @return {@code Tensor} lido.
     */
    public Tensor ler(String caminho) {
		File arquivo = new File(caminho);
		if (!arquivo.getName().toLowerCase().endsWith(FORMATO)) {
			throw new IllegalArgumentException(
				"\nO caminho deve conter a extensão " + FORMATO
			);
		}

        Tensor tensor = null; 

        try (BufferedReader br = new BufferedReader(new FileReader(caminho))) {
            String[] shapeStr = br.readLine().split(" ");
            int n = shapeStr.length;
            
            int tamanho = 1;
            int[] shape = new int[n];
            for (int i = 0; i < n; i++) {
                shape[i] = Integer.parseInt(shapeStr[i]);
                tamanho *= shape[i];
            }

            double[] dados = new double[tamanho];
            for (int i = 0; i < tamanho; i++) {
                double valor = Double.parseDouble(br.readLine());
                dados[i] = valor;
            }

            tensor = new Tensor(dados).reshape(shape);

        } catch (IOException e) {
            System.out.println("\nErro ao ler dados do tensor:");
			System.out.println(e.getMessage());       
        }

        return tensor;
    }

}
