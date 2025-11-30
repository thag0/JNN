package jnn.io.seriais;

import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;

import javax.imageio.ImageIO;

import jnn.core.tensor.Tensor;
import jnn.core.tensor.TensorConverter;

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

        try (DataOutputStream out = new DataOutputStream(new FileOutputStream(arquivo))) {
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
            // shape
            int[] shape = lerArrInt(in);

            // dados
            double[] dados = lerArrDouble(in);
            t = new Tensor(dados).reshape(shape);

        } catch (IOException e) {
            System.out.println("Erro ao ler tensor: " + e.getMessage());
            e.printStackTrace();
        }

        return t;
    }

    /**
     * Lê uma imagem de um arquivo externo e converte em um {@code Tensor}.
     * @param caminho caminho da imagem, deve conter a extensão do arquivo.
     * @return {@code Tensor} convertido.
     */
    public Tensor lerImagem(String caminho) {
        try {
            BufferedImage base = ImageIO.read(new File(caminho));
            
            int largura = base.getWidth();
            int altura = base.getHeight();
            
            BufferedImage img = new BufferedImage(
                largura,
                altura,
                BufferedImage.TYPE_INT_ARGB
            );
            
            Graphics2D g2 = img.createGraphics();
            g2.drawImage(base, 0, 0, null);
            g2.dispose();
            
            int[] buff = img.getRGB(0, 0, largura, altura, null, 0, largura);
            int[][][] data = new int[3][altura][largura];
            boolean cinza = true;
            
            for (int y = 0; y < altura; y++) {
                for (int x = 0; x < largura; x++) {
                    int pixel = buff[y * largura + x];

                    int r = (pixel >> 16) & 0xFF;
                    int g = (pixel >> 8) & 0xFF;
                    int b = pixel & 0xFF;

                    data[0][y][x] = r;
                    data[1][y][x] = g;
                    data[2][y][x] = b;
                
                    if (cinza && !(r == g && g == b)) cinza = false;
                }
            }

            return TensorConverter.tensor(cinza ? data[0] : data);

        } catch (IOException e) {
            throw new RuntimeException("Erro ao ler imagem: " + caminho, e);
        }
    }

}
