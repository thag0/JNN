package testes;

import java.awt.image.BufferedImage;

import ged.Ged;
import geim.Geim;
import rna.avaliacao.perda.*;
import rna.estrutura.*;
import rna.inicializadores.*;
import rna.modelos.Sequencial;
import rna.otimizadores.*;

public class TesteConv{
   static Ged ged = new Ged();
   static Geim geim = new Geim();

   public static void main(String[] args){
      ged.limparConsole();
      
      int[] formEntrada = {28, 28, 1};
      int[] formFiltro = {3, 3};

      double[][][][] entradas = new double[10][formEntrada[2]][formEntrada[0]][formEntrada[1]];
      double[][] saidas = new double[10][10];
      for(int i = 0; i < 10; i++){
         entradas[i][0] = imagemParaMatriz("/dados/mnist/" + i + ".png");
         saidas[i][i] = 1;
      }

      Inicializador ini = new Xavier();
      Convolucional conv1 = new Convolucional(formEntrada, formFiltro, 8, "sigmoid");
      conv1.inicializar(ini, ini, 0);

      Convolucional conv2 = new Convolucional(conv1.formatoSaida(), formFiltro, 8, "sigmoid");
      Flatten flat = new Flatten(conv2.formatoSaida());
      Densa densa1 = new Densa(flat.tamanhoSaida(), 20, "leakyrelu");
      Densa densa2 = new Densa(densa1.tamanhoSaida(), 10, "softmax");

      Sequencial cnn = new Sequencial(new Camada[]{
         conv1,
         conv2,
         flat,
         densa1,
         densa2,
      });

      cnn.inicializar(ini);
   
      Perda perda = new EntropiaCruzada();
      Otimizador otm = new SGD(0.001, 0.995);

      cnn.treinar(entradas, saidas, 300, perda, otm, true);

      for(int i = 0; i < 10; i++){
         System.out.println("Real: " + i + ", Pred: " + testarImagem(cnn, entradas[i][0]));
      }
   }

   public static double[][] imagemParaMatriz(String caminho){
      BufferedImage img = geim.lerImagem(caminho);
      double[][] imagem = new double[img.getHeight()][img.getWidth()];

      int[][] cinza = geim.obterCinza(img);

      for(int y = 0; y < imagem.length; y++){
         for(int x = 0; x < imagem[y].length; x++){
            imagem[y][x] = cinza[y][x];
         }
      }

      return imagem;
   }

   public static int testarImagem(Sequencial cnn, double[][] entrada){
      double[][][] e = new double[1][][];
      e[0] = entrada;

      cnn.calcularSaida(e);
      double[] prev = cnn.obterSaida();

      for(int i = 0; i < prev.length; i++){
         if(prev[i] > 0.95){
            return i;
         }
      }

      return -1;
   }
}
