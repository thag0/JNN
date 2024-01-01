package testes.modelos;

import java.awt.image.BufferedImage;

import lib.ged.Ged;
import lib.geim.Geim;
import rna.core.OpMatriz;
import rna.modelos.Sequencial;
import rna.serializacao.Serializador;

public class TreinoConv{
   static Ged ged = new Ged();
   static Geim geim = new Geim();
   static OpMatriz opmat = new OpMatriz();
   static Serializador serializador = new Serializador();
   
   public static void main(String[] args){
      ged.limparConsole();
      
      Sequencial modelo = serializador.lerSequencial("./modelo-convolucional.txt");
      testarPorbabilidade(modelo, "0_teste_1");
      testarPorbabilidade(modelo, "0_teste_2");
      testarPorbabilidade(modelo, "1_teste_1");
      testarPorbabilidade(modelo, "1_teste_2");
      testarPorbabilidade(modelo, "2_teste_1");
      testarPorbabilidade(modelo, "2_teste_2");
   }

   static void testarPorbabilidade(Sequencial modelo, String imagemTeste){
      System.out.println("\nTestando: " + imagemTeste);
      double[][][] teste1 = new double[1][][];
      teste1[0] = imagemParaMatriz("/dados/mnist/teste/" + imagemTeste + ".jpg");
      modelo.calcularSaida(teste1);
      double[] previsao = modelo.saidaParaArray();
      for(int i = 0; i < previsao.length; i++){
         System.out.println("Prob: " + i + ": " + (int)(previsao[i]*100) + "%");
      }
   }

   static double[][] imagemParaMatriz(String caminho){
      BufferedImage img = geim.lerImagem(caminho);
      double[][] imagem = new double[img.getHeight()][img.getWidth()];

      int[][] cinza = geim.obterCinza(img);

      for(int y = 0; y < imagem.length; y++){
         for(int x = 0; x < imagem[y].length; x++){
            imagem[y][x] = (double)cinza[y][x] / 255;
         }
      }
      return imagem;
   }
}
