import java.awt.image.BufferedImage;
import java.util.concurrent.TimeUnit;
import ged.Ged;
import geim.Geim;
import rna.avaliacao.perda.*;
import rna.estrutura.*;
import rna.inicializadores.*;
import rna.modelos.Sequencial;
import rna.otimizadores.*;

public class MainConv{
   static Ged ged = new Ged();
   static Geim geim = new Geim();

   public static void main(String[] args){
      ged.limparConsole();

      int amostras = 5;
      int digitos = 2;
      
      double[][][][] entradas = new double[amostras*digitos][digitos][][];
      double[][] saidas = new double[amostras*digitos][digitos];
      entradas = carregarDadosMNIST(amostras, digitos);
      System.out.println("Imagens carregadas.");
      saidas = carregarRotulosMNIST(amostras, digitos);
      System.out.println("Rótulos carregados.");

      Sequencial cnn = criarModelo();

      //treinar e marcar tempo
      long t1, t2;
      long horas, minutos, segundos;

      System.out.println("Treinando.");
      t1 = System.nanoTime();
      cnn.treinar(entradas, saidas, 200);
      t2 = System.nanoTime();

      long tempoDecorrido = t2 - t1;
      long segundosTotais = TimeUnit.NANOSECONDS.toSeconds(tempoDecorrido);
      horas = segundosTotais / 3600;
      minutos = (segundosTotais % 3600) / 60;
      segundos = segundosTotais % 60;
      System.out.println("Tempo de treinamento: " + horas + "h " + minutos + "m " + segundos + "s");
      testes.TesteSequencial.exportarHistoricoPerda(cnn);

      System.out.println(cnn.info());

      //-------------------------------------
      // for(int i = 0; i < 10; i++){
      //    System.out.println("Real: " + i + ", Pred: " + testarImagem(cnn, entradas[i][0]));
      // }

      System.out.println("\nTeste 0");
      double[][][] teste0 = new double[1][][];
      teste0[0] = imagemParaMatriz("/dados/mnist/teste/0_teste.jpg");
      cnn.calcularSaida(teste0);
      double[] previsao = cnn.saidaParaArray();
      for(int i = 0; i < previsao.length; i++){
         System.out.println("Prob: " + i + ": " + (int)(previsao[i]*100) + "%");
      }

      System.out.println("\nTeste 1");
      double[][][] teste1 = new double[1][][];
      teste1[0] = imagemParaMatriz("/dados/mnist/teste/1_teste.jpg");
      cnn.calcularSaida(teste1);
      previsao = cnn.saidaParaArray();
      for(int i = 0; i < previsao.length; i++){
         System.out.println("Prob: " + i + ": " + (int)(previsao[i]*100) + "%");
      }
   } 

   public static Sequencial criarModelo(){
      int[] formEntrada = {28, 28, 1};
      
      Sequencial modelo = new Sequencial(new Camada[]{
         new Convolucional(formEntrada, new int[]{4, 4}, 10, "tanh"),
         new Convolucional(new int[]{3, 3}, 10, "tanh"),
         new Flatten(),
         new Densa(100, "tanh"),
         new Densa(2, "softmax"),
      });

      modelo.compilar(new SGD(0.001, 0.95), new EntropiaCruzada(), new Xavier());
      modelo.configurarHistorico(true);

      return modelo;
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

   public static int testarImagem(Sequencial modelo, double[][] entrada){
      double[][][] e = new double[1][][];
      e[0] = entrada;

      modelo.calcularSaida(e);
      double[] prev = modelo.saidaParaArray();

      for(int i = 0; i < prev.length; i++){
         if(prev[i] > 0.66){
            return i;
         }
      }

      return -1;
   }

   public static double[][][][] carregarDadosMNIST(int amostras, int digitos){
      String caminho = "/dados/mnist/treino/";

      double[][][][] entradas = new double[digitos * amostras][1][][];
      for(int i = 0; i < entradas.length; i++){
         for(int j = 0; j < amostras; j++){
            double[][] imagem = imagemParaMatriz(caminho + j + "/img_" + j + ".jpg");
            entradas[i][0] = imagem;
         }
      }

      return entradas;
   }

   public static double[][] carregarRotulosMNIST(int amostras, int digitos){
      double[][] rotulos = new double[digitos * amostras][digitos];
      for(int numero = 0; numero < digitos; numero++){
         for(int i = 0; i < amostras; i++){
            int indice = numero * amostras + i;
            rotulos[indice][numero] = 1;
         }
      }
  
      return rotulos;
   }
}
