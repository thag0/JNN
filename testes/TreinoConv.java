package testes;

import java.awt.image.BufferedImage;
import java.util.concurrent.TimeUnit;

import lib.ged.Ged;
import lib.geim.Geim;
import rna.camadas.Convolucional;
import rna.camadas.Densa;
import rna.camadas.Flatten;
import rna.camadas.MaxPooling;
import rna.core.Mat;
import rna.core.OpMatriz;
import rna.modelos.Sequencial;
import rna.serializacao.Serializador;

public class TreinoConv{
   static Ged ged = new Ged();
   static Geim geim = new Geim();
   static OpMatriz opmat = new OpMatriz();
   static Serializador serializador = new Serializador();
   static int digitos = 10;
   static int amostras = 10;
   
   public static void main(String[] args){
      ged.limparConsole();
      
      Sequencial modelo = serializador.lerSequencial("./dados/modelosMNIST/conv-mnist-89.txt");
      // testarModelo(modelo, digitos, amostras);
      // testarTodosDados(modelo);

      double[][] img = imagemParaMatriz("/dados/mnist/teste/0/img_0.jpg");
      double[][][] entrada = new double[1][][];
      entrada[0] = img;

      Convolucional conv1 = (Convolucional) modelo.camada(0);
      MaxPooling max1 = (MaxPooling) modelo.camada(1);
      Convolucional conv2 = (Convolucional) modelo.camada(2);
      MaxPooling max2 = (MaxPooling) modelo.camada(3);
      Flatten flat = (Flatten) modelo.camada(4);
      Densa d1 = (Densa) modelo.camada(5);
      Densa d2 = (Densa) modelo.camada(6);

      double[] grad = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

      long t;
      t = medirTempo(() -> d2.calcularGradiente(new Mat(grad)));
      System.out.println("backward den2:  \t " + TimeUnit.NANOSECONDS.toMillis(t) + "ms");
      
      t = medirTempo(() -> d1.calcularGradiente(d2.obterGradEntrada()));
      System.out.println("backward den1:  \t " + TimeUnit.NANOSECONDS.toMillis(t) + "ms");
      
      t = medirTempo(() -> flat.calcularGradiente(d1.obterGradEntrada()));
      System.out.println("backward flat:  \t " + TimeUnit.NANOSECONDS.toMillis(t) + "ms");
      
      t = medirTempo(() -> max2.calcularGradiente(flat.obterGradEntrada()));
      System.out.println("backward max2:  \t " + TimeUnit.NANOSECONDS.toMillis(t) + "ms");
      
      t = medirTempo(() -> conv2.calcularGradiente(max2.obterGradEntrada()));
      System.out.println("backward conv2:  \t " + TimeUnit.NANOSECONDS.toMillis(t) + "ms");
      
      t = medirTempo(() -> max1.calcularGradiente(conv2.obterGradEntrada()));
      System.out.println("backward max1:  \t " + TimeUnit.NANOSECONDS.toMillis(t) + "ms");
      
      t = medirTempo(() -> conv1.calcularGradiente(max1.obterGradEntrada()));
      System.out.println("backward conv1:  \t " + TimeUnit.NANOSECONDS.toMillis(t) + "ms");
   }

   static void testarTodosDados(Sequencial modelo){
      for(int i = 0; i < digitos; i++){
         for(int j = 0; j < amostras; j++){
            testarPrevisao(modelo, (i + "/img_" + j), false);
         }
         System.out.println();
      }
   }

   static long medirTempo(Runnable func){
      long t1 = System.nanoTime();
      func.run();
      return System.nanoTime() - t1;
   }

   static void testarModelo(Sequencial modelo, int digitos, int amostras){
      var testeX = carregarDadosMNIST("/dados/mnist/teste/", amostras, digitos);
      var testeY = criarRotulosMNIST(amostras, digitos);

      double acuraria = modelo.avaliador.acuracia(testeX, testeY);
      System.out.println("Perda: " + modelo.avaliar(testeX, testeY));
      System.out.println("Acurácia: " + acuraria + "%");
   }

   static void testarPrevisao(Sequencial modelo, String imagemTeste, boolean prob){
      double[][][] entrada = new double[1][][];
      String extensao = ".jpg";
      entrada[0] = imagemParaMatriz("/dados/mnist/teste/" + imagemTeste + extensao);
      modelo.calcularSaida(entrada);
      double[] previsao = modelo.saidaParaArray();
      
      System.out.print("\nTestando: " + imagemTeste + extensao);
      if(prob){
         System.out.println();
         for(int i = 0; i < previsao.length; i++){
            System.out.println("Prob: " + i + ": " + (int)(previsao[i]*100) + "%");
         }
      }else{
         System.out.print(" -> Prev: " + maiorIndice(previsao));
      }

   }

   static int maiorIndice(double[] arr){
      int id = 0;
      double maior = arr[0];

      for(int i = 1; i < arr.length; i++){
         if(arr[i] > maior){
            id = i;
            maior = arr[i];
         }
      }

      return id;
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

   public static double[][][][] carregarDadosMNIST(String caminho, int amostras, int digitos){
      double[][][][] entradas = new double[digitos * amostras][1][][];

      int id = 0;
      for(int i = 0; i < digitos; i++){
         for(int j = 0; j < amostras; j++){
            String caminhoCompleto = caminho + i + "/img_" + j + ".jpg";
            double[][] imagem = imagemParaMatriz(caminhoCompleto);
            entradas[id++][0] = imagem;
         }
      }

      System.out.println("Imagens carregadas. (" + entradas.length + ")");
      return entradas;
   }

   public static double[][] criarRotulosMNIST(int amostras, int digitos){
      double[][] rotulos = new double[digitos * amostras][digitos];
      for(int numero = 0; numero < digitos; numero++){
         for(int i = 0; i < amostras; i++){
            int indice = numero * amostras + i;
            rotulos[indice][numero] = 1;
         }
      }
      
      System.out.println("Rótulos gerados de 0 a " + (digitos-1) + ".");
      return rotulos;
   }
}
