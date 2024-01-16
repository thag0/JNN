package testes;

import java.awt.image.BufferedImage;
import java.util.concurrent.TimeUnit;

import lib.ged.Ged;
import lib.geim.Geim;
import rna.camadas.Convolucional;
import rna.camadas.Densa;
import rna.camadas.Flatten;
import rna.camadas.MaxPooling;
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

      long t;
      t = medirTempo(() -> modelo.calcularSaida(entrada));
      System.out.println("forward mod: \t" + TimeUnit.NANOSECONDS.toMillis(t) + "ms");

      t = medirTempo(() -> conv1.calcularSaida(entrada));
      System.out.println("forward conv1: \t" + TimeUnit.NANOSECONDS.toMillis(t) + "ms");
      
      t = medirTempo(() -> max1.calcularSaida(conv1.saida()));
      System.out.println("forward max1: \t" + TimeUnit.NANOSECONDS.toMillis(t) + "ms");
      
      t = medirTempo(() -> conv2.calcularSaida(max1.saida()));
      System.out.println("forward conv2: \t" + TimeUnit.NANOSECONDS.toMillis(t) + "ms");
      
      t = medirTempo(() -> max2.calcularSaida(conv2.saida()));
      System.out.println("forward conv2: \t" + TimeUnit.NANOSECONDS.toMillis(t) + "ms");    
      
      t = medirTempo(() -> flat.calcularSaida(max2.saida()));
      System.out.println("forward flat: \t" + TimeUnit.NANOSECONDS.toMillis(t) + "ms");    
      
      t = medirTempo(() -> d1.calcularSaida(flat.saida()));
      System.out.println("forward d1: \t" + TimeUnit.NANOSECONDS.toMillis(t) + "ms");    
      
      t = medirTempo(() -> d2.calcularSaida(d1.saida()));
      System.out.println("forward d2: \t" + TimeUnit.NANOSECONDS.toMillis(t) + "ms");

      System.out.println("Prev: " + maiorIndice(d2.saida.paraArray()));
   }

   static long medirTempo(Runnable func){
      long t1 = System.nanoTime();
      func.run();
      return System.nanoTime() - t1;
   }

   static void testarModelo(Sequencial modelo, int digitos, int amostras){
      var testeX = carregarDadosMNIST("/dados/mnist/teste/", amostras, digitos);
      var testeY = criarRotulosMNIST(amostras, digitos);

      double perda = modelo.avaliador.entropiaCruzada(testeX, testeY);
      double acuraria = modelo.avaliador.acuracia(testeX, testeY);
      System.out.println("Perda: " + perda);
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
