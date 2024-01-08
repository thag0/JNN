package testes;

import java.awt.image.BufferedImage;
import java.util.concurrent.TimeUnit;

import lib.ged.Ged;
import lib.geim.Geim;
import rna.core.OpMatriz;
import rna.modelos.Sequencial;
import rna.serializacao.Serializador;
import rna.treinamento.AuxiliarTreino;

public class TreinoConv{
   static Ged ged = new Ged();
   static Geim geim = new Geim();
   static OpMatriz opmat = new OpMatriz();
   static Serializador serializador = new Serializador();
   
   public static void main(String[] args){
      ged.limparConsole();
      
      Sequencial modelo = serializador.lerSequencial("./dados/modelosMNIST/conv-mnist-89.txt");
      // modelo.info();

      int digitos = 10;
      int amostras = 10;
      var testeX = carregarDadosMNIST("/dados/mnist/teste/", amostras, digitos);
      var testeY = criarRotulosMNIST(amostras, digitos);

      long t;
      t = medirTempo(() -> modelo.calcularSaida(testeX[0]));
      System.out.println("Tempo forward: " + TimeUnit.NANOSECONDS.toMillis(t) + "ms");
      
      t = medirTempo(() -> modelo.otimizador().atualizar(modelo.camadas()));
      System.out.println("Tempo otimizador: " + TimeUnit.NANOSECONDS.toMillis(t) + "ms");
      
      AuxiliarTreino aux = new AuxiliarTreino();
      t = medirTempo(() -> {
         aux.backpropagation(modelo.camadas(), modelo.perda(), new double[]{0, 0, 0, 0, 0, 0, 0, 0, 0, 1});
      });
      System.out.println("Tempo backprop: " + TimeUnit.NANOSECONDS.toMillis(t) + "ms");

      System.exit(0);

      double perda = modelo.avaliador.entropiaCruzada(testeX, testeY);
      double acuraria = modelo.avaliador.acuracia(testeX, testeY);
      System.out.println("Perda: " + perda);
      System.out.println("Acurácia: " + acuraria + "%");
   }

   static long medirTempo(Runnable func){
      long t1 = System.nanoTime();
      func.run();
      return System.nanoTime() - t1;
   }

   static void testarPorbabilidade(Sequencial modelo, String imagemTeste){
      double[][][] teste1 = new double[1][][];
      teste1[0] = imagemParaMatriz("/dados/mnist/teste/" + imagemTeste + ".jpg");
      modelo.calcularSaida(teste1);
      double[] previsao = modelo.saidaParaArray();

      System.out.println("\nTestando: " + imagemTeste);
      System.out.println("Prev: " + maiorIndice(previsao));
      // for(int i = 0; i < previsao.length; i++){
         // System.out.println("Prob: " + i + ": " + (int)(previsao[i]*100) + "%");
         // System.out.printf("Prob: %d: %.2f\n", i, (float)(previsao[i]*100));
      // }
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
