package testes;

import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.concurrent.TimeUnit;

import lib.ged.Dados;
import lib.ged.Ged;
import lib.geim.Geim;
import rna.camadas.Camada;
import rna.core.Mat;
import rna.core.OpMatriz;
import rna.modelos.Sequencial;
import rna.serializacao.Serializador;

public class TreinoConv{
   static Ged ged = new Ged();
   static Geim geim = new Geim();
   static OpMatriz opmat = new OpMatriz();
   static Serializador serializador = new Serializador();
   static int amostras = 100;
   static int digitos = 10;
   
   public static void main(String[] args){
      ged.limparConsole();
      
      // Sequencial modelo = serializador.lerSequencial("./dados/modelosMNIST/modelo-convolucional.txt");
      Sequencial modelo = serializador.lerSequencial("./dados/modelosMNIST/conv-mnist-89.txt");
      // testarModelo(modelo, digitos, amostras);
      // testarTodosDados(modelo);

      tempoForward(modelo);//keras += 30ms
      tempoBackward(modelo);

      // long t = medirTempo(() -> modelo.otimizador().atualizar(modelo.camadas()));
      // System.out.println(
      //    "Tempo otimizador (" + modelo.otimizador().nome() + 
      //    "): " + TimeUnit.NANOSECONDS.toMillis(t) + "ms"
      // ); 
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

   static void tempoForward(Sequencial modelo){
      //arbritário
      double[][] img = imagemParaMatriz("/dados/mnist/teste/0/img_0.jpg");
      double[][][] entrada = new double[1][][];
      entrada[0] = img;

      int n = modelo.numCamadas();
      long t, total = 0;

      Dados dados = new Dados();
      dados.editarNome("Tempos Forward");
      ArrayList<String[]> conteudo = new ArrayList<>();

      t = medirTempo(() -> modelo.camada(0).calcularSaida(entrada));
      conteudo.add(new String[]{
         modelo.camada(0).nome(),
         String.valueOf(TimeUnit.NANOSECONDS.toMillis(t)) + " ms"        
      });
      total += t;
      
      for(int i = 1; i < n; i++){
         Camada atual = modelo.camada(i);
         Camada anterior = modelo.camada(i-1);
         t = medirTempo(() -> {
            atual.calcularSaida(anterior.saida());
         });
         total += t;

         conteudo.add(new String[]{
            modelo.camada(i).nome(),
            String.valueOf(TimeUnit.NANOSECONDS.toMillis(t)) + " ms"        
         });
      }
      conteudo.add(new String[]{
         "Tempo total", 
         String.valueOf(TimeUnit.NANOSECONDS.toMillis(total)) + " ms"
      });

      dados.atribuir(conteudo);
      dados.imprimir();
   }

   static void tempoBackward(Sequencial modelo){
      //arbritário
      double[] grad = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

      int n = modelo.numCamadas();
      long t, total = 0;

      Dados dados = new Dados();
      dados.editarNome("Tempos Backward");
      ArrayList<String[]> conteudo = new ArrayList<>();

      t = medirTempo(() -> modelo.camada(n-1).calcularGradiente(new Mat(grad)));
      conteudo.add(new String[]{
         modelo.camada(n-1).nome(),
         String.valueOf(TimeUnit.NANOSECONDS.toMillis(t)) + " ms"        
      });
      total += t;
      
      for(int i = n-2; i >= 0; i--){
         final int id = i;
         Camada atual = modelo.camada(id);
         Camada proxima = modelo.camada(id+1);
         t = medirTempo(() -> {
            atual.calcularGradiente(proxima.obterGradEntrada());
         });
         total += t;

         conteudo.add(new String[]{
            modelo.camada(i).nome(),
            String.valueOf(TimeUnit.NANOSECONDS.toMillis(t)) + " ms"        
         });
      }
      conteudo.add(new String[]{
         "Tempo total",
         String.valueOf(TimeUnit.NANOSECONDS.toMillis(total)) + " ms"        
      });

      dados.atribuir(conteudo);
      dados.imprimir();
   }

   static void testarModelo(Sequencial modelo, int digitos, int amostras){
      var testeX = carregarDadosMNIST("/dados/mnist/teste/", amostras, digitos);
      var testeY = criarRotulosMNIST(amostras, digitos);

      double acuraria = modelo.avaliador.acuracia(testeX, testeY);
      System.out.println("Perda: " + modelo.avaliar(testeX, testeY));
      System.out.println("Acurácia: " + (acuraria * 100) + "%");
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
