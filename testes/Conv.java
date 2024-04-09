package testes;

import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.concurrent.TimeUnit;

import lib.ged.Dados;
import lib.ged.Ged;
import lib.geim.Geim;
import rna.camadas.*;
import rna.core.OpMatriz;
import rna.core.OpTensor4D;
import rna.core.Tensor4D;
import rna.inicializadores.GlorotUniforme;
import rna.inicializadores.Inicializador;
import rna.inicializadores.Zeros;
import rna.modelos.Sequencial;
import rna.otimizadores.Otimizador;
import rna.serializacao.Serializador;

public class Conv{
   static Ged ged = new Ged();
   static Geim geim = new Geim();
   static OpMatriz opmat = new OpMatriz();
   static OpTensor4D optensor = new OpTensor4D();
   static Serializador serializador = new Serializador();
   static final int amostras = 100;
   static final int digitos = 10;

   static final String CAMINHO_MODELOS = "./dados/modelos/";
   
   public static void main(String[] args){
      ged.limparConsole();

      // String nomeModelo = "mlp-mnist-89";
      // String nomeModelo = "conv-mnist-95-5";
      String nomeModelo = "modelo-treinado";
      Sequencial modelo = serializador.lerSequencial(CAMINHO_MODELOS + nomeModelo + ".nn");

      testarPrevisao(modelo, "treino/3/img_1", true);
      testarPrevisao(modelo, "3_deslocado", true);

      // modelo.info();
      // testarAcertosMNIST(modelo);
      // testarTodosDados(modelo);

      // Dados forward = tempoForward(modelo);//media 30/40 ms
      // Dados backward = tempoBackward(modelo);//media 20/25 ms
      // forward = ged.filtrar(forward, 1, "Convolucional");
      // backward = ged.filtrar(backward, 1, "Convolucional");
      // forward.imprimir();
      // backward.imprimir();

      testarForward();
      testarBackward();
   }

   static void testarAcertosMNIST(Sequencial modelo){
      final String caminho = "/dados/mnist/teste/";
      
      double media = 0;
      for(int digito = 0; digito < digitos; digito++){

         double acertos = 0;
         for(int amostra = 0; amostra < amostras; amostra++){
            String caminhoImagem = caminho + digito + "/img_" + amostra + ".jpg";
            Tensor4D img = new Tensor4D(imagemParaMatriz(caminhoImagem));
            
            double[] previsoes = modelo.forward(img).paraArray();
            if(maiorIndice(previsoes) == digito){
               acertos++;
            }
         }

         double porcentagem = acertos / (double)amostras;
         media += porcentagem;
         System.out.println("Acertos " + digito + " -> " + porcentagem);
      }
      System.out.println("média acertos: " + String.format("%.2f", (media/digitos)*100) + "%");
   }

   /**
    * Testar com multithread
    */
   static void testarForward(){
      int[] formEntrada = {12, 16, 16};
      Inicializador iniKernel = new GlorotUniforme(12345);
      Inicializador iniBias = new Zeros();
      Convolucional conv = new Convolucional(formEntrada, new int[]{3, 3}, 16, "linear", iniKernel, iniBias);
      conv.inicializar();

      Tensor4D entrada = new Tensor4D(conv.formatoEntrada());
      entrada.map((x) -> Math.random());

      //simulação de propagação dos dados numa camada convolucional sem bias
      Tensor4D filtros = new Tensor4D(conv.kernel());
      Tensor4D saidaEsperada = new Tensor4D(conv.formatoSaida());
      int[] idEntrada = {0, 0};
      int[] idKernel = {0, 0};
      int[] idSaida = {0, 0};
      saidaEsperada.preencher(0.0);
      for(int i = 0; i < filtros.dim1(); i++){
         idSaida[1] = i;
         for(int j = 0; j < filtros.dim2(); j++){
            idEntrada[1] = j;
            idKernel[0] = i;
            idKernel[1] = j;
            optensor.correlacao2D(entrada, filtros, saidaEsperada, idEntrada, idKernel, idSaida);
         }
      }

      conv.forward(entrada);

      System.out.println("Forward esperado: " + conv.saida().comparar(saidaEsperada));
   }

   /**
    * Testar com multithread
    */
   static void testarBackward(){
      int[] formEntrada = {26, 24, 24};
      Inicializador iniKernel = new GlorotUniforme(12345);
      Inicializador iniBias = new Zeros();
      Convolucional conv = new Convolucional(formEntrada, new int[]{3, 3}, 26, "linear", iniKernel, iniBias);
      
      Tensor4D entrada = new Tensor4D(conv.formatoEntrada());
      entrada.map((x) -> Math.random());
      
      Tensor4D filtros = new Tensor4D(conv.kernel().shape());
      filtros.map((x) -> Math.random());
      conv.setKernel(filtros.paraArray());

      Tensor4D grad = new Tensor4D(conv.gradSaida);
      grad.map((x) -> Math.random());
      
      //backward
      conv.forward(entrada);
      Tensor4D gradEntrada = conv.backward(grad);

      Tensor4D gradSaida = new Tensor4D(conv.gradSaida);
      Tensor4D gradFiltroEsperado = new Tensor4D(conv.gradKernel().shape());
      Tensor4D gradEntradaEsperado = new Tensor4D(gradEntrada.shape());

      gradEntradaEsperado.preencher(0);
      gradFiltroEsperado.preencher(0);
      for(int i = 0; i < filtros.dim1(); i++){
         for(int j = 0; j < filtros.dim2(); j++){
            int[] idEntrada = {0, j};
            int[] idDerivada = {0, i};
            int[] idGradKernel = {i, j};
            int[] idKernel = {i, j};
            int[] idGradEntrada = {0, j};
            gradFiltroEsperado.preencher2D(idKernel[0], idKernel[1], 0.0);
            optensor.correlacao2D(entrada, gradSaida, gradFiltroEsperado, idEntrada, idDerivada, idGradKernel);
            optensor.convolucao2DFull(gradSaida, filtros, gradEntradaEsperado, idDerivada, idKernel, idGradEntrada);
         }
      }

      boolean gradF = conv.gradKernel().equals(gradFiltroEsperado);
      boolean gradE = gradEntrada.equals(gradEntradaEsperado);
      
      if(gradE && gradF){
         System.out.println("Backward esperado: " + (gradE && gradF));
      
      }else{
         System.out.println("Backward inesperado ->  gradFiltro: " +  gradF + ", gradEntrada: " + gradE);
      }
   }

   static void testarTodosDados(Sequencial modelo){
      for(int i = 0; i < digitos; i++){
         for(int j = 0; j < amostras; j++){
            testarPrevisao(modelo, ("teste/" + i + "/img_" + j), false);
         }
         System.out.println();
      }
   }

   static long medirTempo(Runnable func){
      long t1 = System.nanoTime();
      func.run();
      return System.nanoTime() - t1;
   }

   static Dados tempoForward(Sequencial modelo){
      //arbritário
      double[][] img = imagemParaMatriz("/dados/mnist/teste/1/img_0.jpg");
      double[][][] entrada = new double[1][][];
      entrada[0] = img;

      int n = modelo.numCamadas();
      long t, total = 0;

      Dados dados = new Dados();
      dados.editarNome("Tempos Forward");
      ArrayList<String[]> conteudo = new ArrayList<>();

      t = medirTempo(() -> modelo.camada(0).forward(entrada));
      conteudo.add(new String[]{
         "0",
         modelo.camada(0).nome(),
         String.valueOf(TimeUnit.NANOSECONDS.toMillis(t)) + " ms"        
      });
      total += t;
      
      for(int i = 1; i < n; i++){
         Camada atual = modelo.camada(i);
         Camada anterior = modelo.camada(i-1);
         t = medirTempo(() -> {
            atual.forward(anterior.saida());
         });
         total += t;

         conteudo.add(new String[]{
            String.valueOf(i),
            modelo.camada(i).nome(),
            String.valueOf(TimeUnit.NANOSECONDS.toMillis(t)) + " ms"        
         });
      }
      conteudo.add(new String[]{
         "-",
         "Tempo total", 
         String.valueOf(TimeUnit.NANOSECONDS.toMillis(total)) + " ms"
      });

      dados.atribuir(conteudo);
      dados.editarNome("tempos forward");
      return dados;
   }

   static Dados tempoBackward(Sequencial modelo){
      //arbritário
      double[] grad = new double[modelo.saidaParaArray().length];
      grad[0] = 1;
      for(int i = 1; i < grad.length; i++){
         grad[i] = 0.02;
      }

      int n = modelo.numCamadas();
      long t, total = 0;

      Dados dados = new Dados();
      dados.editarNome("Tempos Backward");
      ArrayList<String[]> conteudo = new ArrayList<>();

      t = medirTempo(() -> modelo.camada(n-1).backward(new Tensor4D(grad)));
      conteudo.add(new String[]{
         String.valueOf(n-1),
         modelo.camada(n-1).nome(),
         String.valueOf(TimeUnit.NANOSECONDS.toMillis(t)) + " ms"        
      });
      total += t;
      
      for(int i = n-2; i >= 0; i--){
         final int id = i;
         Camada atual = modelo.camada(id);
         Camada proxima = modelo.camada(id+1);
         t = medirTempo(() -> {
            atual.backward(proxima.gradEntrada());
         });
         total += t;

         conteudo.add(new String[]{
            String.valueOf(i),
            modelo.camada(i).nome(),
            String.valueOf(TimeUnit.NANOSECONDS.toMillis(t)) + " ms"        
         });
      }
      conteudo.add(new String[]{
         "-",
         "Tempo total",
         String.valueOf(TimeUnit.NANOSECONDS.toMillis(total)) + " ms"        
      });

      dados.atribuir(conteudo);
      dados.editarNome("tempos backward");
      return dados;
   }

   static void tempoOtimizador(Sequencial modelo){
      Otimizador otm = modelo.otimizador();

      //arbritário
      double[] grad = new double[modelo.saidaParaArray().length];
      grad[0] = 1;
      for(int i = 1; i < grad.length; i++){
         grad[i] = 0.02;
      }

      //backward simples
      modelo.camadaSaida().backward(new Tensor4D(grad));
      for(int i = modelo.numCamadas()-2; i >= 0; i--){
         modelo.camada(i).backward(modelo.camada(i+1).gradEntrada());
      }

      long t = System.nanoTime();
      otm.atualizar(modelo.camadas());
      t = System.nanoTime() - t;

      System.out.println(
         "Tempo otimizador (" + otm.nome() + "): " + TimeUnit.NANOSECONDS.toMillis(t) + "ms"
      );
   }

   static void testarModelo(Sequencial modelo, int digitos, int amostras){
      var testeX = carregarDadosMNIST("/dados/mnist/teste/", amostras, digitos);
      var testeY = criarRotulosMNIST(amostras, digitos);

      double acuraria = modelo.avaliador().acuracia(testeX, testeY);
      System.out.println("Perda: " + modelo.avaliar(testeX, testeY));
      System.out.println("Acurácia: " + (acuraria * 100) + "%");
   }

   static void testarPrevisao(Sequencial modelo, String caminhoImagem, boolean prob){
      String extensao = ".jpg";
      var entrada = new Tensor4D(imagemParaMatriz("/dados/mnist/" + caminhoImagem + extensao));
      double[] previsao = modelo.forward(entrada).paraArray();
      
      System.out.print("\nTestando: " + caminhoImagem + extensao);
      if(prob){
         System.out.println();
         for(int i = 0; i < previsao.length; i++){
            System.out.println("Prob: " + i + ": " + (int)(previsao[i]*100) + "%");
         }
      }else{
         System.out.println(" -> Prev: " + maiorIndice(previsao));
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
