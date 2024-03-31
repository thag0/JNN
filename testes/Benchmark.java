package testes;

import java.util.Random;
import java.util.concurrent.TimeUnit;

import lib.ged.Ged;
import rna.camadas.Convolucional;
import rna.core.OpTensor4D;
import rna.core.Tensor4D;
import rna.inicializadores.GlorotNormal;
import rna.inicializadores.GlorotUniforme;
import rna.inicializadores.Inicializador;
import rna.inicializadores.Zeros;

public class Benchmark{
   static OpTensor4D optensor = new OpTensor4D();

   public static void main(String[] args){
      Ged ged = new Ged();
      ged.limparConsole();

      convForward();
      convBackward();
   }

   /**
    * Calcula o tempo de propagação direta da camada convolucional.
    * <p>
    *    Os valores são arbritários e podem ser ajustados para testar
    *    diferentes cenários de estresse da camada.
    * </p>
    */
   static void convForward(){
      final int tamFiltro = 3;
      final int numFiltros = 16;
      final int tamEntrada = 26;
      final int profEntrada = 16;

      final int[] formatoEntrada = {profEntrada, tamEntrada, tamEntrada};
      final int[] formatoFiltro = {tamFiltro, tamFiltro};

      String ativacao = "sigmoid";

      final long seed = 123456789;
      final Inicializador iniKernel = new GlorotNormal(seed);
      final Inicializador iniBias = new GlorotNormal(seed);

      Convolucional conv = new Convolucional(
         formatoEntrada, formatoFiltro, numFiltros, ativacao, iniKernel, iniBias
      );
      conv.inicializar();

      Tensor4D entrada = new Tensor4D(conv.entrada.shape());
      entrada.preencherContador(true);

      long tempo = 0;
      tempo = medirTempo(() -> conv.forward(entrada));
   
      //resultados
      StringBuilder sb = new StringBuilder();
      String pad = "   ";

      sb.append("Config conv = [\n");
         sb.append(pad).append("filtros: " + conv.filtros.shapeStr() + "\n");
         sb.append(pad).append("entrada: " + conv.entrada.shapeStr() + "\n");
         sb.append(pad).append("saida: " + conv.saida.shapeStr() + "\n");
         sb.append(pad).append("Tempo forward: " + TimeUnit.NANOSECONDS.toMillis(tempo) + "ms\n");
      sb.append("]\n");

      System.out.println(sb.toString());
   }

   /**
    * Calcula o tempo de retropropagação da camada convolucional.
    * <p>
    *    Os valores são arbritários e podem ser ajustados para testar
    *    diferentes cenários de estresse da camada.
    * </p>
    */
   static void convBackward(){
      final int tamFiltro = 3;
      final int numFiltros = 64;
      final int tamEntrada = 28;
      final int profEntrada = 16;

      final int[] formatoEntrada = {profEntrada, tamEntrada, tamEntrada};
      final int[] formatoFiltro = {tamFiltro, tamFiltro};

      String ativacao = "sigmoid";

      final long seed = 123456789;
      final Inicializador iniKernel = new GlorotNormal(seed);
      final Inicializador iniBias = new GlorotNormal(seed);

      Convolucional conv = new Convolucional(
         formatoEntrada, formatoFiltro, numFiltros, ativacao, iniKernel, iniBias
      );
      conv.inicializar();

      //preparar dados pra retropropagar
      long randSeed = 99999;
      Random rand = new Random(randSeed);
      conv.entrada.map((x) -> rand.nextDouble());

      Tensor4D grad = new Tensor4D(conv.gradSaida.shape());
      grad.map((x) -> rand.nextDouble());

      long tempo = medirTempo(() -> conv.backward(grad));

      //resultados
      StringBuilder sb = new StringBuilder();
      String pad = "   ";

      sb.append("Config conv = [\n");
         sb.append(pad).append("filtros: " + conv.filtros.shapeStr() + "\n");
         sb.append(pad).append("entrada: " + conv.entrada.shapeStr() + "\n");
         sb.append(pad).append("saida: " + conv.saida.shapeStr() + "\n");
         sb.append(pad).append("Tempo backward: " + TimeUnit.NANOSECONDS.toMillis(tempo) + "ms\n");
      sb.append("]\n");

      System.out.println(sb.toString());
      
      testarForward();
      testarBackward();
   }

   /**
    * Calcula o tempo de execução (nanosegundos) de uma função
    * @param func função desejada.
    * @return tempo de processamento.
    */
   static long medirTempo(Runnable func){
      long t = System.nanoTime();
      func.run();
      return System.nanoTime() - t;
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

      Tensor4D entrada = new Tensor4D(conv.entrada.shape());
      entrada.map((x) -> Math.random());

      //simulação de propagação dos dados numa camada convolucional sem bias
      Tensor4D filtros = new Tensor4D(conv.filtros);
      Tensor4D saidaEsperada = new Tensor4D(conv.saida);
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
      
      Tensor4D entrada = new Tensor4D(conv.entrada);
      entrada.map((x) -> Math.random());
      
      Tensor4D filtros = new Tensor4D(conv.filtros);
      filtros.map((x) -> Math.random());
      conv.filtros.copiar(filtros);

      Tensor4D grad = new Tensor4D(conv.gradSaida);
      grad.map((x) -> Math.random());
      
      //backward
      conv.entrada.copiar(entrada);
      conv.backward(grad);

      Tensor4D gradSaida = new Tensor4D(conv.gradSaida);
      Tensor4D gradFiltroEsperado = new Tensor4D(conv.gradFiltros);
      Tensor4D gradEntradaEsperado = new Tensor4D(conv.gradEntrada);

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
            optensor.convolucao2DFull(gradSaida, conv.filtros, gradEntradaEsperado, idDerivada, idKernel, idGradEntrada);
         }
      }

      boolean gradF = conv.gradFiltros.equals(gradFiltroEsperado);
      boolean gradE = conv.gradEntrada.equals(gradEntradaEsperado);
      
      if(gradE && gradF){
         System.out.println("Backward esperado: " + (gradE && gradF));
      
      }else{
         System.out.println("Backward inesperado ->  gradFiltro: " +  gradF + ", gradEntrada: " + gradE);
      }
   }
}
