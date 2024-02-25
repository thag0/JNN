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

      Tensor4D entrada = new Tensor4D(conv.entrada.dimensoes());
      entrada.preencherContador(true);

      long tempo = 0;
      tempo = medirTempo(() -> conv.calcularSaida(entrada));
   
      //resultados
      StringBuilder sb = new StringBuilder();
      String pad = "   ";

      sb.append("Config conv = [\n");
         sb.append(pad).append("filtros: " + conv.filtros.dimensoesStr() + "\n");
         sb.append(pad).append("entrada: " + conv.entrada.dimensoesStr() + "\n");
         sb.append(pad).append("saida: " + conv.saida.dimensoesStr() + "\n");
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

      Tensor4D grad = new Tensor4D(conv.gradSaida.dimensoes());
      grad.map((x) -> rand.nextDouble());

      long tempo = medirTempo(() -> conv.calcularGradiente(grad));

      //resultados
      StringBuilder sb = new StringBuilder();
      String pad = "   ";

      sb.append("Config conv = [\n");
         sb.append(pad).append("filtros: " + conv.filtros.dimensoesStr() + "\n");
         sb.append(pad).append("entrada: " + conv.entrada.dimensoesStr() + "\n");
         sb.append(pad).append("saida: " + conv.saida.dimensoesStr() + "\n");
         sb.append(pad).append("Tempo backward: " + TimeUnit.NANOSECONDS.toMillis(tempo) + "ms\n");
      sb.append("]\n");

      System.out.println(sb.toString());      
   }

   static void testarForward(){
      int[] formEntrada = {5, 8, 8};
      Inicializador iniKernel = new GlorotUniforme(12345);
      Inicializador iniBias = new Zeros();
      Convolucional conv = new Convolucional(formEntrada, new int[]{2, 2}, 3, "linear", iniKernel, iniBias);
      conv.inicializar();

      Tensor4D entrada = new Tensor4D(conv.entrada.dimensoes());
      entrada.preencherContador(true);

      //simulação de propagação dos dados numa camada convolucional sem bias
      Tensor4D filtros = new Tensor4D(conv.filtros);
      Tensor4D saidaEsperada = new Tensor4D(conv.saida);
      int[] idEntrada = {0, 0};
      int[] idKernel = {0, 0};
      int[] idSaida = {0, 0};
      for(int i = 0; i < filtros.dim1(); i++){
         idSaida[1] = i;
         for(int j = 0; j < filtros.dim2(); j++){
            idEntrada[1] = j;
            idKernel[0] = i;
            idKernel[1] = j;
            optensor.correlacao2D(entrada, filtros, saidaEsperada, idEntrada, idKernel, idSaida, true);
         }
      }

      conv.calcularSaida(entrada);

      System.out.println("Forward esperado: " + conv.somatorio.comparar(saidaEsperada));
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
}
