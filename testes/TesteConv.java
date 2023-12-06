package testes;

import ged.Ged;
import rna.avaliacao.perda.ErroMedioQuadrado;
import rna.core.Mat;
import rna.estrutura.*;
import rna.inicializadores.Inicializador;
import rna.inicializadores.Xavier;
import rna.otimizadores.Otimizador;
import rna.otimizadores.SGD;

public class TesteConv{
   static Ged ged = new Ged();

   public static void main(String[] args){
      ged.limparConsole();
      
      double[][] e = {
         {1, 6, 2},
         {5, 3, 1},
         {7, 0, 4},
      };
      
      double[][] f1 = {
         {1, 2},
         {-1, 0},
      };
      
      int[] formatoEntrada = {e.length, e[0].length, 1};
      int[] formatoFiltro = {f1.length, f1[0].length};

      double[][][] entrada = new double[1][][];
      entrada[0] = e;

      // ---------------------------------------------------------
      Inicializador ini = new Xavier();

      Convolucional conv = new Convolucional(formatoEntrada, formatoFiltro, 2, false);
      conv.inicializar(ini, ini, 0);
      conv.filtros[0][0] = new Mat(f1);

      Flatten flat = new Flatten(conv.formatoSaida());
      
      Densa densa1 = new Densa(conv.tamanhoSaida(), 5);
      densa1.configurarAtivacao("tanh");
      densa1.inicializar(ini, ini, 0);
      
      Densa densa2 = new Densa(densa1.tamanhoSaida(), 3);
      densa2.configurarAtivacao("sigmoid");
      densa2.inicializar(ini, ini, 0);

      ConvNet cnn = new ConvNet();
      cnn.add(conv);
      cnn.add(flat);
      cnn.add(densa1);
      cnn.add(densa2);

      cnn.calcularSaida(entrada);
      
      double[] real = {0.0, 1.0, 0.0};
      ErroMedioQuadrado emq = new ErroMedioQuadrado();
      Otimizador otm = new SGD(0.001, 0.995);

      cnn.treinar(entrada, real, 3000, emq, otm);
   

      ged.imprimirArray(cnn.obterSaida(), "saida cnn");
   }
}
