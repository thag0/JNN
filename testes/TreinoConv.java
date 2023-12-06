package testes;

import ged.Ged;
import rna.core.Mat;
import rna.estrutura.Camada;
import rna.estrutura.Convolucional;
import rna.otimizadores.*;

public class TreinoConv{
   static Ged ged = new Ged();
   public static void main(String[] args){
      ged.limparConsole();

      int[] formEntrada = {4, 4, 1};
      int[] formFiltro =  {2, 2};
      Convolucional camada = new Convolucional(formEntrada, formFiltro, 2);
      camada.filtros[0][0] = new Mat(new double[][]{{1., 2.}, {3., 4.}});
      camada.filtros[1][0] = new Mat(new double[][]{{5., 6.}, {7., 8.}});

      double[][] grad1 = {
         {1, 2, 3},
         {4, 5, 6},
         {7, 8, 9}
      };
      double[][] grad2 = {
         {1, 2, 3},
         {4, 5, 6},
         {7, 8, 9}
      };
      double[][][] gradSaida = {grad1, grad2};

      double[][] entrada1 = {
         {2, 2, 2, 2},
         {2, 2, 2, 2},
         {2, 2, 2, 2},
         {2, 2, 2, 2},
      };

      camada.entrada[0] = new Mat(entrada1);

      Camada[] redec = {camada};
      Otimizador otm = new SGD();
      otm.inicializar(redec);
      
      ged.imprimirArray(camada.obterKernel(), "kernel");
      for(int i = 0; i < 100; i++){
         camada.calcularGradiente(gradSaida);
         otm.atualizar(redec);
      }
      ged.imprimirArray(camada.obterKernel(), "kernel");
   }
}
