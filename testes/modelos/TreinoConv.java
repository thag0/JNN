package testes.modelos;

import ged.Ged;
import rna.core.Mat;
import rna.core.OpMatriz;
import rna.estrutura.*;

public class TreinoConv{
   static Ged ged = new Ged();
   static OpMatriz opmat = new OpMatriz();
   public static void main(String[] args){
      ged.limparConsole();
      
      double[][] e1 = {
         {1, 6, 2},
         {5, 3, 1},
         {7, 0, 4},
      };
      
      double[][] f1 = {
         { 1, 2},
         {-1, 0}
      };

      double[][] g1 = {
         {2, 3},
         {4, 5},
      };

      double[][][] entrada = new double[1][][];
      entrada[0] = e1;

      Convolucional camada = new Convolucional(new int[]{2, 2}, 1, "linear");
      camada.construir(new int[]{3, 3, 1});
      camada.filtros[0][0] = new Mat(f1);
      camada.bias[0] = new Mat(new double[][]{
         {1, 2},
         {3, 4}
      });

      camada.calcularSaida(entrada);
      camada.calcularGradiente(new Mat[]{new Mat(g1)});

      //valores de saída e gradientes da camada convolucional estão corretos
      //parâmetros dos inicializadores corretos
   }
}
