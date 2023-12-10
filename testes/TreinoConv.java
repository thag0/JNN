package testes;

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
      double[][] e2 = {
         {1, 6, 2},
         {5, 3, 1},
         {7, 0, 4},
      };
      
      double[][] f1 = {
         {1, 2},
         {-1, 0}
      };
      double[][] f2 = {
         {3, 5},
         {7, 8}
      };

      double[][][] entrada = new double[2][][];
      entrada[0] = e1;
      entrada[1] = e2;

      Convolucional camada = new Convolucional(new int[]{2, 2}, 1, "linear");
      camada.construir(new int[]{3, 3, 2});
      camada.filtros[0][0] = new Mat(f1);
      camada.filtros[0][1] = new Mat(f2);

      camada.calcularSaida(entrada);
      for(int i = 0; i < camada.saida.length; i++){
         camada.somatorio[i].print("saida " + i);
      }
   }
}
