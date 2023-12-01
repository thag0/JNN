package testes;

import ged.Ged;
import rna.core.Mat;
import rna.estrutura.Convolucional;

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

      Convolucional camada = new Convolucional(formatoEntrada, formatoFiltro, 1);
      camada.filtros[0][0] = new Mat(f1);
      camada.calcularSaida(entrada);

      for(int i = 0; i < camada.saida.length; i++){
         camada.saida[i].print("Saida " + i);
      }
   }
}
