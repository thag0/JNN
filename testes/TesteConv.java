package testes;

import ged.Ged;
import rna.core.Mat;
import rna.estrutura.Convolucional;
import rna.inicializadores.Inicializador;
import rna.inicializadores.Xavier;

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

      Inicializador ini = new Xavier();
      ini.configurarSeed(1234);

      Convolucional camada = new Convolucional(formatoEntrada, formatoFiltro, 1, false);
      camada.inicializar(ini, ini, 0);
      camada.filtros[0][0] = new Mat(f1);
      camada.configurarAtivacao("relu");
      camada.calcularSaida(entrada);

      for(int i = 0; i < camada.filtros.length; i++){
         for(int j = 0; j < camada.filtros[i].length; j++){
            camada.filtros[i][j].print("Filtro " + i + " entrada " + j);
         }
      }

      System.out.println();
      for(int i = 0; i < camada.saida.length; i++){
         camada.saida[i].print("Saida " + i);
      }
   }
}
