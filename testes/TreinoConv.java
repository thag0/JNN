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

      camada.calcularSaida(entrada);
      camada.calcularGradiente(new Mat[]{new Mat(g1)});

      //gradient descent "manual"
      double lr = 1;
      opmat.escalar(camada.gradFiltros[0][0], lr, camada.gradFiltros[0][0]);
      opmat.escalar(camada.gradBias[0], lr, camada.gradBias[0]);

      camada.gradFiltros[0][0].print("grad kernel");
      camada.gradBias[0].print("grad bias");
      camada.gradEntrada[0].print("grad entrada");
      
      opmat.sub(camada.filtros[0][0], camada.gradFiltros[0][0], camada.filtros[0][0]);
      opmat.sub(camada.bias[0], camada.gradBias[0], camada.bias[0]);
   }
}
