package testes;

import ged.Ged;
import rna.avaliacao.perda.EntropiaCruzada;
import rna.core.Mat;
import rna.core.Matriz;
import rna.estrutura.CamadaDensa;
import rna.inicializadores.Inicializador;
import rna.inicializadores.Xavier;

@SuppressWarnings("unused")
public class MatrizTeste{
   static Ged ged = new Ged();
   static Matriz mat = new Matriz();
   
   public static void main(String[] args){
      ged.limparConsole();

      double[] d = {
         1, 2, 3,
         4, 5, 6
      };

      double[][] m = {
         {1, 2, 3},
         {4, 5, 6},
         {7, 8, 9},
      };

      Mat a = new Mat(m);
      a.print();
   }
}
