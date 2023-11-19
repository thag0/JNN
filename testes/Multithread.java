package testes;

import ged.Ged;
import rna.core.Matriz;

public class Multithread {
   public static void main(String[] args){
      Ged ged = new Ged();

      int lin = 1024;
      int col = lin;

      double[][] a = new double[lin][col];
      double[][] b = new double[lin][col];
      double[][] r = new double[lin][col];

      for(int i = 0; i < a.length; i++){
         for(int j = 0; j < a[i].length; j++){
            a[i][j] = (i* a.length) + j + 1;
         }
      }
      // ged.imprimirMatriz(a);

      ged.matIdentidade(b);
      Matriz.multT(a, b, r);
      // ged.imprimirMatriz(r);
      System.out.println("concluído.");
   }

}
