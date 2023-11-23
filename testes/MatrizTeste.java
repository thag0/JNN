package testes;

import ged.Ged;
import rna.core.Mat;
import rna.core.Matriz;

public class MatrizTeste{
   static Ged ged = new Ged();
   
   public static void main(String[] args){
      Matriz m = new Matriz();

      Mat a = new Mat(2, 2, new double[]{
         1, 2, 
         3, 4
      });
      Mat b = new Mat(2, 2, new double[]{
         1, 2, 
         3, 4
      });
      Mat c = new Mat(2, 2);

      m.add(a, b, c);
      c.print();
   }
}
