package testes;

import lib.ged.Ged;
import lib.geim.Geim;
import rna.ativacoes.Ativacao;
import rna.ativacoes.Softmax;
import rna.core.OpArray;
import rna.core.OpMatriz;
import rna.core.OpTensor4D;
import rna.core.Tensor4D;
import rna.core.Utils;

public class Playground{
   static Ged ged = new Ged();
   static OpArray oparr = new OpArray();
   static OpMatriz opmat = new OpMatriz();
   static OpTensor4D optensor = new OpTensor4D();
   static Geim geim = new Geim();
   static Utils utils = new Utils();
   
   public static void main(String[] args){
      ged.limparConsole();

      Ativacao atv = new Softmax();
      Tensor4D tensor = new Tensor4D(new double[][]{
         {1, 2, 3},
         {1, 2, 3}
      });
      atv.forward(tensor, tensor);

      tensor.print(4);
   }

   /**
    * Mede o tempo de execução da função fornecida.
    * @param func função.
    * @return tempo em nanosegundos.
    */
   static long medirTempo(Runnable func){
      long t = System.nanoTime();
      func.run();
      return System.nanoTime() - t;
   }

}