package testes;

import ged.Ged;
import rna.core.OpTensor;
import rna.core.Tensor;

public class TesteTensor{
   static Ged ged = new Ged();
   static OpTensor opten = new OpTensor();

   public static void main(String[] args){
      ged.limparConsole();

      Tensor t1 = new Tensor(new int[]{2, 2, 2});
      Tensor t2 = new Tensor(new int[]{2, 2, 2});
      Tensor t3 = new Tensor(new int[]{2, 2, 2});

      t1.preencher(1);
      t2.preencher(2);
      
      opten.addEixo(t1, t2, t3, 1);

      t3.print();

      ged.imprimirArray(t3.paraArray(), "Tensor");
   }
}
