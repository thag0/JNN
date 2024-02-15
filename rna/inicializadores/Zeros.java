package rna.inicializadores;

import rna.core.Mat;
import rna.core.Tensor4D;

public class Zeros extends Inicializador{

   /**
    * Inizialiciador para matrizes com valor zero.
    */
   public Zeros(){}
   
   /**
    * Inicializa todos os valores da matriz como zero.
    * @param m matriz que será inicializada.
    */
   @Override
   public void inicializar(Mat m){
      m.preencher(0);
   }

   @Override
   public void inicializar(Tensor4D tensor, int dim1, int dim2){
      for(int i = 0; i < tensor.dim3(); i++){
         for(int j = 0; j < tensor.dim4(); j++){
            tensor.editar(dim1, dim2, i, j, 0);
         }
      }
   }
}
