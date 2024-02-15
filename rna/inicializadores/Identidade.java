package rna.inicializadores;

import rna.core.Mat;
import rna.core.Tensor4D;

/**
 * Inicializador de matriz identidade para uso dentro da biblioteca.
 */
public class Identidade extends Inicializador{

   /**
    * Instância um inicializador de matriz identidade.
    */
   public Identidade(){}

   /**
    * Inicializa todos os valores da matriz no formato de identidade.
    * @param m matriz que será inicializada.
    */
   @Override
   public void inicializar(Mat m){
      m.forEach((i, j) -> {
         m.editar(i, j, (i == j ? 1 : 0));
      });
   }

   @Override
   public void inicializar(Tensor4D tensor, int dim1, int dim2){

      for(int i = 0; i < tensor.dim3(); i++){
         for(int j = 0; j < tensor.dim4(); j++){
            tensor.editar(dim1, dim2, i, j, (
               (i == j ? 1 : 0)
            ));
         }
      }
   }
}
