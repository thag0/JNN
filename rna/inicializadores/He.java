package rna.inicializadores;

import rna.core.Tensor4D;

/**
 * Inicializador He para uso dentro da biblioteca.
 */
public class He extends Inicializador {

   /**
    * Instância um inicializador He para matrizes com seed
    * aleatória.
    */
   public He() {}

   @Override
   public void inicializar(Tensor4D tensor) {
      double a = Math.sqrt(2.0 / tensor.dim3());

      tensor.map(x -> a * super.random.nextGaussian());
   }

   @Override
   public void inicializar(Tensor4D tensor, int dim1) {
      double a = Math.sqrt(2.0 / tensor.dim3());
      
      tensor.map3D(dim1, 
         x ->  a * super.random.nextGaussian()
      );
   }

   @Override
   public void inicializar(Tensor4D tensor, int dim1, int dim2) {
      double a = Math.sqrt(2.0 / tensor.dim3());
      
      tensor.map2D(dim1, dim2, 
         x ->  a * super.random.nextGaussian()
      );
   }

   @Override
   public void inicializar(Tensor4D tensor, int dim1, int dim2, int dim3) {
      double a = Math.sqrt(2.0 / tensor.dim3());
      
      tensor.map1D(dim1, dim2, dim3, 
         x ->  a * super.random.nextGaussian()
      );
   }
}
