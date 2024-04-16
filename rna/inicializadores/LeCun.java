package rna.inicializadores;

import rna.core.Tensor4D;

/**
 * Inicializador LeCun para uso dentro da biblioteca.
 */
public class LeCun extends Inicializador {
   
   /**
    * Instância um inicializador LeCun para matrizes com seed
    * aleatória.
    */
   public LeCun() {}
   
   /**
    * Instância um inicializador LeCun para matrizes.
    * @param seed seed usada pelo gerador de números aleatórios.
    */
   public LeCun(long seed) {
      super(seed);
   }

   @Override
   public void inicializar(Tensor4D tensor) {
      double variancia = Math.sqrt(1.0 / tensor.dim3());

      tensor.map(x -> super.random.nextGaussian() * variancia);
   }

   @Override
   public void inicializar(Tensor4D tensor, int dim1) {
      double variancia = Math.sqrt(1.0 / tensor.dim3());

      tensor.map3D(dim1, 
         x -> super.random.nextGaussian() * variancia
      );
   }

   @Override
   public void inicializar(Tensor4D tensor, int dim1, int dim2) {
      double variancia = Math.sqrt(1.0 / tensor.dim3());

      tensor.map2D(dim1, dim2, 
         x -> super.random.nextGaussian() * variancia
      );
   }

   @Override
   public void inicializar(Tensor4D tensor, int dim1, int dim2, int dim3) {
      double variancia = Math.sqrt(1.0 / tensor.dim3());

      tensor.map1D(dim1, dim2, dim3, 
         x -> super.random.nextGaussian() * variancia
      );
   }

}
