package rna.inicializadores;

import rna.core.Mat;
import rna.core.Tensor4D;

/**
 * Inicializador He para uso dentro da biblioteca.
 */
public class He extends Inicializador{

   /**
    * Instância um inicializador He para matrizes com seed
    * aleatória.
    */
   public He(){}
   
   /**
    * Instância um inicializador He para matrizes.
    * @param seed seed usada pelo gerador de números aleatórios.
    */
   public He(long seed){
      super(seed);
   }

   /**
    * Aplica o algoritmo de inicialização He nos pesos.
    * @param m matriz que será inicializada.
    */
   @Override
   public void inicializar(Mat m){
      double a = Math.sqrt(2.0 / m.lin());
      m.map((x) -> (
         a * super.random.nextGaussian()
      ));
   }

   @Override
   public void inicializar(Tensor4D tensor, int dim1, int dim2){
      double a = Math.sqrt(2.0 / tensor.dim3());

      for(int i = 0; i < tensor.dim3(); i++){
         for(int j = 0; j < tensor.dim4(); j++){
            tensor.editar(dim1, dim2, i, j, (
               a * super.random.nextGaussian()
            ));
         }
      }
   }
}
