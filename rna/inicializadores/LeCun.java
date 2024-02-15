package rna.inicializadores;

import rna.core.Mat;
import rna.core.Tensor4D;

/**
 * Inicializador LeCun para uso dentro da biblioteca.
 */
public class LeCun extends Inicializador{
   
   /**
    * Instância um inicializador LeCun para matrizes com seed
    * aleatória.
    */
   public LeCun(){}
   
   /**
    * Instância um inicializador LeCun para matrizes.
    * @param seed seed usada pelo gerador de números aleatórios.
    */
   public LeCun(long seed){
      super(seed);
   }

   /**
    * Aplica o algoritmo de inicialização LeCun na matriz fornecida.
    * @param m matriz que será inicializada.
    */
   @Override
   public void inicializar(Mat m){
      double variancia = Math.sqrt(1.0 / m.lin());
      m.map((x) -> (
         super.random.nextGaussian() * variancia
      ));
   }

   @Override
   public void inicializar(Tensor4D tensor, int dim1, int dim2){
      double variancia = Math.sqrt(1.0 / tensor.dim3());

      for(int i = 0; i < tensor.dim3(); i++){
         for(int j = 0; j < tensor.dim4(); j++){
            tensor.editar(dim1, dim2, i, j, (
               super.random.nextGaussian() * variancia
            ));
         }
      }
   }
}
