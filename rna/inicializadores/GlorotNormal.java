package rna.inicializadores;

import rna.core.Mat;
import rna.core.Tensor4D;

/**
 * Inicializador Glorot normalizado para uso dentro da biblioteca.
 */
public class GlorotNormal extends Inicializador{

   /**
    * Instância um inicializador Glorot normalizado para matrizes 
    * com seed
    * aleatória.
    */
   public GlorotNormal(){}

   /**
    * Instância um inicializador Glorot normalizado para matrizes.
    * @param seed seed usada pelo gerador de números aleatórios.
    */
   public GlorotNormal(long seed){
      super(seed);
   }

   /**
    * Aplica o algoritmo de inicialização Glorot normalizado na matriz 
    * fornecida.
    * @param m matriz que será inicializada.
    */
   @Override
   public void inicializar(Mat m){
      double desvio = Math.sqrt(2.0 / (m.lin() + m.col()));
      m.map((x) -> (
         super.random.nextGaussian() * desvio
      ));
   }

   @Override
   public void inicializar(Tensor4D tensor, int dim1, int dim2){
      double desvio = Math.sqrt(2.0 / (tensor.dim3() + tensor.dim4()));

      for(int i = 0; i < tensor.dim3(); i++){
         for(int j = 0; j < tensor.dim4(); j++){
            tensor.editar(dim1, dim2, i, j, (
               super.random.nextGaussian() * desvio
            ));
         }
      }
   }
}
