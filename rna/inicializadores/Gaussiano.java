package rna.inicializadores;

import rna.core.Mat;
import rna.core.Tensor4D;

/**
 * Inicializador Gaussiano para uso dentro da biblioteca.
 */
public class Gaussiano extends Inicializador{

   /**
    * Instância um inicializador Gaussiano para matrizes com seed
    * aleatória.
    */
   public Gaussiano(){}

   /**
    * Instância um inicializador Gaussiano para matrizes.
    * @param seed seed usada pelo gerador de números aleatórios.
    */
   public Gaussiano(long seed){
      super(seed);
   }

   /**
    * Aplica o algoritmo de inicialização gaussiano/normal nos pesos.
    * @param m matriz que será inicializada
    */
   @Override
   public void inicializar(Mat m){
      m.map((x) -> (
         super.random.nextGaussian()
      ));
   }

   @Override
   public void inicializar(Tensor4D tensor, int dim1, int dim2){
      for(int i = 0; i < tensor.dim3(); i++){
         for(int j = 0; j < tensor.dim4(); j++){
            tensor.editar(dim1, dim2, i, j, (
               super.random.nextGaussian()
            ));
         }
      }
   }
}
