package rna.inicializadores;

import rna.core.Tensor4D;

/**
 * Inicializador de valores aleatórios para uso dentro da biblioteca.
 */
public class Aleatorio extends Inicializador{

   /**
    * Valor mínimo de aleatorização.
    */
   private double min;

   /**
    * Valor máximo de aleatorização.
    */
   private double max;

   /**
    * Instancia um inicializador de valores aleatórios com seed
    * também aleatória.
    * @param min valor mínimo de aleatorização.
    * @param max valor máximo de aleatorização.
    */
   public Aleatorio(double min, double max){
      if(min >= max){
         throw new IllegalArgumentException(
            "O valor mínimo (" + min +") " +
            "deve ser menor que o valor máximo (" + max + ")."
         );
      }

      this.min = min;
      this.max = max;
   }

   /**
    * Instancia um inicializador de valores aleatórios.
    * @param min valor mínimo de aleatorização.
    * @param max valor máximo de aleatorização.
    * @param seed seed usada pelo gerador de números aleatórios.
    */
   public Aleatorio(double min, double max, long seed){
      super(seed);

      if(min >= max){
         throw new IllegalArgumentException(
            "O valor mínimo (" + min +") " +
            "deve ser menor que o valor máximo (" + max + ")."
         );
      }

      this.min = min;
      this.max = max;
   }

   /**
    * Instancia um inicializador de valores aleatórios com seed
    * também aleatória.
    */
   public Aleatorio(){
      this(-1, 1);
   }

   /**
    * Instancia um inicializador de valores aleatórios.
    * @param seed seed usada pelo gerador de números aleatórios.
    */
   public Aleatorio(long seed){
      super(seed);
      this.min = -1;
      this.max =  1;
   }

   @Override
   public void inicializar(Tensor4D tensor){
      tensor.map((x) -> {
         return super.random.nextDouble(min, max);
      });
   }

   @Override
   public void inicializar(Tensor4D tensor, int dim1){
      tensor.map3D(dim1, (x) -> {
         return super.random.nextDouble(min, max);
      });
   }

   @Override
   public void inicializar(Tensor4D tensor, int dim1, int dim2){
      tensor.map2D(dim1, dim2, (x) -> {
         return super.random.nextDouble(min, max);
      });
   }

   @Override
   public void inicializar(Tensor4D tensor, int dim1, int dim2, int dim3){
      tensor.map1D(dim1, dim2, dim3, (x) -> {
         return super.random.nextDouble(min, max);
      });
   }
}
