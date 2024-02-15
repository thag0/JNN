package rna.inicializadores;

import rna.core.Mat;
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

   /**
    * Inicializa os valores aleatoriamente dentro do intervalo {@code min : max}
    * @param m matriz que será inicializada.
    */
   @Override
   public void inicializar(Mat m){
      m.map((x) -> (
         super.random.nextDouble(min, max)
      ));
   }

   @Override
   public void inicializar(Tensor4D tensor, int dim1, int dim2){
      for(int i = 0; i < tensor.dim3(); i++){
         for(int j = 0; j < tensor.dim4(); j++){
            tensor.editar(dim1, dim2, i, j, (
               super.random.nextDouble(min, max)
            ));
         }
      }
   }
}
