package rna.inicializadores;

import rna.core.Mat;

/**
 * Inicializador de valores aleatórios positivos para uso dentro da biblioteca.
 */
public class AleatorioPositivo extends Inicializador{

   private double max = 0;

   /**
    * Instancia um inicializador de valores aleatórios positivos
    * com seed aleatória.
    * @param max valor máximo de aleatorização.
    */
   public AleatorioPositivo(double max){
      if(max <= 0){
         throw new IllegalArgumentException(
            "O valor máximo deve ser maior que zero."
         );
      }

      this.max = max;
   }

   /**
    * Instancia um inicializador de valores aleatórios positivos
    * com seed aleatória.
    */
   public AleatorioPositivo(){
      this(1.0);
   }

   /**
    * Instancia um inicializador de valores aleatórios positivos.
    * @param seed seed usada pelo gerador de números aleatórios.
    */
   public AleatorioPositivo(long seed){
      super(seed);
   }

   /**
    * Inicializa os valores aleatoriamente dentro do intervalo {@code 0 : max}
    * @param m matriz que será inicializada.
    */
   @Override
   public void inicializar(Mat m){
      m.map(val -> super.random.nextDouble(0, max));
   }
}
