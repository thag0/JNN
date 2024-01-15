package rna.inicializadores;

import rna.core.Mat;

/**
 * Inicializador de valores aleatórios para uso dentro da biblioteca.
 */
public class Aleatorio extends Inicializador{

   /**
    * Instancia um inicializador de valores aleatórios com seed
    * também aleatória.
    */
   public Aleatorio(){}

   /**
    * Instancia um inicializador de valores aleatórios.
    * @param seed seed usada pelo gerador de números aleatórios.
    */
   public Aleatorio(long seed){
      super(seed);
   }

   /**
    * Inicializa os valores aleatoriamente dentro do intervalo {@code -x : +x}
    * @param m matriz que será inicializada.
    * @param x valor que será usado como máximo e mínimo na aleatorização.
    */
   @Override
   public void inicializar(Mat m, double x){
      m.forEach((i, j) -> {
         m.editar(i, j, super.random.nextDouble(-x, x));
      });
   }
}
