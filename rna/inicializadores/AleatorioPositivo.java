package rna.inicializadores;

import rna.core.Mat;

/**
 * Inicializador de valores aleatórios positivos para uso dentro da biblioteca.
 */
public class AleatorioPositivo extends Inicializador{

   /**
    * Instancia um inicializador de valores aleatórios positivos
    * com seed aleatória.
    */
   public AleatorioPositivo(){}

   /**
    * Instancia um inicializador de valores aleatórios positivos.
    * @param seed seed usada pelo gerador de números aleatórios.
    */
   public AleatorioPositivo(long seed){
      super(seed);
   }

   /**
    * Inicializa os valores aleatoriamente dentro do intervalo {@code 0 : x}
    * @param m matriz que será inicializada.
    * @param x valor que será usado como máximo e mínimo na aleatorização.
    */
   @Override
   public void inicializar(Mat m, double x){
      m.forEach((i, j) -> {
         m.editar(i, j, super.random.nextDouble(0, x));
      });
   }
}
