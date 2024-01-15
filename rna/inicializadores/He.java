package rna.inicializadores;

import rna.core.Mat;

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
    * @param x valor usado apenas por outros inicializadores.
    */
   @Override
   public void inicializar(Mat m, double x){
      double a = Math.sqrt(2.0 / m.lin());
      m.forEach((i, j) -> {
         m.editar(i, j, (a * super.random.nextGaussian()));
      });
   }
}
