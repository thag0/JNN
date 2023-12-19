package rna.inicializadores;

import rna.core.Mat;

public class He extends Inicializador{

   /**
    * Aplica o algoritmo de inicialização He nos pesos.
    * @param m matriz que será inicializada.
    * @param x valor utilizado apenas por outros otimizadores.
    */
   @Override
   public void inicializar(Mat m, double x){
      double a = Math.sqrt(2.0 / m.lin());

      for(int i = 0; i < m.lin(); i++){
         for(int j = 0; j < m.col(); j++){
            m.editar(i, j, super.random.nextDouble(-a, a));
         }
      }
   }

   @Override
   public void configurarSeed(long seed){
      super.configurarSeed(seed);
   }
}
