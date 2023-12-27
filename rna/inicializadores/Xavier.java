package rna.inicializadores;

import rna.core.Mat;

public class Xavier extends Inicializador{

   /**
    * Aplica o algoritmo de inicialização Xavier/Glorot nos pesos.
    * @param m matriz que será inicializada.
    * @param x valor utilizado apenas por outros otimizadores.
    */
   @Override
   public void inicializar(Mat m, double x){
      double desvio = Math.sqrt(2.0 / (m.lin() + m.col()));
      for(int i = 0; i < m.lin(); i++){
         for(int j = 0; j < m.col(); j++){
            m.editar(i, j, (desvio * super.random.nextGaussian()));
         }
      }
   }

   @Override
   public void configurarSeed(long seed){
      super.configurarSeed(seed);
   }
}
