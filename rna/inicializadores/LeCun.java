package rna.inicializadores;

import rna.core.Mat;

public class LeCun extends Inicializador{
   
   /**
    * Aplica o algoritmo de inicialização LeCun nos pesos.
    * @param m matriz que será inicializada.
    * @param x valor utilizado apenas por outros otimizadores.
    */
   @Override
   public void inicializar(Mat m, double x){
      double a = Math.sqrt(1.0 / m.lin());

      for(int i = 0; i < m.lin(); i++){
         for(int j = 0; j < m.col(); j++){
            // m.editar(i, j, super.random.nextDouble(-a, a));
            m.editar(i, j, (a * super.random.nextGaussian()));
         }
      }
   }
}
