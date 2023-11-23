package rna.inicializadores;

import rna.core.Mat;

public class LeCun extends Inicializador{
   
   /**
    * Aplica o algoritmo de inicialização LeCun nos pesos.
    * @param m matriz que será inicializada.
    * @param alcance valor utilizado apenas por outros otimizadores.
    */
   @Override
   public void inicializar(Mat m, double alcance){
      double a = Math.sqrt(1.0 / m.lin);

      for(int i = 0; i < m.lin; i++){
         for(int j = 0; j < m.col; j++){
            m.editar(i, j, super.random.nextDouble(-a, a));
         }
      }
   }

   @Override
   public void configurarSeed(long seed){
      super.configurarSeed(seed);
   }
}
