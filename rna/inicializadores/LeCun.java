package rna.inicializadores;

public class LeCun extends Inicializador{
   
   /**
    * Aplica o algoritmo de inicialização LeCun nos pesos.
    * @param array array de pesos do neurônio
    * @param entradas quantidade de conexões do neurônio também corresponde a quantidade de saídas
    * da camada anterior.
    */
   @Override
   public void inicializar(double[][] m, double alcance){
      double a = Math.sqrt(1.0 / m.length);
      for(int i = 0; i < m.length; i++){
         for(int j = 0; j < m[i].length; j++){
            m[i][j] = super.random.nextDouble(-a, a);
         }
      }
   }

   @Override
   public void configurarSeed(long seed){
      super.configurarSeed(seed);
   }
}
