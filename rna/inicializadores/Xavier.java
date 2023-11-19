package rna.inicializadores;

public class Xavier extends Inicializador{

   /**
    * Aplica o algoritmo de inicialização Xavier/Glorot nos pesos.
    * @param array array de pesos do neurônio
    * @param entradas quantidade de conexões do neurônio também corresponde a quantidade de saídas
    * da camada anterior.
    * @param saidas quantidade de saídas da camada em que o neurônio está presente.
    */
   @Override
   public void inicializar(double[][] m, double alcance){
      double a = Math.sqrt(2.0 / (m.length + m[0].length));

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
