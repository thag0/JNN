package rna.inicializadores;

public class Aleatorio extends Inicializador{

   /**
    * Inicializa os pesos aleatoriamente dentro do intervalo {@code -alcance : +alcance}
    * @param array array de pesos do neurônio
    * @param alcance valor que será usado como máximo e mínimo na aleatorização.
    */
   @Override
   public void inicializar(double[][] m, double alcance){
      for(int i = 0; i < m.length; i++){
         for(int j = 0; j < m[i].length; j++){
            m[i][j] = super.random.nextDouble(-alcance, alcance);
         }
      }
   }

   @Override
   public void configurarSeed(long seed){
      super.configurarSeed(seed);
   }
}
