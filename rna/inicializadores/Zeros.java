package rna.inicializadores;

public class Zeros extends Inicializador{
   
   /**
    * Inicializa todos os valores da matriz como zero.
    * @param array array de pesos do neurônio
    * @param entradas quantidade de conexões do neurônio também corresponde a quantidade de saídas
    * da camada anterior.
    */
   @Override
   public void inicializar(double[][] m, double alcance){
      for(int i = 0; i < m.length; i++){
         for(int j = 0; j < m[i].length; j++){
            m[i][j] = 0;
         }
      }
   }
}
