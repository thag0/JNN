package rna.inicializadores;

import rna.core.Mat;

public class Gaussiano extends Inicializador{

   /**
    * Aplica o algoritmo de inicialização gaussiano/normal nos pesos.
   * @param m matriz que será inicializada
   * @param x valor de alcance da aleatorização
   */
   @Override
   public void inicializar(Mat m, double x){
      for(int i = 0; i < m.lin(); i++){
         for(int j = 0; j < m.col(); j++){
            m.editar(i, j, (2 * x * super.random.nextGaussian()));
         }
      }
   }
}
