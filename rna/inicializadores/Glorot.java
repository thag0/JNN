package rna.inicializadores;

import rna.core.Mat;

public class Glorot extends Inicializador{

   /**
    * Aplica o algoritmo de inicialização Glorot na matriz fornecida.
    * @param m matriz que será inicializada.
    * @param x valor utilizado apenas por outros otimizadores.
    */
   @Override
   public void inicializar(Mat m, double x){
      double desvio = Math.sqrt(6.0 / (m.lin() + m.col()));
      for(int i = 0; i < m.lin(); i++){
         for(int j = 0; j < m.col(); j++){
            double val = super.random.nextDouble() * (2 * desvio) - desvio;
            m.editar(i, j, val);
         }
      }
   }
}
