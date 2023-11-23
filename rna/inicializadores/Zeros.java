package rna.inicializadores;

import rna.core.Mat;

public class Zeros extends Inicializador{
   
   /**
    * Inicializa todos os valores da matriz como zero.
    * @param m matriz que será inicializada.
    * @param alcance valor usado apenas por outros otimizadores.
    */
   @Override
   public void inicializar(Mat m, double alcance){
      for(int i = 0; i < m.lin; i++){
         for(int j = 0; j < m.col; j++){
            m.editar(i, j, 0);
         }
      }
   }
}
