package rna.inicializadores;

import rna.core.Mat;

public class Constante extends Inicializador{
   
   /**
    * Inicializa todos os valores da matriz como zero.
    * @param m matriz que será inicializada.
    * @param alcance valor usado para preencher a matriz.
    */
   @Override
   public void inicializar(Mat m, double alcance){
      for(int i = 0; i < m.lin; i++){
         for(int j = 0; j < m.col; j++){
            m.editar(i, j, alcance);
         }
      }
   }
}
