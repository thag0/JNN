package rna.inicializadores;

import rna.core.Mat;

public class Zeros extends Inicializador{

   /**
    * Inizialiciador para matrizes com valor zero.
    */
   public Zeros(){}
   
   /**
    * Inicializa todos os valores da matriz como zero.
    * @param m matriz que será inicializada.
    */
   @Override
   public void inicializar(Mat m){
      m.preencher(0);
   }
}
