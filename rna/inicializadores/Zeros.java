package rna.inicializadores;

import rna.core.Mat;

public class Zeros extends Inicializador{
   
   /**
    * Inicializa todos os valores da matriz como zero.
    * @param m matriz que será inicializada.
    * @param x valor usado apenas por outros otimizadores.
    */
   @Override
   public void inicializar(Mat m, double x){
      m.preencher(0);
   }
}
