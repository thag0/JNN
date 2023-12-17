package rna.inicializadores;

import rna.core.Mat;

public class Constante extends Inicializador{
   
   /**
    * Inicializa todos os valores da matriz com um valor constante.
    * @param m matriz que será inicializada.
    * @param x valor usado para preencher a matriz.
    */
   @Override
   public void inicializar(Mat m, double x){
      m.preencher(x);
   }
}
