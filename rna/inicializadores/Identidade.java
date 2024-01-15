package rna.inicializadores;

import rna.core.Mat;

/**
 * Inicializador de matriz identidade para uso dentro da biblioteca.
 */
public class Identidade extends Inicializador{

   /**
    * Instância um inicializador de matriz identidade.
    */
   public Identidade(){}

   /**
    * Inicializa todos os valores da matriz no formato de identidade.
    * @param m matriz que será inicializada.
    * @param x valor usado apenas por outros inicializadores.
    */
   @Override
   public void inicializar(Mat m, double x){
      m.forEach((i, j) -> {
         m.editar(i, j, (i == j ? 1 : 0));
      });
   }  
}
