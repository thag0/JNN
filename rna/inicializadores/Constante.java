package rna.inicializadores;

import rna.core.Mat;
import rna.core.Tensor4D;

/**
 * Inicializador de valor constante para uso dentro da biblioteca.
 */
public class Constante extends Inicializador{
   
   /**
    * Valor de preenchimento.
    */
   private double val = 0;

   /**
    * Instância um inicializador de valor constante.
    * @param val valor usado de constante na inicialização.
    */
   public Constante(double val){
      this.val = val;
   }

   /**
    * Instância um inicializador de valor constante.
    */
   public Constante(){
      this(0);
   }

   /**
    * Inicializa todos os valores da matriz com um valor constante.
    * @param m matriz que será inicializada.
    */
   @Override
   public void inicializar(Mat m){
      m.preencher(val);
   }

   @Override
   public void inicializar(Tensor4D tensor, int dim1, int dim2){
      for(int i = 0; i < tensor.dim3(); i++){
         for(int j = 0; j < tensor.dim4(); j++){
            tensor.editar(dim1, dim2, i, j, val);
         }
      }
   }
}
