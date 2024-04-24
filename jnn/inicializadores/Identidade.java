package jnn.inicializadores;

import jnn.core.Tensor4D;

/**
 * Inicializador de matriz identidade para uso dentro da biblioteca.
 */
public class Identidade extends Inicializador {

   /**
    * Instância um inicializador de matriz identidade.
    */
   public Identidade() {}

   @Override
   public void inicializar(Tensor4D tensor) {
      int canais = tensor.dim1();
      int profundidade = tensor.dim2();

      for (int c = 0; c < canais; c++) {
         for (int p = 0; p < profundidade; p++) {
            for (int i = 0; i < tensor.dim3(); i++) {
               for (int j = 0; j < tensor.dim4(); j++) {
                  tensor.set(c, p, i, j, (
                     (i == j ? 1 : 0)
                  ));
               }
            }
         }
      }
   }

   @Override
   public void inicializar(Tensor4D tensor, int dim1) {
      int profundidade = tensor.dim2();

      for (int p = 0; p < profundidade; p++) {
         for (int i = 0; i < tensor.dim3(); i++) {
            for (int j = 0; j < tensor.dim4(); j++) {
               tensor.set(dim1, p, i, j, (
                  (i == j ? 1 : 0)
               ));
            }
         }
      }
   }

   @Override
   public void inicializar(Tensor4D tensor, int dim1, int dim2) {
      for (int i = 0; i < tensor.dim3(); i++) {
         for (int j = 0; j < tensor.dim4(); j++) {
            tensor.set(dim1, dim2, i, j, (
               (i == j ? 1 : 0)
            ));
         }
      }
   }

   @Override
   public void inicializar(Tensor4D tensor, int dim1, int dim2, int dim3) {
      for (int j = 0; j < tensor.dim4(); j++) {
         tensor.set(dim1, dim2, dim3, j, (
            (dim3 == j ? 1 : 0)
         ));
      }
   }
}
