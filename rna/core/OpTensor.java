package rna.core;

import java.util.Arrays;

public class OpTensor{
   public OpTensor(){

   }

   private void verificarCompatibilidade(Tensor a, Tensor b){
      if(Arrays.equals(a.formato(), b.formato()) == false){
         throw new IllegalArgumentException(
            "Os tensores devem conter o mesmo formato."
         );
      }
   }

   /**
    * Adiciona a soma de todos os elementos do tensor A e B no tensor R.
    * @param a
    * @param b
    * @param r
    */
   public void add(Tensor a, Tensor b, Tensor r){
      verificarCompatibilidade(a, b);
      verificarCompatibilidade(a, r);

      int[] indices = new int[a.formato().length];
      addNivel(a, b, r, indices, 0);
   }

   private void addNivel(Tensor a, Tensor b, Tensor res, int[] indices, int nivel){
      if(nivel == a.formato().length - 1){
         for(int i = 0; i < a.formato()[nivel]; i++){
            indices[nivel] = i;
            res.add(a.dado(indices) + b.dado(indices), indices);
         }
      
      }else{
         for(int i = 0; i < a.formato()[nivel]; i++){
            indices[nivel] = i;
            addNivel(a, b, res, indices, nivel + 1);
         }
      }
   }

   public void addEixo(Tensor a, Tensor b, Tensor r, int dim){
      // Verifica se os tensores são compatíveis
      verificarCompatibilidade(a, b);
      verificarCompatibilidade(a, r);
    
      // Itera sobre todas as dimensões do tensor, começando pela mais externa
      for (int i = a.formato().length - 1; i >= 0; i--) {
        // Verifica se a dimensão atual é igual ao eixo especificado
        if (i == dim) {
          // Itera sobre todos os elementos da dimensão atual
          for (int j = 0; j < a.formato()[i]; j++) {
            // Adiciona os dados e edita o tensor r
            r.editar(a.dado(i, j) + b.dado(i, j), i, j);
          }
        }
      }
    }


}
