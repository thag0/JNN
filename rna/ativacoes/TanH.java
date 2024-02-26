package rna.ativacoes;

import rna.core.Tensor4D;

public class TanH extends Ativacao{

   public TanH(){
      super.construir(
         (x) -> { return (2 / (1 + Math.exp(-2*x))) - 1; }, 
         (x) -> {
            double tanh = (2 / (1 + Math.exp(-2*x))) - 1;
            return 1 - (tanh * tanh);
         }
      );
   }

   @Override
   public void derivada(Tensor4D entrada, Tensor4D gradiente, Tensor4D saida){
      double[] e = entrada.paraArray();
      double[] g = gradiente.paraArray();
      
      //mais rápido
      for(int i = 0; i < e.length; i++){
         e[i] = dx.applyAsDouble(e[i]);
      }
      for(int i = 0; i < e.length; i++){
         e[i] *= g[i];
      }
      
      saida.copiarElementos(e);
   }
}
