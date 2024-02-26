package rna.ativacoes;

import rna.core.Tensor4D;

public class Sigmoid extends Ativacao{

   public Sigmoid(){
      super.construir(
         (x) -> { return 1 / (1 + Math.exp(-x)); },
         (x) -> { 
            double s = 1 / (1 + Math.exp(-x));
            return s * (1 - s);
         }
      );
   }

   @Override
   public void derivada(Tensor4D entrada, Tensor4D gradiente, Tensor4D saida){
      //mais eficiente
      double[] e = entrada.paraArray();
      double[] g = gradiente.paraArray();

      for(int i = 0; i < e.length; i++){
         e[i] = dx.applyAsDouble(e[i]);
         e[i] *= g[i];
      }
      
      saida.copiarElementos(e);
   }
}
