package rna.ativacoes;

import rna.core.Tensor4D;

public class TanH extends Ativacao{

   public TanH(){
      super.construir(
         (x) -> { return (2 / (1 + Math.exp(-2*x))) - 1; }, 
         (x) -> {
            double t = (2 / (1 + Math.exp(-2*x))) - 1;
            return 1 - (t * t);
         }
      );
   }

   @Override
   public void derivada(Tensor4D entrada, Tensor4D gradiente, Tensor4D saida){
      double[] e = entrada.paraArray();
      double[] g = gradiente.paraArray();
      double[] s = new double[e.length];
   
      for(int i = 0; i < e.length; i++){
         s[i] = dx.applyAsDouble(e[i]) * g[i];
      }
      
      saida.copiarElementos(s);
   }
}
