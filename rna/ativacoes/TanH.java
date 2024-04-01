package rna.ativacoes;

import rna.camadas.Densa;

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
   public void derivada(Densa densa){
      //aproveitar os resultados pre calculados

      double[] e = densa.saida().paraArray();
      double[] g = densa.gradSaida.paraArray();
      double[] d = new double[e.length];

      for(int i = 0; i < d.length; i++){
         d[i] = (1 - (e[i]*e[i])) * g[i];
      }

      densa.gradSaida.copiarElementos(d);
   }
}
