package rna.ativacoes;

import rna.camadas.Densa;

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
   public void backward(Densa densa){
      //aproveitar os resultados pre calculados

      double[] e = densa.saidaParaArray();
      double[] g = densa._gradSaida.paraArray();
      double[] d = new double[e.length];

      for(int i = 0; i < d.length; i++){
         d[i] = e[i]*(1 - e[i]) * g[i];
      }

      densa._gradSaida.copiarElementos(d);
   }
}
