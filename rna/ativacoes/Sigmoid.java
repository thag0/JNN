package rna.ativacoes;

import rna.camadas.Densa;

/**
 * Implementação da função de ativação Sigmóide para uso dentro 
 * dos modelos.
 * <p>
 *    A função Sigmóide é uma função de ativação bastante utilizada 
 *    em redes neurais, mapeando qualquer valor para o intervalo [0, 1].
 * </p>
 */
public class Sigmoid extends Ativacao {

   /**
    * Instancia a função de ativação Sigmoid.
    */
   public Sigmoid() {
      construir(
         x -> 1 / (1 + Math.exp(-x)),
         x -> { 
            double s = 1 / (1 + Math.exp(-x));
            return s * (1 - s);
         }
      );
   }

   @Override
   public void backward(Densa densa) {
      //aproveitar os resultados pre calculados

      double[] e = densa.saidaParaArray();
      double[] g = densa._gradSaida.paraArray();
      double[] d = new double[e.length];

      for (int i = 0; i < d.length; i++) {
         d[i] = e[i]*(1 - e[i]) * g[i];
      }

      densa._gradSaida.copiarElementos(d);
   }
}
