package rna.ativacoes;

import rna.camadas.Convolucional;
import rna.camadas.Densa;

public class TanH extends Ativacao{

   public TanH(){
      super.construir(
         (x) -> { return (2 / (1 + Math.exp(-2*x))) - 1; }, 
         null
      );
   }

   @Override
   public void derivada(Densa camada){
      //forma manual pra aproveitar os valores pre calculados
      double[] grads = camada.gradSaida.paraArray();
      double[] deriv = camada.saida.paraArray();

      int i, tamanho = grads.length;
      for(i = 0; i < tamanho; i++){
         deriv[i] = 1 - (deriv[i] * deriv[i]);
         deriv[i] *= grads[i];
      }
      
      camada.derivada.copiar(deriv, 0, 0, 0);
   }

   @Override
   public void derivada(Convolucional camada){
      //forma manual pra aproveitar os valores pre calculados
      int i, j, k;
      double grad, d;

      int prof = camada.somatorio.dim2();
      int alt = camada.somatorio.dim3();
      int larg = camada.somatorio.dim4();
      for(i = 0; i < prof; i++){
         for(j = 0; j < alt; j++){
            for(k = 0; k < larg; k++){
               grad = camada.gradSaida.elemento(0, i, j, k);
               d = camada.saida.elemento(0, i, j, k);
               d = 1 - (d * d);

               camada.derivada.editar(0, i, j, k, (grad * d));
            }
         }
      }
   }
}
