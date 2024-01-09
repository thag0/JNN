package rna.ativacoes;

import rna.camadas.Convolucional;
import rna.camadas.Densa;

public class Sigmoid extends Ativacao{

   public Sigmoid(){
      super.construir(
         (x) -> { return 1 / (1 + Math.exp(-x)); },
         null
      );
   }

   @Override
   public void derivada(Densa camada){
      //forma manual pra aproveitar os valores pre calculados
      double grads[] = camada.gradSaida.paraArray();
      double deriv[] = camada.saida.paraArray();

      for(int i = 0; i < camada.saida.col(); i++){
         deriv[i] = deriv[i] * (1 - deriv[i]);
         deriv[i] *= grads[i];
      }
      
      camada.derivada.copiar(0, deriv);
   }

   @Override
   public void derivada(Convolucional camada){
      //forma manual pra aproveitar os valores pre calculados
      int i, j, k;
      double grad, d;

      for(i = 0; i < camada.somatorio.length; i++){
         for(j = 0; j < camada.somatorio[i].lin(); j++){
            for(k = 0; k < camada.somatorio[i].col(); k++){
               grad = camada.gradSaida[i].elemento(j, k);
               d = camada.saida[i].elemento(j, k);
               d = d * (1 - d);

               camada.derivada[i].editar(j, k, (grad * d));
            }
         }
      }
   }
}
