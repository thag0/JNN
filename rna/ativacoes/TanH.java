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
      double grads[] = camada.gradSaida.paraArray();
      double deriv[] = camada.saida.paraArray();

      for(int i = 0; i < camada.saida.col(); i++){
         deriv[i] = 1 - (deriv[i] * deriv[i]);
         deriv[i] *= grads[i];
      }
      
      camada.derivada.copiar(0, deriv);
   }

   @Override
   public void derivada(Convolucional camada){
      //forma manual pra aproveitar os valores pre calculados
      int i, j, k;
      double grad, d;
      int linhas, colunas;

      for(i = 0; i < camada.somatorio.length; i++){
         linhas = camada.somatorio[i].lin();
         colunas = camada.somatorio[i].col();
         for(j = 0; j < linhas; j++){
            for(k = 0; k < colunas; k++){
               grad = camada.gradSaida[i].elemento(j, k);
               d = camada.saida[i].elemento(j, k);
               d = 1 - (d * d);

               camada.derivada[i].editar(j, k, (grad * d));
            }
         }
      }
   }
}
