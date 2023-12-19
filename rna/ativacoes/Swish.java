package rna.ativacoes;

import rna.estrutura.Convolucional;
import rna.estrutura.Densa;

/**
 * Implementação da função de ativação Swish para uso 
 * dentro da {@code Rede Neural}.
 */
public class Swish extends Ativacao{

   /**
    * Instancia a função de ativação Swish.
    */
   public Swish(){
      super.construir(
         (x) -> { return x * sigmoid(x); }, 
         (x) -> {
            double sig = sigmoid(x);
            return sig + (x * sig * (1 - sig));
         }
      );
   }

   private double sigmoid(double x){
      return 1 / (1 + Math.exp(-x));
   }

   @Override
   public void calcular(Densa camada){
      super.aplicarFx(camada.somatorio, camada.saida);
   }

   @Override
   public void derivada(Densa camada){
      super.aplicarDx(camada.gradSaida, camada.somatorio, camada.derivada);
   }

   @Override
   public void calcular(Convolucional camada){
      for(int i = 0; i < camada.somatorio.length; i++){
         super.aplicarFx(camada.somatorio[i], camada.saida[i]);
      }
   }

   @Override
   public void derivada(Convolucional camada){
      for(int i = 0; i < camada.somatorio.length; i++){
         super.aplicarDx(camada.gradSaida[i], camada.somatorio[i], camada.derivada[i]);
      }
   }
}
