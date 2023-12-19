package rna.ativacoes;

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
      super.construir(this::swish, this::swishd);
   }

   @Override
   public void calcular(Densa camada){
      super.aplicarFx(camada.somatorio, camada.saida);
   }

   @Override
   public void derivada(Densa camada){
      super.aplicarDx(camada.gradSaida, camada.somatorio, camada.derivada);
   }

   private double swish(double x){
      return x * sigmoid(x);
   }

   private double swishd(double x){
      double sig = sigmoid(x);
      return sig + (x * sig * (1 - sig));
   }

   private double sigmoid(double x){
      return 1 / (1 + Math.exp(-x));
   }
}
