package rna.ativacoes;

import rna.estrutura.Densa;

/**
 * Implementação da função de ativação SoftPlus para uso 
 * dentro da {@code Rede Neural}.
 */
public class SoftPlus extends Ativacao{

   /**
    * Instancia a função de ativação SoftPlus.
    */
   public SoftPlus(){
      super.construir(this::softplus, this::softplusd);
   }
   @Override
   public void calcular(Densa camada){
      super.aplicarFuncao(camada.somatorio, camada.saida);
   }

   @Override
   public void derivada(Densa camada){
      super.aplicarDerivada(camada.gradSaida, camada.somatorio, camada.derivada);
   }

   private double softplus(double x){
      return Math.log(1 + Math.exp(x));
   }

   private double softplusd(double x){
      double exp = Math.exp(x);
      return (exp / (1 + exp));
   }
}
