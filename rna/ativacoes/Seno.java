package rna.ativacoes;

import rna.estrutura.Densa;

/**
 * Implementação da função de ativação Seno para uso dentro 
 * da {@code Rede Neural}.
 */
public class Seno extends Ativacao{

   /**
    * Instancia a função de ativação Seno.
    */
   public Seno(){

   }

   @Override
   public void calcular(Densa camada){
      super.aplicarFuncao(camada.somatorio, this::seno, camada.saida);
   }

   @Override
   public void derivada(Densa camada){
      super.aplicarDerivada(camada.gradSaida, camada.somatorio, this::senod, camada.derivada);
   }

   private double seno(double x){
      return Math.sin(x);
   }

   private double senod(double x){
      return Math.cos(x);
   }
   
}
