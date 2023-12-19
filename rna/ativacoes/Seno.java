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
      super.construir(this::seno, this::senod);
   }

   @Override
   public void calcular(Densa camada){
      super.aplicarFx(camada.somatorio, camada.saida);
   }

   @Override
   public void derivada(Densa camada){
      super.aplicarDx(camada.gradSaida, camada.somatorio, camada.derivada);
   }

   private double seno(double x){
      return Math.sin(x);
   }

   private double senod(double x){
      return Math.cos(x);
   }
   
}
