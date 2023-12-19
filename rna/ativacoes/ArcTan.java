package rna.ativacoes;

import rna.estrutura.Densa;

public class ArcTan extends Ativacao{

   public ArcTan(){
      super.construir(this::arctan, this::arctand);
   }

   @Override
   public void calcular(Densa camada){
      super.aplicarFx(camada.somatorio, camada.saida);
   }

   @Override
   public void derivada(Densa camada){
      super.aplicarDx(camada.gradSaida, camada.somatorio, camada.derivada);
   }

   private double arctan(double x){
      return Math.atan(x);
   }

   private double arctand(double x){
      return 1.0 / (1.0 + (x * x));
   }
}
