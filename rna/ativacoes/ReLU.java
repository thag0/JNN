package rna.ativacoes;

import rna.estrutura.Convolucional;
import rna.estrutura.Densa;

public class ReLU extends Ativacao{

   public ReLU(){
      super.construir(
         (x) -> { return (x > 0) ? x : 0; },
         (x) -> { return (x > 0) ? 1 : 0; }
      );
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
      for(int i = 0; i < camada.saida.length; i++){
         super.aplicarFx(camada.somatorio[i], camada.saida[i]);
      }
   }

   @Override
   public void derivada(Convolucional camada){
      for(int i = 0; i < camada.saida.length; i++){
         super.aplicarDx(camada.gradSaida[i], camada.somatorio[i], camada.derivada[i]);
      }
   }
}
