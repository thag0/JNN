package rna.ativacoes;

import rna.estrutura.Convolucional;
import rna.estrutura.Densa;

/**
 * Implementação da função de ativação GELU para uso dentro 
 * da {@code Rede Neural}.
 */
public class GELU extends Ativacao{

   private final double RAIZ_2_POR_PI = Math.sqrt(2 / Math.PI); 
   private final double ALFA = 0.044715;

   /**
    * Intancia uma nova função de ativação GELU.
    */
   public GELU(){
      super.construir(this::gelu, this::gelud);
   }

   private double gelu(double x){
      double xCubo = x * x * x;
      double tanh = Math.tanh(RAIZ_2_POR_PI * (x + ALFA * xCubo));
      return 0.5 * x * (1.0 + tanh);
   }

   private double gelud(double x){
      double xCubo = x * x * x;
      double tanh = Math.tanh(RAIZ_2_POR_PI * (x + ALFA * xCubo));
      double exp = Math.exp(-0.5 * x * x) / RAIZ_2_POR_PI;
      return 0.5 * (1.0 + tanh + x * exp);
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
