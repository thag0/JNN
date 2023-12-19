package rna.ativacoes;

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

   @Override
   public void calcular(Densa camada){
      super.aplicarFuncao(camada.somatorio, camada.saida);
   }

   @Override
   public void derivada(Densa camada){
      super.aplicarDerivada(camada.gradSaida, camada.somatorio, camada.derivada);
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
}
