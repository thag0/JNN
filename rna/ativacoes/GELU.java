package rna.ativacoes;

import rna.estrutura.CamadaDensa;

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

   }

   @Override
   public void calcular(CamadaDensa camada){
      int i, j;
      double s;

      for(i = 0; i < camada.saida.lin; i++){
         for(j = 0; j < camada.saida.col; j++){
            s = gelu(camada.somatorio.dado(i, j));
            camada.saida.editar(i, j, s);
         }
      }
   }

   @Override
   public void derivada(CamadaDensa camada){
      int i, j;
      double d;

      for(i = 0; i < camada.derivada.lin; i++){
         for(j = 0; j < camada.derivada.col; j++){
            d = derivada(camada.somatorio.dado(i, j));
            camada.derivada.editar(i, j, d);
         }
      }
   }

   private double gelu(double x){
      double xCubo = x * x * x;
      double tanh = Math.tanh(RAIZ_2_POR_PI * (x + ALFA * xCubo));
      return 0.5 * x * (1.0 + tanh);
   }

   private double derivada(double x){
      double xCubo = x * x * x;
      double tanh = Math.tanh(RAIZ_2_POR_PI * (x + ALFA * xCubo));
      double exp = Math.exp(-0.5 * x * x) / RAIZ_2_POR_PI;
      return 0.5 * (1.0 + tanh + x * exp);
   }
}
