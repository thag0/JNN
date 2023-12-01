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

   }
   @Override
   public void calcular(Densa camada){
      int i, j;
      double s;

      for(i = 0; i < camada.saida.lin; i++){
         for(j = 0; j < camada.saida.col; j++){
            s = softplus(camada.somatorio.dado(i, j));
            camada.saida.editar(i, j, s);
         }
      }
   }

   @Override
   public void derivada(Densa camada){
      int i, j;
      double grad, d;

      for(i = 0; i < camada.derivada.lin; i++){
         for(j = 0; j < camada.derivada.col; j++){
            grad = camada.gradienteSaida.dado(i, j);
            d = derivada(camada.somatorio.dado(i, j));
            camada.derivada.editar(i, j, (grad * d));
         }
      }
   }

   private double softplus(double x){
      return Math.log(1 + Math.exp(x));
   }

   private double derivada(double x){
      double exp = Math.exp(x);
      return (exp / (1 + exp));
   }
}
