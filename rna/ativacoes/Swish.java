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

   }

   @Override
   public void calcular(Densa camada){
      int i, j;
      double s;

      for(i = 0; i < camada.saida.lin; i++){
         for(j = 0; j < camada.saida.col; j++){
            s = swish(camada.somatorio.dado(i, j));
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
            grad = camada.gradSaida.dado(i, j);
            d = derivada(camada.somatorio.dado(i, j));
            camada.derivada.editar(i, j, (grad * d));
         }
      }
   }

   private double swish(double x){
      return x * sigmoid(x);
   }

   private double derivada(double x){
      double sig = sigmoid(x);
      return sig + (x * sig * (1 - sig));
   }

   private double sigmoid(double x){
      return 1 / (1 + Math.exp(-x));
   }
}
