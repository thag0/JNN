package rna.ativacoes;

import rna.estrutura.CamadaDensa;

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
   public void calcular(CamadaDensa camada){
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
