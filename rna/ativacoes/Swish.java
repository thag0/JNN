package rna.ativacoes;

/**
 * Implementação da função de ativação Swish para uso 
 * dentro dos modelos.
 */
public class Swish extends Ativacao{

   /**
    * Instancia a função de ativação Swish.
    */
   public Swish(){
      super.construir(
         (x) -> { return x * sigmoid(x); }, 
         (x) -> {
            double sig = sigmoid(x);
            return sig + (x * sig * (1 - sig));
         }
      );
   }

   private double sigmoid(double x){
      return 1 / (1 + Math.exp(-x));
   }
}
