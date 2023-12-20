package rna.ativacoes;

/**
 * Implementação da função de ativação SoftPlus para uso 
 * dentro dos modelos.
 */
public class SoftPlus extends Ativacao{

   /**
    * Instancia a função de ativação SoftPlus.
    */
   public SoftPlus(){
      super.construir(
         (x) -> { return Math.log(1 + Math.exp(x)); },
         (x) -> { 
            double exp = Math.exp(x);
            return (exp / (1 + exp));
         }
      );
   }
}
