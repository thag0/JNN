package jnn.ativacoes;

/**
 * Implementação da função de ativação SoftPlus para uso dentro dos modelos.
 * <p>
 *    A função SoftPlus é uma função de ativação suave, que suaviza a transição
 *    entre valores positivos e negativos.
 * </p>
 */
public class SoftPlus extends Ativacao {

   /**
    * Instancia a função de ativação SoftPlus.
    */
   public SoftPlus() {
      construir(
         x -> Math.log(1 + Math.exp(x)),
         x -> { 
            double exp = Math.exp(x);
            return (exp / (1 + exp));
         }
      );
   }
}
