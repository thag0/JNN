package jnn.ativacoes;

/**
 * Implementação da função de ativação Swish para uso dentro dos modelos.
 * <p>
 *    A função Swish é uma função de ativação que gradualmente se aproxima 
 *    da função identidade à medida que seu argumento se torna positivo e 
 *    se assemelha à função sigmoid quando o argumento é negativo.
 * </p>
 */
public class Swish extends Ativacao {

   /**
    * Instancia a função de ativação Swish.
    */
   public Swish() {
      construir(
         x -> x * sigmoid(x), 
         x -> {
            double sig = sigmoid(x);
            return sig + (x * sig * (1 - sig));
         }
      );
   }

   private double sigmoid(double x) {
      return 1 / (1 + Math.exp(-x));
   }
}
