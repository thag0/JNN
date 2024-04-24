package jnn.ativacoes;

/**
 * Implementação da função de ativação Seno para uso dentro 
 * dos modelos.
 * <p>
 *    A função Seno retorna o seno do valor recebido como entrada.
 * </p>
 */
public class Seno extends Ativacao {

   /**
    * Instancia a função de ativação Seno.
    */
   public Seno() {
      construir(
         x -> Math.sin(x), 
         x -> Math.cos(x)
      );
   }
}
