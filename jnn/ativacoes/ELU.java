package jnn.ativacoes;

/**
 * Implementação da função de ativação ELU para uso dentro 
 * dos modelos.
 * <p>
 *    É possível configurar o valor de {@code alfa} para obter
 *    melhores resultados.
 * </p>
 */
public class ELU extends Ativacao {

   /**
    * Instancia a função de ativação ELU com 
    * seu valor de alfa configurável.
    * @param alfa novo valor alfa.
    */
   public ELU(double alfa) {
      construir(
         x -> (x > 0) ? x : alfa * (Math.exp(x) - 1),
         x -> (x > 0) ? 1 : alfa * Math.exp(x)
      );
   }

   /**
    * Instancia a função de ativação ELU com 
    * seu valor de alfa padrão.
    * <p>
    *    O valor padrão para o alfa é {@code 0.01}.
    * </p>
    */
   public ELU() {
      this(0.01d);
   }
}
