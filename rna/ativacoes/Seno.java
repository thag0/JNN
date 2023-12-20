package rna.ativacoes;

/**
 * Implementação da função de ativação Seno para uso dentro 
 * dos modelos.
 */
public class Seno extends Ativacao{

   /**
    * Instancia a função de ativação Seno.
    */
   public Seno(){
      super.construir(
         (x) -> { return Math.sin(x); }, 
         (x) -> { return Math.cos(x); }
      );
   }
}
