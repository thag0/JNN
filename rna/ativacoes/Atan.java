package rna.ativacoes;

public class Atan extends Ativacao{

   public Atan(){
      super.construir(
         (x) -> { return Math.atan(x); },
         (x) -> { return 1.0 / (1.0 + (x * x)); }
      );
   }
}
