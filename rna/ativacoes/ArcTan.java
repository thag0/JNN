package rna.ativacoes;

public class ArcTan extends Ativacao{

   public ArcTan(){
      super.construir(
         (x) -> { return Math.atan(x); },
         (x) -> { return 1.0 / (1.0 + (x * x)); }
      );
   }

   public double tanh(double x){
      return (2 / (1 + Math.exp(-2*x))) - 1;
   }
}
