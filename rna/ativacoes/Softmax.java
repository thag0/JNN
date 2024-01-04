package rna.ativacoes;

import rna.core.Mat;
import rna.core.OpMatriz;
import rna.estrutura.Densa;

/**
 * Implementação da função de ativação Softmax para uso
 * dentro dos modelos.
 */
public class Softmax extends Ativacao{

   /**
    * Operador matricial.
    */
   OpMatriz opm = new OpMatriz();

   /**
    * Instancia a função de ativação Softmax.
    * <p>
    * A função Softmax transforma os valores de entrada em probabilidades
    * normalizadas,
    * permitindo que o neurônio com a maior saída tenha uma probabilidade mais
    * alta.
    * </p>
    */
   public Softmax(){

   }

   @Override
   public void calcular(Densa camada){
      double somaExp = 0;

      for(int i = 0; i < camada.somatorio.col(); i++){
         somaExp += Math.exp(camada.somatorio.dado(0, i));
      }

      for(int i = 0; i < camada.saida.col(); i++){
         double s = Math.exp(camada.somatorio.dado(0, i)) / somaExp;
         camada.saida.editar(0, i, s);
      }
   }

   @Override
   public void derivada(Densa camada){
      int n = camada.somatorio.col();
      Mat tmp = camada.saida.bloco(0, n);
      Mat ident = opm.identidade(n);
      Mat transp = tmp.transpor();

      opm.mult(
         camada.gradSaida, 
         opm.hadamard(tmp, opm.sub(ident, transp)), 
         camada.derivada
      );
   }

}
