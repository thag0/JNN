package rna.ativacoes;

import rna.camadas.Densa;
import rna.core.OpTensor4D;
import rna.core.Tensor4D;

/**
 * Implementação da função de ativação Softmax para uso
 * dentro dos modelos.
 */
public class Softmax extends Ativacao{

   /**
    * Operador para tensores.
    */
   OpTensor4D optensor = new OpTensor4D();

   /**
    * Instancia a função de ativação Softmax.
    * <p>
    * A função Softmax transforma os valores de entrada em probabilidades
    * normalizadas,
    * permitindo que o neurônio com a maior saída tenha uma probabilidade mais
    * alta.
    * </p>
    */
   public Softmax(){}

   @Override
   public void calcular(Densa camada){
      double somaExp = 0;
      int colunas = camada.somatorio.dim4();

      for(int i = 0; i < colunas; i++){
         somaExp += Math.exp(camada.somatorio.elemento(0, 0, 0, i));
      }

      for(int i = 0; i < colunas; i++){
         double s = Math.exp(camada.somatorio.elemento(0, 0, 0, i)) / somaExp;
         camada.saida.editar(0, 0, 0, i, s);
      }
   }

   @Override
   public void derivada(Densa camada){
      int n = camada.somatorio.dim4();
      Tensor4D tmp = camada.saida.bloco2D(0, 0, 0, n);
      Tensor4D ident = new Tensor4D(1, 1, n, camada.somatorio.dim4());
      ident.identidade2D(0, 0);
      Tensor4D transp = optensor.matTranspor(tmp, 0, 0);

      optensor.matMult(
         camada.gradSaida, 
         optensor.matHadamard(
            tmp,
            optensor.matSub(ident, transp, 0),
            0, 
            0
         ), 
         camada.derivada,
         0,
         0
      );
   }

}
