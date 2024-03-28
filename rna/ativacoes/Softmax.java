package rna.ativacoes;

import rna.camadas.Convolucional;
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
    *    A função Softmax transforma os valores de entrada em probabilidades
    *    normalizadas,
    *    permitindo que a unidade com a maior saída tenha uma probabilidade mais
    *    alta.
    * </p>
    * <p>
    *    A ativação atua em cada linha do tensor individualmente
    * </p>
    * Exemplo:
    * <pre>
    * tensor = [[[
    *    [1, 2, 3], 
    *    [2, 3, 1], 
    *    [3, 1, 2], 
    *]]]
    *
    *softmax.calcular(tensor, tensor);
    *
    *tensor = [[[
    *    [0.1, 0.2, 0.7], 
    *    [0.2, 0.7, 0.1], 
    *    [0.7, 0.1, 0.2], 
    *]]]
    * </pre>
    */
   public Softmax(){}

   @Override
   public void calcular(Tensor4D entrada, Tensor4D saida){
      int canais = entrada.dim1();
      int profundidade = entrada.dim2();
      int linhas = entrada.dim3();
      int colunas = entrada.dim4();
   
      for(int can = 0; can < canais; can++){
         for(int prof = 0; prof < profundidade; prof++){
            for(int lin = 0; lin < linhas; lin++){
               double somaExp = 0;

               for(int i = 0; i < colunas; i++){
                  somaExp += Math.exp(entrada.get(can, prof, lin, i));
               }

               for(int i = 0; i < colunas; i++){
                  double s = Math.exp(entrada.get(can, prof, lin, i)) / somaExp;
                  saida.set(s, can, prof, lin, i);
               }
            }
         }
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
         camada.gradSaida
      );
   }

   @Override
   public void derivada(Convolucional camada){
      throw new UnsupportedOperationException(
         "\nSem suporte para derivada " + nome() + " em camadas convolucionais."
      );
   }

}
