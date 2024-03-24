package rna.ativacoes;

import java.util.function.DoubleUnaryOperator;

import rna.camadas.Convolucional;
import rna.camadas.Densa;
import rna.core.Tensor4D;

/**
 * Classe base para a implementação das funções de ativação.
 * <p>
 *    As funções de ativação percorres todos os elementos contendo os
 *    resultados de cada operação dos kernels das camadas, e aplica sua
 *    operação correspondente nas suas saídas.
 * </p>
 * <p>
 *    Funções de ativação podem fazer uso dos métodos {@code aplicarFx()} e 
 *    {@code aplicarDx()}, sendo necessário informar nos seus constritures 
 *    uma interface funcional que fará o cálculo da saída de acordo com uma
 *    entrada redebida.
 * </p>
 * Exemplo com a função ReLU:
 * <pre>
 *public class ReLU extends Ativacao{
 *  public ReLU(){
 *    super.construir(
 *       (x) -> { return x > 0 ? x : 0; },
 *       (x) -> { return x > 0 ? 1 : 0; }
 *    );
 *  }
 *
 *}
 * </pre>
 * <p>
 *    Novas funções de ativações devem sobrescrever os métodos existentes {@code ativar()} e {@code derivada()}.
 * </p>
 */
public abstract class Ativacao{

   /**
    * Função de ativação.
    */
   protected DoubleUnaryOperator fx;

   /**
    * Derivada da função de ativação.
    */
   protected DoubleUnaryOperator dx;

   /**
    * Configura a função de ativação e sua derivada para uso.
    * @param fx função de ativação.
    * @param dx deriviada da função de ativação
    */
   public void construir(DoubleUnaryOperator fx, DoubleUnaryOperator dx){
      if(fx == null){
         throw new IllegalArgumentException(
            "É necessário que ao menos a função de ativação seja configurada, recebido null."
         );
      }
      this.fx = fx;
      this.dx = dx;
   }

   /**
    * Calcula o resultado da ativação de acordo com a função configurada.
    * @param entrada tensor de entrada.
    * @param saida tensor de destino.
    */
   public void calcular(Tensor4D entrada, Tensor4D saida){
      if(entrada.comparar4D(saida) == false){
         throw new IllegalArgumentException(
            "\nAs dimensões do tensor de entrada " + entrada.shapeStr() +
            " e saída " + saida.shapeStr() + " devem ser iguais."
         );
      }

      saida.map(entrada, fx);
   }

   /**
    * Calcula o resultado da derivada da função de ativação de acordo 
    * com a função configurada
    * @param entrada tensor de entrada.
    * @param entrada tensor com os gradientes.
    * @param saida tensor de destino.
    */
   public void derivada(Tensor4D entrada, Tensor4D gradiente, Tensor4D saida){
      if(entrada.comparar4D(saida) == false){
         throw new IllegalArgumentException(
            "\nAs dimensões do tensor de entrada " + entrada.shapeStr() +
            " e saída " + saida.shapeStr() + " devem ser iguais."
         );
      }

      int dim1 = entrada.dim1();
      int dim2 = entrada.dim2();
      int dim3 = entrada.dim3();
      int dim4 = entrada.dim4();
      double e, g;
      for(int i = 0; i < dim1; i++){
         for(int j = 0; j < dim2; j++){
            for(int k = 0; k < dim3; k++){
               for(int l = 0; l < dim4; l++){
                  e = entrada.get(i, j, k, l);
                  g = gradiente.get(i, j, k, l);
                  saida.set(i, j, k, l, (
                     dx.applyAsDouble(e) * g
                  ));
               }
            }
         }
      }
   }

   /**
    * Implementação especifíca para camadas densas.
    * <p>
    *    Função criada para dar suporte a ativações especiais.
    * </p>
    * @param camada camada densa.
    */
   public void derivada(Densa camada){
      derivada(camada.somatorio, camada.gradSaida, camada.gradSaida);
   }

   /**
    * Implementação especifíca para camadas convolucionais.
    * <p>
    *    Função criada para dar suporte a ativações especiais.
    * </p>
    * @param camada camada convolucional.
    */
   public void derivada(Convolucional camada){
      derivada(camada.somatorio, camada.gradSaida, camada.gradSaida);
   }

   /**
    * Retorna o nome da função de atvação.
    * @return nome da função de ativação.
    */
   public String nome(){
      return getClass().getSimpleName();
   }
}
