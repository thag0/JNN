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
    * @param entrada {@code Tensor} de entrada.
    * @param dest {@code Tensor} de destino.
    */
   public void forward(Tensor4D entrada, Tensor4D dest){
      if(entrada.comparar4D(dest) == false){
         throw new IllegalArgumentException(
            "\nAs dimensões do tensor de entrada " + entrada.shapeStr() +
            " e saída " + dest.shapeStr() + " devem ser iguais."
         );
      }

      dest.map(entrada, fx);
   }

   /**
    * Calcula o resultado da derivada da função de ativação de acordo 
    * com a função configurada
    * @param entrada {@code Tensor} de entrada.
    * @param gradiente {@code Tensor} contendo os gradientes.
    * @param dest {@code Tensor} de destino.
    */
   public void backward(Tensor4D entrada, Tensor4D gradiente, Tensor4D dest){
      if(entrada.comparar4D(dest) == false){
         throw new IllegalArgumentException(
            "\nAs dimensões do tensor de entrada " + entrada.shapeStr() +
            " e saída " + dest.shapeStr() + " devem ser iguais."
         );
      }

      int d1 = entrada.dim1();
      int d2 = entrada.dim2();
      int d3 = entrada.dim3();
      int d4 = entrada.dim4();
      int i, j, k, l;
      double e, g;

      for(i = 0; i < d1; i++){
         for(j = 0; j < d2; j++){
            for(k = 0; k < d3; k++){
               for(l = 0; l < d4; l++){
                  e = entrada.get(i, j, k, l);
                  g = gradiente.get(i, j, k, l);
                  dest.set(
                     (dx.applyAsDouble(e) * g),
                     i, j, k, l
                  );
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
   public void backward(Densa camada){
      //por padrão chamar o método da própria ativação
      backward(camada.somatorio, camada.gradSaida, camada.gradSaida);
   }

   /**
    * Implementação especifíca para camadas convolucionais.
    * <p>
    *    Função criada para dar suporte a ativações especiais.
    * </p>
    * @param camada camada convolucional.
    */
   public void backward(Convolucional camada){
      //por padrão chamar o método da própria ativação
      backward(camada.somatorio, camada.gradSaida, camada.gradSaida);
   }

   /**
    * Retorna o nome da função de atvação.
    * @return nome da função de ativação.
    */
   public String nome(){
      return getClass().getSimpleName();
   }
}
