package rna.ativacoes;

import java.util.function.DoubleUnaryOperator;

import rna.camadas.Convolucional;
import rna.camadas.Densa;
import rna.core.Mat;

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
   private DoubleUnaryOperator fx;

   /**
    * Derivada da função de ativação.
    */
   private DoubleUnaryOperator dx;

   /**
    * Executa a função de ativação para cada elemento da entrada.
    * @param entrada matriz com os dados de entrada.
    * @param fx função de ativação desejada.
    * @param saida resultado das ativações.
    */
   protected void aplicarFx(Mat entrada, Mat saida){
      saida.aplicarFuncao(entrada, fx);
   }

   /**
    * Executa a derivada da função de ativação para cada elemento da entrada.
    * @param entrada matriz com os valores de gradientes da camada.
    * @param entrada matriz com os dados de entrada.
    * @param fx derivada de função de ativação desejada.
    * @param saida resultado das derivadas.
    */
   protected void aplicarDx(Mat gradientes, Mat entrada, Mat saida){
      saida.aplicarFuncao(entrada, dx);
      saida.mult(gradientes);
   }

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
    * Calcula o resultado da ativação de acordo com a função configurada
    * <p>
    *    O resultado da ativação da camada é salvo na propriedade {@code camada.saida}.
    * </p>
    * @param camada camada densa usada.
    */
   public void calcular(Densa camada){
      int linhas = camada.somatorio.dim3();
      int colunas = camada.somatorio.dim4();
      for(int i = 0; i < linhas; i++){
         for(int j = 0; j < colunas; j++){
            camada.saida.editar(0, 0, i, j, (
               fx.applyAsDouble(camada.somatorio.elemento(0, 0, i, j)
            )));
         }
      }
   }

   /**
    * Calcula o resultado da derivada da função de ativação de acordo com a 
    * função configurada
    * <p>
    *    O resultado da derivada da camada é salvo na propriedade {@code camada.derivada}.
    * </p>
    * @param camada camada densa usada.
    */
   public void derivada(Densa camada){
      double grad, entrada;
      int linhas = camada.somatorio.dim3();
      int colunas = camada.somatorio.dim4();

      for(int i = 0; i < linhas; i++){
         for(int j = 0; j < colunas; j++){
            entrada = camada.somatorio.elemento(0, 0, i, j);
            grad = camada.gradSaida.elemento(0, 0, i, j);
            camada.derivada.editar(0, 0, i, j, (
               dx.applyAsDouble(entrada) * grad
            ));
         }
      }
   }


   /**
    * Calcula o resultado da ativação de acordo com a função configurada
    * <p>
    *    O resultado da ativação da camada é salvo na propriedade {@code camada.saida}.
    * </p>
    * @param camada camada convolucional usada.
    */
   public void calcular(Convolucional camada){
      int prof = camada.somatorio.dim2();
      int altura = camada.somatorio.dim3();
      int largura = camada.somatorio.dim4();

      for(int i = 0; i < prof; i++){
         for(int j = 0; j < altura; j++){
            for(int k = 0; k < largura; k++){
               camada.saida.editar(0, i, j, k, (
                  fx.applyAsDouble(camada.somatorio.elemento(0, i, j, k))
               ));
            }
         }
      }
   }

   /**
    * Calcula o resultado da derivada da função de ativação de acordo com a 
    * função configurada
    * <p>
    *    O resultado da derivada da camada é salvo na propriedade {@code camada.derivada}.
    * </p>
    * @param camada camada convolucional usada.
    */
   public void derivada(Convolucional camada){
      int prof = camada.somatorio.dim2();
      int altura = camada.somatorio.dim3();
      int largura = camada.somatorio.dim4();
      double entrada, grad;

      for(int i = 0; i < prof; i++){
         for(int j = 0; j < altura; j++){
            for(int k = 0; k < largura; k++){
               entrada = camada.somatorio.elemento(0, i, j, k);
               grad = camada.gradSaida.elemento(0, i, j, k);
               camada.derivada.editar(0, i, j, k, (
                  dx.applyAsDouble(entrada) * grad
               ));
            }
         }
      }
   }

   /**
    * Retorna o nome da função de atvação.
    * @return nome da função de ativação.
    */
   public String nome(){
      return getClass().getSimpleName();
   }
}
