package rna.ativacoes;

import java.util.function.DoubleUnaryOperator;

import rna.core.Mat;
import rna.estrutura.Convolucional;
import rna.estrutura.Densa;

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
 *  public void calcular(Densa camada){
 *    super.aplicarFx(camada.somatorio, camada.saida) 
 *  }
 * 
 *  public void derivada(Densa camada){
 *    super.aplicarDx(camada.gradSaida, camada.somatorio, camada.derivada) 
 *  }
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
      int lin = entrada.lin();
      int col = entrada.col();
      for(int i = 0; i < lin; i++){
         for(int j = 0; j < col; j++){
            saida.editar(i, j, fx.applyAsDouble(entrada.dado(i, j)));
         }
      }
   }

   /**
    * Executa a derivada da função de ativação para cada elemento da entrada.
    * @param entrada matriz com os valores de gradientes da camada.
    * @param entrada matriz com os dados de entrada.
    * @param fx derivada de função de ativação desejada.
    * @param saida resultado das derivadas.
    */
   protected void aplicarDx(Mat gradientes, Mat entrada, Mat saida){
      double[] e = entrada.paraArray();
      double[] g = gradientes.paraArray();
      for(int i = 0; i < e.length; i++){
         e[i] = g[i] * dx.applyAsDouble(e[i]);
      }
      saida.copiar(e);
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
      this.aplicarFx(camada.somatorio, camada.saida);
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
      this.aplicarDx(camada.gradSaida, camada.somatorio, camada.derivada);
   }


   /**
    * Calcula o resultado da ativação de acordo com a função configurada
    * <p>
    *    O resultado da ativação da camada é salvo na propriedade {@code camada.saida}.
    * </p>
    * @param camada camada convolucional usada.
    */
   public void calcular(Convolucional camada){
      for(int i = 0; i < camada.somatorio.length; i++){
         this.aplicarFx(camada.somatorio[i], camada.saida[i]);
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
      for(int i = 0; i < camada.somatorio.length; i++){
         this.aplicarDx(camada.gradSaida[i], camada.somatorio[i], camada.derivada[i]);
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
