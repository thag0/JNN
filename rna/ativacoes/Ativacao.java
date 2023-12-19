package rna.ativacoes;

import java.util.function.DoubleUnaryOperator;

import rna.core.Mat;
import rna.estrutura.Convolucional;
import rna.estrutura.Densa;

/**
 * Classe base para a implementação das funções de ativação.
 * <p>
 *    Novas funções de ativações devem sobrescrever os métodos existentes {@code ativar()} e {@code derivada()}.
 * </p>
 */
public abstract class Ativacao{

   /**
    * Executa a função de ativação para cada elemento da entrada.
    * @param entrada matriz com os dados de entrada.
    * @param funcao função de ativação desejada.
    * @param saida resultado das ativações.
    */
   protected void aplicarFuncao(Mat entrada, DoubleUnaryOperator funcao, Mat saida){
      int linhas = entrada.lin();
      int colunas = entrada.col();
      int i, j;
      for(i = 0; i < linhas; i++){
         for (j = 0; j < colunas; j++){
            double valor = entrada.dado(i, j);
            saida.editar(i, j, funcao.applyAsDouble(valor));
         }
      }
   }

   /**
    * Executa a derivada da função de ativação para cada elemento da entrada.
    * @param entrada matriz com os valores de gradientes da camada.
    * @param entrada matriz com os dados de entrada.
    * @param funcao derivada de função de ativação desejada.
    * @param saida resultado das derivadas.
    */
   protected void aplicarDerivada(Mat gradientes, Mat entrada, DoubleUnaryOperator derivada, Mat saida){
      int linhas = entrada.lin();
      int colunas = entrada.col();
      int i, j;
      for(i = 0; i < linhas; i++){
         for (j = 0; j < colunas; j++){
            double grad = gradientes.dado(i, j);
            double valor = entrada.dado(i, j);
            saida.editar(i, j, grad * derivada.applyAsDouble(valor));
         }
      }
   }

   /**
    * Calcula o resultado da ativação de acordo com a função configurada
    * <p>
    *    O resultado da ativação da camada é salvo na propriedade {@code camada.saida}.
    * </p>
    * @param camada camada densa usada.
    */
   public void calcular(Densa camada){
      throw new UnsupportedOperationException(
         "Implementar ativação para camada densa."
      );
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
      throw new UnsupportedOperationException(
         "Implementar derivada da ativação para camada densa."
      );
   }


   /**
    * Calcula o resultado da ativação de acordo com a função configurada
    * <p>
    *    O resultado da ativação da camada é salvo na propriedade {@code camada.saida}.
    * </p>
    * @param camada camada convolucional usada.
    */
   public void calcular(Convolucional camada){
      throw new UnsupportedOperationException(
         "Implementar ativação para camada convolucional."
      );
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
      throw new UnsupportedOperationException(
         "Implementar derivada da ativação para camada convolucional."
      );
   }

   public String nome(){
      return getClass().getSimpleName();
   }
}
