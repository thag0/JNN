package rna.ativacoes;

import rna.estrutura.CamadaDensa;

/**
 * Classe base para a implementação das funções de ativação.
 * <p>
 *    Novas funções de ativações devem sobrescrever os métodos existentes {@code ativar()} e {@code derivada()}.
 * </p>
 */
public abstract class Ativacao{

   /**
    * Calcula o resultado da ativação de acordo com a função configurada
    * <p>
    *    O resultado da ativação da camada é salvo na propriedade {@code camada.saida}.
    * </p>
    * @param camada camada densa usada.
    */
   public void calcular(CamadaDensa camada){
      throw new UnsupportedOperationException("Implementar ativação.");
   }

   /**
    * Calcula o resultado da derivada da função de ativação de acordo com a 
    * função configurada
    * <p>
    *    O resultado da derivada da camada é salvo na propriedade {@code camada.derivada}.
    * </p>
    * @param camada camada densa usada.
    */
   public void derivada(CamadaDensa camada){
      throw new UnsupportedOperationException("Implementar ativação derivada.");
   }
}
