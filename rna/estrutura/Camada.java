package rna.estrutura;

/**
 * Classe base para as camadas dentro dos modelos de Rede Neural.
 */
public abstract class Camada{
   
   public void calcularSaida(Object entrada){
      throw new IllegalArgumentException(
         "Implementar cálculo de saída."
      );
   }

   public void calcularGradiente(Object gradSeguinte){
      throw new IllegalArgumentException(
         "Implementar cálculo de gradientes."
      );
   }
   
   public int[] formatoEntrada(){
      throw new IllegalArgumentException(
         "Implementar formato de entrada."
      );
   }

   public int[] formatoSaida(){
      throw new IllegalArgumentException(
         "Implementar formato de saída."
      );
   }
}
