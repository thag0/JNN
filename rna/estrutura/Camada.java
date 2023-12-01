package rna.estrutura;

public abstract class Camada{
   
   public void calcularSaida(double[] entrada){
      throw new IllegalArgumentException(
         "Implementar cálculo de saída para double[]"
      );
   }

   public void calcularSaida(double[][][] entrada){
      throw new IllegalArgumentException(
         "Implementar cálculo de saída para double[][][]"
      );
   }

   public void calcularGradientes(double[] gradSeguinte){
      throw new IllegalArgumentException(
         "Implementar cálculo de gradientes para double[]"
      );
   }

   public void calcularGradientes(double[][] gradSeguinte){
      throw new IllegalArgumentException(
         "Implementar cálculo de gradientes para double[][]"
      );
   }
}
