package rna.estrutura;

import rna.core.Mat;

public class Flatten extends Camada{
   int[] formEntrada;
   int[] formSaida;

   public Mat[] entrada;
   private double[] saida;

   public Flatten(int[] formEntrada){
      this.formEntrada = formEntrada;

      int tamanho = 1;
      for(int i : formEntrada){
         tamanho *= i;
      }
      this.saida = new double[tamanho];
   }

   @Override
   public void calcularSaida(Object entrada){
      if(entrada instanceof double[] == false){
         throw new IllegalArgumentException(
            "Os dados de entrada para a camada Flatten devem ser do tipo \"double[]\", " +
            "objeto recebido é do tipo \"" + entrada.getClass().getTypeName() + "\""
         );
      }

      double[] e = (double[]) entrada;
      System.arraycopy(e, 0, this.saida, 0, this.saida.length);
   }

   @Override
   public void calcularGradiente(Object gradSeguinte){
      if(gradSeguinte instanceof double[] == false){
         throw new IllegalArgumentException(
            "O gradiente seguinte para a camada Flatten deve ser do tipo \"double[]\""
         );
      }

      double[] grad = (double[]) gradSeguinte;
      int id = 0;

      double[][][] entrada = new double[this.entrada.length][this.entrada[0].lin][this.entrada[0].col];
      for(int i = 0; i < formEntrada[0]; i++){
         for(int j = 0; j < formEntrada[1]; j++){
            for(int k = 0; k < formEntrada[2]; k++){
               entrada[i][j][k] = grad[id++];
            }
         }
      }

      for(int i = 0; i < this.entrada.length; i++){
         this.entrada[i].copiar(entrada[i]);
      }
   }
  
   @Override
   public double[] obterSaida(){
      return this.saida;
   }
}
