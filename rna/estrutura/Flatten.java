package rna.estrutura;

import rna.core.Mat;

public class Flatten extends Camada{
   int[] formEntrada;
   int[] formSaida;
   public double[] saida;

   public Flatten(){

   }

   @Override
   public void calcularSaida(Object entrada){
      if(entrada instanceof Mat[]){
         saida((Mat[]) entrada);
      }
      if(entrada instanceof Mat){
         saida((Mat) entrada);
      }
      if(entrada instanceof double[][]){
         saida((double[][]) entrada);
      }
   }

   @Override
   public void calcularGradiente(Object gradSeguinte){
      if(gradSeguinte instanceof double[]){
         entrada((double[]) gradSeguinte);
      }
   }

   private void saida(double[][] entrada){
      this.formEntrada = new int[]{entrada.length, entrada[0].length};
      
      int id = 0;
      this.saida = new double[entrada.length * entrada[0].length];
      for(int i = 0; i < entrada.length; i++){
         for(int j = 0; j < entrada[i].length; j++){
            this.saida[id++] = entrada[i][j];
         }
      }

      this.formSaida = new int[]{1, this.saida.length};
   }

   private void saida(Mat entrada){
      this.formEntrada = new int[]{entrada.lin, entrada.col};
      this.saida = entrada.paraArray();
      this.formSaida = new int[]{1, this.saida.length};
   }

   private void saida(Mat[] entrada){
      int tamanho = entrada[0].lin * entrada[0].col;
      int n = entrada.length;
      this.formEntrada = new int[]{entrada[0].lin, entrada[0].col, n};
      this.saida = new double[n * tamanho];

      for(int i = 0; i < n; i++){
         double[] arr = entrada[i].paraArray();
         System.arraycopy(arr, 0, this.saida, i * tamanho, tamanho);
      }

      this.formSaida = new int[]{1, this.saida.length};
   }

   private void entrada(double[] entrada){

   }
  
}
