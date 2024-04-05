package testes;

import lib.ged.Ged;
import lib.geim.Geim;
import rna.camadas.Camada;
import rna.camadas.Convolucional;
import rna.camadas.Densa;
import rna.camadas.Entrada;
import rna.core.OpArray;
import rna.core.OpMatriz;
import rna.core.OpTensor4D;
import rna.core.Utils;
import rna.modelos.Sequencial;

public class Playground{
   static Ged ged = new Ged();
   static OpArray oparr = new OpArray();
   static OpMatriz opmat = new OpMatriz();
   static OpTensor4D optensor = new OpTensor4D();
   static Geim geim = new Geim();
   static Utils utils = new Utils();
   
   public static void main(String[] args){
      ged.limparConsole();

      Sequencial modelo = new Sequencial(new Camada[]{
         new Entrada(28, 28),
         new Convolucional(new int[]{3, 3}, 2),
         new Densa(2, 2),
         new Densa(2),
      });
      modelo.sub();
      modelo.sub();
      modelo.sub();
      modelo.compilar("sgd", "mse");
      modelo.info();
   }

   /**
    * Mede o tempo de execução da função fornecida.
    * @param func função.
    * @return tempo em nanosegundos.
    */
   static long medirTempo(Runnable func){
      long t = System.nanoTime();
      func.run();
      return System.nanoTime() - t;
   }

   static void conv2D(double[][] in, double[][] f, double[][] dst){
      int alturaKernel = f.length;
      int larguraKernel = f[0].length;

      int altEsperada  = in.length    - f.length    + 1;
      int largEsperada = in[0].length - f[0].length + 1;

      for(int i = 0; i < altEsperada; i++){
         for(int j = 0; j < largEsperada; j++){
            
            double soma = 0.0;
            for(int k = 0; k < alturaKernel; k++){
               int posX = i+k;
               for(int l = 0; l < larguraKernel; l++){
                  soma += in[posX][j+l] * f[k][l];
               }
            }

            dst[i][j] += soma;
         }
      }
   }

}