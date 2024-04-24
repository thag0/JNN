package testes;

import jnn.camadas.Convolucional;
import jnn.core.OpArray;
import jnn.core.OpMatriz;
import jnn.core.OpTensor4D;
import jnn.core.Tensor4D;
import jnn.core.Utils;
import lib.ged.Ged;
import lib.geim.Geim;

public class Playground{
   static Ged ged = new Ged();
   static OpArray oparr = new OpArray();
   static OpMatriz opmat = new OpMatriz();
   static OpTensor4D optensor = new OpTensor4D();
   static Geim geim = new Geim();
   static Utils utils = new Utils();
   
   public static void main(String[] args){
      ged.limparConsole();

      Convolucional conv = new Convolucional(new int[]{1, 10, 10}, new int[]{3, 3}, 2);
      
      Tensor4D amostra = new Tensor4D(conv._entrada.shape());
      amostra.map(x -> Math.random());
      Tensor4D grad = new Tensor4D(conv._gradSaida.shape());
      grad.map(x -> Math.random());

      conv.forward(amostra);
      conv.backward(grad);
      System.out.println(conv._gradFiltros);
      conv.backward(grad);
      System.out.println(conv._gradFiltros);
   }

   /**
    * Testes
    */
   static void testeConv2dFull(){
      double[][] a = {
         {1, 6, 2},
         {5, 3, 1},
         {7, 0, 4},
      };
      double[][] b = {
         {1, 2},
         {-1, 0},
      };

      Tensor4D t1 = new Tensor4D(a);
      Tensor4D t2 = new Tensor4D(b);
      Tensor4D t3 = new Tensor4D(t1.dim3()+t2.dim3()-1, t1.dim4()+t2.dim4()-1);
      optensor.convolucao2DFull(t1, t2, t3, new int[]{0, 0}, new int[]{0, 0}, new int[]{0, 0});
      
      Tensor4D esperado = new Tensor4D(new double[][]{
         {1, 8, 14, 4},
         {4, 7, 5, 2},
         {2, 11, 3, 8},
         {-7, 0, -4, 0}
      });

      t3.print(1);
      System.out.println("Resultado esperado: " + t3.equals(esperado));
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

}