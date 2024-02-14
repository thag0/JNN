package testes;

import java.awt.image.BufferedImage;
import java.nio.Buffer;
import java.util.concurrent.TimeUnit;

import lib.ged.Dados;
import lib.ged.Ged;
import lib.geim.Geim;
import rna.avaliacao.perda.EntropiaCruzada;
import rna.camadas.Convolucional;
import rna.camadas.Densa;
import rna.camadas.Dropout;
import rna.camadas.Flatten;
import rna.camadas.MaxPooling;
import rna.core.Mat;
import rna.core.OpArray;
import rna.core.OpMatriz;
import rna.core.OpTensor4D;
import rna.core.Tensor4D;
import rna.inicializadores.AleatorioPositivo;
import rna.inicializadores.Inicializador;
import rna.inicializadores.GlorotUniforme;
import rna.treinamento.AuxiliarTreino;

@SuppressWarnings("unused")
public class MatrizTeste{
   static Ged ged = new Ged();
   static OpArray oparr = new OpArray();
   static OpMatriz opmat = new OpMatriz();
   static OpTensor4D optensor = new OpTensor4D();
   static Geim geim = new Geim();
   
   public static void main(String[] args){
      ged.limparConsole();

      // double[][][] amostra = {
      //    {
      //       {1, 6, 2},
      //       {5, 3, 1},
      //       {7, 0, 4},
      //    },
      //    {
      //       {1, 6, 2},
      //       {5, 3, 1},
      //       {7, 0, 4},
      //    }
      // };

      // double[][][] grad = {
      //    {
      //       {1, 2},
      //       {6, 5},
      //    },
      //    {
      //       {1, 7},
      //       {3, 9},
      //    }
      // };

      // Inicializador iniKernel = new GlorotUniforme(12345);
      // Convolucional conv = new Convolucional(new int[]{3, 3, 2}, new int[]{2, 2}, 2, "leakyrelu", iniKernel);
      // conv.inicializar();

      // conv.calcularSaida(amostra);
      // conv.calcularGradiente(new Mat[]{
      //    new Mat(grad[0]),
      //    new Mat(grad[1]),
      // });
      // for(int i = 0; i < conv.gradEntrada.length; i++){
      //    conv.gradEntrada[i].print("grad entrada " + i, 4);
      // }

      double[][] a = {
         {1, 6, 2},
         {5, 3, 1},
         {7, 0, 4},
      };

      double[][] k = {
         {1, 2},
         {-1, 0},
      };

      Mat entrada = new Mat(a);
      Mat filtro = new Mat(k);
      Mat saida = new Mat(4, 4);

      opmat.convolucao2DFull(entrada, filtro, saida, false);
      saida.print(4);
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

   static void conv2D(Mat entrada, Mat filtro, Mat res){
      double[] e = entrada.paraArray();
      double[] k = filtro.paraArray();
      inverterArray(k);
  
      int lenEntrada = e.length;
      int lenFiltro = k.length;
      int lin = entrada.lin() - filtro.lin() + 1;
      int col = entrada.col() - filtro.col() + 1;
      int lenResultado = lin*col;
      double[] resultado = new double[lenResultado];
  
      for(int i = 0; i < lin; i++){
         for(int j = 0; j < col; j++){
            double sum = 0;
            for(int m = 0; m < filtro.lin(); m++){ 
               for(int n = 0; n < filtro.col(); n++){
                  sum += e[(i + m) * entrada.col() + (j + n)] * k[m * filtro.col() + n];
               }
            }
            resultado[i * col + j] = sum;
         }
      }

      res.copiar(resultado);
   }

   static void correlacao2D(Mat entrada, Mat filtro, Mat res){
      double[] e = entrada.paraArray();
      double[] k = filtro.paraArray();
  
      int lenEntrada = e.length;
      int lenFiltro = k.length;
      int lin = entrada.lin() - filtro.lin() + 1;
      int col = entrada.col() - filtro.col() + 1;
      int lenResultado = lin*col;
      double[] resultado = new double[lenResultado];
  
      for(int i = 0; i < lin; i++){
         for(int j = 0; j < col; j++){
            double sum = 0;
            for(int m = 0; m < filtro.lin(); m++){ 
               for(int n = 0; n < filtro.col(); n++){
                  sum += e[(i + m) * entrada.col() + (j + n)] * k[m * filtro.col() + n];
               }
            }
            resultado[i * col + j] = sum;
         }
      }

      res.copiar(resultado);
   }

   public static void inverterArray(double[] arr){
      int inicio = 0;
      int fim = arr.length - 1;
      while (inicio < fim) {
         // Troca os elementos nas posições start e end
         double temp = arr[inicio];
         arr[inicio] = arr[fim];
         arr[fim] = temp;
         // Move para os próximos elementos
         inicio++;
         fim--;
      }
   }
}