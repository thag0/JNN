package testes;

import java.awt.image.BufferedImage;
import java.nio.Buffer;
import java.sql.Time;
import java.util.concurrent.TimeUnit;

import lib.ged.Dados;
import lib.ged.Ged;
import lib.geim.Geim;
import rna.avaliacao.perda.EntropiaCruzada;
import rna.avaliacao.perda.EntropiaCruzadaBinaria;
import rna.camadas.AvgPooling;
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
import rna.core.Utils;
import rna.inicializadores.AleatorioPositivo;
import rna.inicializadores.Inicializador;
import rna.inicializadores.Zeros;
import rna.inicializadores.GlorotUniforme;
import rna.inicializadores.Identidade;
import rna.treinamento.AuxiliarTreino;

@SuppressWarnings("unused")
public class MatrizTeste{
   static Ged ged = new Ged();
   static OpArray oparr = new OpArray();
   static OpMatriz opmat = new OpMatriz();
   static OpTensor4D optensor = new OpTensor4D();
   static Geim geim = new Geim();
   static Utils utils = new Utils();
   
   public static void main(String[] args){
      ged.limparConsole();
      AuxiliarTreino aux = new AuxiliarTreino();

      double[][][][] entrada = {
         {
            {
               {1, 1},
               {1, 1},
            }
         },
         {
            {
               {2, 2},
               {2, 2},
            }
         },
         {
            {
               {3, 3},
               {3, 3},
            }
         },
      };

      Tensor4D tensor = new Tensor4D(entrada);
      System.out.println("tensor = " + tensor.dimensoesStr() + "\n");
      Object[] arr = utils.transformarParaArray(tensor);
      
      for(var elemento : arr){
         Tensor4D t = (Tensor4D) elemento;
         t.print(1);
      }
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