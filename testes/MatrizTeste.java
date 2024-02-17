package testes;

import java.awt.image.BufferedImage;
import java.nio.Buffer;
import java.util.concurrent.TimeUnit;

import lib.ged.Dados;
import lib.ged.Ged;
import lib.geim.Geim;
import rna.avaliacao.perda.EntropiaCruzada;
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
      
      double[][][] amostra = {
         {
            {1, 2, 3, 4},
            {8, 7, 6, 5},
            {1, 2, 3, 4},
            {8, 7, 6, 5},
         }
      };

      Tensor4D entrada = new Tensor4D(amostra);
      Convolucional conv = new Convolucional(new int[]{1, 4, 4}, new int[]{3, 3}, 2);
      conv.inicializar();
      conv.filtros.print(2);
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