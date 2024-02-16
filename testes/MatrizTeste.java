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
      
      double[][][] entrada1 = {
         {
            {1, 2},
            {3, 4}
         }
      };
      double[][][] grad1 = {
         {
            {1, 2},
            {3, 4}
         }
      };

      Tensor4D entrada2 = new Tensor4D(entrada1);

      AvgPooling camada = new AvgPooling(new int[]{1, 2, 2}, new int[]{2, 2}, new int[]{2, 2});
      camada.calcularSaida(entrada1);
      camada.calcularGradiente(new Tensor4D(grad1));
      camada.saida.print();
      camada.gradEntrada.print();
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