package testes;

import java.util.concurrent.TimeUnit;

import lib.ged.Dados;
import lib.ged.Ged;
import rna.avaliacao.perda.EntropiaCruzada;
import rna.camadas.Densa;
import rna.camadas.Dropout;
import rna.camadas.Flatten;
import rna.camadas.MaxPooling;
import rna.core.Mat;
import rna.core.OpMatriz;
import rna.inicializadores.AleatorioPositivo;
import rna.inicializadores.Inicializador;
import rna.inicializadores.GlorotUniforme;
import rna.treinamento.AuxiliarTreino;

@SuppressWarnings("unused")
public class MatrizTeste{
   static Ged ged = new Ged();
   static OpMatriz opmat = new OpMatriz();
   
   public static void main(String[] args){
      ged.limparConsole();

      Mat m = new Mat(new double[][]{
         {1, -1},
         {-1, 1}
      });

      System.out.println(m.somarElementos());
   }

   static long medirTempo(Runnable func){
      long t = System.nanoTime();
      func.run();
      return System.nanoTime() - t;
   }
}