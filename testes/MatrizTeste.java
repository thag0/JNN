package testes;

import java.util.concurrent.TimeUnit;

import lib.ged.Dados;
import lib.ged.Ged;
import rna.avaliacao.perda.EntropiaCruzada;
import rna.camadas.Convolucional;
import rna.camadas.Densa;
import rna.camadas.Dropout;
import rna.camadas.Flatten;
import rna.camadas.MaxPooling;
import rna.core.Mat;
import rna.core.OpMatriz;
import rna.inicializadores.Inicializador;
import rna.inicializadores.Xavier;
import rna.treinamento.AuxiliarTreino;

@SuppressWarnings("unused")
public class MatrizTeste{
   static Ged ged = new Ged();
   static OpMatriz opmat = new OpMatriz();
   
   public static void main(String[] args){
      ged.limparConsole();

      int lin = 1;
      int col = 1000;
      Mat mat = new Mat(lin, col);
      mat.forEach((i, j) -> {
         mat.editar(i, j, (
            i*mat.col() + mat.col()
         ));
      });

      Densa densa = new Densa(col, 200);
      System.out.println(densa.pesos.tamanho());

      long t;
      t = medirTempo(() -> densa.calcularSaida(mat));
      System.out.println("Tempo forward: " + TimeUnit.NANOSECONDS.toMillis(t) + "ms");
   }

   static long medirTempo(Runnable func){
      long t = System.nanoTime();
      func.run();
      return System.nanoTime() - t;
   }
}