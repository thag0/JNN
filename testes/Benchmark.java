package testes;

import java.util.concurrent.TimeUnit;

import lib.ged.Ged;
import rna.core.Mat;
import rna.core.OpMatriz;

public class Benchmark {
   public static void main(String[] args){
      Ged ged = new Ged();
      OpMatriz opmat = new OpMatriz();
      ged.limparConsole();

      ged.limparConsole();

      int fator = 256;
      int lin = fator * 4;
      int col = fator * 4;

      Mat a = new Mat(lin, col);
      Mat b = new Mat(lin, col);
      Mat r = new Mat(lin, col);

      a.forEach((i, j) -> a.editar(i, j, i*a.col() + j));
      b.forEach((i, j) -> b.editar(i, j, (i==j) ? 1 : 0));
      
      long tempo;
      tempo = medirTempo(() -> opmat.mult(a, b, r, 4));
      System.out.println("Tempo: " + TimeUnit.NANOSECONDS.toMillis(tempo) + "ms");
      System.out.println("Resultado esperado: " + r.comparar(a));
   }

   static long medirTempo(Runnable func){
      long t = System.nanoTime();
      func.run();
      return System.nanoTime() - t;
   }
}
