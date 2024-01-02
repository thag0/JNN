package testes;

import java.util.concurrent.TimeUnit;

import lib.ged.Ged;
import rna.core.Mat;
import rna.core.OpMatriz;

public class Benchmark {
   public static void main(String[] args){
      Ged ged = new Ged();
      OpMatriz opamt = new OpMatriz();
      ged.limparConsole();

      int lin = 1024;
      int col = 1024;

      Mat a = new Mat(lin, col);
      a.preencher(1);
      Mat b = new Mat(lin, col);
      b.preencher(2);
      Mat r = new Mat(lin, col);

      //treinar e marcar tempo
      long t1, t2;
      long minutos, segundos;

      t1 = System.nanoTime();
      opamt.multT(a, b, r, 4);
      t2 = System.nanoTime();
      
      long tempoDecorrido = t2 - t1;
      long segundosTotais = TimeUnit.NANOSECONDS.toSeconds(tempoDecorrido);
      minutos = (segundosTotais % 3600) / 60;
      segundos = segundosTotais % 60;
      System.out.println("Concluído em: " + minutos + "m " + segundos + "s " + (t2-t1) + "ms");
      // c.print();
   }

}
