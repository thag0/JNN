package testes;

import java.util.concurrent.TimeUnit;

import ged.Ged;
import rna.core.Mat;
import rna.core.OpMatriz;

public class Benchmark {
   public static void main(String[] args){
      Ged ged = new Ged();
      OpMatriz opamt = new OpMatriz();
      ged.limparConsole();

      Mat a = new Mat(28, 28);
      Mat b = new Mat(5, 5);
      Mat c = new Mat(24, 24);

      //treinar e marcar tempo
      long t1, t2;
      long minutos, segundos;

      t1 = System.nanoTime();
      for(int i = 0; i < 1000; i++){
         opamt.correlacaoCruzada(a, b, c, true);
      }
      t2 = System.nanoTime();
      
      long tempoDecorrido = t2 - t1;
      long segundosTotais = TimeUnit.NANOSECONDS.toSeconds(tempoDecorrido);
      minutos = (segundosTotais % 3600) / 60;
      segundos = segundosTotais % 60;
      System.out.println("Concluído em: " + minutos + "m " + segundos + "s");
      // c.print();
   }

}
