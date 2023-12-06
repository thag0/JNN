package testes;

import java.util.concurrent.TimeUnit;

import ged.Ged;
import rna.core.Mat;
import rna.core.OpMatriz;

public class Benchmark {
   public static void main(String[] args){
      Ged ged = new Ged();
      OpMatriz mat = new OpMatriz();
      ged.limparConsole();
      
      int lin = 1024;
      // int lin = 10;
      int col = lin;
      Mat a = new Mat(lin, col);
      Mat b = new Mat(lin, col);
      Mat c = new Mat(lin, col);
      for(int i = 0; i < a.lin; i++){
         for(int j = 0; j < a.col; j++){
            a.editar(i, j, ((i* a.lin) + j + 1));
            
            b.editar(i, j, ((i == j) ? 1 : 0));
         }
      }

      //treinar e marcar tempo
      long t1, t2;
      long minutos, segundos;

      t1 = System.nanoTime();
      mat.mult(a, b, c);
      t2 = System.nanoTime();
      
      long tempoDecorrido = t2 - t1;
      long segundosTotais = TimeUnit.NANOSECONDS.toSeconds(tempoDecorrido);
      minutos = (segundosTotais % 3600) / 60;
      segundos = segundosTotais % 60;
      System.out.println("Concluído em: " + minutos + "m " + segundos + "s");
      // c.print();
   }

}
