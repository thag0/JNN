package testes;

import java.util.concurrent.TimeUnit;

import ged.Ged;
import rna.core.MatrizT;

public class Multithread {
   public static void main(String[] args){
      Ged ged = new Ged();
      ged.limparConsole();

      int lin = 1024 + 256;
      // int lin = 10;
      int col = lin;

      double[][] a = new double[lin][col];
      double[][] b = new double[lin][col];
      double[][] r = new double[lin][col];

      for(int i = 0; i < a.length; i++){
         for(int j = 0; j < a[i].length; j++){
            a[i][j] = (i* a.length) + j + 1;
         }
      }
      ged.matIdentidade(b);

      MatrizT multM = new MatrizT(2);

      //treinar e marcar tempo
      long t1, t2;
      long minutos, segundos;

      t1 = System.nanoTime();
      multM.mult(a, b, r);
      t2 = System.nanoTime();
      
      long tempoDecorrido = t2 - t1;
      long segundosTotais = TimeUnit.NANOSECONDS.toSeconds(tempoDecorrido);
      minutos = (segundosTotais % 3600) / 60;
      segundos = segundosTotais % 60;
      System.out.println("Concluído em: " + minutos + "m " + segundos + "s");

      // ged.imprimirMatriz(r);
   }

}
