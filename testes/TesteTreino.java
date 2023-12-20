package testes;

import ged.Ged;
import rna.avaliacao.perda.ErroMedioQuadrado;
import rna.inicializadores.Xavier;
import rna.estrutura.*;
import rna.modelos.*;
import rna.otimizadores.SGD;

public class TesteTreino{
   static Ged ged = new Ged();

   public static void main(String[] args){
      ged.limparConsole();
      double[][] entrada = {
         {0, 0},
         {0, 1},
         {1, 0},
         {1, 1}
      };
      double[][] saida = {
         {0},
         {1},
         {1},
         {0}
      };

      Sequencial rede = new Sequencial();
      rede.add(new Densa(2, 3, "tanh"));
      rede.add(new Densa(1, "sigmoid"));
      
      rede.compilar(
         new SGD(0.1),
         new ErroMedioQuadrado(),
         new Xavier()
      );
      rede.treinar(entrada, saida, 10_000);

      for(int i = 0; i < entrada.length; i++){
         rede.calcularSaida(entrada[i]);
         System.out.println(
            entrada[i][0] + " - " + entrada[i][1] + 
            " R:" + saida[i][0] + 
            " P:" + rede.saidaParaArray()[0]
         );
      }
      
      System.out.println("\nPerda: " + rede.avaliador.erroMedioQuadrado(entrada, saida));
   }
}
