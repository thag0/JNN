package testes;

import ged.Ged;
import rna.avaliacao.perda.ErroMedioQuadrado;
import rna.inicializadores.Xavier;
import rna.modelos.RedeNeural;
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

      RedeNeural rede = new RedeNeural(new int[]{2, 3, 1});
      rede.compilar(
         new ErroMedioQuadrado(),
         new SGD(0.01, 0.95),
         new Xavier()
      );
      // rede.configurarAtivacao(rede.obterCamada(0), "tanh");
      rede.configurarAtivacao("sigmoid");
      rede.treinar(entrada, saida, 10_000);

      for(int i = 0; i < entrada.length; i++){
         rede.calcularSaida(entrada[i]);
         System.out.println(
            entrada[i][0] + " - " + entrada[i][1] + 
            " R:" + saida[i][0] + 
            " P:" + rede.obterCamada(1).saida.dado(0 ,0)
         );
      }
      
      System.out.println("\nPerda: " + rede.avaliador.erroMedioQuadrado(entrada, saida));
   }
}
