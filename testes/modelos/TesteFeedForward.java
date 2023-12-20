package testes.modelos;
import ged.Ged;
import rna.modelos.RedeNeural;

public class TesteFeedForward {
   public static void main(String[] args){
      Ged ged = new Ged();
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

      RedeNeural rede = new RedeNeural(new int[]{2, 2, 1});
      rede.compilar();
      rede.configurarAtivacao("sigmoid");

      double[] pN1 = {-7.577308710973026, 7.34560483929071};
      double   bN1 = -3.8770293408497674;

      double[] pN2 = {-7.39212040295605, 7.661754069020709};
      double   bN2 = 3.681963491508159;

      double[] pN3 = {14.662523120653429, -14.163951965348884};
      double   bN3 = 6.7686160132656585;

      rede.camada(0).configurarPesos(0, pN1);
      rede.camada(0).configurarBias(0, bN1);
      rede.camada(0).configurarPesos(1, pN2);
      rede.camada(0).configurarBias(1, bN2);

      rede.camada(1).configurarPesos(0, pN3);
      rede.camada(1).configurarBias(0, bN3);

      for(int i = 0; i < entrada.length; i++){
         rede.calcularSaida(entrada[i]);
         System.out.println(
            entrada[i][0] + " - " + entrada[i][1] + 
            " R:" + saida[i][0] + 
            " P:" + rede.camada(1).saida.dado(0 ,0)
         );
      }

      System.out.println("\nPerda: " + rede.avaliador.erroMedioQuadrado(entrada, saida));

   }
}
