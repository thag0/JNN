package testes.modelos;
import lib.ged.Ged;
import rna.avaliacao.perda.MSE;
import rna.camadas.Densa;
import rna.inicializadores.Xavier;
import rna.modelos.Sequencial;
import rna.otimizadores.SGD;

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

      //teste a propagação direta do modelo
      //usando os dados da porta lógica XOR

      Sequencial modelo = new Sequencial();
      modelo.add(new Densa(2, 2, "sigmoid"));
      modelo.add(new Densa(1, "sigmoid"));

      modelo.compilar(
         new SGD(),
         new MSE(),
         new Xavier()
      );
      
      double[] pN1 = {-7.577308710973026, 7.34560483929071};
      double   bN1 = -3.8770293408497674;
      
      double[] pN2 = {-7.39212040295605, 7.661754069020709};
      double   bN2 = 3.681963491508159;

      double[] pN3 = {14.662523120653429, -14.163951965348884};
      double   bN3 = 6.7686160132656585;

      Densa d1 = (Densa) modelo.camada(0);
      d1.configurarPesos(0, pN1);
      d1.configurarBias(0, bN1);
      d1.configurarPesos(1, pN2);
      d1.configurarBias(1, bN2);
      
      Densa d2 = (Densa) modelo.camada(1);
      d2.configurarPesos(0, pN3);
      d2.configurarBias(0, bN3);

      double[][] predicoes = (double[][]) modelo.calcularSaidas(entrada);

      for(int i = 0; i < entrada.length; i++){
         StringBuilder sb = new StringBuilder();
         sb.append("(" + entrada[i][0] + ", " + entrada[i][1] + ") ");
         sb.append("Real: " + saida[i][0] + " ");
         sb.append("Pred: " + predicoes[i][0]);
         System.out.println(sb.toString());
      }

      System.out.println("\nPerda: " + modelo.avaliar(entrada, saida));
   }
}
