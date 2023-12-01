package testes;

import ged.Dados;
import ged.Ged;
import rna.avaliacao.perda.EntropiaCruzada;
import rna.core.Mat;
import rna.core.OpMatriz;
import rna.estrutura.Densa;
import rna.inicializadores.Inicializador;
import rna.inicializadores.Xavier;

@SuppressWarnings("unused")
public class MatrizTeste{
   static Ged ged = new Ged();
   static OpMatriz opmat = new OpMatriz();
   
   public static void main(String[] args){
      ged.limparConsole();

      double[][] e = {
         {1, 6, 2},
         {5, 3, 1},
         {7, 0, 4},
      };
      double[][] f = {
         {1, 2},
         {-1, 0},
      };

      Mat entrada = new Mat(e);
      Mat filtro = new Mat(f);
      Mat resCorr = new Mat(2, 2);
      Mat resConv = new Mat(2, 2);

      opmat.correlacaoCruzada(entrada, filtro, resCorr);
      opmat.convolucao(entrada, filtro, resConv);
      resCorr.print("Corr");
      resConv.print("Conv");
   }

   static void derivadaSoftmax(){
      double[] saida = {1.0, 2.0, 3.0, 4.0};
      double[] grad  = {6.0, 6.0, 6.0, 6.0};
   
      Mat s = new Mat(1, 4, saida);
      Mat g = new Mat(1, 4, grad);
      
      int n = s.col;
      Mat tmp = s.bloco(0, n);
      Mat ident = opmat.identidade(n);
      Mat trans = tmp.transpor();

      Mat derivada = new Mat(1, 4);
      
      Mat resSub = new Mat(n, n);
      opmat.sub(ident, trans, resSub);
      opmat.hadamard(tmp, resSub, resSub);

      opmat.mult(g, resSub, derivada);
      derivada.print();
   }
}