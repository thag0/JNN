package testes;

import ged.Dados;
import ged.Ged;
import rna.avaliacao.perda.EntropiaCruzada;
import rna.core.Mat;
import rna.core.OpMatriz;
import rna.estrutura.Densa;
import rna.estrutura.Flatten;
import rna.inicializadores.Inicializador;
import rna.inicializadores.Xavier;

@SuppressWarnings("unused")
public class MatrizTeste{
   static Ged ged = new Ged();
   static OpMatriz opmat = new OpMatriz();
   
   public static void main(String[] args){
      ged.limparConsole();

      double[][] e1 = {
         {1, 6, 2},
         {5, 3, 1},
         {7, 0, 4},
      };
      double[][] f1 = {
         {1, 2},
         {-1, 0},
      };
      
      double[][] grad1 = {
         {1, 2, 3},
         {4, 5, 6},
         {7, 8, 9}
      };

      double[][] filtro = {
         {1, 2},
         {3, 4}
      };

      Mat g = new Mat(grad1);
      Mat f = new Mat(filtro);
      Mat res = new Mat(2, 2);
      opmat.correlacaoCruzada(g, f, res);
      res.print();
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