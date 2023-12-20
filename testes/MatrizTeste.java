package testes;

import ged.Dados;
import ged.Ged;
import rna.avaliacao.perda.EntropiaCruzada;
import rna.core.Mat;
import rna.core.OpMatriz;
import rna.estrutura.Densa;
import rna.estrutura.Flatten;
import rna.estrutura.MaxPooling;
import rna.inicializadores.Inicializador;
import rna.inicializadores.Xavier;

@SuppressWarnings("unused")
public class MatrizTeste{
   static Ged ged = new Ged();
   static OpMatriz opmat = new OpMatriz();
   
   public static void main(String[] args){
      ged.limparConsole();

      double[][] e1 = {
         {1, 2, 3, 4},
         {5, 6, 7, 8},
         {4, 3, 2, 1},
         {8, 7, 6, 5}
      };
      
      double[][] f1 = {
         {3, 2, 1},
         {1, 2, 3},
         {4, 5, 6}
      };

      double[][] g1 = {
         {9, 8, 7},
         {3, 2, 1},
         {3, 4, 6}
      };

      Mat entrada = new Mat(e1);
      Mat filtro = new Mat(f1);
      Mat gradiente = new Mat(g1);

      Mat saida = new Mat(2, 2);
      Mat gradK = new Mat(2, 2);
      Mat gradE = new Mat(5, 5);

      opmat.correlacaoCruzada(entrada, filtro, saida, false);
      opmat.correlacaoCruzada(entrada, gradiente, gradK, false);
      opmat.convolucaoFull(gradiente, filtro, gradE, false);
      saida.print("saida");
      gradK.print("gradk");
      gradE.print("gradE");
   }

   static void derivadaSoftmax(){
      double[] saida = {1.0, 2.0, 3.0, 4.0};
      double[] grad  = {6.0, 6.0, 6.0, 6.0};
   
      Mat s = new Mat(1, 4, saida);
      Mat g = new Mat(1, 4, grad);
      
      int n = s.col();
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