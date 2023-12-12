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
         {5, 2, 8, 1},
         {1, 7, 3, 6},
         {9, 8, 4, 1},
         {3, 4, 2, 6}
      };
      double[][] e2 = {
         {0.77, 0.00, 0.11, 0.33, 0.55, 0.00, 0.33},
         {0.00, 1.00, 0.00, 0.33, 0.55, 0.11, 0.33},
         {0.11, 0.00, 1.00, 0.00, 0.11, 0.00, 0.55},
         {0.33, 0.33, 0.00, 0.55, 0.11, 0.33, 0.33},
         {0.55, 0.00, 0.11, 0.00, 1.00, 0.00, 0.11},
         {0.00, 0.11, 0.00, 0.33, 0.00, 1.00, 0.00},
         {0.33, 0.00, 0.55, 0.33, 0.11, 0.00, 0.77}
      };

      double[][] g = {
         {0.4, 0.5, 0.6},
         {0.2, 0.7, 0.9},
         {0.1, 0.4, 0.5}
      };

      Mat[] entradas = new Mat[1];
      entradas[0] = new Mat(e2);
      entradas[0].print();
      
      MaxPooling mp = new MaxPooling(new int[]{e2.length, e2[0].length, 1}, new int[]{2, 2});
      int[] formSaida = mp.formatoSaida();
      mp.calcularSaida(entradas);

      double[] saidas = mp.saidaParaArray();
      Mat saida = new Mat(formSaida[0], formSaida[1]);
      saida.copiar(saidas);
      saida.print("saída max pooling");

      Mat[] grads = new Mat[1];
      grads[0] = new Mat(g);
      mp.calcularGradiente(grads);

      Mat[] gE = mp.obterGradEntrada();
      gE[0].print("grad entrada");
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