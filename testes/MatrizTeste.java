package testes;

import ged.Ged;
import rna.avaliacao.perda.EntropiaCruzada;
import rna.core.Mat;
import rna.core.OpMatriz;
import rna.estrutura.CamadaDensa;
import rna.inicializadores.Inicializador;
import rna.inicializadores.Xavier;

@SuppressWarnings("unused")
public class MatrizTeste{
   static Ged ged = new Ged();
   static OpMatriz mat = new OpMatriz();
   
   public static void main(String[] args){
      ged.limparConsole();
      derivadaSoftmax();

   }

   static void derivadaSoftmax(){
      double[] saida = {1.0, 2.0, 3.0, 4.0};
      double[] grad  = {6.0, 6.0, 6.0, 6.0};
   
      Mat s = new Mat(1, 4, saida);
      Mat g = new Mat(1, 4, grad);
      
      int n = s.col;
      Mat tmp = s.bloco(0, n);
      Mat ident = mat.identidade(n);
      Mat trans = tmp.transpor();

      Mat derivada = new Mat(1, 4);
      
      Mat resSub = new Mat(n, n);
      mat.sub(ident, trans, resSub);
      mat.hadamard(tmp, resSub, resSub);

      mat.mult(g, resSub, derivada);
      derivada.print();
   }
}