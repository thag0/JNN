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

      double[][] m = {
         {1, 2, 3},
         {4, 5, 6},
         {7, 8, 9},
      };

      Mat a = new Mat(m);
      Mat b = new Mat(m);
      Mat c = new Mat(m);

      Teste teste = new Teste(a, b, c);
      teste.multiplicar(a, b, c);
   }
}

class Teste{
   public Mat a;
   public Mat b;
   public Mat c;
   private OpMatriz opm = new OpMatriz();

   public Teste(Mat a, Mat b, Mat c){
      this.a = a;
      this.b = b;
      this.c = c;
   }

   public void multiplicar(Mat a, Mat b, Mat c){
      opm.multT(a, b, c, 2);
   }
}