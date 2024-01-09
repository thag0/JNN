package testes;

import lib.ged.Dados;
import lib.ged.Ged;
import rna.avaliacao.perda.EntropiaCruzada;
import rna.camadas.Densa;
import rna.camadas.Flatten;
import rna.camadas.MaxPooling;
import rna.core.Mat;
import rna.core.OpMatriz;
import rna.inicializadores.Inicializador;
import rna.inicializadores.Xavier;
import rna.treinamento.AuxiliarTreino;

@SuppressWarnings("unused")
public class MatrizTeste{
   static Ged ged = new Ged();
   static OpMatriz opmat = new OpMatriz();
   
   public static void main(String[] args){
      ged.limparConsole();

      double[][] e = {
         {1, 1, 1},
         {2, 2, 2},
         {3, 3, 3},
         {4, 4, 4},
         {5, 5, 5},
         {6, 6, 6},
         {7, 7, 7},
      };

      AuxiliarTreino aux = new AuxiliarTreino();
      Object[] sub = aux.obterSubMatriz(e, 1, 3);
      ged.imprimirMatriz(sub);
   }
}