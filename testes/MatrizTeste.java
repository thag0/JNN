package testes;

import lib.ged.Dados;
import lib.ged.Ged;
import rna.avaliacao.perda.EntropiaCruzada;
import rna.camadas.Convolucional;
import rna.camadas.Densa;
import rna.camadas.Dropout;
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

      Convolucional conv = new Convolucional(new int[]{2, 2}, 1);
      conv.construir(new int[]{2, 2, 1});
      conv.inicializar(new Xavier(), new Xavier(), 0);

      Convolucional clone = conv.clonar();

      System.out.println(conv.entrada);
      System.out.println(clone.entrada);
   }
}