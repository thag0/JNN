package testes;

import lib.ged.Dados;
import lib.ged.Ged;
import rna.avaliacao.perda.EntropiaCruzada;
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

      Mat entrada = new Mat(new double[][]{
         {1, 2, 3},
         {4, 5, 6},
         {7, 8, 9},
      });

      Dropout dropout = new Dropout(0.5);
      dropout.construir(new int[]{entrada.lin(), entrada.col()});
      dropout.configurarTreino(true);

      dropout.calcularSaida(entrada);
      dropout.mascara.print("mascara");
      dropout.saida.print("saida");
   }
}