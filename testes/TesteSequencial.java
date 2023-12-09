package testes;

import ged.Ged;
import rna.avaliacao.perda.ErroMedioQuadrado;
import rna.core.OpMatriz;
import rna.estrutura.Camada;
import rna.estrutura.Densa;
import rna.inicializadores.Xavier;
import rna.modelos.Sequencial;
import rna.otimizadores.SGD;

class TesteSequencial{
   static Ged ged = new Ged();
   static OpMatriz opm = new OpMatriz();

   public static void main(String[] args){
      ged.limparConsole();

      double[][] e = {
         {0, 0},
         {0, 1},
         {1, 0},
         {1, 1}
      };

      double[][] s = {
         {0},
         {1},
         {1},
         {0}
      };

      Sequencial modelo = new Sequencial(new Camada[]{
         new Densa(2, 3, "tanh"),
         new Densa(1, "sigmoid")
      });
      modelo.compilar(new SGD(0.0001, 0.999), new ErroMedioQuadrado(), new Xavier());

      System.out.println(modelo.info());

      modelo.treinar(e, s, 5000, false);

      for(int i = 0; i < 2; i++){
         for(int j = 0; j < 2; j++){
            double[] amostra = {i, j};
            modelo.calcularSaida(amostra);
            double[] previsao = modelo.obterSaida();
            System.out.println(i + " " + j + " = " + previsao[0]);
         }
      }
   }
}