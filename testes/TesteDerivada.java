package testes;

import ged.Ged;
import rna.core.Mat;
import rna.core.OpMatriz;
import rna.estrutura.CamadaDensa;

class TesteDerivada{
   static Ged ged = new Ged();
   static OpMatriz opm = new OpMatriz();

   public static void main(String[] args){
      ged.limparConsole();

      Mat pesos = new Mat(new double[][]{
         {1, 2, 3},
         {4, 5, 6},
         {7, 8, 9},
      });

      CamadaDensa camada = new CamadaDensa(3, 3);
      camada.configurarAtivacao("softmax");

      double[] entrada = {2, 3, 4};
      double[] grad = {0.01, 0.23, 0.45};

      camada.pesos = pesos;

      camada.calcularSaida(entrada);
      camada.calcularGradiente(grad);

      camada.somatorio.print("Somatorio");
      camada.saida.print("Saida");
      camada.derivada.print("Derivada");
      camada.gradienteEntrada.print("Grad Entrada");
   }
}