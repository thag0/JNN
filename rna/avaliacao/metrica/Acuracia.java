package rna.avaliacao.metrica;

import rna.estrutura.RedeNeural;

public class Acuracia extends Metrica{

   @Override
   public double calcular(RedeNeural rede, double[][] entrada, double[][] saida){
      int numAmostras = entrada.length;
      int acertos = 0;
      double[] dadosEntrada = new double[entrada[0].length];
      double[] dadosSaida = new double[saida[0].length];

      for(int i = 0; i < numAmostras; i++){
         System.arraycopy(entrada[i], 0, dadosEntrada, 0, dadosEntrada.length);
         System.arraycopy(saida[i], 0, dadosSaida, 0, dadosSaida.length);

         rede.calcularSaida(dadosEntrada);

         int indiceCalculado = super.indiceMaiorValor(rede.obterSaidas());
         int indiceEsperado = super.indiceMaiorValor(dadosSaida);

         if(indiceCalculado == indiceEsperado){
            acertos++;
         }
      }

      double acuracia = (double)acertos / numAmostras;
      return acuracia;
   }
}
