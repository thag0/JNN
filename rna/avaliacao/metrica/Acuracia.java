package rna.avaliacao.metrica;

import rna.modelos.Modelo;

public class Acuracia extends Metrica{

   @Override
   public double calcular(Modelo rede, Object[] entrada, Object[] saida){
      int numAmostras = entrada.length;
      int acertos = 0;

      if(saida instanceof double[][] == false){
         throw new IllegalArgumentException(
            "Objeto esperado para saída é double[][], recebido " + saida.getClass().getTypeName()
         );
      }

      for(int i = 0; i < numAmostras; i++){

         rede.calcularSaida(entrada[i]);

         int indiceCalculado = super.indiceMaiorValor(rede.saidaParaArray());
         int indiceEsperado = super.indiceMaiorValor((double[])saida[i]);

         if(indiceCalculado == indiceEsperado){
            acertos++;
         }
      }

      double acuracia = (double)acertos / numAmostras;
      return acuracia;
   }
}
