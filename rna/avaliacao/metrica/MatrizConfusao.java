package rna.avaliacao.metrica;

import rna.modelos.RedeNeural;

public class MatrizConfusao extends Metrica{

   @Override
   public int[][] calcularMatriz(RedeNeural rede, double[][] entradas, double[][] saidas){
      return super.matrizConfusao(rede, entradas, saidas);
   } 
}
