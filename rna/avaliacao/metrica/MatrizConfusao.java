package rna.avaliacao.metrica;

import rna.modelos.Modelo;

public class MatrizConfusao extends Metrica{

   @Override
   public int[][] calcularMatriz(Modelo rede, Object entradas, double[][] saidas){
      return super.matrizConfusao(rede, entradas, saidas);
   } 
}
