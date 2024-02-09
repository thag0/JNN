package rna.inicializadores;

import rna.core.Mat;

/**
 * Inicializador Glorot normalizado para uso dentro da biblioteca.
 */
public class GlorotNormal extends Inicializador{

   /**
    * Instância um inicializador Glorot normalizado para matrizes 
    * com seed
    * aleatória.
    */
   public GlorotNormal(){}

   /**
    * Instância um inicializador Glorot normalizado para matrizes.
    * @param seed seed usada pelo gerador de números aleatórios.
    */
   public GlorotNormal(long seed){
      super(seed);
   }

   /**
    * Aplica o algoritmo de inicialização Glorot normalizado na matriz 
    * fornecida.
    * @param m matriz que será inicializada.
    */
   @Override
   public void inicializar(Mat m){
      double desvio = Math.sqrt(2.0 / (m.lin() + m.col()));
      m.map((x) -> (
         super.random.nextGaussian() * desvio
      ));
   }
}
