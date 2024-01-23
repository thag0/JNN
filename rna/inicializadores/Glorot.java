package rna.inicializadores;

import rna.core.Mat;

/**
 * Inicializador Glorot para uso dentro da biblioteca.
 */
public class Glorot extends Inicializador{

   /**
    * Instância um inicializador Glorot para matrizes com seed
    * aleatória.
    */
   public Glorot(){}

   /**
    * Instância um inicializador Glorot para matrizes.
    * @param seed seed usada pelo gerador de números aleatórios.
    */
   public Glorot(long seed){
      super(seed);
   }

   /**
    * Aplica o algoritmo de inicialização Glorot na matriz fornecida.
    * @param m matriz que será inicializada.
    */
   @Override
   public void inicializar(Mat m){
      double desvio = Math.sqrt(6.0 / (m.lin() + m.col()));
      m.map((x) -> (
         super.random.nextDouble() * (2 * desvio) - desvio
      ));
   }
}
