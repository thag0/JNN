package rna.inicializadores;

import rna.core.Mat;

/**
 * Inicializador Gaussiano para uso dentro da biblioteca.
 */
public class Gaussiano extends Inicializador{

   /**
    * Instância um inicializador Gaussiano para matrizes com seed
    * aleatória.
    */
   public Gaussiano(){}

   /**
    * Instância um inicializador Gaussiano para matrizes.
    * @param seed seed usada pelo gerador de números aleatórios.
    */
   public Gaussiano(long seed){
      super(seed);
   }

   /**
    * Aplica o algoritmo de inicialização gaussiano/normal nos pesos.
    * @param m matriz que será inicializada
    */
   @Override
   public void inicializar(Mat m){
      m.map(val -> super.random.nextGaussian());
   }
}
