package rna.inicializadores;

import java.util.Random;

import rna.core.Mat;
import rna.core.Tensor4D;

/**
 * Classe responsável pelas funções de inicialização dos pesos
 * da Rede Neural.
 */
public abstract class Inicializador{

   /**
    * Gerador de números pseudo aleatórios compartilhado
    * para as classes filhas.
    */
   protected Random random = new Random();
   
   /**
    * Inicialização com seed aleatória
    */
   protected Inicializador(){}

   /**
    * Inicialização com seed configurada.
    * @param seed seed. 
    */
   protected Inicializador(long seed){
      this.random.setSeed(seed);
   }

   /**
    * Configura o início do gerador aleatório.
    * @param seed nova seed de início.
    */
   public void configurarSeed(long seed){
      this.random.setSeed(seed);
   }

   /**
    * Inicializa os valores do array de acordo com o inicializador configurado.
    * @param m matriz de dados.
    */
   public abstract void inicializar(Mat m);

   /**
    * Inicializa os valores tensor.
    * @param tensor tensor desejado.
    */
   public abstract void inicializar(Tensor4D tensor, int dim1, int dim2);

   /**
    * Retorna o nome do inicializador.
    * @return nome do inicializador.
    */
   public String nome(){
      return getClass().getSimpleName();
   }
}
