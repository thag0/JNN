package rna.inicializadores;

import java.util.Random;
import java.util.function.DoubleUnaryOperator;

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
    * Função usada pelo inicializador.
    */
   DoubleUnaryOperator func;

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
    * Inicializa os valores tensor de acordo com o índice especificado.
    * @param tensor tensor desejado.
    * @param dim1 índice da primeira dimensão.
    */
   public abstract void inicializar(Tensor4D tensor, int dim1);

   /**
    * Inicializa todos os valores tensor.
    * @param tensor tensor desejado.
    */
   public abstract void inicializar(Tensor4D tensor);

   /**
    * Inicializa os valores tensor de acordo com os índices especificados.
    * @param tensor tensor desejado.
    * @param dim1 índice da primeira dimensão.
    * @param dim2 índice da segunda dimensão.
    */
   public abstract void inicializar(Tensor4D tensor, int dim1, int dim2);

   /**
    * Inicializa os valores tensor de acordo com os índices especificados.
    * @param tensor tensor desejado.
    * @param dim1 índice da primeira dimensão.
    * @param dim2 índice da segunda dimensão.
    * @param dim3 índice da terceira dimensão.
    */
   public abstract void inicializar(Tensor4D tensor, int dim1, int dim2, int dim3);

   /**
    * Retorna o nome do inicializador.
    * @return nome do inicializador.
    */
   public String nome(){
      return getClass().getSimpleName();
   }
}
