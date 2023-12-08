package rna.modelos;

import rna.avaliacao.perda.ErroMedioQuadrado;
import rna.avaliacao.perda.Perda;
import rna.core.Mat;
import rna.estrutura.Camada;
import rna.inicializadores.Inicializador;
import rna.otimizadores.Otimizador;
import rna.otimizadores.SGD;

/**
 * Modelo básico ainda.
 */
public class Sequencial{

   /**
    * Lista de camadas do modelo.
    */
   public Camada[] camadas;

   /**
    * Função de perda do modelo.
    */
   public Perda perda = new ErroMedioQuadrado();
   
   /**
    * Otimizador do modelo.
    */
   public Otimizador otimizador = new SGD();

   /**
    * Inicializa um modelo sequencial vazio.
    */
   public Sequencial(){
      this.camadas = new Camada[0];
   }

   /**
    * Inicializa um modelo sequencial a partir de um conjunto de camadas
    * definido
    * @param camadas camadas que serão usadas pelo modelo.
    */
   public Sequencial(Camada[] camadas){
      this.camadas = camadas;
   }

   /**
    * Adiciona novas camadas ao final da lista de camadas do modelo.
    * @param camada nova camada.
    */
   public void add(Camada camada){
      Camada[] c = this.camadas;
      this.camadas = new Camada[c.length+1];

      for(int i = 0; i < c.length; i++){
         this.camadas[i] = c[i];
      }
      this.camadas[this.camadas.length-1] = camada;
   }

   /**
    * Apaga a última camada contida no modelo.
    */
   public void sub(){
      Camada[] c = this.camadas;
      this.camadas = new Camada[this.camadas.length-1];
      for(int i = 0; i < this.camadas.length; i++){
         this.camadas[i] = c[i];
      }
   }

   /**
    * Configura a função de perda que será utilizada durante o processo
    * de treinamento da Rede Neural.
    * @param perda nova função de perda.
    */
   public void configurarPerda(Perda perda){
      if(perda == null){
         throw new IllegalArgumentException("A função de perda não pode ser nula.");
      }

      this.perda = perda;
   }

   /**
    * Configura o novo otimizador da Rede Neural com base numa nova instância de otimizador.
    * <p>
    *    Configurando o otimizador passando diretamente uma nova instância permite configurar
    *    os hiperparâmetros do otimizador fora dos valores padrão, o que pode ajudar a
    *    melhorar o desempenho de aprendizado da Rede Neural em cenário específicos.
    * </p>
    * Otimizadores disponíveis.
    * <ol>
    *    <li> GradientDescent  </li>
    *    <li> SGD (Gradiente Descendente Estocástico) </li>
    *    <li> AdaGrad </li>
    *    <li> RMSProp </li>
    *    <li> Adam  </li>
    *    <li> Nadam </li>
    *    <li> AMSGrad </li>
    *    <li> Adamax  </li>
    *    <li> Lion   </li>
    *    <li> Adadelta </li>
    * </ol>
    * <p>
    *    {@code O otimizador padrão é o SGD}
    * </p>
    * @param otimizador novo otimizador.
    * @throws IllegalArgumentException se o novo otimizador for nulo.
    */
   public void configurarOtimizador(Otimizador otimizador){
      if(otimizador == null){
         throw new IllegalArgumentException("O novo otimizador não pode ser nulo.");
      }
      this.otimizador = otimizador;
   }

   /**
    * Inicializa os parâmetros necessários para cada camada do modelo,
    * além de aleatorizar os kernels e bias.
    * @param otimizador otimizador usando para ajustar os parâmetros treinavéis do modelo.
    * @param perda função de perda usada para o treinamento do modelo.
    * @param iniKernel inicializador para os kernels.
    */
   public void compilar(Otimizador otimizador, Perda perda, Inicializador iniKernel){
      this.compilar(otimizador, perda, iniKernel, null);
   }

   /**
    * Inicializa os parâmetros necessários para cada camada do modelo,
    * além de aleatorizar os kernels e bias.
    * @param otimizador
    * @param perda
    * @param iniKernel inicializador para os kernels.
    * @param iniBias inicializador para os bias.
    */
   public void compilar(Otimizador otimizador, Perda perda, Inicializador iniKernel, Inicializador iniBias){
      if(iniKernel == null){
         throw new IllegalArgumentException(
            "O inicializador para o kernel não pode ser nulo."
         );
      }

      for(int i = 0; i < this.camadas.length; i++){
         this.camadas[i].inicializar(iniKernel, iniBias, 0);
         this.camadas[i].configurarId(i);
      }

      this.otimizador.inicializar(this.camadas);
   }

   /**
    * Feedforward
    * @param entrada entrada.
    */
   public void calcularSaida(Object entrada){
      this.camadas[0].calcularSaida(entrada);
      for(int i = 1; i < this.camadas.length; i++){
         this.camadas[i].calcularSaida(this.camadas[i-1].obterSaida());
      }
   }

   public void treinar(double[][][][] entrada, double[][] saida, int epochs, boolean logs){
      for(int e = 0; e < epochs; e++){
         double perdaEpoca = 0;
         for(int i = 0; i < entrada.length; i++){
            this.calcularSaida(entrada[i]);
            perdaEpoca += this.perda.calcular(this.obterSaida(), saida[i]);

            double[] g = this.perda.derivada(this.obterSaida(), saida[i]);
            Mat gradientes = new Mat(g);
            this.calcularGradientes(gradientes);
            this.otimizador.atualizar(this.camadas);
         }

         perdaEpoca /= entrada.length;

         if(logs & (e % 100 == 0)){
            System.out.println("Perda (" + e + "): " + perdaEpoca);
         }
      }
   }

   public void calcularGradientes(Object gradSeguinte){
      Camada saida = this.camadas[this.camadas.length-1];
      saida.calcularGradiente(gradSeguinte);

      for(int i = this.camadas.length-2; i >= 0; i--){
         this.camadas[i].calcularGradiente(this.camadas[i+1].obterGradEntrada());
      }
   }

   public double[] obterSaida(){
      double[] saida = (double[]) this.camadas[this.camadas.length-1].obterSaida();
      return  saida;
   }
}
