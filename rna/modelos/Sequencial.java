package rna.modelos;

import rna.avaliacao.perda.ErroMedioQuadrado;
import rna.avaliacao.perda.Perda;
import rna.core.Mat;
import rna.estrutura.Camada;
import rna.estrutura.Convolucional;
import rna.estrutura.Densa;
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
      if(this.camadas[0].construida == false){
         throw new IllegalArgumentException(
            "É necessário que a primeira camada seja construída."
         );
      }

      for(int i = 1; i < this.camadas.length; i++){
         this.camadas[i].construir(this.camadas[i-1].formatoSaida());
      }

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
         this.camadas[i].calcularSaida(this.camadas[i-1].saidaParaArray());
      }
   }

   /**
    * 
    * @param entrada
    * @param saida
    * @param epochs
    * @param logs
    */
   public void treinar(Object entradas, double[][] saida, int epochs, boolean logs){
      if(entradas instanceof double[][][][] && this.camadas[0] instanceof Convolucional){
         treinoConv((double[][][][]) entradas, saida, epochs, logs);
         
      }else if(entradas instanceof double[][] && this.camadas[0] instanceof Densa){
         treinoDensa((double[][]) entradas, saida, epochs, logs);

      }else{
         throw new IllegalArgumentException("Formato de entrada não suportado");
      }

   }

   private void treinoConv(double[][][][] entrada, double[][] saida, int epochs, boolean logs){
      for(int e = 0; e < epochs; e++){
         double perdaEpoca = 0;
         for(int i = 0; i < entrada.length; i++){
            this.calcularSaida(entrada[i]);
            double[] previsao = this.obterSaida();

            if(logs){
               perdaEpoca += this.perda.calcular(previsao, saida[i]);
            }

            this.backpropagation(previsao, saida[i]);
            this.otimizador.atualizar(this.camadas);
         }

         perdaEpoca /= entrada.length;

         if(logs & (e % 20 == 0)){
            System.out.println("Perda (" + e + "): " + perdaEpoca);
         }
      }
   }

   private void treinoDensa(double[][] entrada, double[][] saida, int epochs, boolean logs){
      for(int e = 0; e < epochs; e++){
         double perdaEpoca = 0;
         for(int i = 0; i < entrada.length; i++){
            this.calcularSaida(entrada[i]);
            double[] previsao = this.obterSaida();

            if(logs){
               perdaEpoca += this.perda.calcular(previsao, saida[i]);
            }

            this.backpropagation(previsao, saida[i]);
            this.otimizador.atualizar(this.camadas);
         }

         perdaEpoca /= entrada.length;

         if(logs & (e % 20 == 0)){
            System.out.println("Perda (" + e + "): " + perdaEpoca);
         }
      }
   }

   /**
    * Realiza a retroprogação dos gradientes para as camadas do modelo.
    * @param previsto saídas previstas pelo modelo.
    * @param real rótulos reais.
    */
   public void backpropagation(double[] previsto, double[] real){
      double[] grads = this.perda.derivada(previsto, real);
      this.obterCamadaSaida().calcularGradiente(new Mat(grads));
      
      for(int i = this.camadas.length-2; i >= 0; i--){
         this.camadas[i].calcularGradiente(this.camadas[i+1].obterGradEntrada());
      }
   }

   /**
    * Retorna a {@code camada} do Modelo correspondente ao índice fornecido.
    * @param id índice da busca.
    * @return camada baseada na busca.
    * @throws IllegalArgumentException se o índice estiver fora do alcance do tamanho 
    * das camadas.
    */
   public Camada obterCamada(int id){
      if((id < 0) || (id >= this.camadas.length)){
         throw new IllegalArgumentException(
            "O índice fornecido (" + id + 
            ") é inválido ou fora de alcance."
         );
      }
   
      return this.camadas[id];
   }

   /**
    * Retorna a {@code camada de saída} do modelo.
    * @return camada de saída.
    */
   public Camada obterCamadaSaida(){
      return this.camadas[this.camadas.length-1];
   }

   /**
    * Retorna um array contendo a saída do modelo.
    * @return saída do modelo.
    */
   public double[] obterSaida(){
      double[] saida = (double[]) this.camadas[this.camadas.length-1].saidaParaArray();
      return  saida;
   }
}
