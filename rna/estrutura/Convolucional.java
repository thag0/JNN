package rna.estrutura;

import rna.ativacoes.Ativacao;
import rna.ativacoes.ReLU;
import rna.core.Mat;
import rna.core.OpMatriz;
import rna.inicializadores.Constante;
import rna.inicializadores.Inicializador;
import rna.serializacao.DicionarioAtivacoes;

/**
 * Implementação em andamento da camada convolucional.
 */
public class Convolucional extends Camada{

   /**
    * Operador matricial para a camada.
    */
   OpMatriz opmat = new OpMatriz();

   private int altEntrada;
   private int largEntrada;
   private int profEntrada;

   private int altFiltro;
   private int largFiltro;
   private int numFiltros;

   private int altSaida;
   private int largSaida;

   /**
    * Array de matrizes contendo os valores de entrada para a camada,
    * que serão usados para o processo de feedforward.
    * <p>
    *    O formato da entrada é dado por:
    * </p>
    * <pre>
    *entrada = [profundidade entrada]
    *entrada[n] = [alturaEntrada][larguraEntrada]
    * </pre>
    */
   public Mat[] entrada;

   /**
    * Array bidimensional de matrizes contendo os filtros (ou kernels)
    * da camada.
    * <p>
    *    O formato dos filtros é dado por:
    * </p>
    * <pre>
    *filtros = [numFiltro][profundidadeEntrada]
    *filtros[i][j] = [alturaFiltro][larguraFiltro]
    * </pre>
    */
   public Mat[][] filtros;

   /**
    * Array de matrizes contendo os bias (vieses) para cada valor de 
    * saída da camada.
    * <p>
    *    O formato do bias é dado por:
    * </p>
    * <pre>
    *bias = [numeroFiltros]
    *bias[n] = [alturaSaida][larguraSaida]
    * </pre>
    */
   public Mat[] bias;

   /**
    * Auxiliar na verificação de uso do bias.
    */
   private boolean usarBias;

   /**
    * Array de matrizes contendo valores de somatório para cada valor de 
    * saída da camada.
    * <p>
    *    O formato somatório é dado por:
    * </p>
    * <pre>
    *somatorio = [numeroFiltros]
    *somatorio[n] = [alturaSaida][larguraSaida]
    * </pre>
    */
   public Mat[] somatorio;
   
   /**
    * Array de matrizes contendo os valores de saídas da camada.
    * <p>
    *    O formato da saída é dado por:
    * </p>
    * <pre>
    *saida = [numeroFiltros]
    *saida[n] = [alturaSaida][larguraSaida]
    * </pre>
    */
   public Mat[] saida;

   /**
    * Função de ativação da camada.
    */
   Ativacao ativacao = new ReLU();

   /**
    * Instancia uma camada convolucional de acordo com os formatos fornecidos.
    * <p>
    *    A disposição do formato de entrada deve ser da seguinte forma:
    * </p>
    * <pre>
    *    formEntrada = (altura, largura, profundidade)
    * </pre>
    * Onde largura e altura devem corresponder as dimensões dos dados de entrada
    * que serão processados pela camada e a profundidade diz respeito a quantidade
    * de entradas que a camada deve processar.
    * <p>
    *    A disposição do formato do filtro deve ser da seguinte forma:
    * </p>
    * <pre>
    *    formFiltro = (altura, largura)
    * </pre>
    * Onde largura e altura correspondem as dimensões que os filtros devem assumir.
    * @param formEntrada formato de entrada da camada.
    * @param formFiltro formato dos filtros da camada.
    * @param numFiltros quantidade de filtros.
    * @param usarBias adicionar uso do bias para a camada.
    * @throws IllegalArgumentException se as dimensões fornecidas não correspondenrem
    * ao padrão desejado ou se o número de filtros for menor que 1.
    */
   public Convolucional(int[] formEntrada, int[] formFiltro, int numFiltros, boolean usarBias){
      if(formEntrada.length != 3){
         throw new IllegalArgumentException(
            "O formato de entrada deve conter 3 elementos (altura, largura, profundidade)" 
         );
      }
      for(int i = 0; i < formEntrada.length; i++){
         if(formEntrada[i] < 1){
            throw new IllegalArgumentException(
               "O formato de entrada deve conter valores maiores do que zero." 
            ); 
         }
      }
      if(formFiltro.length != 2){
         throw new IllegalArgumentException(
            "O formato do filtro deve conter 2 elementos (altura, largura)" 
         );
      }
      for(int i = 0; i < formFiltro.length; i++){
         if(formFiltro[i] < 1){
            throw new IllegalArgumentException(
               "O formato do filtro deve conter valores maiores do que zero." 
            ); 
         }
      }
      if(numFiltros < 1){
         throw new IllegalArgumentException(
            "A camada deve conter ao menos 1 filtro."
         );
      }

      //formato da entrada da camada
      this.altEntrada  = formEntrada[0];
      this.largEntrada = formEntrada[1];
      this.profEntrada = formEntrada[2];

      //formato dos filtros da camada
      this.altFiltro  = formFiltro[0];
      this.largFiltro = formFiltro[1];
      this.numFiltros = numFiltros;
      
      this.entrada = new Mat[profEntrada];
      for(int i = 0; i < this.profEntrada; i++){
         this.entrada[i] = new Mat(this.altEntrada, this.largEntrada);
      }
      
      this.filtros = new Mat[numFiltros][this.profEntrada];
      this.somatorio = new Mat[numFiltros];
      this.saida = new Mat[numFiltros];

      this.usarBias = usarBias;
      if(this.usarBias){
         this.bias = new Mat[numFiltros];
      }

      this.altSaida = this.altEntrada - this.altFiltro + 1;
      this.largSaida = this.largEntrada - this.largFiltro + 1;

      for(int i = 0; i < this.numFiltros; i++){
         this.filtros[i] = new Mat[this.profEntrada];
         for(int j = 0; j < this.profEntrada; j++){
            this.filtros[i][j] = new Mat(this.altFiltro, this.largFiltro);
         }

         this.somatorio[i] = new Mat(this.altSaida, this.largSaida);
         this.saida[i] = new Mat(this.altSaida, this.largSaida);

         if(this.usarBias){
            this.bias[i] = new Mat(this.altSaida, this.largSaida);
         }
      }
   }

   /**
    * Instancia uma camada convolucional de acordo com os formatos fornecidos.
    * <p>
    *    A disposição do formato de entrada deve ser da seguinte forma:
    * </p>
    * <pre>
    *    formEntrada = (altura, largura, profundidade)
    * </pre>
    * Onde largura e altura devem corresponder as dimensões dos dados de entrada
    * que serão processados pela camada e a profundidade diz respeito a quantidade
    * de entradas que a camada deve processar.
    * <p>
    *    A disposição do formato do filtro deve ser da seguinte forma:
    * </p>
    * <pre>
    *    formFiltro = (altura, largura)
    * </pre>
    * Onde largura e altura correspondem as dimensões que os filtros devem assumir.
    * <p>
    *    O valor de uso do bias será usado como {@code true} por padrão.
    * <p>
    * @param formEntrada formato de entrada da camada.
    * @param formFiltro formato dos filtros da camada.
    * @param numFiltros quantidade de filtros.
    * @throws IllegalArgumentException se as dimensões fornecidas não correspondenrem
    * ao padrão desejado ou se o número de filtros for menor que 1.
    */
   public Convolucional(int[] formEntrada, int[] formFiltro, int numFiltros){
      this(formEntrada, formFiltro, numFiltros, true);
   }

   /**
    * Configura a função de ativação da camada através do nome fornecido, letras 
    * maiúsculas e minúsculas não serão diferenciadas.
    * <p>
    *    Ativações disponíveis:
    * </p>
    * <ul>
    *    <li> ReLU. </li>
    *    <li> Sigmoid. </li>
    *    <li> TanH. </li>
    *    <li> Leaky ReLU. </li>
    *    <li> ELU .</li>
    *    <li> Swish. </li>
    *    <li> GELU. </li>
    *    <li> Linear. </li>
    *    <li> Seno. </li>
    *    <li> Argmax. </li>
    *    <li> Softmax. </li>
    *    <li> Softplus. </li>
    *    <li> ArcTan. </li>
    * </ul>
    * @param ativacao nome da nova função de ativação.
    * @throws IllegalArgumentException se o valor fornecido não corresponder a nenhuma 
    * função de ativação suportada.
    */
   public void configurarAtivacao(String ativacao){
      DicionarioAtivacoes dic = new DicionarioAtivacoes();
      this.ativacao = dic.obterAtivacao(ativacao);
   }

   /**
    * Configura a função de ativação da camada através de uma instância de 
    * {@code FuncaoAtivacao} que será usada para ativar seus neurônios.
    * <p>
    *    Configurando a ativação da camada usando uma instância de função 
    *    de ativação aumenta a liberdade de personalização dos hiperparâmetros
    *    que algumas funções podem ter.
    * </p>
    * @param ativacao nova função de ativação.
    * @throws IllegalArgumentException se a função de ativação fornecida for nula.
    */
   public void configurarAtivacao(Ativacao ativacao){
      if(ativacao == null){
         throw new IllegalArgumentException(
            "A função de ativação não pode ser nula."
         );
      }

      this.ativacao = ativacao;
   }

   /**
    * Inicaliza os pesos e bias (caso tenha) da camada de acordo com o 
    * inicializador configurado.
    * @param iniFiltros inicializador de filtros.
    * @param iniBias inicializador de bias.
    * @param x valor usado pelos inicializadores, dependendo do que for usado
    * pode servir de alcance na aleatorização, valor de constante, entre outros.
    */
   public void inicializar(Inicializador iniFiltros, Inicializador iniBias, double x){
      if(iniFiltros == null){
         throw new IllegalArgumentException(
            "O inicializador não pode ser nulo."
         );
      }

      for(int i = 0; i < numFiltros; i++){
         for(int j = 0; j < profEntrada; j++){
            iniFiltros.inicializar(this.filtros[i][j], x);
         }
      }
      
      if(this.usarBias){
         for(Mat b : this.bias){
            if(iniBias == null) new Constante().inicializar(b, 0);
            else iniBias.inicializar(b, x);
         }
      }
   }

   /**
    * Inicaliza os pesos da camada de acordo com o inicializador configurado.
    * @param iniPesos inicializador de pesos.
    * @param x valor usado pelos inicializadores, dependendo do que for usado
    * pode servir de alcance na aleatorização, valor de constante, entre outros.
    */
   public void inicializar(Inicializador iniPesos, double x){
      this.inicializar(iniPesos, null, x);
   }

   /**
    * Realiza a operação de correlação cruzada entre os valores de entrada
    * e os filtros da camada.
    * @param entrada array de matrizes contendo os valores de entrada.
    */
   @Override
   public void calcularSaida(Object entrada){
      if(entrada instanceof double[][][] == false){
         throw new IllegalArgumentException(
            "Os dados de entrada para a camada Convolucional devem ser " +
            "do tipo \"double[][][]\", objeto recebido é do tipo \"" + 
            entrada.getClass().getSimpleName() + "\""
         );
      }

      double[][][] e = (double[][][]) entrada;
      if(e.length != this.profEntrada){
         throw new IllegalArgumentException(
            "A profundidade de entrada (" + e.length + 
            ") dos dados fornecidos é diferente da profundidade " + 
            " de entrada da camada (" + this.profEntrada + ")."
         );
      }
      if(e[0].length != this.largEntrada){
         throw new IllegalArgumentException(
            "A largura de entrada (" + e[0].length + 
            ") dos dados fornecidos é diferente da largura " + 
            " de entrada da camada (" + this.largEntrada + ")."
         );
      }
      if(e[0][0].length != this.altEntrada){
         throw new IllegalArgumentException(
            "A altura de entrada (" + e[0][0].length + 
            ") dos dados fornecidos é diferente da altura " + 
            " de entrada da camada (" + this.altEntrada + ")."
         );
      }

      for(int i = 0; i < this.profEntrada; i++){
         this.entrada[i].copiar(e[i]);
      }

      //feedforward
      for(int i = 0; i < this.numFiltros; i++){
         this.saida[i].preencher(0);

         for(int j = 0; j < this.profEntrada; j++){
            opmat.correlacaoCruzada(this.entrada[j], this.filtros[i][j], this.somatorio[i]);
            opmat.add(this.somatorio[i], this.saida[i], this.saida[i]);
         }

         if(this.usarBias){
            opmat.add(this.saida[i], this.bias[i], this.saida[i]);
         }
      }

      //deixar os valores calculados no somatório pra saída 
      //ficar com o resultado das ativações
      for(int i = 0; i < this.saida.length; i++){
         this.somatorio[i].copiar(this.saida[i]);
      }

      this.ativacao.calcular(this);
   }


   public void calcularGradientes(Object gradSeguinte){
      //TODO
   }

   /**
    * Retorna o array de saídas da camada.
    * @return array de matrizes contendo as saídas da camada.
    */
   public Mat[] obterSaida(){
      return this.saida;
   }

   /**
    * Retorna as saídas da camada no formato de um array trimensional.
    * @return saída da camada.
    */
   public double[][][] saidaParaDouble(){
      double[][][] saida = new double[this.numFiltros][][];
      for(int i = 0; i < saida.length; i++){
         saida[i] = this.saida[i].paraDouble();
      }

      return saida;
   }

   /**
    * Retorna um array contendo o formato da saída da camada que 
    * é disposto do seguinte formato:
    * <pre>
    *    formato = (altura, largura, profundidade)
    * </pre>
    * @return array contendo as dimensões de saída da camada.
    */
   public int[] obterFormatoSaida(){
      return new int[]{this.altSaida, this.largSaida, this.numFiltros};
   }

   /**
    * Retorda a quantidade de parâmetros totais da camada, em outras palavras, 
    * retorna o somatório da quantidade de filtros e bias presentes na camada.
    * @return a quantidade de parâmetros.
    */
   public int numParametros(){
      int parametros = 0;

      parametros += this.numFiltros * this.profEntrada * this.largFiltro * this.altFiltro;
      parametros += this.bias.length * this.altSaida * this.altSaida;

      return parametros;
   }

   /**
    * Calcula o formato de entrada da camada Densa, que é disposto da
    * seguinte forma:
    * <pre>
    *    formato = (entrada.altura, entrada.largura, entrada.profundidade)
    * </pre>
    * @return formato de entrada da camada.
    */
   @Override
   public int[] formatoEntrada(){
      return new int[]{
         this.entrada[0].lin, 
         this.entrada[0].col, 
         this.entrada.length
      };
   }
 
    /**
     * Calcula o formato de saída da camada Densa, que é disposto da
     * seguinte forma:
     * <pre>
     *    formato = (saida.altura, saida.largura, saida.profundidade)
     * </pre>
     * @return formato de saída da camada
     */
    @Override
   public int[] formatoSaida(){
      return new int[]{
         this.saida[0].lin, 
         this.saida[0].col, 
         this.saida.length
      };
   }
}