package rna.camadas;

import rna.ativacoes.Ativacao;
import rna.ativacoes.Linear;
import rna.core.Dicionario;
import rna.core.Mat;
import rna.core.OpMatriz;
import rna.core.Utils;
import rna.inicializadores.Inicializador;
import rna.inicializadores.Xavier;
import rna.inicializadores.Zeros;

/**
 * <h2>
 *    Camada Convolucional
 * </h2>
 * <p>
 *    A camada convolucional realiza operações de convolução sobre a entrada
 *    utilizando filtros (kernels) para extrair características locais, dada 
 *    pela expressão:.
 * </p>
 * <pre>
 *    somatorio = convolucao(entrada, filtros) + bias
 * </pre>
 * Após a propagação dos dados, a função de ativação da camada é aplicada ao 
 * resultado do somatório, que por fim é salvo na saída da camada.
 * <pre>
 *    saida = ativacao(somatorio)
 * </pre>
 * <p>
 *    Detalhe adicional:
 * </p>
 * Na realidade a operação que é feita dentro da camada convolucional é chamada de
 * correlação cruzada, é nela que aplicamos os kernels pela entrada recebida. A 
 * verdadeira operação de convolução tem a peculiaridade de rotacionar o filtro 180° 
 * antes de ser executada.
 */
public class Convolucional extends Camada implements Cloneable{

   /**
    * Operador matricial para a camada.
    */
   OpMatriz opmat = new OpMatriz();

   /**
    * Utilitário.
    */
   Utils utils = new Utils();

   /**
    * Altura da cada entrada da camada.
    */
   private int altEntrada;

   /**
    * Largura de cada entrada da camada.
    */
   private int largEntrada;

   /**
    * Profundidade da camada, também podendo dizer quantas
    * entradas a camada suporta.
    */
   private int profEntrada;

   /**
    * Altura de cada filtro presente na camada.
    */
   private int altFiltro;

   /**
    * Largura de cada filtro presente na camada.
    */
   private int largFiltro;

   /**
    * Números de filtros presentes na camada.
    */
   private int numFiltros;

   /**
    * Auxiliar na contagem de parâmetros para os filtros.
    * <p>
    *    O numero de kernels corresponde a quantidade de parametros total
    *    presente em cada filtro da camada.
    * </p>
    * Exemplo:
    * <p>
    *    Um filtro 3x3x1 possui 9 parâmetros.
    * </p>
    */
   private int numParamsKernel;

   /**
    * Altura da saída da camada.
    */
   private int altSaida;

   /**
    * Largura da saída da camada.
    */
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
    * Array de matrizes contendo os valores relativos a derivada da função de
    * ativação da camada.
    * <p>
    *    O formato da derivada é dado por:
    * </p>
    * <pre>
    *derivada = [numeroFiltros]
    *derivada[n] = [alturaSaida][larguraSaida]
    * </pre>
    */
   public Mat[] derivada;

   /**
    * Array de matrizes contendo os valores dos gradientes usados para 
    * a retropropagação para camadas anteriores.
    */
   public Mat[] gradEntrada;

   /**
    * Array de matrizes contendo os valores dos gradientes relativos a saída
    * da camada.
    * <p>
    *    O formato dos gradientes da saída é dado por:
    * </p>
    * <pre>
    *gradSaida = [numFiltros]
    *gradSaida[n] = [alturaSaida, larguraSaida]
    * </pre>
    */
   public Mat[] gradSaida;

   /**
    * Array de matrizes contendo os valores dos gradientes relativos a cada
    * filtro da camada.
    * <p>
    *    O formato dos gradientes para os filtros é dado por:
    * </p>
    * <pre>
    *gradFiltros = [numFiltros]
    *gradFiltros[n] = [alturaFiltro, larguraFiltro]
    * </pre>
    */
   public Mat[][] gradFiltros;

   /**
    * Array de matrizes contendo os valores dos gradientes relativos a cada
    * bias da camada.
    * <p>
    *    O formato dos gradientes para os bias é dado por:
    * </p>
    * <pre>
    *gradBias = [numFiltros]
    *gradBias[n] = [alturaSaida, larguraSaida]
    * </pre>
    */
   public Mat[] gradBias;

   /**
    * Função de ativação da camada.
    */
   Ativacao ativacao = new Linear();

   /**
    * Inicializador para os pesos da camada.
    */
   private Inicializador iniKernel = new Xavier();

   /**
    * Inicializador para os bias da camada.
    */
   private Inicializador iniBias = new Zeros();

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
    * @param filtros quantidade de filtros.
    * @param ativacao função de ativação.
    * @param iniKernel inicializador para os filtros.
    * @param iniBias inicializador para os bias.
    */
   public Convolucional(int[] formEntrada, int[] formFiltro, int filtros, String ativacao, Object iniKernel, Object iniBias){
      this(formFiltro, filtros, ativacao, iniKernel, iniBias);

      if(formEntrada == null){
         throw new IllegalArgumentException(
            "O formato de entrada não pode ser nulo."
         );
      }

      int[] e = (int[]) formEntrada;
      if(e.length != 3){
         throw new IllegalArgumentException(
            "O formato de entrada deve conter 3 elementos (altura, largura, profundidade), " +
            "recebido: " + e.length
         );
      }
      if(utils.apenasMaiorZero(e) == false){
         throw new IllegalArgumentException(
            "Os valores do formato de entrada devem ser maiores que zero."
         );
      }
      this.altEntrada  = e[0];
      this.largEntrada = e[1];
      this.profEntrada = e[2];

      construir(new int[]{
         this.altEntrada,
         this.largEntrada,
         this.profEntrada
      });
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
    * @param formEntrada formato de entrada da camada.
    * @param formFiltro formato dos filtros da camada.
    * @param filtros quantidade de filtros.
    * @param ativacao função de ativação.
    * @param iniKernel inicializador para os filtros.
    */
   public Convolucional(int[] formEntrada, int[] formFiltro, int filtros, String ativacao, Object iniKernel){
      this(formEntrada, formFiltro, filtros, ativacao, iniKernel, null);
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
    * @param formEntrada formato de entrada da camada.
    * @param formFiltro formato dos filtros da camada.
    * @param filtros quantidade de filtros.
    * @param ativacao função de ativação.
    */
   public Convolucional(int[] formEntrada, int[] formFiltro, int filtros, String ativacao){
      this(formEntrada, formFiltro, filtros, ativacao, null, null);
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
    * @param formFiltro formato dos filtros da camada.
    * @param filtros quantidade de filtros.
    * @param ativacao função de ativação.
    * @param iniKernel inicializador para os filtros.
    * @param iniBias inicializador para os bias.
    */
   public Convolucional(int[] formFiltro, int filtros, String ativacao, Object iniKernel, Object iniBias){
      if(formFiltro == null){
         throw new IllegalArgumentException(
            "O formato do filtro não pode ser nulo."
         );
      }

      //formado dos filtros
      int[] f = (int[]) formFiltro;
      if(f.length != 2){
         throw new IllegalArgumentException(
            "O formato dos filtros deve conter 2 elementos (altura, largura), " +
            "recebido: " + f.length
         );
      }
      if(utils.apenasMaiorZero(f) == false){
         throw new IllegalArgumentException(
            "Os valores de formato para os filtros devem ser maiores que zero."
         );      
      }
      this.altFiltro  = f[0];
      this.largFiltro = f[1];

      //número de filtros
      if(filtros <= 0){
         throw new IllegalArgumentException(
            "O número de filtro deve ser maior que zero, recebido: " + filtros
         );
      }
      this.numFiltros = filtros;

      if(ativacao != null){
         configurarAtivacao(ativacao);
      }

      Dicionario dic = new Dicionario();

      if(iniKernel != null){
         this.iniKernel = dic.obterInicializador(iniKernel);
      }

      if(iniBias != null){
         this.iniBias = dic.obterInicializador(iniBias);
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
    * @param formFiltro formato dos filtros da camada.
    * @param filtros quantidade de filtros.
    * @param ativacao função de ativação.
    * @param iniKernel inicializador para os filtros.
    */
   public Convolucional(int[] formFiltro, int filtros, String ativacao, Object iniKernel){
      this(formFiltro, filtros, ativacao, iniKernel, null);
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
    * @param formFiltro formato dos filtros da camada.
    * @param filtros quantidade de filtros.
    * @param ativacao função de ativação.
    */
   public Convolucional(int[] formFiltro, int filtros, String ativacao){
      this(formFiltro, filtros, ativacao, null, null);
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
    * @param formFiltro formato dos filtros da camada.
    * @param filtros quantidade de filtros.
    */
   public Convolucional(int[] formFiltro, int filtros){
      this(formFiltro, filtros, null, null, null);
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
    * @param filtros quantidade de filtros.
    */
   public Convolucional(int[] formEntrada, int[] formFiltro, int filtros){
      this(formEntrada, formFiltro, filtros, null, null, null);
   }
   
   /**
    * Inicializa os parâmetros necessários para a camada Convolucional.
    * <p>
    *    O formato de entrada deve ser um array contendo o tamanho de 
    *    cada dimensão e entrada da camada, e deve estar no formato:
    * </p>
    * <pre>
    *    entrada = (altura, largura, profundidade)
    * </pre>
    * @param entrada formato de entrada para a camada.
    */
   @Override
   public void construir(Object entrada){
      if(entrada == null){
         throw new IllegalArgumentException(
            "Formato de entrada fornecida para camada Convolucional é nulo."
         );
      }
      if(entrada instanceof int[] == false){
         throw new IllegalArgumentException(
            "Objeto esperado para entrada da camada Convolucional é do tipo int[], " +
            "objeto recebido é do tipo " + entrada.getClass().getTypeName()
         );
      }

      int[] formEntrada = (int[]) entrada;
      if(formEntrada.length != 3){
         throw new IllegalArgumentException(
            "O formato de entrada para a camada Convolucional deve conter três " + 
            "elementos (altura, largura, profundidade), objeto recebido possui " + formEntrada.length
         );
      }

      if(utils.apenasMaiorZero(formEntrada) == false){
         throw new IllegalArgumentException(
            "Os valores de dimensões de entrada para a camada Convolucional não " +
            "podem conter valores menores que 1."
         );
      }

      //inicialização de parâmetros

      //inicialização da entrada
      this.altEntrada  = formEntrada[0];
      this.largEntrada = formEntrada[1];
      this.profEntrada = formEntrada[2];

      this.altSaida = this.altEntrada - this.altFiltro + 1;
      this.largSaida = this.largEntrada - this.largFiltro + 1;

      //inicialização dos parâmetros necessários
      this.entrada = new Mat[this.profEntrada];
      this.gradEntrada = new Mat[this.profEntrada];
      for(int i = 0; i < this.profEntrada; i++){
         this.entrada[i] = new Mat(this.altEntrada, this.largEntrada);
         this.gradEntrada[i] = new Mat(this.altEntrada, this.largEntrada);
      }

      this.filtros = new Mat[this.numFiltros][this.profEntrada];
      this.gradFiltros = new Mat[this.numFiltros][this.profEntrada];
      this.somatorio = new Mat[this.numFiltros];
      this.saida = new Mat[this.numFiltros];
      this.derivada = new Mat[this.numFiltros];
      this.gradSaida = new Mat[this.numFiltros];

      if(this.usarBias){
         this.bias = new Mat[this.numFiltros];
         this.gradBias = new Mat[this.numFiltros];
      }

      for(int i = 0; i < this.numFiltros; i++){
         for(int j = 0; j < this.profEntrada; j++){
            this.filtros[i][j] = new Mat(this.altFiltro, this.largFiltro);
            this.gradFiltros[i][j] = new Mat(this.altFiltro, this.largFiltro);
         }

         this.somatorio[i] = new Mat(this.altSaida, this.largSaida);
         this.saida[i] = new Mat(this.altSaida, this.largSaida);
         this.derivada[i] = new Mat(this.altSaida, this.largSaida);
         this.gradSaida[i] = new Mat(this.altSaida, this.largSaida);

         if(this.usarBias){
            this.bias[i] = new Mat(this.altSaida, this.largSaida);
            this.gradBias[i] = new Mat(this.altSaida, this.largSaida);
         }
      }

      //auxiliar
      this.numParamsKernel = 0;
      for(Mat[] filtro : this.filtros){
         for(Mat camada : filtro){
            this.numParamsKernel += camada.tamanho();
         }
      }
      
      this.treinavel = true;
      this.construida = true;//camada pode ser usada.
   }

   @Override
   public void inicializar(double x){   
      for(int i = 0; i < numFiltros; i++){
         for(int j = 0; j < profEntrada; j++){
            this.iniKernel.inicializar(this.filtros[i][j], x);
         }
      }

      if(this.usarBias){
         for(Mat b : this.bias){
            this.iniBias.inicializar(b, x);
         }
      }
   }

   @Override
   public void configurarAtivacao(Object ativacao){
      Dicionario dic = new Dicionario();
      this.ativacao = dic.obterAtivacao(ativacao);
   }

   @Override
   public void configurarBias(boolean usarBias){
      this.usarBias = usarBias;
   }

   /**
    * Propagação direta dos dados de entrada através da camada convolucional.
    * Realiza a correlação cruzada entre os filtros da camada e os dados de entrada,
    * somando os resultados ponderados. Caso a camada tenha configurado o uso do bias, ele
    * é adicionado após a operação. Por fim é aplicada a função de ativação aos resultados
    * que serão salvos da saída da camada.
    * <p>
    *    A expressão que define a saída para cada filtro é dada por:
    * </p>
    * <pre>
    *somatorio[i] = correlacaoCruzada(filtros[i][j], entrada[j]) + bias[i]
    *saida[i] = ativacao(somatorio[i])
    * </pre>
    * onde {@code i} é o índice do filtro e {@code j} é o índice dos dados de entrada.
    * <p>
    *    Após a propagação dos dados, a função de ativação da camada é aplicada
    *    ao resultado do somatório e o resultado é salvo da saída da camada.
    * </p>
    * @param entrada dados de entrada que serão processados, deve ser um array 
    * tridimensional do tipo {@code double[][][]}.
    * @throws IllegalArgumentException caso a entrada fornecida não seja suportada 
    * pela camada.
    * @throws IllegalArgumentException caso haja alguma incompatibilidade entre a entrada
    * fornecida e a capacidade de entrada da camada.
    */
   @Override
   public void calcularSaida(Object entrada){
      super.verificarConstrucao();

      if(entrada instanceof double[]){
         utils.copiar((double[]) entrada, this.entrada);
      
      }else if(entrada instanceof double[][][]){
         double[][][] e = (double[][][]) entrada;
         if(e.length != this.profEntrada || e[0].length != this.largEntrada || e[0][0].length != this.altEntrada){
            throw new IllegalArgumentException(
               "As dimensões da entrada " + 
               "(" + e[0].length + ", " + e[0][0].length + ", " + e.length + ") " +
               "não correspondem as dimensões de entrada da camada Convolucional " + 
               "(" + this.altEntrada +", " + this.largEntrada + ", " + this.profEntrada + ")"
            );
         }
         utils.copiar(e, this.entrada);
      
      }else if(entrada instanceof Mat[]){
         utils.copiar((Mat[]) entrada, this.entrada);

      }else{         
         throw new IllegalArgumentException(
            "Os dados de entrada para a camada Convolucional devem ser " +
            "do tipo \"double[][][]\", \"double[]\" ou \"Mat[]\", objeto recebido é do tipo \"" + 
            entrada.getClass().getTypeName() + "\""
         );
      }

      //feedforward
      if(this.usarBias){
         utils.copiar(this.bias, this.somatorio);
      }

      this.opmat.correlacaoCruzada(this.entrada, this.filtros, this.somatorio, true);

      this.ativacao.calcular(this);
   }

   /**
    * Calcula os gradientes da camada para os pesos e bias baseado nos
    * gradientes fornecidos.
    * <p>
    *    Após calculdos, os gradientes em relação a entrada da camada são
    *    calculados e salvos em {@code gradEntrada} para serem retropropagados 
    *    para as camadas anteriores da Rede Neural em que a camada estiver.
    * </p>
    * Resultados calculados ficam salvos nas prorpiedades {@code camada.gradFiltros} e
    * {@code camada.gradBias}.
    * @param gradSeguinte gradiente da camada seguinte.
    */
   @Override
   public void calcularGradiente(Object gradSeguinte){
      super.verificarConstrucao();

      if(gradSeguinte instanceof Mat[] == false){
         throw new IllegalArgumentException(
            "Os gradientes para a camada Convolucional devem ser " +
            "do tipo \"" + this.gradSaida.getClass().getTypeName() + 
            "\", objeto recebido é do tipo \"" + gradSeguinte.getClass().getTypeName() + "\""
         );
      }

      Mat[] grads = (Mat[]) gradSeguinte;
      if(grads.length != this.gradSaida.length){
         throw new IllegalArgumentException(
            "Incompatibilidade entre o número de gradientes fornecidos (" + grads.length + 
            ") e o suportado pela camada (" + this.gradSaida.length + ")"
         );
      }
      utils.copiar(grads, this.gradSaida);

      this.ativacao.derivada(this);
      for(Mat gradE : this.gradEntrada){
         gradE.preencher(0);
      }

      //backward
      for(int i = 0; i < this.numFiltros; i++){
         for(int j = 0; j < this.profEntrada; j++){
            opmat.correlacaoCruzada(this.entrada[j], this.derivada[i], this.gradFiltros[i][j], false);
            opmat.convolucaoFull(this.derivada[i], this.filtros[i][j], this.gradEntrada[j], true);
         }
      }

      if(this.usarBias){
         for(int i = 0; i < this.gradBias.length; i++){
            this.gradBias[i].copiar(this.derivada[i]);
         }
      }
   }

   /**
    * Retorna a quantidade de filtros presentes na camada.
    * @return quantiadde de filtros presentes na camada.
    */
   public int numFiltros(){
      return this.numFiltros;
   }

   @Override
   public Ativacao obterAtivacao(){
      return this.ativacao;
   }

   @Override
   public Mat[] saida(){
      return this.saida;
   }

   @Override
   public boolean temBias(){
      return this.usarBias;
   }

   @Override
   public int numParametros(){
      super.verificarConstrucao();
      int parametros = 0;

      parametros += this.numFiltros * this.profEntrada * this.altFiltro * this.largFiltro;
      if(this.usarBias){
         parametros += this.bias.length * this.altSaida * this.altSaida;
      }

      return parametros;
   }

   @Override
   public double[] saidaParaArray(){
      super.verificarConstrucao();

      int id = 0;
      double[] saida = new double[this.tamanhoSaida()];

      for(int i = 0; i < this.saida.length; i++){
         double[] s = this.saida[i].paraArray();
         for(double valor : s){
            saida[id++] = valor;
         }
      }

      return saida;
   }

   @Override 
   public int tamanhoSaida(){
      return this.numFiltros * this.altSaida * this.largSaida;
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

   @Override
   public Convolucional clonar(){
      super.verificarConstrucao();

      try{
         Convolucional clone = (Convolucional) super.clone();

         clone.ativacao = this.ativacao;

         clone.usarBias = this.usarBias;
         if(this.usarBias){
            clone.bias = this.bias.clone();
            clone.gradBias = this.gradBias.clone();
         }

         clone.entrada = new Mat[this.profEntrada];
         for(int i = 0; i < this.profEntrada; i++){
            clone.entrada[i] = this.entrada[i].clone();
         }
         
         clone.filtros     = new Mat[this.numFiltros][];
         clone.gradFiltros = new Mat[this.numFiltros][];
         for(int i = 0; i < this.numFiltros; i++){
            clone.filtros[i]     = new Mat[this.profEntrada];
            clone.gradFiltros[i] = new Mat[this.profEntrada];
            for(int j = 0; j < this.profEntrada; j++){
               clone.filtros[i][j] = this.filtros[i][j].clone();
               clone.gradFiltros[i][j] = this.gradFiltros[i][j].clone();
            }
         }

         clone.somatorio   = new Mat[this.numFiltros];
         clone.saida       = new Mat[this.numFiltros];
         clone.gradSaida   = new Mat[this.numFiltros];
         clone.derivada    = new Mat[this.numFiltros];
         for(int i = 0; i < this.numFiltros; i++){
            clone.somatorio[i] = this.somatorio[i].clone();
            clone.saida[i] = this.saida[i].clone();
            clone.gradSaida[i] = this.gradSaida[i].clone();
            clone.derivada[i] = this.derivada[i].clone();
         }

         return clone;
      }catch(Exception e){
         throw new RuntimeException(e);
      }
   }

   /**
    * Calcula o formato de entrada da camada Convolucional, que é disposto da
    * seguinte forma:
    * <pre>
    *    formato = (entrada.altura, entrada.largura, entrada.profundidade)
    * </pre>
    * @return formato de entrada da camada.
    */
   @Override
   public int[] formatoEntrada(){
      return new int[]{
         this.entrada[0].lin(), 
         this.entrada[0].col(), 
         this.entrada.length
      };
   }
 
   /**
    * Calcula o formato de saída da camada Convolucional, que é disposto da
    * seguinte forma:
    * <pre>
    *    formato = (altura, largura, profundidade)
    * </pre>
    * @return formato de saída da camada.
    */
   @Override
   public int[] formatoSaida(){
      super.verificarConstrucao();

      return new int[]{
         this.altSaida,
         this.largSaida,
         this.numFiltros
      };
   }

   /**
    * Retorna o formato dos filtros contidos na camada.
    * @return formato de cada filtro (altura, largura).
    */
   public int[] formatoFiltro(){
      return new int[]{
         this.altFiltro,
         this.largFiltro
      };
   }

   @Override
   public double[] obterKernel(){
      int cont = 0, i, j;
      double[] kernel = new double[this.numParamsKernel];
      for(i = 0; i < numFiltros; i++){
         for(j = 0; j < this.profEntrada; j++){
            double[] arr = this.filtros[i][j].paraArray();
            System.arraycopy(arr, 0, kernel, cont, arr.length);
            cont += arr.length;
         }
      }

      return kernel;
   }

   @Override
   public double[] obterGradKernel(){
      int cont = 0, i, j;
      double[] grad = new double[this.numParamsKernel];
      
      for(i = 0; i < numFiltros; i++){
         for(j = 0; j < this.profEntrada; j++){
            double[] arr = this.gradFiltros[i][j].paraArray();
            System.arraycopy(arr, 0, grad, cont, arr.length);
            cont += arr.length;
         }
      }

      return grad;
   }

   @Override
   public double[] obterBias(){
      double[] bias = new double[this.numFiltros * this.altSaida * this.largSaida];
      int cont = 0;

      for(int i = 0; i < this.bias.length; i++){
         double[] arr = this.bias[i].paraArray();
         System.arraycopy(arr, 0, bias, cont, arr.length);
         cont += arr.length;
      }

      return bias;
   }

   @Override
   public double[] obterGradBias(){
      double[] grad = new double[this.numFiltros * this.altEntrada * this.largEntrada];
      int cont = 0;

      for(int i = 0; i < this.gradBias.length; i++){
         double[] arr = this.gradBias[i].paraArray();
         System.arraycopy(arr, 0, grad, cont, arr.length);
         cont += arr.length;
      }

      return grad;
   }

   @Override
   public Object obterGradEntrada(){
      return this.gradEntrada; 
   }

   @Override
   public void editarKernel(double[] kernel){
      if(kernel.length != this.numParamsKernel){
         throw new IllegalArgumentException(
            "A dimensão do kernel fornecido (" + kernel.length + ") não é igual a quantidade de " +
            " parâmetros para os kernels da camada ("+ this.numParamsKernel + ")."
         );
      }
         
      utils.copiar(kernel, this.filtros);
   }

   @Override
   public void editarBias(double[] bias){
      if(bias.length != (this.altSaida * this.largSaida * this.numFiltros)){
         throw new IllegalArgumentException(
            "A dimensão do bias fornecido não é igual a quantidade de " +
            " parâmetros para os bias da camada."
         );
      }
      
      utils.copiar(bias, this.bias);
   }

}
