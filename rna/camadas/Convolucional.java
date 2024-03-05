package rna.camadas;

import rna.ativacoes.Ativacao;
import rna.ativacoes.Linear;
import rna.core.Dicionario;
import rna.core.OpTensor4D;
import rna.core.Tensor4D;
import rna.core.Utils;
import rna.inicializadores.Inicializador;
import rna.inicializadores.GlorotUniforme;
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
    * Operador de tensores para a camada.
    */
   OpTensor4D optensor = new OpTensor4D();

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
    * Altura da saída da camada.
    */
   private int altSaida;

   /**
    * Largura da saída da camada.
    */
   private int largSaida;

   /**
    * Tensor contendo os valores de entrada para a camada,
    * que serão usados para o processo de feedforward.
    * <p>
    *    O formato da entrada é dado por:
    * </p>
    * <pre>
    *    entrada = (1, profundidade, altura, largura)
    * </pre>
    */
   public Tensor4D entrada;

   /**
    * Tensor contendo os filtros (ou kernels)
    * da camada.
    * <p>
    *    O formato dos filtros é dado por:
    * </p>
    * <pre>
    *    entrada = (numFiltros, profundidadeEntrada, alturaFiltro, larguraFiltro)
    * </pre>
    */
   public Tensor4D filtros;

   /**
    * Tensor contendo os bias (vieses) para cada valor de 
    * saída da camada.
    * <p>
    *    O formato do bias é dado por:
    * </p>
    * <pre>
    *    bias = (1, 1, 1, numFiltros)
    * </pre>
    */
   public Tensor4D bias;

   /**
    * Auxiliar na verificação de uso do bias.
    */
   private boolean usarBias = true;

   /**
    * Tensor contendo valores resultantes do cálculo de correlação cruzada
    * entre a entrada e os filtros, com o bias adicionado (se houver).
    * <p>
    *    O formato somatório é dado por:
    * </p>
    * <pre>
    *    somatorio = (1, numeroFiltros, alturaSaida, larguraSaida)
    * </pre>
    */
   public Tensor4D somatorio;
   
   /**
    * Tensor contendo os valores de saídas da camada.
    * <p>
    *    O formato da saída é dado por:
    * </p>
    * <pre>
    *    saida = (1, numeroFiltros, alturaSaida, larguraSaida)
    * </pre>
    */
   public Tensor4D saida;

   /**
    * Tensor contendo os valores relativos a derivada da função de
    * ativação da camada.
    * <p>
    *    O formato da derivada é dado por:
    * </p>
    * <pre>
    *    derivada = (1, numFiltros, alturaSaida, larguraSaida)
    * </pre>
    */
   public Tensor4D derivada;

   /**
    * Tensor contendo os valores dos gradientes usados para 
    * a retropropagação para camadas anteriores.
    * <p>
    *    O formato do gradiente de entrada é dado por:
    * </p>
    * <pre>
    *    gradEntrada = (1, profEntrada, alturaEntrada, larguraEntrada)
    * </pre>
    */
   public Tensor4D gradEntrada;

   /**
    * Tensor contendo os valores dos gradientes relativos a saída
    * da camada.
    * <p>
    *    O formato dos gradientes da saída é dado por:
    * </p>
    * <pre>
    *    gradSaida = (1, numFiltros, alturaSaida, larguraSaida)
    * </pre>
    */
   public Tensor4D gradSaida;

   /**
    * Tensor contendo os valores dos gradientes relativos a cada
    * filtro da camada.
    * <p>
    *    O formato dos gradientes para os filtros é dado por:
    * </p>
    * <pre>
    * gradFiltros = (numFiltros, profundidadeEntrada, alturaFiltro, larguraFiltro)
    * </pre>
    */
   public Tensor4D gradFiltros;

   /**
    * Tensor contendo os valores dos gradientes relativos a cada
    * bias da camada.
    * <p>
    *    O formato dos gradientes para os bias é dado por:
    * </p>
    * <pre>
    *    gradBias = (1, 1, 1, numFiltros)
    * </pre>
    */
   public Tensor4D gradBias;

   /**
    * Função de ativação da camada.
    */
   Ativacao ativacao = new Linear();

   /**
    * Inicializador para os filtros da camada.
    */
   private Inicializador iniKernel = new GlorotUniforme();

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
    *    formEntrada = (profundidade, altura, largura)
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
            "\nO formato de entrada não pode ser nulo."
         );
      }

      if(formEntrada.length != 3){
         throw new IllegalArgumentException(
            "\nO formato de entrada deve conter 3 elementos (altura, largura, profundidade), " +
            "recebido: " + formEntrada.length
         );
      }

      if(utils.apenasMaiorZero(formEntrada) == false){
         throw new IllegalArgumentException(
            "\nOs valores do formato de entrada devem ser maiores que zero."
         );
      }

      this.profEntrada = formEntrada[0];
      this.altEntrada  = formEntrada[1];
      this.largEntrada = formEntrada[2];

      construir(formEntrada);
   }

   /**
    * Instancia uma camada convolucional de acordo com os formatos fornecidos.
    * <p>
    *    A disposição do formato de entrada deve ser da seguinte forma:
    * </p>
    * <pre>
    *    formEntrada = (profundidade, altura, largura)
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
    *    formEntrada = (profundidade, altura, largura)
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
    * Instancia uma camada convolucional de acordo com os formatos fornecidos.
    * <p>
    *    A disposição do formato de entrada deve ser da seguinte forma:
    * </p>
    * <pre>
    *    formEntrada = (profundidade, altura, largura)
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
            "\nO formato do filtro não pode ser nulo."
         );
      }

      //formado dos filtros
      int[] f = (int[]) formFiltro;
      if(f.length != 2){
         throw new IllegalArgumentException(
            "\nO formato dos filtros deve conter 2 elementos (altura, largura), " +
            "recebido: " + f.length
         );
      }
      if(utils.apenasMaiorZero(f) == false){
         throw new IllegalArgumentException(
            "\nOs valores de formato para os filtros devem ser maiores que zero."
         );      
      }
      this.altFiltro  = f[0];
      this.largFiltro = f[1];

      //número de filtros
      if(filtros <= 0){
         throw new IllegalArgumentException(
            "\nO número de filtro deve ser maior que zero, recebido: " + filtros
         );
      }
      this.numFiltros = filtros;
      
      Dicionario dic = new Dicionario();
      if(ativacao != null) this.ativacao = dic.obterAtivacao(ativacao);
      if(iniKernel != null) this.iniKernel = dic.obterInicializador(iniKernel);
      if(iniBias != null) this.iniBias = dic.obterInicializador(iniBias);
   }

   /**
    * Instancia uma camada convolucional de acordo com os formatos fornecidos.
    * <p>
    *    A disposição do formato de entrada deve ser da seguinte forma:
    * </p>
    * <pre>
    *    formEntrada = (profundidade, altura, largura)
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
    *    formEntrada = (profundidade, altura, largura)
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
    *    formEntrada = (profundidade, altura, largura)
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
    * Inicializa os parâmetros necessários para a camada Convolucional.
    * <p>
    *    O formato de entrada deve ser um array contendo o tamanho de 
    *    cada dimensão de entrada da camada, e deve estar no formato:
    * </p>
    * <pre>
    *    entrada = (profundidade, altura, largura)
    * </pre>
    * @param entrada formato de entrada para a camada.
    */
   @Override
   public void construir(Object entrada){
      if(entrada == null){
         throw new IllegalArgumentException(
            "\nFormato de entrada fornecida para camada Convolucional é nulo."
         );
      }

      if(entrada instanceof int[] == false){
         throw new IllegalArgumentException(
            "\nObjeto esperado para entrada da camada Convolucional é do tipo int[], " +
            "objeto recebido é do tipo " + entrada.getClass().getTypeName()
         );
      }

      int[] fEntrada = (int[]) entrada;
      if(fEntrada.length == 4){
         this.profEntrada = fEntrada[1];
         this.altEntrada  = fEntrada[2];
         this.largEntrada = fEntrada[3];
      
      }else if(fEntrada.length == 3){
         this.profEntrada = fEntrada[0];
         this.altEntrada  = fEntrada[1];
         this.largEntrada = fEntrada[2];
         
      }else{
         throw new IllegalArgumentException(
            "\nO formato de entrada para a camada Convolucional deve conter três " + 
            "elementos (profundidade, altura, largura), ou quatro elementos (primeiro desconsiderado)" + 
            "objeto recebido possui " + fEntrada.length
         );
      }

      if(utils.apenasMaiorZero(fEntrada) == false){
         throw new IllegalArgumentException(
            "\nOs valores de dimensões de entrada para a camada Convolucional não " +
            "podem conter valores menores que 1."
         );
      }

      this.altSaida = this.altEntrada - this.altFiltro + 1;
      this.largSaida = this.largEntrada - this.largFiltro + 1;

      //inicialização dos parâmetros necessários
      this.entrada      = new Tensor4D(1, profEntrada, altEntrada, largEntrada);
      this.gradEntrada  = new Tensor4D(this.entrada);
      this.filtros      = new Tensor4D(numFiltros, profEntrada, altFiltro, largFiltro);
      this.saida        = new Tensor4D(1, numFiltros, altSaida, largSaida);
      this.gradFiltros  = new Tensor4D(this.filtros);
      this.somatorio    = new Tensor4D(this.saida);
      this.derivada     = new Tensor4D(this.saida);
      this.gradSaida    = new Tensor4D(this.saida);

      if(usarBias){
         this.bias      = new Tensor4D(1, 1, 1, numFiltros);
         this.gradBias  = new Tensor4D(this.bias);
      }

      configurarNomes();
      
      this.treinavel = true;
      this.construida = true;//camada pode ser usada.
   }

   @Override
   public void inicializar(){
      verificarConstrucao();
      
      for(int i = 0; i < numFiltros; i++){
         for(int j = 0; j < profEntrada; j++){
            iniKernel.inicializar(filtros, i, j);
         }
      }

      if(usarBias){
         iniBias.inicializar(bias, 0, 0);
      }
   }

   @Override
   public void configurarAtivacao(Object ativacao){
      this.ativacao = new Dicionario().obterAtivacao(ativacao);
   }

   @Override
   public void configurarBias(boolean usarBias){
      this.usarBias = usarBias;
   }

   @Override
   protected void configurarNomes(){
      entrada.nome("entrada");
      gradEntrada.nome("gradiente entrada");
      filtros.nome("kernel");
      saida.nome("saída");
      gradFiltros.nome("gradiente kernel");
      somatorio.nome("somatório");
      derivada.nome("derivada");
      gradSaida.nome("gradiente saída");

      if(usarBias){
         bias.nome("bias");
         gradBias.nome("gradiente bias");
      }
   }

   /**
    * <h2>
    *    Propagação direta através da camada Convolucional
    * </h2>
    * <p>
    *    Realiza a correlação cruzada entre os dados de entrada e os filtros da 
    *    camada, somando os resultados ponderados. Caso a camada tenha configurado 
    *    o uso do bias, ele é adicionado após a operação. Por fim é aplicada a função 
    *    de ativação aos resultados que serão salvos da saída da camada.
    * </p>
    * <h3>
    *    A expressão que define a saída da camada é dada por:
    * </h3>
    * <pre>
    *somatorio = correlacaoCruzada(filtros, entrada)
    *somatorio.add(bias)
    *saida = ativacao(somatorio)
    * </pre>
    * <h3>
    *    Nota
    * </h3>
    * <p>
    *    Caso a entrada seja um {@code Tensor4D}, é considerada apenas a primeira dimensão do
    *    tensor.
    * </p>
    * @param entrada dados de entrada que serão processados, tipos aceitos são,
    * {@code double[][][]} ou {@code Tensor4D}.
    * @throws IllegalArgumentException caso a entrada fornecida não seja suportada 
    * pela camada.
    * @throws IllegalArgumentException caso haja alguma incompatibilidade entre a entrada
    * fornecida e a capacidade de entrada da camada.
    */
   @Override
   public void calcularSaida(Object entrada){
      verificarConstrucao();

      if(entrada instanceof double[][][]){
         double[][][] e = (double[][][]) entrada;
         if(e.length != this.profEntrada || e[0].length != this.largEntrada || e[0][0].length != this.altEntrada){
            throw new IllegalArgumentException(
               "\nAs dimensões da entrada recebida " + 
               "(" + e.length + ", " + e[0].length + ", " + e[0][0].length + ") " +
               "são incompatíveis com as dimensões da entrada da camada " + 
               "(" + profEntrada + ", " + altEntrada + ", " + largEntrada + ")"
            );
         }

         this.entrada.copiar(e, 0);
      
      }else if(entrada instanceof Tensor4D){
         Tensor4D e = (Tensor4D) entrada;
         if(this.entrada.comparar3D(e) == false){
            throw new IllegalArgumentException(
               "\nAs dimensões da entrada recebida " + e.dimensoesStr() + 
               " são incompatíveis com as dimensões da entrada da camada " + this.entrada.dimensoesStr()
            );
         }

         this.entrada.copiar(e, 0);

      }else{
         throw new IllegalArgumentException(
            "\nOs dados de entrada para a camada Convolucional devem ser " +
            "do tipo " + this.entrada.getClass().getSimpleName() + 
            " ou double[][][] objeto recebido é do tipo \"" + 
            entrada.getClass().getTypeName() + "\"."
         );
      }

      //feedforward
      somatorio.preencher(0);

      optensor.convForward(this.entrada, filtros, somatorio);
      
      if(usarBias){
         for(int i = 0; i < numFiltros; i++){
            double b = bias.elemento(0, 0, 0, i);
            somatorio.add2D(0, i, b);
         }
      }

      ativacao.calcular(somatorio, saida);
   }

   /**
    * <h2>
    *    Propagação reversa através da camada Convolucional
    * </h2>
    * <p>
    *    Calcula os gradientes da camada para os filtros e bias baseado nos
    *    gradientes fornecidos.
    * </p>
    * <p>
    *    Após calculdos, os gradientes em relação a entrada da camada são
    *    calculados e salvos em {@code gradEntrada} para serem retropropagados 
    *    para as camadas anteriores do modelo em que a camada estiver.
    * </p>
    * Resultados calculados ficam salvos nas prorpiedades {@code camada.gradFiltros} e
    * {@code camada.gradBias}.
    * <h3>
    *    Nota
    * </h3>
    * <p>
    *    Caso o gradiente seja um {@code Tensor4D}, é considerada apenas a primeira dimensão 
    *    do tensor.
    * </p>
    * @param gradSeguinte gradiente da camada seguinte.
    */
   @Override
   public void calcularGradiente(Object gradSeguinte){
      verificarConstrucao();

      if(gradSeguinte instanceof Tensor4D){
         Tensor4D g = (Tensor4D) gradSeguinte;
         if(gradSaida.comparar3D(g) == false){
            throw new IllegalArgumentException(
               "\nAs três dimensões finais do tensor recebido " + g.dimensoesStr() +
               "são imcompatíveis as três primeira dimensões do tensor de gradiente"
            );
         }

         gradSaida.copiar(g, 0);

      }else{
         throw new IllegalArgumentException(
            "Os gradientes para a camada Convolucional devem ser " +
            "do tipo \"" + gradSaida.getClass().getTypeName() + 
            "\", objeto recebido é do tipo \"" + gradSeguinte.getClass().getTypeName() + "\""
         );
      }

      ativacao.derivada(this);
      gradEntrada.preencher(0);

      //backward
      Tensor4D tempGrad = new Tensor4D(gradFiltros.dimensoes());
      optensor.convBackward(this.entrada, this.filtros, this.derivada, tempGrad, this.gradEntrada);
      gradFiltros.add(tempGrad);

      if(usarBias){
         for(int i = 0; i < numFiltros; i++){
            gradBias.add(0, 0, 0, i, derivada.somarElementos2D(0, i));
         }
      }
   }

   @Override
   public void zerarGradientes(){
      verificarConstrucao();

      gradFiltros.zerar();
      gradBias.zerar();
   }

   /**
    * Retorna a quantidade de filtros presentes na camada.
    * @return quantiadde de filtros presentes na camada.
    */
   public int numFiltros(){
      return this.numFiltros;
   }

   @Override
   public Ativacao ativacao(){
      return this.ativacao;
   }

   @Override
   public Tensor4D saida(){
      return this.saida;
   }

   @Override
   public boolean temBias(){
      return usarBias;
   }

   @Override
   public int numParametros(){
      verificarConstrucao();

      int parametros = filtros.tamanho();
      
      if(usarBias){
         parametros += bias.tamanho();
      }

      return parametros;
   }

   @Override
   public double[] saidaParaArray(){
      verificarConstrucao();

      return saida.paraArray();
   }

   @Override 
   public int tamanhoSaida(){
      return saida.tamanho();
   }

   @Override
   public Convolucional clonar(){
      verificarConstrucao();

      try{
         Convolucional clone = (Convolucional) super.clone();

         clone.ativacao = this.ativacao;
         clone.usarBias = this.usarBias;

         clone.entrada     = this.entrada.clone();
         clone.filtros     = this.filtros.clone();
         clone.gradFiltros = this.gradFiltros.clone();

         if(this.usarBias){
            clone.bias     = this.bias.clone();
            clone.gradBias = this.gradBias.clone();
         }

         clone.somatorio   = this.somatorio.clone();
         clone.saida       = this.saida.clone();
         clone.gradSaida   = this.gradSaida.clone();
         clone.derivada    = this.derivada.clone();

         return clone;
      }catch(Exception e){
         throw new RuntimeException(e);
      }
   }

   /**
    * Calcula o formato de entrada da camada Convolucional, que é disposto da
    * seguinte forma:
    * <pre>
    *    formato = (profundidade, altura, largura)
    * </pre>
    * @return formato de entrada da camada.
    */
   @Override
   public int[] formatoEntrada(){
      verificarConstrucao();

      return new int[]{
         profEntrada, 
         altEntrada, 
         largEntrada
      };
   }
 
   /**
    * Calcula o formato de saída da camada Convolucional, que é disposto da
    * seguinte forma:
    * <pre>
    *    formato = (profundidade, altura, largura)
    * </pre>
    * @return formato de saída da camada.
    */
   @Override
   public int[] formatoSaida(){
      verificarConstrucao();

      return new int[]{
         numFiltros,
         altSaida,
         largSaida
      };
   }

   /**
    * Retorna o formato dos filtros contidos na camada.
    * @return formato de cada filtro (altura, largura).
    */
   public int[] formatoFiltro(){
      verificarConstrucao();

      return new int[]{
         altFiltro,
         largFiltro
      };
   }

   @Override
   public Tensor4D kernel(){
      return filtros;
   }

   @Override
   public double[] kernelParaArray(){
      return kernel().paraArray();
   }

   @Override
   public double[] gradKernelParaArray(){
      return gradFiltros.paraArray();
   }

   @Override
   public Tensor4D bias(){
      if(usarBias){
         return bias;
      }

      throw new IllegalStateException(
         "\nA camada " + nome() + " (" + id + ") não possui bias configurado."
      );
   }

   @Override
   public double[] biasParaArray(){
      return bias.paraArray();
   }

   @Override
   public double[] gradBias(){
      return gradBias.paraArray();
   }

   @Override
   public Object gradEntrada(){
      return gradEntrada; 
   }

   @Override
   public void editarKernel(double[] kernel){
      if(kernel.length != this.filtros.tamanho()){
         throw new IllegalArgumentException(
            "A dimensão do kernel fornecido (" + kernel.length + ") não é igual a quantidade de " +
            " parâmetros para os kernels da camada ("+ this.filtros.tamanho() + ")."
         );
      }
         
      this.filtros.copiarElementos(kernel);
   }

   @Override
   public void editarBias(double[] bias){
      if(bias.length != this.bias.tamanho()){
         throw new IllegalArgumentException(
            "A dimensão do bias fornecido não é igual a quantidade de " +
            " parâmetros para os bias da camada."
         );
      }
      
      this.bias.copiarElementos(bias);
   }

}
