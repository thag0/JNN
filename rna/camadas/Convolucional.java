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
   private OpTensor4D optensor = new OpTensor4D();

   /**
    * Utilitário.
    */
   private Utils utils = new Utils();

   /**
    * Formato de entrada da camada convolucional, dado por:
    * <pre>
    *    form = (profundidade, altura, largura)
    * </pre>
    */
   private final int[] shapeEntrada = {1, 1, 1};

   /**
    * Formato de cada filtro da camada convolucional, dado por:
    * <pre>
    *    form = (altura, largura)
    * </pre>
    */
   private final int[] shapeFiltro = {1, 1};

   /**
    * Formato de saída da camada convolucional, dado por:
    * <pre>
    *    form = (numFiltros, altura, largura)
    * </pre>
    */
   private final int[] shapeSaida = {1, 1, 1};

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
   public Tensor4D _entrada;

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
   public Tensor4D _filtros;

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
   public Tensor4D _bias;

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
    * <p>
    *    Atributo visível devido as funções de ativação.
    * </p>
    */
   public Tensor4D _somatorio;
   
   /**
    * Tensor contendo os valores de saídas da camada.
    * <p>
    *    O formato da saída é dado por:
    * </p>
    * <pre>
    *    saida = (1, numeroFiltros, alturaSaida, larguraSaida)
    * </pre>
    */
   public Tensor4D _saida;

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
   public Tensor4D _gradEntrada;

   /**
    * Tensor contendo os valores dos gradientes relativos a saída
    * da camada.
    * <p>
    *    O formato dos gradientes da saída é dado por:
    * </p>
    * <pre>
    *    gradSaida = (1, numFiltros, alturaSaida, larguraSaida)
    * </pre>
    * <p>
    *    Atributo visível devido as funções de ativação.
    * </p>
    */
   public Tensor4D _gradSaida;

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
   public Tensor4D _gradFiltros;

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
   public Tensor4D _gradBias;

   /**
    * Função de ativação da camada.
    */
   private Ativacao ativacao = new Linear();

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
    * @param entrada formato de entrada da camada.
    * @param filtro formato dos filtros da camada.
    * @param filtros quantidade de filtros.
    * @param ativacao função de ativação.
    * @param iniKernel inicializador para os filtros.
    * @param iniBias inicializador para os bias.
    */
   public Convolucional(int[] entrada, int[] filtro, int filtros, Object ativacao, Object iniKernel, Object iniBias){
      this(filtro, filtros, ativacao, iniKernel, iniBias);

      if(entrada == null){
         throw new IllegalArgumentException(
            "\nO formato de entrada não pode ser nulo."
         );
      }

      if(entrada.length != 3){
         throw new IllegalArgumentException(
            "\nO formato de entrada deve conter 3 elementos (altura, largura, profundidade), " +
            "recebido: " + entrada.length
         );
      }

      if(utils.apenasMaiorZero(entrada) == false){
         throw new IllegalArgumentException(
            "\nOs valores do formato de entrada devem ser maiores que zero."
         );
      }

      shapeEntrada[0] = entrada[0];//profundidade
      shapeEntrada[1] = entrada[1];//altura
      shapeEntrada[2] = entrada[2];//largura

      construir(entrada);
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
    * @param entrada formato de entrada da camada.
    * @param filtro formato dos filtros da camada.
    * @param filtros quantidade de filtros.
    * @param ativacao função de ativação.
    * @param iniKernel inicializador para os filtros.
    */
   public Convolucional(int[] entrada, int[] filtro, int filtros, String ativacao, Object iniKernel){
      this(entrada, filtro, filtros, ativacao, iniKernel, null);
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
    * @param entrada formato de entrada da camada.
    * @param filtro formato dos filtros da camada.
    * @param filtros quantidade de filtros.
    * @param ativacao função de ativação.
    */
   public Convolucional(int[] entrada, int[] filtro, int filtros, String ativacao){
      this(entrada, filtro, filtros, ativacao, null, null);
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
    * @param entrada formato de entrada da camada.
    * @param filtro formato dos filtros da camada.
    * @param filtros quantidade de filtros.
    */
   public Convolucional(int[] entrada, int[] filtro, int filtros){
      this(entrada, filtro, filtros, null, null, null);
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
    * @param filtro formato dos filtros da camada.
    * @param filtros quantidade de filtros.
    * @param ativacao função de ativação.
    * @param iniKernel inicializador para os filtros.
    * @param iniBias inicializador para os bias.
    */
   public Convolucional(int[] filtro, int filtros, Object ativacao, Object iniKernel, Object iniBias){
      if(filtro == null){
         throw new IllegalArgumentException(
            "\nO formato do filtro não pode ser nulo."
         );
      }

      //formado dos filtros
      int[] formFiltro = (int[]) filtro;
      if(formFiltro.length != 2){
         throw new IllegalArgumentException(
            "\nO formato dos filtros deve conter 2 elementos (altura, largura), " +
            "recebido: " + formFiltro.length
         );
      }
      if(utils.apenasMaiorZero(formFiltro) == false){
         throw new IllegalArgumentException(
            "\nOs valores de formato para os filtros devem ser maiores que zero."
         );      
      }

      shapeFiltro[0] = formFiltro[0];
      shapeFiltro[1] = formFiltro[1];

      //número de filtros
      if(filtros < 1){
         throw new IllegalArgumentException(
            "\nO número de filtro deve ser maior que zero, recebido: " + filtros
         );
      }

      shapeSaida[0] = filtros;
      
      Dicionario dicio = new Dicionario();
      if(ativacao != null) this.ativacao = dicio.getAtivacao(ativacao);
      if(iniKernel != null) this.iniKernel = dicio.getInicializador(iniKernel);
      if(iniBias != null) this.iniBias = dicio.getInicializador(iniBias);
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
    * @param filtro formato dos filtros da camada.
    * @param filtros quantidade de filtros.
    * @param ativacao função de ativação.
    * @param iniKernel inicializador para os filtros.
    */
   public Convolucional(int[] filtro, int filtros, String ativacao, Object iniKernel){
      this(filtro, filtros, ativacao, iniKernel, null);
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
    * @param filtro formato dos filtros da camada.
    * @param filtros quantidade de filtros.
    * @param ativacao função de ativação.
    */
   public Convolucional(int[] filtro, int filtros, String ativacao){
      this(filtro, filtros, ativacao, null, null);
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
    * @param filtro formato dos filtros da camada.
    * @param filtros quantidade de filtros.
    */
   public Convolucional(int[] filtro, int filtros){
      this(filtro, filtros, null, null, null);
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

      if(fEntrada.length != 3 && fEntrada.length != 4){
         throw new IllegalArgumentException(
            "\nO formato de entrada para a camada Convolucional deve conter três " + 
            "elementos (profundidade, altura, largura), ou quatro elementos (primeiro desconsiderado) " + 
            "objeto recebido possui " + fEntrada.length
         );
      }

      shapeEntrada[0] = (fEntrada.length == 4) ? fEntrada[1] : fEntrada[0];//profundidade
      shapeEntrada[1] = (fEntrada.length == 4) ? fEntrada[2] : fEntrada[1];//altura
      shapeEntrada[2] = (fEntrada.length == 4) ? fEntrada[3] : fEntrada[2];//largura

      if(utils.apenasMaiorZero(fEntrada) == false){
         throw new IllegalArgumentException(
            "\nOs valores de dimensões de entrada para a camada Convolucional não " +
            "podem conter valores menores que 1."
         );
      }

      //dim -> ((entrada - filtro) / stride) + 1
      shapeSaida[1] = shapeEntrada[1] - shapeFiltro[0] + 1;
      shapeSaida[2] = shapeEntrada[2] - shapeFiltro[1] + 1;

      if(shapeSaida[1] < 1 || shapeSaida[2] < 1){
         throw new IllegalArgumentException(
            "\nFormato de entrada " + utils.shapeStr(fEntrada) +
            " e formato dos filtros " + 
            utils.shapeStr(new int[]{shapeSaida[0], shapeFiltro[0], shapeFiltro[1]}) +
            " resultam num formato de saída inválido " + utils.shapeStr(shapeSaida)
         );
      }

      //inicialização dos parâmetros necessários
      this._entrada      = new Tensor4D(shapeEntrada);
      this._gradEntrada  = new Tensor4D(this._entrada.shape());
      this._filtros      = new Tensor4D(shapeSaida[0], shapeEntrada[0], shapeFiltro[0], shapeFiltro[1]);
      this._gradFiltros  = new Tensor4D(_filtros.shape());
      this._saida        = new Tensor4D(shapeSaida);
      this._somatorio    = new Tensor4D(_saida.shape());
      this._gradSaida    = new Tensor4D(_saida.shape());

      if(usarBias){
         this._bias      = new Tensor4D(shapeSaida[0]);
         this._gradBias  = new Tensor4D(_bias.shape());
      }

      setNomes();
      
      this.treinavel = true;
      this._construida = true;//camada pode ser usada.
   }

   @Override
   public void inicializar(){
      verificarConstrucao();
      
      int numFiltros = shapeSaida[0];
      int profEntrada = shapeEntrada[0];
      for(int i = 0; i < numFiltros; i++){
         for(int j = 0; j < profEntrada; j++){
            iniKernel.inicializar(_filtros, i, j);
         }
      }

      if(usarBias){
         iniBias.inicializar(_bias, 0, 0, 0);
      }
   }

   @Override
   public void setAtivacao(Object ativacao){
      this.ativacao = new Dicionario().getAtivacao(ativacao);
   }

   @Override
   public void setBias(boolean usarBias){
      this.usarBias = usarBias;
   }

   @Override
   protected void setNomes(){
      _entrada.nome("entrada");
      _gradEntrada.nome("gradiente entrada");
      _filtros.nome("kernel");
      _saida.nome("saída");
      _gradFiltros.nome("gradiente kernel");
      _somatorio.nome("somatório");
      _gradSaida.nome("gradiente saída");

      if(usarBias){
         _bias.nome("bias");
         _gradBias.nome("gradiente bias");
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
   public Tensor4D forward(Object entrada){
      verificarConstrucao();

      if(entrada instanceof double[][][]){
         double[][][] e = (double[][][]) entrada;
         if(e.length != shapeEntrada[0] || e[0].length != shapeEntrada[1] || e[0][0].length != shapeEntrada[2]){
            throw new IllegalArgumentException(
               "\nAs dimensões da entrada recebida " + 
               "(" + e.length + ", " + e[0].length + ", " + e[0][0].length + ") " +
               "são incompatíveis com as dimensões da entrada da camada " + 
               utils.shapeStr(shapeEntrada)
            );
         }

         this._entrada.copiar(e, 0);
      
      }else if(entrada instanceof Tensor4D){
         Tensor4D e = (Tensor4D) entrada;
         if(this._entrada.comparar3D(e) == false){
            throw new IllegalArgumentException(
               "\nAs dimensões da entrada recebida " + e.shapeStr() + 
               " são incompatíveis com as dimensões da entrada da camada " + this._entrada.shapeStr()
            );
         }

         this._entrada.copiar(e, 0);

      }else{
         throw new IllegalArgumentException(
            "\nOs dados de entrada para a camada Convolucional devem ser " +
            "do tipo " + this._entrada.getClass().getSimpleName() + 
            " ou double[][][] objeto recebido é do tipo \"" + 
            entrada.getClass().getTypeName() + "\"."
         );
      }

      //feedforward

      //zerar os valores calculados anteiormente
      _somatorio.preencher(0.0d);

      optensor.convForward(this._entrada, _filtros, _somatorio);
      
      if(usarBias){
         double b;
         int numFiltros = shapeSaida[0];
         for(int i = 0; i < numFiltros; i++){
            b = _bias.get(0, 0, 0, i);
            _somatorio.add2D(0, i, b);
         }
      }

      ativacao.forward(_somatorio, _saida);

      return _saida;
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
    * @param grad gradiente da camada seguinte.
    */
   @Override
   public Tensor4D backward(Object grad){
      verificarConstrucao();

      if(grad instanceof Tensor4D){
         Tensor4D g = (Tensor4D) grad;
         if(_gradSaida.comparar3D(g) == false){
            throw new IllegalArgumentException(
               "\nAs três dimensões finais do tensor recebido " + g.shapeStr() +
               "são imcompatíveis as três primeira dimensões do tensor de gradiente"
            );
         }

         _gradSaida.copiar(g, 0);

      }else{
         throw new IllegalArgumentException(
            "Os gradientes para a camada Convolucional devem ser " +
            "do tipo \"" + _gradSaida.getClass().getTypeName() + 
            "\", objeto recebido é do tipo \"" + grad.getClass().getTypeName() + "\""
         );
      }

      ativacao.backward(this);
      
      //backward
      Tensor4D tempGrad = new Tensor4D(_gradFiltros.shape());
      _gradEntrada.preencher(0.0d);

      optensor.convBackward(_entrada, _filtros, _gradSaida, tempGrad, _gradEntrada);
      _gradFiltros.add(tempGrad);

      if(usarBias){
         int numFiltros = shapeSaida[0];
         for(int i = 0; i < numFiltros; i++){
            _gradBias.add(0, 0, 0, i, _gradSaida.somar2D(0, i));
         }
      }

      return _gradEntrada;
   }

   @Override
   public void zerarGradientes(){
      verificarConstrucao();

      _gradFiltros.zerar();
      _gradBias.zerar();
   }

   /**
    * Retorna a quantidade de filtros presentes na camada.
    * @return quantiadde de filtros presentes na camada.
    */
   public int numFiltros(){
      return this.shapeSaida[0];
   }

   @Override
   public Ativacao ativacao(){
      return this.ativacao;
   }

   @Override
   public Tensor4D saida(){
      return this._saida;
   }

   @Override
   public boolean temBias(){
      return usarBias;
   }

   @Override
   public int numParametros(){
      verificarConstrucao();

      int parametros = _filtros.tamanho();
      
      if(usarBias){
         parametros += _bias.tamanho();
      }

      return parametros;
   }

   @Override
   public double[] saidaParaArray(){
      verificarConstrucao();

      return _saida.paraArray();
   }

   @Override 
   public int tamanhoSaida(){
      return _saida.tamanho();
   }

   @Override
   public String info() {
      verificarConstrucao();

      StringBuilder sb = new StringBuilder();
      String pad = " ".repeat(4);
      
      sb.append(nome() + " (id " + this.id + ") = [\n");

      sb.append(pad + "Ativação: " + ativacao.nome() + "\n");
      sb.append(pad + "Entrada: " + utils.shapeStr(shapeEntrada) + "\n");
      sb.append(pad + "Filtros: " + numFiltros() + "\n");
      sb.append(pad + "Saida: " + utils.shapeStr(shapeSaida) + "\n");
      sb.append("\n");

      sb.append(pad + "Kernel: " + _filtros.shapeStr() + "\n");

      sb.append(pad + "Bias: ");
      if(temBias()){
         sb.append("(" + _bias.dim3() + ", "   + _bias.dim4() + ")\n");
      }else{
         sb.append(" N/A\n");
      }

      sb.append("]\n");

      return sb.toString();
   }

   @Override
   public String toString() {
      StringBuilder sb = new StringBuilder(info());
      int tamanho = sb.length();

      sb.delete(tamanho-1, tamanho);//remover ultimo "\n"    
      
      sb.append(" <hash: " + Integer.toHexString(hashCode()) + ">");
      sb.append("\n");
      
      return sb.toString();
   }

   @Override
   public Convolucional clone(){
      verificarConstrucao();

      try{
         Convolucional clone = (Convolucional) super.clone();

         clone.ativacao = this.ativacao;
         clone.usarBias = this.usarBias;

         clone._entrada     = this._entrada.clone();
         clone._filtros     = this._filtros.clone();
         clone._gradFiltros = this._gradFiltros.clone();

         if(this.usarBias){
            clone._bias     = this._bias.clone();
            clone._gradBias = this._gradBias.clone();
         }

         clone._somatorio   = this._somatorio.clone();
         clone._saida       = this._saida.clone();
         clone._gradSaida   = this._gradSaida.clone();

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
         shapeEntrada[0],
         shapeEntrada[1],
         shapeEntrada[2]
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
         shapeSaida[0],
         shapeSaida[1],
         shapeSaida[2]
      };
   }

   /**
    * Retorna o formato dos filtros contidos na camada.
    * @return formato de cada filtro (altura, largura).
    */
   public int[] formatoFiltro(){
      verificarConstrucao();

      return new int[]{
         shapeFiltro[0],
         shapeFiltro[1],
      };
   }

   @Override
   public Tensor4D kernel(){
      return _filtros;
   }

   @Override
   public double[] kernelParaArray(){
      return kernel().paraArray();
   }

   @Override
   public Tensor4D gradKernel(){
      return _gradFiltros;
   }

   @Override
   public double[] gradKernelParaArray(){
      return _gradFiltros.paraArray();
   }

   @Override
   public Tensor4D bias(){
      if(usarBias){
         return _bias;
      }

      throw new IllegalStateException(
         "\nA camada " + nome() + " (" + id + ") não possui bias configurado."
      );
   }

   @Override
   public double[] biasParaArray(){
      return _bias.paraArray();
   }

   @Override
   public double[] gradBias(){
      return _gradBias.paraArray();
   }

   @Override
   public Tensor4D gradEntrada(){
      return _gradEntrada; 
   }

   @Override
   public void setKernel(double[] kernel){
      if(kernel.length != _filtros.tamanho()){
         throw new IllegalArgumentException(
            "A dimensão do kernel fornecido (" + kernel.length + ") não é igual a quantidade de " +
            " parâmetros para os kernels da camada (" + _filtros.tamanho() + ")."
         );
      }
         
      _filtros.copiarElementos(kernel);
   }

   @Override
   public void setBias(double[] bias){
      if(bias.length != this._bias.tamanho()){
         throw new IllegalArgumentException(
            "A dimensão do bias fornecido não é igual a quantidade de " +
            " parâmetros para os bias da camada."
         );
      }
      
      this._bias.copiarElementos(bias);
   }

}
