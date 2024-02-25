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
 *    Camada Densa ou Totalmente conectada
 * </h2>
 * <p>
 *    A camada densa é um tipo de camada que está profundamente conectada
 *    com a camada anterior, onde cada conexão da camada anterior se conecta
 *    com todas as conexões de saída da camada densa.
 * </p>
 * <p>
 *    Ela funciona realizando a operação de produto entre a {@code entrada} e 
 *    seus {@code pesos}, adicionando os bias caso sejam configurados, de acordo 
 *    com a expressão:
 * </p>
 * <pre>
 *    somatorio = matMult(pesos * entrada) + bias
 * </pre>
 * Após a propagação dos dados, a função de ativação da camada é aplicada ao 
 * resultado do somatório, que por fim é salvo na saída da camada.
 * <pre>
 *    saida = ativacao(somatorio)
 * </pre>
 */
public class Densa extends Camada implements Cloneable{

   /**
    * Operador para tensores.
    */
   private OpTensor4D optensor = new OpTensor4D();

   /**
    * Utilitário.
    */
   private Utils utils = new Utils();

   /**
    * Variável controlador para o tamanho de entrada da camada densa.
    */
    private int tamEntrada;
    
   /**
    * Variável controlador para a quantidade de neurônios (unidades) 
    * da camada densa.
    */
   private int numNeuronios;

   //core

   /**
    * Tensor contendo os valores dos pesos de cada conexão da
    * entrada com a saída da camada.
    * <p>
    *    O formato da matriz de pesos é definido por:
    * </p>
    * <pre>
    *    pesos = (1, 1, entrada, neuronios)
    * </pre>
    * Assim, a disposição dos pesos é dada da seguinte forma:
    * <pre>
    * pesos = [
    *    n1p1, n2p1, n3p1, nNpN
    *    n1p2, n2p2, n3p2, nNpN
    *    n1p3, n2p3, n3p3, nNpN
    * ]
    * </pre>
    * Onde <strong>n</strong> é o neurônio (ou unidade) e <strong>p</strong>
    * é seu peso correspondente.
    */
   public Tensor4D pesos;

   /**
    * Tensor contendo os viéses da camada, seu formato se dá por:
    * <pre>
    * b = (1, 1, 1, neuronios)
    * </pre>
    */
   public Tensor4D bias;

   /**
    * Auxiliar na verificação do uso do bias na camada.
    */
   boolean usarBias = true;

   // auxiliares

   /**
    * Tensor contendo os valores de entrada da camada, seu formato se dá por:
    * <pre>
    *    entrada = (1, 1, 1, tamEntrada)
    * </pre>
    */
   public Tensor4D entrada;

   /**
    * Tensor contendo os valores de resultado da multiplicação matricial entre
    * os pesos e a entrada da camada adicionados com o bias, seu formato se dá por:
    * <pre>
    *    somatorio = (1, 1, 1, neuronios)
    * </pre>
    */
   public Tensor4D somatorio;

   /**
    * Tensor contendo os valores de resultado da soma entre os valores 
    * da matriz de somatório com os valores da matriz de bias da camada, seu 
    * formato se dá por:
    * <pre>
    *    saida = (1, 1, 1, neuronios)
    * </pre>
    */
   public Tensor4D saida;
   
   /**
    * Tensor contendo os valores de gradientes de cada neurônio da 
    * camada, seu formato se dá por:
    * <pre>
    *    gradSaida = (1, 1, 1, neuronios)
    * </pre>
    */
   public Tensor4D gradSaida;

   /**
    * Gradientes usados para retropropagar os erros para camadas anteriores.
    * <p>
    *    O formato do gradiente de entrada é definido por:
    * </p>
    * <pre>
    *    gradEntrada = (1, 1, 1, tamEntrada)
    * </pre>
    */
   public Tensor4D gradEntrada;

   /**
    * Tensor contendo os valores dos gradientes para os pesos da camada.
    * <p>
    *    O formato da matriz de gradiente dos pesos é definido por:
    * </p>
    * <pre>
    *    gradPesos = (1, 1, entrada, neuronios)
    * </pre>
    */
   public Tensor4D gradPesos;

   /**
    * Tensor contendo os valores dos gradientes para os bias da camada.
    * <p>
    *    O formato da matriz de gradientes dos bias é definido por:
    * </p>
    * <pre>
    *    gradBias = (1, 1, 1, neuronios)
    * </pre>
    */
   public Tensor4D gradBias;

   /**
    * Matriz coluna contendo os valores de derivada da função de ativação.
    * <p>
    *    O formato da matriz de derivada é definido por:
    * </p>
    * <pre>
    *    gradBias = (1, 1, 1, neuronios)
    * </pre>
    */
   public Tensor4D derivada;

   /**
    * Função de ativação da camada
    */
   private Ativacao ativacao = new Linear();

   /**
    * Inicializador para os pesos da camada.
    */
   private Inicializador iniKernel = new GlorotUniforme();

   /**
    * Inicializador para os bias da camada.
    */
   private Inicializador iniBias = new Zeros();

   /**
    * Instancia uma nova camada densa de neurônios de acordo com os dados fornecidos.
    * Após a inicialização os pesos e bias da Camada estarão zerados e devem ser 
    * inicializados com o método {@code inicializar()}.
    * @param e quantidade de conexões de entrada.
    * @param n quantidade de neurônios.
    * @param ativacao função de ativação que será usada pela camada.
    * @param usarBias uso de viés na camada.
    * @param iniKernel inicializador para os pesos da camada.
    * @param iniBias inicializador para os bias da camada.
    */
   public Densa(int e, int n, String ativacao, Object iniKernel, Object iniBias){
      this(n, ativacao, iniKernel, iniBias);

      if(e < 1){
         throw new IllegalArgumentException(
            "\nA camada deve conter ao menos uma entrada."
         );
      }
      if(e <= 0){
         throw new IllegalArgumentException(
            "\nO valor de entrada deve ser maior que zero."
         );
      }
   
      construir(new int[]{1, e});//construir automaticamente
   }

   /**
    * Instancia uma nova camada densa de neurônios de acordo com os dados fornecidos.
    * Após a inicialização os pesos e bias da Camada estarão zerados e devem ser 
    * inicializados com o método {@code inicializar()}.
    * @param e quantidade de conexões de entrada.
    * @param n quantidade de neurônios.
    * @param ativacao função de ativação que será usada pela camada.
    * @param iniKernel inicializador para os pesos da camada.
    */
   public Densa(int e, int n, String ativacao, Object iniKernel){
      this(e, n, ativacao, iniKernel, null);
   }

   /**
    * Instancia uma nova camada densa de neurônios de acordo com os dados fornecidos.
    * Após a inicialização os pesos e bias da Camada estarão zerados e devem ser 
    * inicializados com o método {@code inicializar()}.
    * @param e quantidade de conexões de entrada.
    * @param n quantidade de neurônios.
    * @param ativacao função de ativação que será usada pela camada.
    */
   public Densa(int e, int n, String ativacao){
      this(e, n, ativacao, null, null);
   }

   /**
    * Instancia uma nova camada densa de neurônios de acordo com os dados fornecidos.
    * Após a inicialização os pesos e bias da Camada estarão zerados e devem ser 
    * inicializados com o método {@code inicializar()}.
    * @param e quantidade de conexões de entrada.
    * @param n quantidade de neurônios.
    */
   public Densa(int e, int n){
      this(e, n, null, null, null);
   }

   /**
    * Instancia uma nova camada densa de neurônios de acordo com os dados fornecidos.
    * Após a inicialização os pesos e bias da Camada estarão zerados e devem ser 
    * inicializados com o método {@code inicializar()}.
    * @param n quantidade de neurônios.
    * @param ativacao função de ativação que será usada pela camada.
    * @param iniKernel inicializador para os pesos da camada.
    * @param iniBias inicializador para os bias da camada.
    */
   public Densa(int n, String ativacao, Object iniKernel, Object iniBias){
      if(n < 1){
         throw new IllegalArgumentException(
            "A camada deve conter ao menos um neurônio."
         );
      }
      this.numNeuronios = n;

      //usar os valores padrão se necessário
      Dicionario dic = new Dicionario();
      if(ativacao != null) this.ativacao = dic.obterAtivacao(ativacao);
      if(iniKernel != null) this.iniKernel = dic.obterInicializador(iniKernel);
      if(iniBias != null)  this.iniBias = dic.obterInicializador(iniBias);
   }

   /**
    * Instancia uma nova camada densa de neurônios de acordo com os dados fornecidos.
    * Após a inicialização os pesos e bias da Camada estarão zerados e devem ser 
    * inicializados com o método {@code inicializar()}.
    * @param n quantidade de neurônios.
    * @param ativacao função de ativação que será usada pela camada.
    * @param iniKernel inicializador para os pesos da camada.
    */
   public Densa(int n, String ativacao, Object iniKernel){
      this(n, ativacao, iniKernel, null);
   }

   /**
    * Instancia uma nova camada densa de neurônios de acordo com os dados fornecidos.
    * Após a inicialização os pesos e bias da Camada estarão zerados e devem ser 
    * inicializados com o método {@code inicializar()}.
    * @param n quantidade de neurônios.
    * @param ativacao função de ativação que será usada pela camada.
    */
   public Densa(int n, String ativacao){
      this(n, ativacao, null, null);
   }

   /**
    * Instancia uma nova camada densa de neurônios de acordo com os dados fornecidos.
    * Após a inicialização os pesos e bias da Camada estarão zerados e devem ser 
    * inicializados com o método {@code inicializar()}.
    * @param n quantidade de neurônios.
    */
   public Densa(int n){
      this(n, null, null, null);
   }

   /**
    * Inicializa os parâmetros necessários para a camada Densa.
    * <p>
    *    O formato de entrada deve ser um array contendo o tamanho de 
    *    cada dimensão e entrada da camada, e deve estar no formato:
    * </p>
    * <pre>
    *    entrada = (1, tamEntrada)
    * </pre>
    * @param entrada formato de entrada para a camada.
    */
   @Override
   public void construir(Object entrada){
      if(entrada instanceof int[] == false){
         throw new IllegalArgumentException(
            "\nObjeto esperado para entrada da camada Densa é do tipo int[], " +
            "objeto recebido é do tipo " + entrada.getClass().getTypeName()
         );
      }

      int[] formatoEntrada = (int[]) entrada;
      if(utils.apenasMaiorZero(formatoEntrada) == false){
         throw new IllegalArgumentException(
            "\nOs valores recebidos para o formato de entrada devem ser maiores que zero."
         );       
      }

      this.tamEntrada = formatoEntrada[utils.ultimoIndice(formatoEntrada)];

      if(this.numNeuronios <= 0){
         throw new IllegalArgumentException(
            "\nO número de neurônios para a camada Densa não foi definido."
         );
      }

      //inicializações
      this.entrada =    new Tensor4D(1, 1, 1, this.tamEntrada);
      this.saida =      new Tensor4D(1, 1, 1, this.numNeuronios);
      this.pesos =      new Tensor4D(1, 1, this.tamEntrada, this.numNeuronios);
      this.gradPesos =  new Tensor4D(pesos.dimensoes());

      if(usarBias){
         this.bias =     new Tensor4D(saida.dimensoes());
         this.gradBias = new Tensor4D(saida.dimensoes());
      }

      this.somatorio =   new Tensor4D(saida.dimensoes());
      this.derivada =    new Tensor4D(saida.dimensoes());
      this.gradSaida =   new Tensor4D(saida.dimensoes());
      this.gradEntrada = new Tensor4D(1, 1, this.entrada.dim3(), this.entrada.dim4());
      
      this.treinavel = true;//camada pode ser treinada.
      this.construida = true;//camada pode ser usada.
   }

   @Override
   public void configurarSeed(long seed){
      this.iniKernel.configurarSeed(seed);
      this.iniBias.configurarSeed(seed);
   }

   @Override
   public void inicializar(){
      verificarConstrucao();
      iniKernel.inicializar(pesos, 0, 0);
      iniBias.inicializar(bias, 0, 0);
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
      this.entrada.nome("entrada");
      this.pesos.nome("kernel");
      this.saida.nome("saida");
      this.somatorio.nome("somatório");
      this.derivada.nome("derivada");
      this.gradSaida.nome("gradiente saída");
      this.gradEntrada.nome("gradiente entrada");
      this.gradPesos.nome("gradiente kernel");

      if(usarBias){
         this.bias.nome("bias");
         this.gradBias.nome("gradiente bias");
      }
   }

   /**
    * Alimenta os dados de entrada para a saída da camada por meio da 
    * multiplicação matricial entre os pesos da camada e os dados de 
    * entrada, em seguida é adicionado o bias caso ele seja configurado 
    * no momento da inicialização.
    * <p>
    *    A pressão que define a saída é dada por:
    * </p>
    * <pre>
    *somatorio = (pesos * entrada) + bias
    *saida = ativacao(somatorio)
    * </pre>
    * Após a propagação dos dados, a função de ativação da camada é aplicada
    * ao resultado do somatório e o resultado é salvo da saída da camada.
    * @param entrada dados de entrada que serão processados, objetos aceitos incluem:
    * {@code Tensor4D}, ou {@code double[]}.
    * @throws IllegalArgumentException caso a entrada fornecida não seja suportada 
    * pela camada.
    * @throws IllegalArgumentException caso o tamanho dos dados de entrada seja diferente
    * da capacidade de entrada da camada.
    */
   @Override
   public void calcularSaida(Object entrada){
      verificarConstrucao();

      if(entrada instanceof Tensor4D){
         Tensor4D e = (Tensor4D) entrada;
         if(this.entrada.dim4() != e.dim4()){
            throw new IllegalArgumentException(
               "Dimensões da entrada " + e.dimensoesStr() + " incompatível com entrada da " +
               "camada Densa " + this.entrada.dimensoesStr()
            );
         }

         this.entrada.copiar(
            e.array1D(0, 0, 0), 
            0, 0, 0
         );

      }else if(entrada instanceof double[]){
         double[] e = (double[]) entrada;
         if(e.length != this.entrada.dim4()){
            throw new IllegalArgumentException(
               "Dimensões incompatíveis entre a entrada recebida (" + e.length +") e a" +
               " entrada da camada " + this.entrada.dim4()
            );
         }

         this.entrada.copiar(e, 0, 0, 0);

      }else{
         throw new IllegalArgumentException(
            "Tipo de entrada \"" + entrada.getClass().getTypeName() + "\"" +
            " não suportada."
         );
      }

      //feedforward
      optensor.matMult(this.entrada, pesos, somatorio, 0, 0);

      if(this.usarBias){
         somatorio.add(bias);
      }

      ativacao.calcular(this);
   }

   /**
    * Calcula os gradientes da camada para os pesos e bias baseado nos
    * gradientes fornecidos.
    * <p>
    *    Após calculdos, os gradientes em relação a entrada da camada são
    *    calculados e salvos em {@code gradEntrada} para serem retropropagados 
    *    para as camadas anteriores da Rede Neural em que a camada estiver.
    * </p>
    * Resultados calculados ficam salvos nas prorpiedades {@code camada.gradPesos} e
    * {@code camada.gradBias}.
    * @param gradSeguinte gradiente da camada seguinte, deve ser um objeto do tipo 
    * {@code Tensor4D} ou {@code double[]}.
    */
   @Override
   public void calcularGradiente(Object gradSeguinte){
      verificarConstrucao();

      if(gradSeguinte instanceof double[]){
         double[] grad = (double[]) gradSeguinte;
         if(grad.length != gradSaida.dim4()){
            throw new IllegalArgumentException(
               "\nTamanho do gradiente recebido (" + grad.length + ") incompatível com o " +
               "suportado pela camada Densa (" + gradSaida.dim4() + ")."
            );
         }

         gradSaida.copiar(grad, 0, 0, 0);
      
      }else if(gradSeguinte instanceof Tensor4D){
         Tensor4D grad = (Tensor4D) gradSeguinte;
         gradSaida.copiar(
            grad.array1D(0, 0, 0),
            0, 0, 0
         );

      }else{
         throw new IllegalArgumentException(
            "\nO gradiente para a camada Densa deve ser do tipo " + this.gradSaida.getClass() +
            " ou \"double[]\", objeto recebido é do tipo \"" + gradSeguinte.getClass().getTypeName() + "\""
         );
      }

      //backward
      ativacao.derivada(this);

      //gradiente temporário para usar como acumulador.
      Tensor4D tempGrad = new Tensor4D(gradPesos.dimensoes());
      
      optensor.matMult(
         optensor.matTranspor(this.entrada, 0, 0), derivada, tempGrad,
         0, 0
      );
      gradPesos.add(tempGrad);

      if(usarBias){
         gradBias.add(derivada);
      }

      optensor.matMult(
         derivada, optensor.matTranspor(pesos, 0, 0), gradEntrada, 0, 0
      );
   }

   @Override
   public Tensor4D saida(){
      return this.saida;
   }

   /**
    * Retorna a quantidade de neurônios presentes na camada.
    * @return quantidade de neurônios presentes na camada.
    */
   public int numNeuronios(){
      super.verificarConstrucao();

      return this.pesos.dim4();
   }

   @Override
   public Ativacao obterAtivacao(){
      return this.ativacao;
   }

   /**
    * Retorna a capacidade de entrada da camada.
    * @return tamanho de entrada da camada.
    */
   public int tamanhoEntrada(){
      super.verificarConstrucao();

      return this.entrada.dim4();
   }

   /**
    * Retorna a capacidade de saída da camada.
    * @return tamanho de saída da camada.
    */
   public int tamanhoSaida(){
      return this.numNeuronios;
   }

   @Override
   public boolean temBias(){
      return this.usarBias;
   }

   @Override
   public int numParametros(){
      super.verificarConstrucao();

      int parametros = 0;
      
      parametros += this.pesos.tamanho();
      if(this.temBias()){
         parametros += this.bias.tamanho();
      }

      return parametros;
   }

   @Override
   public double[] saidaParaArray(){
      super.verificarConstrucao();
      return this.saida.array1D(0, 0, 0);
   }

   /**
    * Indica algumas informações sobre a camada, como:
    * <ul>
    *    <li>Id da camada dentro do Modelo em que foi criada.</li>
    *    <li>Função de ativação.</li>
    *    <li>Quantidade de neurônios.</li>
    *    <li>Formato da entrada, pessos, bias e saída.</li>
    * </ul>
    * Algumas informações não estarão disponíveis caso a camada não esteja
    * inicializada.
    * @return buffer formatado contendo as informações da camada.
    */
   public String info(){
      super.verificarConstrucao();

      String buffer = "";
      String espacamento = "    ";
      
      buffer += "\nInfo " + this.getClass().getSimpleName() + " " + this.id + " = [\n";

      buffer += espacamento + "Ativação: " + this.ativacao.getClass().getSimpleName() + "\n";
      buffer += espacamento + "Quantidade neurônios: " + this.numNeuronios() + "\n";
      buffer += "\n";

      buffer += espacamento + "Entrada: [" + this.entrada.dim3() + ", " + this.entrada.dim4() + "]\n";
      buffer += espacamento + "Pesos:   [" + this.pesos.dim3() + ", "   + this.pesos.dim4() + "]\n";
      if(this.temBias()){
         buffer += espacamento + "Bias:    [" + this.bias.dim3() + ", "   + this.bias.dim4() + "]\n";
      }
      buffer += espacamento + "Saida:   [" + this.saida.dim3() + ", "   + this.saida.dim4() + "]\n";

      buffer += "]\n";

      return buffer;
   }

   @Override
   public Densa clonar(){
      super.verificarConstrucao();

      try{
         Densa clone = (Densa) super.clone();

         clone.optensor = new OpTensor4D();
         clone.ativacao = new Dicionario().obterAtivacao(this.ativacao.getClass().getSimpleName());
         clone.treinavel = this.treinavel;

         clone.usarBias = this.usarBias;
         if(this.usarBias){
            clone.bias = this.bias.clone();
            clone.gradBias = this.gradBias.clone();
         }

         clone.entrada = this.entrada.clone();
         clone.pesos = this.pesos.clone();
         clone.somatorio = this.somatorio.clone();
         clone.saida = this.saida.clone();
         clone.gradSaida = this.gradSaida.clone();
         clone.derivada = this.derivada.clone();
         clone.gradPesos = this.gradPesos.clone();

         return clone;
      }catch(Exception e){
         throw new RuntimeException(e);
      }
   }

   /**
    * Calcula o formato de entrada da camada Densa, que é disposto da
    * seguinte forma:
    * <pre>
    *    formato = (entrada.altura, entrada.largura)
    * </pre>
    * @return formato de entrada da camada.
    */
   @Override
   public int[] formatoEntrada(){
      return new int[]{
         this.entrada.dim3(), 
         this.entrada.dim4()
      };
   }

   /**
    * Calcula o formato de saída da camada Densa, que é disposto da
    * seguinte forma:
    * <pre>
    *    formato = (saida.altura, saida.largura)
    * </pre>
    * No caso da camada densa, o formato também pode ser descrito como:
    * <pre>
    *    formato = (1, numNeuronios)
    * </pre>
    * @return formato de saída da camada
    */
   @Override
   public int[] formatoSaida(){
      return new int[]{
         this.saida.dim3(), 
         this.saida.dim4()
      };
   }

   @Override
   public double[] obterKernel(){
      return this.pesos.paraArray();
   }

   @Override
   public double[] obterGradKernel(){
      return this.gradPesos.paraArray();
   }

   @Override
   public double[] obterBias(){
      return this.bias.paraArray();
   }

   @Override
   public double[] obterGradBias(){
      return this.gradBias.paraArray();
   }

   @Override
   public Object obterGradEntrada(){
      return this.gradEntrada;
   }

   @Override
   public void editarGradienteKernel(double[] grads){
      if(grads.length != gradPesos.tamanho()){
         throw new IllegalArgumentException(
            "A dimensão dos gradientes fornecidos não é igual a quantidade de " +
            "parâmetros para os kernels da camada (" + gradPesos.tamanho() + ")."
         );         
      }

      int cont = 0, lin = gradPesos.dim3(), col = gradPesos.dim4();
      for(int i = 0; i < lin; i++){
         for(int j = 0; j < col; j++){
            gradPesos.editar(0, 0, i, j, grads[cont++]);
         }
      }
   }

   @Override
   public void editarGradienteBias(double[] grads){
      this.gradBias.copiar(grads, 0, 0, 0);
   }

   @Override
   public void editarKernel(double[] kernel){
      if(kernel.length != this.pesos.tamanho()){
         throw new IllegalArgumentException(
            "A dimensão do kernel fornecido não é igual a quantidade de " +
            "parâmetros para os kernels da camada."
         );         
      }

      int cont = 0;
      for(int i = 0; i < this.pesos.dim3(); i++){
         for(int j = 0; j < this.pesos.dim4(); j++){
            this.pesos.editar(0, 0, i, j, kernel[cont++]);
         }
      }
   }

   @Override
   public void editarBias(double[] bias){
      if(bias.length != (this.bias.dim4())){
         throw new IllegalArgumentException(
            "A dimensão do bias fornecido (" + bias.length + ") não é igual a quantidade de " +
            " parâmetros para os bias da camada (" + this.bias.dim4() + ")."
         );
      }

      int cont = 0;
      for(int i = 0; i < this.bias.dim4(); i++){
         this.bias.editar(0, 0, 0, i, bias[cont++]);
      }
   }

   @Override
   public void zerarGradientes(){
      verificarConstrucao();

      gradPesos.preencher(0);
      gradBias.preencher(0);
   }
}
