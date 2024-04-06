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
    * <h3> Não alterar </h3>
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
    * <h3> Não alterar </h3>
    * Tensor contendo os viéses da camada, seu formato se dá por:
    * <pre>
    * b = (1, 1, 1, neuronios)
    * </pre>
    */
   public Tensor4D bias;

   /**
    * Auxiliar na verificação do uso do bias na camada.
    */
   private boolean usarBias = true;

   // auxiliares

   /**
    * <h3> Não alterar </h3>
    * Tensor contendo os valores de entrada da camada, seu formato se dá por:
    * <pre>
    *    entrada = (1, 1, 1, tamEntrada)
    * </pre>
    */
   public Tensor4D entrada;

   /**
    * <h3> Não alterar </h3>
    * Tensor contendo os valores de resultado da multiplicação matricial entre
    * os pesos e a entrada da camada adicionados com o bias, seu formato se dá por:
    * <pre>
    *    somatorio = (1, 1, 1, neuronios)
    * </pre>
    */
   public Tensor4D somatorio;

   /**
    * <h3> Não alterar </h3>
    * Tensor contendo os valores de resultado da soma entre os valores 
    * da matriz de somatório com os valores da matriz de bias da camada, seu 
    * formato se dá por:
    * <pre>
    *    saida = (1, 1, 1, neuronios)
    * </pre>
    */
   public Tensor4D saida;
   
   /**
    * <h3> Não alterar </h3>
    * Tensor contendo os valores de gradientes de cada neurônio da 
    * camada, seu formato se dá por:
    * <pre>
    *    gradSaida = (1, 1, 1, neuronios)
    * </pre>
    */
   public Tensor4D gradSaida;

   /**
    * <h3> Não alterar </h3>
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
    * <h3> Não alterar </h3>
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
    * <h3> Não alterar </h3>
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
      if(ativacao != null) this.ativacao = dic.getAtivacao(ativacao);
      if(iniKernel != null) this.iniKernel = dic.getInicializador(iniKernel);
      if(iniBias != null)  this.iniBias = dic.getInicializador(iniBias);
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
      this.entrada =    new Tensor4D(this.tamEntrada);
      this.saida =      new Tensor4D(this.numNeuronios);
      this.pesos =      new Tensor4D(this.tamEntrada, this.numNeuronios);
      this.gradPesos =  new Tensor4D(pesos.shape());

      if(usarBias){
         this.bias =     new Tensor4D(saida.shape());
         this.gradBias = new Tensor4D(saida.shape());
      }

      this.somatorio =   new Tensor4D(saida.shape());
      this.gradSaida =   new Tensor4D(saida.shape());
      this.gradEntrada = new Tensor4D(this.entrada.shape());

      setNomes();
      
      this.treinavel = true;//camada pode ser treinada.
      this.construida = true;//camada pode ser usada.
   }

   @Override
   public void setSeed(long seed){
      this.iniKernel.setSeed(seed);
      this.iniBias.setSeed(seed);
   }

   @Override
   public void inicializar(){
      verificarConstrucao();

      iniKernel.inicializar(pesos, 0, 0);

      if(usarBias){
         iniBias.inicializar(bias, 0, 0);
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
      entrada.nome("entrada");
      pesos.nome("kernel");
      saida.nome("saida");
      somatorio.nome("somatório");
      gradSaida.nome("gradiente saída");
      gradEntrada.nome("gradiente entrada");
      gradPesos.nome("gradiente kernel");

      if(usarBias){
         bias.nome("bias");
         gradBias.nome("gradiente bias");
      }
   }

   /**
    * <h2>
    *    Propagação direta através da camada Densa
    * </h2>
    * <p>
    *    Alimenta os dados de entrada para a saída da camada por meio da 
    *    multiplicação matricial entre a entrada recebida da camada e os pesos 
    *    da camada, em seguida é adicionado o bias caso ele seja configurado 
    *    no momento da inicialização.
    * </p>
    * <p>
    *    A expressão que define a saída é dada por:
    * </p>
    * <pre>
    *somatorio = matMult(entrada * pesos)
    *somatorio.add(bias)
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
   public Tensor4D forward(Object entrada){
      verificarConstrucao();

      if(entrada instanceof Tensor4D){
         Tensor4D e = (Tensor4D) entrada;
         if(this.entrada.dim4() != e.dim4()){
            throw new IllegalArgumentException(
               "Dimensões da entrada " + e.shapeStr() + " incompatível com entrada da " +
               "camada Densa " + this.entrada.shapeStr()
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
      optensor.matMult(this.entrada, pesos, somatorio);

      if(usarBias){
         somatorio.add(bias);
      }

      ativacao.forward(somatorio, saida);

      // return saida.clone();
      return saida;
   }

   /**
    * <h2>
    *    Propagação reversa através da camada Densa
    * </h2>
    * <p>
    *    Calcula os gradientes da camada para os pesos e bias baseado nos
    *    gradientes fornecidos.
    * </p>
    * <p>
    *    Após calculdos, os gradientes em relação a entrada da camada são
    *    calculados e salvos em {@code gradEntrada} para serem retropropagados 
    *    para as camadas anteriores do modelo em que a camada estiver.
    * </p>
    * Resultados calculados ficam salvos nas prorpiedades {@code camada.gradPesos} e
    * {@code camada.gradBias}.
    * @param grad gradiente da camada seguinte, deve ser um objeto do tipo 
    * {@code Tensor4D} ou {@code double[]}.
    */
   @Override
   public Tensor4D backward(Object grad){
      verificarConstrucao();

      if(grad instanceof double[]){
         double[] g = (double[]) grad;
         if(g.length != gradSaida.dim4()){
            throw new IllegalArgumentException(
               "\nTamanho do gradiente recebido (" + g.length + ") incompatível com o " +
               "suportado pela camada Densa (" + gradSaida.dim4() + ")."
            );
         }

         gradSaida.copiar(g, 0, 0, 0);
      
      }else if(grad instanceof Tensor4D){
         Tensor4D g = (Tensor4D) grad;
         gradSaida.copiar(
            g.array1D(0, 0, 0),
            0, 0, 0
         );

      }else{
         throw new IllegalArgumentException(
            "\nO gradiente para a camada Densa deve ser do tipo " + this.gradSaida.getClass() +
            " ou \"double[]\", objeto recebido é do tipo \"" + grad.getClass().getTypeName() + "\""
         );
      }

      //backward
      ativacao.backward(this);

      //gradiente temporário para usar como acumulador.
      Tensor4D tempGrad = new Tensor4D(gradPesos.shape());
      
      optensor.matMult(
         optensor.matTranspor(this.entrada, 0, 0), gradSaida, tempGrad
      );
      gradPesos.add(tempGrad);

      if(usarBias){
         gradBias.add(gradSaida);
      }

      optensor.matMult(
         gradSaida, optensor.matTranspor(pesos, 0, 0), gradEntrada
      );

      // return gradEntrada.clone();
      return gradEntrada;
   }

   @Override
   public Tensor4D saida(){
      return saida;
   }

   /**
    * Retorna a quantidade de neurônios presentes na camada.
    * @return quantidade de neurônios presentes na camada.
    */
   public int numNeuronios(){
      verificarConstrucao();

      return pesos.dim4();
   }

   @Override
   public Ativacao ativacao(){
      return this.ativacao;
   }

   /**
    * Retorna a capacidade de entrada da camada.
    * @return tamanho de entrada da camada.
    */
   public int tamanhoEntrada(){
      verificarConstrucao();

      return entrada.dim4();
   }

   /**
    * Retorna a capacidade de saída da camada.
    * @return tamanho de saída da camada.
    */
   public int tamanhoSaida(){
      return numNeuronios;
   }

   @Override
   public boolean temBias(){
      return usarBias;
   }

   @Override
   public int numParametros(){
      verificarConstrucao();

      int parametros = pesos.tamanho();
      
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
      verificarConstrucao();

      String buffer = "";
      String espacamento = "    ";
      
      buffer += "\nInfo " + this.getClass().getSimpleName() + " " + this.id + " = [\n";

      buffer += espacamento + "Ativação: " + this.ativacao.nome() + "\n";
      buffer += espacamento + "Quantidade neurônios: " + this.numNeuronios() + "\n";
      buffer += "\n";

      buffer += espacamento + "Entrada: [" + this.entrada.dim3() + ", " + this.entrada.dim4() + "]\n";
      buffer += espacamento + "Pesos:   [" + this.pesos.dim3() + ", "   + this.pesos.dim4() + "]\n";
      if(bias != null){
         buffer += espacamento + "Bias:    [" + this.bias.dim3() + ", "   + this.bias.dim4() + "]\n";
      }
      buffer += espacamento + "Saida:   [" + this.saida.dim3() + ", "   + this.saida.dim4() + "]\n";

      buffer += "]\n";

      return buffer;
   }

   @Override
   public Densa clonar(){
      verificarConstrucao();

      try{
         Densa clone = (Densa) super.clone();

         clone.optensor = new OpTensor4D();
         clone.ativacao = new Dicionario().getAtivacao(this.ativacao.getClass().getSimpleName());
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
    *    formato = (altura, largura)
    * </pre>
    * @return formato de entrada da camada.
    */
   @Override
   public int[] formatoEntrada(){
      return new int[]{
         entrada.dim3(), 
         entrada.dim4()
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
         saida.dim3(), 
         saida.dim4()
      };
   }

   @Override
   public Tensor4D kernel(){
      return pesos;
   }

   @Override
   public double[] kernelParaArray(){
      return pesos.paraArray();
   }

   @Override
   public Tensor4D gradKernel(){
      return gradPesos;
   }

   @Override
   public double[] gradKernelParaArray(){
      return gradPesos.paraArray();
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
   public Tensor4D gradEntrada(){
      return gradEntrada;
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
            gradPesos.set(grads[cont++], 0, 0, i, j);
         }
      }
   }

   @Override
   public void editarGradienteBias(double[] grads){
      this.gradBias.copiar(grads, 0, 0, 0);
   }

   @Override
   public void editarKernel(double[] kernel){
      if(kernel.length != pesos.tamanho()){
         throw new IllegalArgumentException(
            "A dimensão do kernel fornecido não é igual a quantidade de " +
            "parâmetros para os kernels da camada."
         );         
      }

      int cont = 0;
      for(int i = 0; i < pesos.dim3(); i++){
         for(int j = 0; j < pesos.dim4(); j++){
            this.pesos.set(kernel[cont++], 0, 0, i, j);
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
         this.bias.set(bias[cont++], 0, 0, 0, i);
      }
   }

   @Override
   public void zerarGradientes(){
      verificarConstrucao();

      gradPesos.zerar();
      gradBias.zerar();
   }
}
