package rna.camadas;

import rna.ativacoes.Ativacao;
import rna.ativacoes.Linear;
import rna.core.Dicionario;
import rna.core.Mat;
import rna.core.OpMatriz;
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
 *    somatorio = (pesos * entrada) + bias
 * </pre>
 * Após a propagação dos dados, a função de ativação da camada é aplicada ao 
 * resultado do somatório, que por fim é salvo na saída da camada.
 * <pre>
 *    saida = ativacao(somatorio)
 * </pre>
 */
public class Densa extends Camada implements Cloneable{

   /**
    * Operador matricial para a camada densa.
    */
   private OpMatriz opmat = new OpMatriz();

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
    * Matriz contendo os valores dos pesos de cada conexão da
    * entrada com a saída da camada.
    * <p>
    *    O formato da matriz de pesos é definido por:
    * </p>
    * <pre>
    *    pesos = [entrada][neuronios]
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
    * é seu peso.
    */
   public Mat pesos;

   /**
    * Matriz coluna contendo os viéses da camada, seu formato se dá por:
    * <pre>
    * b = [
    *    b1, b2, bn3, bn
    * ]
    * </pre>
    */
   public Mat bias;

   /**
    * Auxiliar na verificação do uso do bias na camada.
    */
   boolean usarBias = true;

   // auxiliares

   /**
    * Matriz coluna contendo os valores de entrada da camada, seu formato se dá por:
    * <pre>
    * e = [
    *    e1,  e2, e3, en 
    * ]
    * </pre>
    */
   public Mat entrada;

   /**
    * Matriz coluna contendo os valores de resultado da multiplicação matricial entre
    * os pesos e a entrada da camada adicionados com o bias, seu formato se dá por:
    * <pre>
    * som = [
    *    som1, som2, som3, somn  
    * ]
    * </pre>
    */
   public Mat somatorio;

   /**
    * Matriz coluna contendo os valores de resultado da soma entre os valores 
    * da matriz de somatório com os valores da matriz de bias da camada, seu 
    * formato se dá por:
    * <pre>
    * s = [
    *    s1, s2, s3, sn  
    * ]
    * </pre>
    */
   public Mat saida;
   
   /**
    * Matriz coluna contendo os valores de gradientes de cada neurônio da 
    * camada, seu formato se dá por:
    * <pre>
    * grad = [
    *    g1, g2, g3, gn  
    * ]
    * </pre>
    */
   public Mat gradSaida;

   /**
    * Gradientes usados para retropropagar os erros para camadas anteriores.
    * <pre>
    * e = [
    *    e1,  e2, e3, en 
    * ]
    * </pre>
    */
   public Mat gradEntrada;

   /**
    * Matriz contendo os valores dos gradientes para os pesos da camada.
    * <p>
    *    O formato da matriz de gradientes é definido por:
    * </p>
    * <pre>
    *    gradientes = [linPesos][colPesos]
    * </pre>
    */
   public Mat gradPesos;

   /**
    * Matriz contendo os valores dos gradientes para os bias da camada.
    * <p>
    *    O formato da matriz de gradientes é definido por:
    * </p>
    * <pre>
    *    gradientes = [linBias][colBias]
    * </pre>
    */
   public Mat gradBias;
   
   /**
    * Auxiliar no treino em lotes.
    * <p>
    *    Matriz de gradiente para os pesos usada como acumulador
    * </p>
    */
   public Mat gradAcPesos;

   /**
    * Auxiliar no treino em lotes.
    * <p>
    *    Matriz de gradiente para os bias usada como acumulador
    * </p>
    */
   public Mat gradAcBias;

   /**
    * Matriz coluna contendo os valores de derivada da função de ativação.
    * <pre>
    * d = [
    *    d1, d2, d3, dn  
    * ]
    * </pre>
    */
   public Mat derivada;

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
            "A camada deve conter ao menos uma entrada."
         );
      }
      if(e <= 0){
         throw new IllegalArgumentException(
            "O valor de entrada deve ser maior que zero."
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
    *    entrada = (altura, largura)
    * </pre>
    * @param entrada formato de entrada para a camada.
    */
   @Override
   public void construir(Object entrada){
      if(entrada instanceof int[] == false){
         throw new IllegalArgumentException(
            "Objeto esperado para entrada da camada Densa é do tipo int[], " +
            "objeto recebido é do tipo " + entrada.getClass().getTypeName()
         );
      }

      int[] formatoEntrada = (int[]) entrada;
      if(formatoEntrada.length < 2){
         throw new IllegalArgumentException(
            "O formato de entrada para a camada Densa deve conter pelo menos dois " + 
            "elementos (altura, largura), objeto recebido possui " + formatoEntrada.length + "."
         );
      }
      if(formatoEntrada[1] < 1 || formatoEntrada[1] < 1){
         throw new IllegalArgumentException(
            "Os valores recebidos para o formato de entrada devem ser maiores que zero, " +
            "recebido = (" + formatoEntrada[0] + ", " + formatoEntrada[1] + ")."
         );
      }

      this.tamEntrada = formatoEntrada[1];
      if(this.numNeuronios == 0){
         throw new IllegalArgumentException(
            "O número de neurônios para a camada Densa não foi definido."
         );
      }

      //inicializações
    
      this.entrada = new Mat(1, this.tamEntrada);
      this.saida =   new Mat(1, this.numNeuronios);
      this.pesos =   new Mat(this.tamEntrada, this.numNeuronios);

      if(usarBias){
         this.bias =       new Mat(this.saida.lin(), this.saida.col());
         this.gradBias =   new Mat(this.bias.lin(), this.bias.col());
         this.gradAcBias = new Mat(this.bias.lin(), this.bias.col());
      }

      this.somatorio =   new Mat(this.saida.lin(), this.saida.col());
      this.derivada =    new Mat(this.saida.lin(), this.saida.col());
      this.gradSaida =   new Mat(this.saida.lin(), this.saida.col());
      this.gradEntrada = new Mat(this.entrada.lin(), this.entrada.col());

      this.gradPesos =   new Mat(this.pesos.lin(), this.pesos.col());
      this.gradAcPesos = new Mat(this.pesos.lin(), this.pesos.col());
      
      this.treinavel = true;
      this.construida = true;//camada pode ser usada.
   }

   @Override
   public void configurarSeed(long seed){
      this.iniKernel.configurarSeed(seed);
      this.iniBias.configurarSeed(seed);
   }

   @Override
   public void inicializar(){
      super.verificarConstrucao();
      this.iniKernel.inicializar(this.pesos);
      this.iniBias.inicializar(this.bias);
   }

   @Override
   public void configurarAtivacao(Object ativacao){
      this.ativacao = new Dicionario().obterAtivacao(ativacao);
   }

   @Override
   public void configurarBias(boolean usarBias){
      this.usarBias = usarBias;
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
    * {@code Mat[]}, {@code Mat} ou {@code double[]}.
    * @throws IllegalArgumentException caso a entrada fornecida não seja suportada 
    * pela camada.
    * @throws IllegalArgumentException caso o tamanho dos dados de entrada seja diferente
    * da capacidade de entrada da camada.
    */
   @Override
   public void calcularSaida(Object entrada){
      super.verificarConstrucao();

      if(entrada instanceof Mat[]){
         Mat[] en = (Mat[]) entrada;
         if(en.length != 1){
            throw new IllegalArgumentException(
               "A camada densa suporta apenas arrays de matrizes com profundidade = 1."
            );
         }
         this.entrada.copiar(en[0]);
      
      }else if(entrada instanceof Mat){
         this.entrada.copiar((Mat) entrada);

      }else if(entrada instanceof double[]){
         double[] en = (double[]) entrada;
         if(en.length != this.tamanhoEntrada()){
            throw new IllegalArgumentException(
               "Tamanho da entrada fornecida (" + en.length + 
               ") incompatível com a entrada da camada (" + this.tamanhoEntrada() + 
               ")."
            );
         }
         this.entrada.copiar(0, en); 

      }else{
         throw new IllegalArgumentException(
            "A camada Densa não suporta entradas do tipo \"" + entrada.getClass().getTypeName() + "\"."
         );
      }

      //feedforward
      this.opmat.mult(this.entrada, this.pesos, this.somatorio);
      if(this.usarBias){
         this.opmat.add(this.somatorio, this.bias, this.somatorio);
      }

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
    * Resultados calculados ficam salvos nas prorpiedades {@code camada.gradPesos} e
    * {@code camada.gradBias}.
    * @param gradSeguinte gradiente da camada seguinte, deve ser um objeto do tipo {@code Mat}.
    */
   @Override
   public void calcularGradiente(Object gradSeguinte){
      super.verificarConstrucao();

      if(gradSeguinte instanceof Mat[]){
         Mat[] grads = (Mat[]) gradSeguinte;
         if(grads.length != 1){
            throw new IllegalArgumentException(
               "A camada densa suporta apenas arrays de matrizes com profundidade = 1."
            );
         }
         this.gradSaida.copiar(grads[0]);
      
      }else if(gradSeguinte instanceof Mat){
         Mat grads = (Mat) gradSeguinte;
         if(grads.col() != this.gradSaida.col()){
            throw new IllegalArgumentException(
               "Dimensões incompatíveis entre o gradiente fornecido (" + grads.col() + 
               ") e o suportado pela camada (" + this.gradSaida.col() + ")."
            );
         }
         this.gradSaida.copiar(grads);

      }else{
         throw new IllegalArgumentException(
            "O gradiente para a camada Densa deve ser do tipo " + this.gradSaida.getClass() +
            ", objeto recebido é do tipo \"" + gradSeguinte.getClass().getTypeName() + "\""
         );
      }

      //backward
      //derivada da função de ativação em relação ao gradiente de saída
      this.ativacao.derivada(this);
      
      //derivada da função de ativação em relação ao pesos.
      this.opmat.mult(
         this.entrada.transpor(), this.derivada, this.gradPesos
      );

      //gradiente para o bias é apenas a derivada da ativação em relação a saída.
      if(this.temBias()){
         this.gradBias.copiar(this.derivada);
      }

      //derivada da saída em relação aos pesos para retropropagação.
      this.opmat.mult(
         this.derivada, this.pesos.transpor(), this.gradEntrada
      );
   }

   @Override
   public Mat saida(){
      return this.saida;
   }

   /**
    * Retorna a quantidade de neurônios presentes na camada.
    * @return quantidade de neurônios presentes na camada.
    */
   public int numNeuronios(){
      super.verificarConstrucao();

      return this.pesos.col();
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

      return this.entrada.col();
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

   /**
    * Configura os novos pesos para um neurônio específico da camada.
    * @param id índice do neurônio que será configurado.
    * @param pesos novos valores de pesos para o neurônio.
    * @throws IllegalArgumentException se o índice for inválido
    * @throws IllegalArgumentException se a quantidade de pesos fornecida for diferente do
    * da quantidade de pesos suportada pelo neurônio.
    */
   public void configurarPesos(int id, double[] pesos){
      if(id < 0 || id >= this.numNeuronios()){
         throw new IllegalArgumentException(
            "Índice fornecido (" + id +") inválido."
         );
      }
      if(this.pesos.lin() != pesos.length){
         throw new IllegalArgumentException(
            "Dimensões de pesos diferente da capacidade do neurônio."
         );
      }

      for(int i = 0; i < this.pesos.lin(); i++){
         this.pesos.editar(i, id, pesos[i]);
      }

   }

   /**
    * Configura o novo valor de bias para o neurônio especificado.
    * @param id id do neurônio que será configurado.
    * @param bias novo valor de bias/viés.
    * @throws IllegalArgumentException se o índice for inválido.
    */
   public void configurarBias(int id, double bias){
      if(id < 0 || id >= this.numNeuronios()){
         throw new IllegalArgumentException(
            "Índice fornecido (" + id +") inválido."
         );
      }

      this.bias.editar(0, id, bias);
   }

   @Override
   public double[] saidaParaArray(){
      super.verificarConstrucao();
      return this.saida.paraArray();
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

      buffer += espacamento + "Entrada: [" + this.entrada.lin() + ", " + this.entrada.col() + "]\n";
      buffer += espacamento + "Pesos:   [" + this.pesos.lin() + ", "   + this.pesos.col() + "]\n";
      if(this.temBias()){
         buffer += espacamento + "Bias:    [" + this.bias.lin() + ", "   + this.bias.col() + "]\n";
      }
      buffer += espacamento + "Saida:   [" + this.saida.lin() + ", "   + this.saida.col() + "]\n";

      buffer += "]\n";

      return buffer;
   }

   @Override
   public Densa clonar(){
      super.verificarConstrucao();

      try{
         Densa clone = (Densa) super.clone();

         clone.opmat = new OpMatriz();
         clone.ativacao = new Dicionario().obterAtivacao(this.ativacao.getClass().getSimpleName());
         clone.treinavel = this.treinavel;

         clone.usarBias = this.usarBias;
         if(this.usarBias){
            clone.bias = this.bias.clone();
            clone.gradBias = this.gradBias.clone();
            clone.gradAcBias = this.gradAcBias.clone();
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
         this.entrada.lin(), 
         this.entrada.col()
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
         this.saida.lin(), 
         this.saida.col()
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
   public double[] obterAcGradKernel(){
      return this.gradAcPesos.paraArray();    
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
   public double[] obterAcGradBias(){
      return this.gradAcBias.paraArray();       
   }

   @Override
   public Object obterGradEntrada(){
      return this.gradEntrada;
   }

   @Override
   public void editarGradienteKernel(double[] grads){
      this.gradPesos.copiar(grads);
   }

   @Override
   public void editarGradienteBias(double[] grads){
      this.gradBias.copiar(grads);
   }

   @Override
   public void editarAcGradKernel(double[] acumulador){
      this.gradAcPesos.copiar(acumulador);
   }

   @Override
   public void editarAcGradBias(double[] acumulador){
      this.gradAcBias.copiar(acumulador);
   }

   @Override
   public void editarKernel(double[] kernel){
      if(kernel.length != this.pesos.tamanho()){
         throw new IllegalArgumentException(
            "A dimensão do kernel fornecido não é igual a quantidade de " +
            " parâmetros para os kernels da camada."
         );         
      }

      this.pesos.copiar(kernel);
   }

   @Override
   public void editarBias(double[] bias){
      if(bias.length != (this.bias.col())){
         throw new IllegalArgumentException(
            "A dimensão do bias fornecido não é igual a quantidade de " +
            " parâmetros para os bias da camada."
         );
      }

      this.bias.copiar(bias);
   }

   @Override
   public void zerarAcumuladores(){
      super.verificarConstrucao();
      this.gradAcPesos.preencher(0);
      this.gradAcBias.preencher(0);
   }
}
