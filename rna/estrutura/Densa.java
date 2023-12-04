package rna.estrutura;

import rna.ativacoes.Ativacao;
import rna.ativacoes.ReLU;
import rna.core.Mat;
import rna.core.OpMatriz;
import rna.inicializadores.Constante;
import rna.inicializadores.Inicializador;
import rna.serializacao.DicionarioAtivacoes;

/**
 * Camada Densa ou fully-connected.
 * <p>
 *    Ela funciona realizando a operação de produto entre a entrada e 
 *    seus pesos, adicionando os bias caso sejam configurados, de acordo 
 *    com a expressão:
 * </p>
 * <pre>
 *    somatorio = (pesos * entrada) + bias
 * </pre>
 * Após a propagação dos dados pela camada, a função de ativação da é aplicada
 * ao resultado do somatório, que por fim é salvo na saída da camada.
 * <pre>
 *    saida = ativacao(somatorio)
 * </pre>
 */
public class Densa extends Camada implements Cloneable{

   /**
    * Operador matricial para a camada densa.
    */
   private OpMatriz opmat = new OpMatriz();

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
    * Matriz coluna contendo os valores de resultado da soma entre os valores da
    * matriz de somatório com os valores da matriz de bias da camada, seu formato se dá por:
    * <pre>
    * s = [
    *    s1, s2, s3, sn  
    * ]
    * </pre>
    */
   public Mat saida;
   
   /**
    * Matriz coluna contendo os valores de gradientes de cada neurônio da camada, seu 
    * formato se dá por:
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
    * Identificador único da camada dentro da Rede Neural.
    */
   private int id;

   /**
    * Função de ativação da camada
    */
   private Ativacao ativacao = new ReLU();

   /**
    * Instancia uma nova camada densa de neurônios, inicializando seus atributos como:
    * <ul>
    *    <li> Pesos </li>
    *    <li> Bias </li>
    *    <li> Entrada </li>
    *    <li> Somatório </li>
    *    <li> Saída </li>
    *    <li> Gradientes </li>
    *    <li> Saída </li>
    * </ul>
    * Após a inicialização os pesos e bias da Camada estarão zerados e devem ser 
    * inicializados com o método {@code inicializar()}.
    * @param entrada quantidade de conexões de entrada.
    * @param neuronios quantidade de neurônios.
    * @param usarBias adicionar uso do bias para a camada.
    * @throws IllegalArgumentException se os valores de entrada ou neurônios forem 
    * menores que um.
    */
   public Densa(int entrada, int neuronios, boolean usarBias){
      if(entrada < 1){
         throw new IllegalArgumentException(
            "A camada deve conter ao menos uma entrada."
         );
      }
      if(neuronios < 1){
         throw new IllegalArgumentException(
            "A camada deve conter ao menos um neurônio."
         );
      }

      this.usarBias = usarBias;
      this.entrada = new Mat(1, entrada);
      this.saida =   new Mat(1, neuronios);
      this.pesos =   new Mat(entrada, neuronios);

      if(usarBias){
         this.bias =       new Mat(this.saida.lin, this.saida.col);
         this.gradBias =   new Mat(this.bias.lin, this.bias.col);
         this.gradAcBias = new Mat(this.bias.lin, this.bias.col);
      }

      this.somatorio =   new Mat(this.saida.lin, this.saida.col);
      this.derivada =    new Mat(this.saida.lin, this.saida.col);
      this.gradSaida =   new Mat(this.saida.lin, this.saida.col);
      this.gradEntrada = new Mat(this.entrada.lin, this.entrada.col);

      this.gradPesos =   new Mat(this.pesos.lin, this.pesos.col);
      this.gradAcPesos = new Mat(this.pesos.lin, this.pesos.col);
   }

   /**
    * Instancia uma nova camada densa de neurônios, inicializando seus atributos como:
    * <ul>
    *    <li> Pesos </li>
    *    <li> Bias </li>
    *    <li> Entrada </li>
    *    <li> Somatório </li>
    *    <li> Saída </li>
    *    <li> Gradientes </li>
    *    <li> Saída </li>
    * </ul>
    * Após a inicialização os pesos e bias da Camada estarão zerados e devem ser 
    * inicializados com o método {@code inicializar()}.
    * @param entrada quantidade de conexões de entrada.
    * @param neuronios quantidade de neurônios.
    */
   public Densa(int entrada, int neuronios){
      this(entrada, neuronios, true);
   }

   @Override
   public void inicializar(Inicializador iniKernel, Inicializador iniBias, double x){
      if(iniKernel == null){
         throw new IllegalArgumentException(
            "O inicializador não pode ser nulo."
         );
      }

      iniKernel.inicializar(this.pesos, x);
      
      if(this.usarBias){
         if(iniBias == null) new Constante().inicializar(this.bias, 0);
         else iniBias.inicializar(this.bias, x);
      }
   }

   @Override
   public void inicializar(Inicializador iniKernel, double x){
      this.inicializar(iniKernel, null, x);
   }

   @Override
   public void configurarAtivacao(String ativacao){
      DicionarioAtivacoes dicionario = new DicionarioAtivacoes();
      this.ativacao = dicionario.obterAtivacao(ativacao);
   }

   @Override
   public void configurarAtivacao(Ativacao ativacao){
      if(ativacao == null){
         throw new IllegalArgumentException(
            "A função de ativação não pode ser nula."
         );
      }

      this.ativacao = ativacao;
   }

   @Override
   public void configurarId(int id){
      this.id = id;
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
    * @param entrada dados de entrada que serão processados, deve ser um array do
    * tipo {@code double[]}.
    * @throws IllegalArgumentException caso a entrada fornecida não seja suportada 
    * pela camada.
    * @throws IllegalArgumentException caso o tamanho dos dados de entrada seja diferente
    * da capacidade de entrada da camada.
    */
   @Override
   public void calcularSaida(Object entrada){
      if(entrada instanceof double[] == false){
         throw new IllegalArgumentException(
            "Os dados de entrada para a camada Densa devem ser do tipo \"double[]\", " +
            "objeto recebido é do tipo \"" + entrada.getClass().getSimpleName() + "\""
         );
      }

      double[] e = (double[]) entrada;
      if(e.length != this.tamanhoEntrada()){
         throw new IllegalArgumentException(
            "Entradas (" + e.length + 
            ") incompatíveis com a entrada da camada (" + this.tamanhoEntrada() + 
            ")."
         );
      }

      this.entrada.copiar(0, e); 

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
      if(gradSeguinte instanceof Mat == false){
         throw new IllegalArgumentException(
            "O gradiente para a camada Densa deve ser do tipo " + this.gradSaida.getClass() +
            ", objeto recebido é do tipo \"" + gradSeguinte.getClass().getSimpleName() + "\""
         );
      }

      Mat grads = (Mat) gradSeguinte;
      if(grads.col != this.gradSaida.col){
         throw new IllegalArgumentException(
            "Dimensões incompatíveis entre o gradiente fornecido (" + grads.col + 
            ") e o suportado pela camada (" + this.gradSaida.col + ")."
         );
      }

      //transformação do array de gradientes para o objeto matricial
      //usado pela biblioteca
      this.gradSaida.copiar(grads);

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

   /**
    * Retorna a quantidade de neurônios presentes na camada.
    * @return quantidade de neurônios presentes na camada.
    */
   public int numNeuronios(){
      return this.pesos.col;
   }

   /**
    * Retorna a instância da função de ativação configurada para a camada.
    * @return função de ativação da camada.
    */
   public Ativacao obterAtivacao(){
      return this.ativacao;
   }

   /**
    * Retorna a capacidade de entrada da camada.
    * @return tamanho de entrada da camada.
    */
   public int tamanhoEntrada(){
      return this.entrada.col;
   }

   /**
    * Retorna a capacidade de saída da camada.
    * @return tamanho de saída da camada.
    */
   public int tamanhoSaida(){
      return this.saida.col;
   }

   @Override
   public boolean temBias(){
      return this.usarBias;
   }

   @Override
   public int numParametros(){
      int parametros = 0;
      
      parametros += this.pesos.lin * this.pesos.col;
      if(this.temBias()){
         parametros += this.bias.lin * this.bias.col;
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
      if(this.pesos.lin != pesos.length){
         throw new IllegalArgumentException(
            "Dimensões de pesos diferente da capacidade do neurônio."
         );
      }

      for(int i = 0; i < this.pesos.lin; i++){
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


   public double[] obterSaida(){
      return this.saida.linha(0);
   }

   /**
    * Retorna a matriz contendo as saídas da camada.
    * <p>
    *    A saída da camada é uma matriz com uma única linha contendo
    *    os seus resultados de saída.
    * </p>
    * @return matriz de saída da camada.
    */
   public Mat saidaParaMat(){
      return this.saida;
   }

   /**
    * Indica algumas informações sobre a camada, como:
    * <ul>
    *    <li>Id da camada dentro da Rede Neural em que foi criada.</li>
    *    <li>Função de ativação.</li>
    *    <li>Quantidade de neurônios.</li>
    *    <li>Formato da entrada, pessos, bias e saída.</li>
    * </ul>
    * Algumas informações não estarão disponíveis caso a camada não esteja
    * inicializada.
    * @return buffer formatado contendo as informações da camada.
    */
   public String info(){
      String buffer = "";
      String espacamento = "    ";
      
      buffer += "\nInfo " + this.getClass().getSimpleName() + " " + this.id + " = [\n";

      buffer += espacamento + "Ativação: " + this.ativacao.getClass().getSimpleName() + "\n";
      buffer += espacamento + "Quantidade neurônios: " + this.numNeuronios() + "\n";
      buffer += "\n";

      buffer += espacamento + "Entrada: [" + this.entrada.lin + ", " + this.entrada.col + "]\n";
      buffer += espacamento + "Pesos:   [" + this.pesos.lin + ", "   + this.pesos.col + "]\n";
      if(this.temBias()){
         buffer += espacamento + "Bias:    [" + this.bias.lin + ", "   + this.bias.col + "]\n";
      }
      buffer += espacamento + "Saida:   [" + this.saida.lin + ", "   + this.saida.col + "]\n";

      buffer += "]\n";

      return buffer;
   }

   /**
    * Clona a instância da camada, criando um novo objeto com as 
    * mesmas características mas em outro espaço de memória.
    * @return clone da camada.
    */
   @Override
   public Densa clone(){
      try{
         Densa clone = (Densa) super.clone();

         clone.ativacao = this.ativacao;

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
         this.entrada.lin, 
         this.entrada.col
      };
   }

   /**
    * Calcula o formato de saída da camada Densa, que é disposto da
    * seguinte forma:
    * <pre>
    *    formato = (saida.altura, saida.largura)
    * </pre>
    * @return formato de saída da camada
    */
   @Override
   public int[] formatoSaida(){
      return new int[]{
         this.saida.lin, 
         this.saida.col
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
}
