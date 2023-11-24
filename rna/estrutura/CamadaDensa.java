package rna.estrutura;

import rna.ativacoes.*;
import rna.core.Mat;
import rna.core.Matriz;
import rna.inicializadores.Inicializador;
import rna.serializacao.DicionarioAtivacoes;

/**
 * Camada Densa ou fully-connected.
 * <p>
 *    Realiza a operação de produto entre a entrada e seus pesos, adicionando
 *    os bias caso sejam configurados, de acordo com as seguintes expressões:
 * </p>
 * <pre>
 *somatorio = mult(entrada, pesos)
 *saida = add(somatorio, bias)
 * </pre>
 */
public class CamadaDensa implements Cloneable{

   /**
    * Operador matricial para a camada densa.
    */
   private Matriz mat = new Matriz();

   /**
    * Matriz contendo os valores dos pesos de cada conexão da
    * entrada com a saída da camada.
    * <p>
    *    O formato da matriz de pesos é definido por:
    * </p>
    * <pre>
    *    pesos = [entrada][neuronios]
    * </pre>
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
    * Matriz coluna contendo os valores de erro de cada neurônio da camada, seu 
    * formato se dá por:
    * <pre>
    * er = [
    *    er1, er2, er3, ern  
    * ]
    * </pre>
    */
   public Mat erros;

   /**
    * Matriz contendo os valores dos gradientes de cada conexão da
    * entrada com a saída da camada.
    * <p>
    *    O formato da matriz de gradientes é definido por:
    * </p>
    * <pre>
    *    gradientes = [linPesos][colPesos]
    * </pre>
    */
   public Mat gradientes;
   
   /**
    * Auxiliar no treino em lotes.
    */
   public Mat gradientesAcumulados;

   /**
    * Matriz coluna contendo os valores de derivada do resultado do somatório.
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
   Ativacao ativacao = new Sigmoid();

   /**
    * Variável auxilinar na inicialização dos pesos e bias da camada.
    */
   private boolean inicializada;

   /**
    * Instancia uma nova camada densa de neurônios, inicializando seus atributos como:
    * <ul>
    *    <li> Pesos </li>
    *    <li> Bias </li>
    *    <li> Entrada </li>
    *    <li> Somatório </li>
    *    <li> Saída </li>
    *    <li> Erros </li>
    *    <li> Gradientes </li>
    *    <li> Saída </li>
    * </ul>
    * Após a inicialização os pesos e bias da Camada estarão zerados e devem ser inicializados
    * com o método {@code inicializar()}.
    * @param entrada quantidade de conexões de entrada.
    * @param neuronios quantidade de neurônios.
    * @param usarBias adicionar uso do bias para a camada.
    */
   public CamadaDensa(int entrada, int neuronios, boolean usarBias){
      this.usarBias = usarBias;

      this.entrada = new Mat(1, entrada);
      this.pesos =   new Mat(entrada, neuronios);
      this.saida =   new Mat(1, neuronios);
      
      if(usarBias){
         this.bias = new Mat(saida.lin, saida.col);
      }

      this.somatorio =            new Mat(saida.lin, saida.col);
      this.derivada =             new Mat(saida.lin, saida.col);
      this.erros =                new Mat(saida.lin, saida.col);

      this.gradientes =           new Mat(pesos.lin, pesos.col);
      this.gradientesAcumulados = new Mat(pesos.lin, pesos.col);
      this.inicializada = false;
   }

   /**
    * Instancia uma nova camada densa de neurônios, inicializando seus atributos como:
    * <ul>
    *    <li> Pesos </li>
    *    <li> Bias </li>
    *    <li> Entrada </li>
    *    <li> Somatório </li>
    *    <li> Saída </li>
    *    <li> Erros </li>
    *    <li> Gradientes </li>
    *    <li> Saída </li>
    * </ul>
    * Após a inicialização os pesos e bias da Camada estarão zerados e devem ser 
    * inicializados com o método {@code inicializar()}.
    * @param entrada quantidade de conexões de entrada.
    * @param neuronios quantidade de neurônios.
    * @param usarBias adicionar uso do bias para a camada.
    */
   public CamadaDensa(int entrada, int neuronios){
      this(entrada, neuronios, true);
   }

   /**
    * Inicaliza os pesos e bias (caso tenha) da camada de acordo com o inicializador configurado.
    * @param inicializador inicializador de pesos.
    * @param alcance valor de alcance inicial para alguns inicializadores.
    */
   public void inicializar(Inicializador inicializador, double alcance){
      if(inicializador == null){
         throw new IllegalArgumentException(
            "O inicializador não pode ser nulo."
         );
      }

      inicializador.inicializar(this.pesos, alcance);
      if(this.usarBias){
         inicializador.inicializar(this.bias, alcance);
      }

      this.inicializada = true;
   }

   /**
    * Configura a função de ativação da camada através do nome fornecido, letras maiúsculas 
    * e minúsculas não serão diferenciadas.
    * <p>
    *    Ativações disponíveis:
    * </p>
    * <ul>
    *    <li> ReLU. </li>
    *    <li> Sigmoid. </li>
    *    <li> Tangente Hiperbólica. </li>
    *    <li> Leaky ReLU. </li>
    *    <li> ELU .</li>
    *    <li> Swish. </li>
    *    <li> GELU. </li>
    *    <li> Linear. </li>
    *    <li> Seno. </li>
    *    <li> Argmax. </li>
    *    <li> Softmax. </li>
    *    <li> Softplus. </li>
    * </ul>
    * @param ativacao nome da nova função de ativação.
    * @throws IllegalArgumentException se o valor fornecido não corresponder a nenhuma 
    * função de ativação suportada.
    */
   public void configurarAtivacao(String ativacao){
      DicionarioAtivacoes dicionario = new DicionarioAtivacoes();
      this.ativacao = dicionario.obterAtivacao(ativacao);
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
    * Configura o id da camada. O id deve indicar dentro da rede neural, em 
    * qual posição a camada está localizada.
    * @param id id da camada.
    */
   public void configurarId(int id){
      this.id = id;
   }

   /**
    * Alimenta os dados de entrada para a saída da camada por meio da 
    * multiplicação matricial entre os pesos da camada e os dados de 
    * entrada, em seguida é adicionado o bias caso ele seja configurado 
    * no momento da inicialização.
    * <p>
    *    Em resumo a expressão que define a saída é dada por:
    * </p>
    * <pre>
    *    somatorio = mult(pesos, entrada)
    * </pre>
    * <pre>
    *    saida = add(somatorio, bias)
    * </pre>
    * @param entrada dados de entrada que serão processados.
    */
   public void calcularSaida(double[] entrada){
      if(entrada.length != this.tamanhoEntrada()){
         throw new IllegalArgumentException(
            "Entradas (" + entrada.length + 
            ") incompatíveis com a entrada da camada (" + this.tamanhoEntrada() + 
            ")."
         );
      }

      this.entrada.substituir(0, entrada); 

      mat.mult(this.entrada, this.pesos, this.somatorio);
      if(usarBias){
         mat.add(this.somatorio, this.bias, this.somatorio);
      }

      ativacao.calcular(this);
   }

   /**
    * Executa a derivada da função de ativação específica da camada
    * em todos os neurônios dela.
    * <p>
    *    O resultado da derivada é salvo na propriedade {@code camada.derivada}.
    * </p>
    */
   public void calcularDerivadas(){
      this.ativacao.derivada(this);
   }

   /**
    * Retorna a quantidade de neurônios presentes na camada.
    * @return quantidade de neurônios presentes na camada.
    */
   public int quantidadeNeuronios(){
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

   /**
    * Verifica se a camada atual possui o bias configurado para seus neurônios.
    * @return true caso possua bias configurado, false caso contrário.
    */
   public boolean temBias(){
      return this.usarBias;
   }

   /**
    * Retorda a quantidade de conexões totais da camada, em outras palavras, 
    * retorna o somatório da quantidade de pesos e bias presentes na camada.
    * @return a quantidade de conexões totais.
    */
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
      if(id < 0 || id >= this.quantidadeNeuronios()){
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
      if(id < 0 || id >= this.quantidadeNeuronios()){
         throw new IllegalArgumentException(
            "Índice fornecido (" + id +") inválido."
         );
      }

      this.bias.editar(0, id, bias);
   }

   /**
    * Retorna a matriz contendo as saídas da camada.
    * <p>
    *    A saída da camada é uma matriz com uma única linha contendo
    *    os seus resultados de saída.
    * </p>
    * @return matriz de saída da camada.
    */
   public Mat obterSaida(){
      return this.saida;
   }

   /**
    * Indica algumas informações sobre a camada, como:
    * <ul>
    *    <li>Id da camada dentro da Rede Neural em que foi criada.</li>
    *    <li>Status de inicialização.</li>
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

      buffer += espacamento + "Inicializada: " + this.inicializada + "\n";
      buffer += espacamento + "Ativação: " + this.ativacao.getClass().getSimpleName() + "\n";
      buffer += espacamento + "Quantidade neurônios: " + this.quantidadeNeuronios() + "\n";
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
   public CamadaDensa clone(){
      try{
         CamadaDensa clone = (CamadaDensa) super.clone();

         clone.ativacao = this.ativacao;

         clone.usarBias = this.usarBias;
         clone.bias = this.bias.clone();

         clone.entrada = this.entrada.clone();
         clone.pesos = this.pesos.clone();
         clone.somatorio = this.somatorio.clone();
         clone.saida = this.saida.clone();
         clone.erros = this.erros.clone();
         clone.derivada = this.derivada.clone();
         clone.gradientes = this.gradientes.clone();

         return clone;
      }catch(Exception e){
         throw new RuntimeException(e);
      }
   }
}
