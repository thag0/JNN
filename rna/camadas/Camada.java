package rna.camadas;

import rna.ativacoes.Ativacao;

/**
 * <h2>
 *    Modelo base para a representação de uma Camada individual
 *    dentro da biblioteca
 * </h2>
 * <p>
 *    A classe camada serve apenas de molde para criação de novas
 *    camadas e não pode ser especificamente instanciada nem utilizada.
 * </p>
 * <p>
 *    As partes mais importantes de uma camada são {@code calcularSaida()} e 
 *    {@code calcularGradiente()} onde são implementados os métodos básicos 
 *    também conhecidos como "forward" e "backward". 
 * </p>
 * <p>
 *    Para a parte de propagação direta (calcular saída ou forward) os dados
 *    recebidos de entrada são processados de acordo com cada regra individual
 *    de cada camada e ao final os resultados são salvos em sua saída.
 * </p>
 * <p>
 *    Na propagação reversa (calcular gradiente ou backward) são recebidos os 
 *    gradientes da camada anterior e cada camada irá fazer seu processamento 
 *    para calcular os próprios gradientes para seus atributos treináveis. Aqui 
 *    cada camada tem o adicional de calcular os gradientes em relação as suas 
 *    entradas para retropropagar para camadas anteriores usadas pelos modelos.
 * </p>
 * <h2>
 * Existem dois detalhes importantes na implementação das camadas.
 * </h2>
 * <ul>
 *    <li>
 *       Primeiramente que os elementos das camadas devem ser pré inicializados 
 *       para evitar alocações dinâmicas durante a execução dos modelos e isso 
 *       se dá por dois motivos: ter controle das dimensões dos objetos criandos 
 *       durante toda a execução dos algoritmos e também criar uma espécie de cache 
 *       para evitar muitas instanciações em runtime.
 *    </li>
 *    <li>
 *       Segundo, que as funções de ativação não são camadas independentes e sim 
 *       funções que atuam sobre os elementos das camadas, especialmente nos elementos 
 *       chamados "somatório" e guardam os resultados na saída da camada.
 *    </li>
 * </ul>
 */
public abstract class Camada{

   /**
    * Controlador para uso dentro dos algoritmos de treino.
    */
   public boolean treinavel = false;

   /**
    * Controlador de construção da camada.
    */
   public boolean construida = false;

   /**
    * Controlador de treino da camada.
    */
   protected boolean treinando = false;

   /**
    * Identificador único da camada.
    */
   public int id;

   /**
    * Instancia a camada base usada dentro dos modelos de Rede Neural.
    * <p>
    *    A camada base não possui implementação de métodos e é apenas usada
    *    como molde de base para as outras camadas terem suas próprias implementações.
    * </p>
    */
   protected Camada(){}

   /**
    * Monta a estrutura da camada.
    * <p>
    *    A construção da camada envolve inicializar seus atributos como entrada,
    *    kernels, bias, além de elementos auxiliares que são importantes para
    *    o seu funcionamento correto.
    * </p>
    * @param entrada formato de entrada da camada, dependerá do formato de saída
    * da camada anterior, no caso de ser a primeira camada, dependerá do formato
    * dos dados de entrada.
    */
   public abstract void construir(Object entrada);

   /**
    * Verificador de inicialização para evitar problemas.
    */
   protected void verificarConstrucao(){
      if(this.construida == false){
         throw new IllegalArgumentException(
            "Camada " + this.getClass().getSimpleName() + 
            " (" + this.id + ") não foi construída."
         );
      }
   }

   /**
    * Inicaliza os parâmetros treináveis da camada de acordo com os inicializadores
    * definidos.
    */
   public abstract void inicializar();

   /**
    * Configura a função de ativação da camada através de uma instância de 
    * {@code FuncaoAtivacao} que será usada para ativar seus neurônios.
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
    * <p>
    *    Configurando a ativação da camada usando uma instância de função 
    *    de ativação aumenta a liberdade de personalização dos hiperparâmetros
    *    que algumas funções podem ter.
    * </p>
    * @param ativacao nova função de ativação.
    * @throws IllegalArgumentException se a função de ativação fornecida for nula.
    */
   public void configurarAtivacao(Object ativacao){
      throw new UnsupportedOperationException(
         "Implementar configuração da função de ativação da camada " + this.getClass().getTypeName() + "."
      );
   }

   /**
    * Configura o id da camada. O id deve indicar dentro de um modelo, em 
    * qual posição a camada está localizada.
    * @param id id da camada.
    */
   public void configurarId(int id){
      if(id < 0){
         throw new IllegalArgumentException(
            "O id da camada deve conter um positivo também podendo ser zero, " + 
            "recebido: " + id
         );
      }

      this.id = id;
   }

   /**
    * Configura o uso do bias para a camada.
    * <p>
    *    A configuração deve ser feita antes da construção da camada.
    * </p>
    * @param usarBias uso do bias.
    */
   public void configurarBias(boolean usarBias){
      throw new UnsupportedOperationException(
         "Implementar configuração de bias da camada " + this.getClass().getTypeName() + "."
      );     
   }

   /**
    * Configura a camada para treino.
    * @param treinando caso verdadeiro a camada será configurada para
    * treino, caso contrário, será usada para testes/predições.
    */
   public void configurarTreino(boolean treinando){
      this.treinando = true;
   }

   /**
    * Configura uma seed fixa para geradores de números aleatórios da
    * camada.
    * @param seed nova seed.
    */
   public void configurarSeed(long seed){
      throw new UnsupportedOperationException(
         "Implementar configuração de seed da camada " + this.getClass().getTypeName() + "."
      ); 
   }

   /**
    * Lógica para o processamento dos dados recebidos pela camada.
    * <p>
    *    Aqui as classes devem propagar os dados recebidos para
    *    as suas saídas.
    * </p>
    * O método deve levar em consideração o uso das funções de ativação
    * diretamente no seu processo de propagação.
    * @param entrada dados de entrada que poderão ser processados pela camada.
    */
   public abstract void calcularSaida(Object entrada);

   /**
    * Lógica para o cálculos dos gradientes de parâmetros treináveis dentro
    * da camada.
    * <p>
    *    Aqui as classes devem retropropagar os gradientes vindos da camada
    *    posterior, os usando para calcular seus próprios gradientes de parâmetros
    *    treinaveis (kernels, bias, etc).
    * </p>
    * O método deve levar em consideração o uso das funções de ativação
    * diretamente no seu processo de retropropagação.
    * @param gradSeguinte
    */
   public abstract void calcularGradiente(Object gradSeguinte);

   /**
    * Retorna a saída da camada.
    * @return saída da camada.
    */
   public abstract Object saida();

   /**
    * Retorna a função de ativação configurada pela camada.
    * @return função de ativação da camada.
    */
   public Ativacao obterAtivacao(){
      throw new UnsupportedOperationException(
         "Implementar retorno da função de ativação da camada " + this.getClass().getTypeName() + "."
      );
   }
   
   /**
    * Lógica para retornar o formato configurado de entrada da camada.
    * <p>
    *    Nele devem ser consideradas as dimensões dos dados de entrada da
    *    camada, que devem estar disposto como:
    * </p>
    * <pre>
    *    formato = (altura, largura, profundidade ...)
    * </pre>
    * @return array contendo os valores das dimensões de entrada da camada.
    */
   public int[] formatoEntrada(){
      throw new UnsupportedOperationException(
         "Implementar formato de entrada da camada " + this.getClass().getTypeName() + "."
      );
   }

   /**
    * Lógica para retornar o formato configurado de saída da camada.
    * <p>
    *    Nele devem ser consideradas as dimensões dos dados de saída da
    *    camada, que devem estar disposto como:
    * </p>
    * <pre>
    *    formato = (altura, largura, profundidade ...)
    * </pre>
    * @return array contendo os valores das dimensões de saída da camada.
    */
   public abstract int[] formatoSaida();

   /**
    * Retorna a saída da camada no formato de array.
    * @return saída da camada.
    */
   public double[] saidaParaArray(){
      throw new UnsupportedOperationException(
         "Implementar retorno de saída para array da camada" + this.getClass().getTypeName() + "."
      );   
   }

   /**
    * Retorna a quantidade total de elementos presentes na saída da camada.
    * @return tamanho de saída da camada.
    */
   public int tamanhoSaida(){
      throw new UnsupportedOperationException(
         "Implementar retorno de tamanho da saída da camada " + this.getClass().getTypeName() + "."
      );        
   }

   /**
    * Retorna a quantidade de parâmetros treináveis da camada.
    * <p>
    *    Esses parâmetros podem incluir pesos, filtros, bias, entre outros.
    * </p>
    * O resultado deve ser a quantidade total desses elementos.
    * @return número da parâmetros da camada.
    */
   public int numParametros(){
      throw new UnsupportedOperationException(
         "Implementar número de parâmetros da camada " + this.getClass().getTypeName() + "."
      );  
   }

   /**
    * Retorna o verificador de uso do bias dentro da camada.
    * @return uso de bias na camada.
    */
   public boolean temBias(){
      throw new UnsupportedOperationException(
         "Implementar uso do bias na camada " + this.getClass().getTypeName() + "."
      );  
   }

   /**
    * Retorna um array contendo os elementos do kernel presente na camada.
    * <p>
    *    O kernel de uma camada inclui seus atributos mais importantes, como
    *    os pesos de uma camada densa, ou os filtros de uma camada convolucional.
    * </p>
    * @return kernel da camada.
    */
   public double[] obterKernel(){
      throw new UnsupportedOperationException(
         "Implementar retorno do kernel da camada " + this.getClass().getTypeName() + "."
      );       
   }

   /**
    * Retorna um array contendo os elementos usados para armazenar o valor
    * dos gradientes para os kernels da camada.
    * @return gradientes para os kernels da camada.
    */
   public double[] obterGradKernel(){
      throw new UnsupportedOperationException(
         "Implementar retorno do gradiente para o kernel da camada" + this.getClass().getTypeName() + "."
      );       
   }

   /**
    * Retorna um array contendo os elementos acumuladores para o kernel presente na camada.
    * <p>
    *    O kernel de uma camada inclui seus atributos mais importantes, como
    *    os pesos de uma camada densa, ou os filtros de uma camada convolucional.
    * </p>
    * Necessário para o treino em lotes.
    * @return acumulador para o kernel da camada.
    */
   public double[] obterAcGradKernel(){
      throw new UnsupportedOperationException(
         "Implementar retorno do acumulador para o kernel da camada " + this.getClass().getTypeName() + "."
      );       
   }

   /**
    * Retorna um array contendo os elementos dos bias presente na camada.
    * <p>
    *    É importante verificar se a camada foi configurada para suportar
    *    os bias antes de usar os valores retornados por ela. Quando não
    *    configurados, os bias da camada são nulos.
    * </p>
    * @return bias da camada.
    */
   public double[] obterBias(){
      throw new UnsupportedOperationException(
         "Implementar retorno do bias da camada " + this.getClass().getTypeName() + "."
      );        
   }

   /**
    * Retorna um array contendo os elementos usados para armazenar o valor
    * dos gradientes para os bias da camada.
    * @return gradientes para os bias da camada.
    */
   public double[] obterGradBias(){
      throw new UnsupportedOperationException(
         "Implementar retorno do gradiente para o bias da camada" + this.getClass().getTypeName() + "."
      );        
   }

   /**
    * Retorna um array contendo os elementos usados para armazenar o valor
    * dos acumuladores para os bias da camada.
    * @return acumuladores para os bias da camada.
    */
   public double[] obterAcGradBias(){
      throw new UnsupportedOperationException(
         "Implementar retorno do acumulador para o bias da camada" + this.getClass().getTypeName() + "."
      );        
   }

   /**
    * Retorna o gradiente de entrada da camada, dependendo do tipo
    * de camada, esse gradiente pode assumir diferentes tipos de objetos.
    * @return gradiente de entrada da camada.
    */
   public Object obterGradEntrada(){
      throw new UnsupportedOperationException(
         "Implementar retorno do gradiente de entrada da camada " + this.getClass().getTypeName() + "."
      );     
   }

   /**
    * Ajusta os valores dos gradientes para o kernel usando os valores 
    * contidos no array fornecido.
    * @param grads novos valores de gradientes.
    */
   public void editarGradienteKernel(double[] grads){
      throw new UnsupportedOperationException(
         "Implementar edição do gradiente para o kernel para a camada " + this.getClass().getTypeName() + "."
      );
   }

   /**
    * Ajusta os valores do kernel usando os valores contidos no array
    * fornecido.
    * @param kernel novos valores do kernel.
    */
   public void editarKernel(double[] kernel){
      throw new UnsupportedOperationException(
         "Implementar edição do kernel para a camada " + this.getClass().getTypeName() + "."
      );
   }

   /**
    * Ajusta os valores do acumulador do kernel usando os valores contidos no array
    * fornecido.
    * @param acumulador novos valores do acumulador.
    */
   public void editarAcGradKernel(double[] acumulador){
      throw new UnsupportedOperationException(
         "Implementar edição do acumulador do kernel para a camada " + this.getClass().getTypeName() + "."
      );      
   }

   /**
    * Ajusta os valores dos gradientes para o bias usando os valores 
    * contidos no array fornecido.
    * @param grads novos valores de gradientes.
    */
   public void editarGradienteBias(double[] grads){
      throw new UnsupportedOperationException(
         "Implementar edição do gradiente para o bias para a camada " + this.getClass().getTypeName() + "."
      );
   }

   /**
    * Ajusta os valores do bias usando os valores contidos no array
    * fornecido.
    * @param bias novos valores do bias.
    */
   public void editarBias(double[] bias){
      throw new UnsupportedOperationException(
         "Implementar edição do bias para a camada " + this.getClass().getTypeName() + "."
      );
   }

   /**
    * Ajusta os valores do acumulador do bias usando os valores contidos no array
    * fornecido.
    * @param  acumulador valores do acumulador.
    */
   public void editarAcGradBias(double[] acumulador){
      throw new UnsupportedOperationException(
         "Implementar edição do acumulador do bias para a camada " + this.getClass().getTypeName() + "."
      );      
   }

   /**
    * Zera os acumudalores de gradientes para os kernels e bias da camada.
    */
   public void zerarAcumuladores(){
      throw new UnsupportedOperationException(
         "Implementar reset para acumuladores da camada " + this.getClass().getTypeName() + "."
      );
   }

   /**
    * Clona as características principais da camada.
    * @return clone da camada.
    */
   public Camada clonar(){
      throw new IllegalArgumentException(
         "Implementar clonagem para a camada " + this.getClass().getTypeName() + "." 
      );
   }

   /**
    * Retorna o nome da camada.
    * @return nome da camada.
    */
   public String nome(){
      return this.getClass().getSimpleName();
   }
}
