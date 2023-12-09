package rna.modelos;

import rna.avaliacao.Avaliador;
import rna.avaliacao.perda.ErroMedioQuadrado;
import rna.avaliacao.perda.Perda;
import rna.estrutura.Camada;
import rna.estrutura.Densa;
import rna.inicializadores.Inicializador;
import rna.otimizadores.Otimizador;
import rna.otimizadores.SGD;
import rna.treinamento.Treinador;

/**
 * Modelo sequencial de camadas.
 */
public class Sequencial extends Modelo{

   Treinador treinador = new Treinador();

   /**
    * Lista de camadas do modelo.
    */
   private Camada[] camadas;

   /**
    * Função de perda do modelo.
    */
   private Perda perda = new ErroMedioQuadrado();
   
   /**
    * Otimizador do modelo.
    */
   private Otimizador otimizador = new SGD();

   /**
    * 
    */
   public Avaliador avaliador = new Avaliador(this);

   /**
    * Auxiliar na verificação de compilação do modelo.
    */
   private boolean compilado = false;

   /**
    * Auxiliar na verificação para o salvamento do histórico
    * de perda do modelo durante o treinamento.
    */
   private boolean calcularHistorico = false;

   /**
    * Inicializa um modelo sequencial vazio.
    */
   public Sequencial(){
      this.camadas = new Camada[0];
      this.compilado = false;
   }

   /**
    * Inicializa um modelo sequencial a partir de um conjunto de camadas
    * definido
    * @param camadas camadas que serão usadas pelo modelo.
    */
   public Sequencial(Camada[] camadas){
      this.camadas = camadas;
      this.compilado = false;
   }

   /**
    * Adiciona uma nova camada ao final da lista de camadas do modelo.
    * <p>
    *    Novas camadas não precisam estar construídas, a única excessão
    *    é caso seja a primeira camada do modelo, ela deve ser construída
    *    já que é necessário saber o formato de entrada do modelo.
    * </p>
    * @param camada nova camada.
    */
   public void add(Camada camada){
      if(camada == null){
         throw new IllegalArgumentException("Camada fornecida é nula.");
      }

      Camada[] c = this.camadas;
      this.camadas = new Camada[c.length+1];

      for(int i = 0; i < c.length; i++){
         this.camadas[i] = c[i];
      }
      this.camadas[this.camadas.length-1] = camada;

      this.compilado = false;
   }

   /**
    * Apaga a última camada contida no modelo.
    * @throws IllegalArgumentException caso o modelo já não possua nenhuma 
    * camada disponível.
    */
   public void sub(){
      if(this.camadas.length == 0){
         throw new IllegalArgumentException(
            "Não há camadas no modelo."
         );
      }

      Camada[] c = this.camadas;
      this.camadas = new Camada[this.camadas.length-1];
      for(int i = 0; i < this.camadas.length; i++){
         this.camadas[i] = c[i];
      }
   }

   /**
    * 
    * @param calcular
    */
   public void configurarHistorico(boolean calcular){
      this.calcularHistorico = calcular;
      this.treinador.configurarHistoricoCusto(calcular);
   }

   /**
    * Configura a função de perda que será utilizada durante o processo
    * de treinamento da Rede Neural.
    * @param perda nova função de perda.
    */
   public void configurarPerda(Perda perda){
      if(perda == null){
         throw new IllegalArgumentException("A função de perda não pode ser nula.");
      }

      this.perda = perda;
   }

   /**
    * Configura o novo otimizador da Rede Neural com base numa nova instância de otimizador.
    * <p>
    *    Configurando o otimizador passando diretamente uma nova instância permite configurar
    *    os hiperparâmetros do otimizador fora dos valores padrão, o que pode ajudar a
    *    melhorar o desempenho de aprendizado da Rede Neural em cenário específicos.
    * </p>
    * Otimizadores disponíveis.
    * <ol>
    *    <li> GradientDescent  </li>
    *    <li> SGD (Gradiente Descendente Estocástico) </li>
    *    <li> AdaGrad </li>
    *    <li> RMSProp </li>
    *    <li> Adam  </li>
    *    <li> Nadam </li>
    *    <li> AMSGrad </li>
    *    <li> Adamax  </li>
    *    <li> Lion   </li>
    *    <li> Adadelta </li>
    * </ol>
    * <p>
    *    {@code O otimizador padrão é o SGD}
    * </p>
    * @param otimizador novo otimizador.
    * @throws IllegalArgumentException se o novo otimizador for nulo.
    */
   public void configurarOtimizador(Otimizador otimizador){
      if(otimizador == null){
         throw new IllegalArgumentException("O novo otimizador não pode ser nulo.");
      }
      this.otimizador = otimizador;
   }

   /**
    * Inicializa os parâmetros necessários para cada camada do modelo,
    * além de aleatorizar os kernels e bias.
    * @param otimizador otimizador usando para ajustar os parâmetros treinavéis do modelo.
    * @param perda função de perda usada para o treinamento do modelo.
    * @param iniKernel inicializador para os kernels.
    */
   public void compilar(Otimizador otimizador, Perda perda, Inicializador iniKernel){
      this.compilar(otimizador, perda, iniKernel, null);
   }

   /**
    * Inicializa os parâmetros necessários para cada camada do modelo,
    * além de aleatorizar os kernels e bias.
    * @param otimizador otimizador usado durante o treinamento do modelo para
    * ajustar seus parâmetros.
    * @param perda função de perda usada para o treinamento do modelo.
    * @param iniKernel inicializador para os kernels.
    * @param iniBias inicializador para os bias.
    */
   public void compilar(Otimizador otimizador, Perda perda, Inicializador iniKernel, Inicializador iniBias){
      if(this.camadas[0].construida == false){
         throw new IllegalArgumentException(
            "É necessário que a primeira camada seja construída."
         );
      }
      if(iniKernel == null){
         throw new IllegalArgumentException(
            "O inicializador para o kernel não pode ser nulo."
         );
      }

      for(int i = 1; i < this.camadas.length; i++){
         this.camadas[i].construir(this.camadas[i-1].formatoSaida());
      }

      for(int i = 0; i < this.camadas.length; i++){
         this.camadas[i].inicializar(iniKernel, iniBias, 0);
         this.camadas[i].configurarId(i);
      }

      this.perda = perda;

      this.otimizador.inicializar(this.camadas);
      this.compilado = true;
   }

   /**
    * Auxiliar na verificação da compilaçã do modelo.
    */
   private void verificarCompilacao(){
      if(this.compilado == false){
         throw new IllegalArgumentException("O modelo ainda não foi compilado.");
      }
   }

   /**
    * Propaga os dados de entrada pelo modelo.
    * @param entrada entrada.
    */
   @Override
   public void calcularSaida(Object entrada){
      verificarCompilacao();

      this.camadas[0].calcularSaida(entrada);
      for(int i = 1; i < this.camadas.length; i++){
         this.camadas[i].calcularSaida(this.camadas[i-1].saidaParaArray());
      }
   }

   /**
    * Propaga os dados de entrada pelo modelo.
    * @param entradas entrada.
    */
   @Override
   public Object[] calcularSaida(Object[] entradas){
      verificarCompilacao();

      double[][] previsoes = new double[entradas.length][];

      for(int i = 0; i < previsoes.length; i++){
         this.calcularSaida(entradas[i]);
         previsoes[i] = this.saidaParaArray().clone();
      }

      return previsoes;
   }

   /**
    * Treina o modelo de acordo com as configurações predefinidas.
    * <p>
    *    Certifique-se de configurar adequadamente o modelo para obter os 
    *    melhores resultados.
    * </p>
    * @param entradas dados de entrada do treino (features). Dependendo da entrada
    * do modelo, pode assumir diferentes formatos, para camadas convolucionais é
    * {@code double[][][][]}, para camadas densas é {@code double[][]}.
    * @param saidas dados de saída correspondente a entrada (class).
    * @param epochs quantidade de épocas de treinamento.
    * @param logs .
    * @throws IllegalArgumentException se o modelo não foi compilado previamente.
    * @throws IllegalArgumentException se houver alguma inconsistência dos dados de entrada e saída para a operação.
    * @throws IllegalArgumentException se o valor de épocas for menor que um.
    */
   @Override
   public void treinar(Object[] entradas, Object[] saidas, int epochs){
      verificarCompilacao();
      treinador.treino(this, entradas, saidas, epochs);
   }

   /**
    * Retorna a função de perda configurada do modelo.
    * @return função de perda atual do modelo.
    */
   @Override
   public Perda obterPerda(){
      return this.perda;
   }
 
    /**
     * Retorna o otimizador que está sendo usado para o treino do modelo.
     * @return otimizador atual do modelo.
     */
   @Override
   public Otimizador obterOtimizador(){
      return this.otimizador;
   }

   /**
    * Retorna a {@code camada} do Modelo correspondente ao índice fornecido.
    * @param id índice da busca.
    * @return camada baseada na busca.
    * @throws IllegalArgumentException se o índice estiver fora do alcance do tamanho 
    * das camadas.
    */
   @Override
   public Camada obterCamada(int id){
      verificarCompilacao();
   
      if((id < 0) || (id >= this.camadas.length)){
         throw new IllegalArgumentException(
            "O índice fornecido (" + id + 
            ") é inválido ou fora de alcance."
         );
      }
   
      return this.camadas[id];
   }

   /**
    * Retorna todo o conjunto de camadas presente no modelo.
    * @throws IllegalArgumentException se o modelo não foi compilado previamente.
    * @return conjunto de camadas do modelo.
    */
   @Override
   public Camada[] obterCamadas(){
      verificarCompilacao();
      return this.camadas;
   }

   /**
    * Retorna a {@code camada de saída} do modelo.
    * @return camada de saída.
    */
   @Override
   public Camada obterCamadaSaida(){
      this.verificarCompilacao();
      return this.camadas[this.camadas.length-1];
   }

   /**
    * Retorna um array contendo a saída do modelo.
    * @return saída do modelo.
    */
   @Override
   public double[] saidaParaArray(){
      verificarCompilacao();
      return this.obterCamadaSaida().saidaParaArray();
   }

   /**
    * Informa o nome configurado da Rede Neural.
    * @return nome específico da rede.
    */
   @Override
   public String obterNome(){
      return this.nome;
   }

   /**
    * Retorna a quantidade total de parâmetros do modelo.
    * <p>
    *    isso inclui todos os kernels e bias (caso configurados).
    * </p>
    * @return quantiade de parâmetros total do modelo.
    */
   @Override
   public int obterQuantidadeParametros(){
      int parametros = 0;
      for(Camada camada : this.camadas){
         parametros += camada.numParametros();
      }
      return parametros;
   }

   /**
    * Retorna a quantidade de camadas presente no modelo.
    * @return quantidade de camadas do modelo.
    * @throws IllegalArgumentException se o modelo não foi compilado previamente.
    */
   @Override
   public int obterQuantidadeCamadas(){
      this.verificarCompilacao();
      return this.camadas.length;
   }

   /**
    * Disponibiliza o histórico da função de perda do modelo durante cada época
    * de treinamento.
    * <p>
    *    O histórico será o do ultimo processo de treinamento usado, seja ele sequencial ou em
    *    lotes. Sendo assim, por exemplo, caso o treino seja em sua maioria feito pelo modo sequencial
    *    mas logo depois é usado o treino em lotes, o histórico retornado será o do treinamento em lote.
    * </p>
    * @return lista contendo o histórico de perdas durante o treinamento da rede.
    * @throws IllegalArgumentException se não foi habilitado previamente o cálculo do 
    * histórico de custos.
    */
    public double[] obterHistorico(){
      if(this.calcularHistorico){
         return this.treinador.obterHistorico();
      
      }else{
         throw new UnsupportedOperationException(
            "O histórico de treino do modelo deve ser configurado previamente."
         );
      }
   }

   /**
    * 
    * @return
    */
   public String info(){
      verificarCompilacao();

      String buffer = "";
      String espacamento = "   ";

      buffer += this.getClass().getSimpleName() + " = [\n";
      
      for(Camada camada : this.camadas){
         int[] entrada = camada.formatoEntrada();
         int[] saida = camada.formatoSaida();

         if(camada instanceof Densa){
            buffer += espacamento + " " + camada.getClass().getSimpleName() + ": ";
            buffer += "Entrada = (" + entrada[0] + ", " + entrada[1] + ")";
            buffer += " Saída = (" + saida[0] + ", " + saida[1] + ") ";
            buffer += "Ativação = " + camada.obterAtivacao().getClass().getSimpleName() + "\n";
         }
      }
      
      buffer += "]\n";

      return buffer;
   }
}
