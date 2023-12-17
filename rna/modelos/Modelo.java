package rna.modelos;

import rna.avaliacao.Avaliador;
import rna.avaliacao.perda.Perda;
import rna.estrutura.Camada;
import rna.inicializadores.Inicializador;
import rna.otimizadores.Otimizador;
import rna.treinamento.Treinador;

/**
 * Base para os modelos dentro da biblioteca.
 */
public class Modelo{

   /**
    * Nome personalizado do modelo.
    */
   protected String nome = this.getClass().getSimpleName();

   /**
    * Auxiliar no controle da compilação do modelo, ajuda a evitar uso 
    * indevido caso ainda não tenha suas variáveis e dependências inicializadas 
    * previamente.
    */
   protected boolean compilado;

   /**
    * Função de perda usada durante o processo de treinamento.
    */
   protected Perda perda;

    /**
     * Otimizador que será utilizado durante o processo de aprendizagem da
     * da Rede Neural.
     */
   protected Otimizador otimizador;

   /**
    * Ponto inicial para os geradores aleatórios.
    * <p>
    *    0 não é considerado uma seed e ela só é configurada quando esse valor é
    *    alterado.
    * </p>
    */
   protected long seedInicial = 0;

   /**
    * Gerenciador de treino do modelo. contém implementações dos 
    * algoritmos de treino para o ajuste de parâmetros treináveis.
    */
   protected Treinador treinador = new Treinador();

   /**
    * Responsável pelo retorno de desempenho da Rede Neural.
    * Contém implementações de métodos tanto para cálculo de perdas
    * quanto de métricas.
    * <p>
    *    Cada instância de rede neural possui seu próprio avaliador.
    * </p>
    */
   public Avaliador avaliador = new Avaliador(this);
   
   /**
    * Inicializa um modelo vazio, sem implementações de métodos.
    * <p>
    *    Para modelos mais completos, use {@code RedeNeural} ou {@code Sequencial}.
    * </p>
    */
   public Modelo(){

   }

   /**
    * <p>
    *    Altera o nome do modelo.
    * </p>
    * O nome é apenas estético e não influencia na performance ou na 
    * usabilidade do modelo.
    * <p>
    *    O nome padrão é o mesmo nome da classe.
    * </p>
    * @param nome novo nome da rede.
    * @throws IllegalArgumentException se o novo nome for uulo ou inválido.
    */
   public void configurarNome(String nome){
      if(nome == null){
         throw new IllegalArgumentException("O novo nome não pode ser nulo.");
      }
      if(nome.isBlank() || nome.isEmpty()){
         throw new IllegalArgumentException("O novo nome não pode estar vazio.");
      }

      this.nome = nome;
   }

   /**
    * Configura a nova seed inicial para os geradores de números aleatórios utilizados 
    * durante o processo de inicialização de parâmetros treináveis do modelo.
    * <p>
    *    Configurações personalizadas de seed permitem fazer testes com diferentes
    *    parâmetros, buscando encontrar um melhor ajuste para o modelo.
    * </p>
    * <p>
    *    A configuração de seed deve ser feita antes da compilação do modelo.
    * </p>
    * @param seed nova seed.
    */
   public void configurarSeed(long seed){
      this.seedInicial = seed;
   }

   /**
    * Define se durante o processo de treinamento, o modelo vai salvar dados relacionados a 
    * função de custo/perda de cada época.
    * <p>
    *    Calcular a perda é uma operação que pode ser computacionalmente cara dependendo do 
    *    tamanho da rede e do conjunto de dados, então deve ser bem avaliado querer habilitar 
    *    ou não esse recurso.
    * </p>
    * <p>
    *    {@code O valor padrão é false}
    * </p>
    * @param calcular se verdadeiro, o modelo armazenará o histórico de perda durante cada época.
    */
   public void configurarHistorico(boolean calcular){
      this.treinador.configurarHistoricoCusto(calcular);
   }

   /**
    * Configura a função de perda que será utilizada durante o processo
    * de treinamento do modelo.
    * @param perda nova função de perda.
    */
   public void configurarPerda(Perda perda){
      if(perda == null){
         throw new IllegalArgumentException("A função de perda não pode ser nula.");
      }

      this.perda = perda;
   }

   /**
    * Inicializa os parâmetros necessários para cada camada do modelo,
    * além de gerar os valores iniciais para os kernels e bias.
    * @param otimizador otimizador usando para ajustar os parâmetros treinavéis do modelo.
    * @param perda função de perda usada para o treinamento do modelo.
    * @param iniKernel inicializador para os kernels.
    */
   public void compilar(Otimizador otimizador, Perda perda, Inicializador iniKernel){
      throw new IllegalArgumentException(
         "Implementar compilação do modelo."
      );
   }

   /**
    * Inicializa os parâmetros necessários para cada camada do modelo,
    * além de gerar os valores iniciais para os kernels e bias.
    * <p>
    *    Caso nenhuma configuração inicial seja feita, o modelo será inicializado com os argumentos padrões. 
    * </p>
    * <p>
    *    Para treinar o modelo deve-se fazer uso da função função {@code treinar()} informando os 
    *    dados necessários para a rede.
    * </p>
    * @param otimizador otimizador usando para ajustar os parâmetros treinavéis do modelo.
    * @param perda função de perda usada para o treinamento do modelo.
    * @param iniKernel inicializador para os kernels.
    * @param iniBias inicializador para os bias.
    */
   public void compilar(Otimizador otimizador, Perda perda, Inicializador iniKernel, Inicializador iniBias){
      throw new IllegalArgumentException(
         "Implementar compilação do modelo."
      );
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
    */
   public void treinar(Object[] entradas, Object[] saidas, int epochs){
      throw new IllegalArgumentException(
         "Implementar treinamento para o modelo."
      );
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
    * @param tamLote tamanho do lote de treinamento.
    */
   public void treinar(Object[] entradas, Object[] saidas, int epochs, int tamLote){
      throw new IllegalArgumentException(
         "Implementar treinamento em lotes para o modelo."
      );
   }

   /**
    * Propaga os dados de entrada pelo modelo.
    * @param entrada entrada.
    */
   public void calcularSaida(Object entrada){
      throw new IllegalArgumentException(
         "Implementar calculo de saída do modelo."
      );
   }

   /**
    * Propaga os dados de entrada pelo modelo.
    * @param entradas array contendo multiplas entradas.
    * @return array contendo as saídas correspondentes.
    */
   public Object[] calcularSaidas(Object[] entradas){
      throw new IllegalArgumentException(
         "Implementar calculo de saída do modelo para múltiplas entradas."
      );
   }

   /**
    * Retorna o otimizador que está sendo usado para o treino do modelo.
    * @return otimizador atual do modelo.
    */
   public Otimizador obterOtimizador(){
     throw new IllegalArgumentException(
         "Implementar retorno do otimizador do modelo."
      );       
   }

   /**
    * Retorna a função de perda configurada do modelo.
    * @return função de perda atual do modelo.
    */
   public Perda obterPerda(){
     throw new IllegalArgumentException(
         "Implementar retorno da função de perda do modelo."
      );       
   }

   /**
    * Retorna a {@code camada} do Modelo correspondente ao índice fornecido.
    * @param id índice da busca.
    * @return camada baseada na busca.
    */
   public Camada obterCamada(int id){
      throw new IllegalArgumentException(
         "Implementar retorno de camada baseado em índice do modelo."
      ); 
   }

   /**
    * Retorna todo o conjunto de camadas presente no modelo.
    * @return conjunto de camadas do modelo.
    */
   public Camada[] obterCamadas(){
      throw new IllegalArgumentException(
         "Implementar retorno das camadas do modelo."
      ); 
   }

   /**
    * Retorna a {@code camada de saída}, ou última camada, do modelo.
    * @return camada de saída.
    */
   public Camada obterCamadaSaida(){
      throw new IllegalArgumentException(
         "Implementar retorno da camada de saída do modelo."
      ); 
   }
   
   /**
    * Retorna um array contendo a saída serializada do modelo.
    * @return saída do modelo.
    */
   public double[] saidaParaArray(){      
      throw new IllegalArgumentException(
         "Implementar retorno de saída para array do modelo."
      ); 
   }

   /**
    * Informa o nome configurado do modelo.
    * @return nome do modelo.
    */
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
   public int obterQuantidadeParametros(){
      throw new IllegalArgumentException(
         "Implementar retorno da quantidade de parâmetros do modelo."
      );     
   }

   /**
    * Retorna a quantidade de camadas presente no modelo.
    * @return quantidade de camadas do modelo.
    */
   public int obterQuantidadeCamadas(){
      throw new IllegalArgumentException(
         "Implementar retorno da quantidade de camadas do modelo."
      ); 
   }

   /**
    * Disponibiliza o histórico da função de perda do modelo durante cada época
    * de treinamento.
    * <p>
    *    O histórico será o do ultimo processo de treinamento usado, seja ele sequencial ou em
    *    lotes. Sendo assim, por exemplo, caso o treino seja em sua maioria feito pelo modo sequencial
    *    mas logo depois é usado o treino em lotes, o histórico retornado será o do treinamento em lote.
    * </p>
    * @return array contendo o valor de perda durante cada época de treinamento do modelo.
    */
   public double[] obterHistorico(){
      throw new IllegalArgumentException(
         "Implementar retorno do histórico de perdas do modelo."
      ); 
   }

   /**
    * Mostra as informações sobre o modelo.
    * @return buffer formatado contendo as informações do modelo.
    */
   public String info(){
      return "Modelo base.";
   }
}
