package rna.modelos;

import rna.avaliacao.perda.Perda;
import rna.estrutura.Camada;
import rna.estrutura.Convolucional;
import rna.estrutura.Densa;
import rna.estrutura.Flatten;
import rna.inicializadores.Inicializador;
import rna.otimizadores.Otimizador;

//TODO implementar serialização do modelo

/**
 * Modelo sequencial de camadas.
 * <p>
 *    Funciona usando empilhamento de camadas para dar mais flexibilidade
 *    na construção de modelos.
 * </P>
 * <p>
 *    Para qualquer modelo novo, é sempre necessário informar o formato
 *    de entrada da primeira camada contida nele.
 * </p>
 * <p>
 *    Exemplo de criação de modelos:
 * </p>
 * <pre>
 *modelo = Sequencial();
 *modelo.add(new Densa(2, 3));
 *modelo.add(new Densa(2));
 * </pre>
 * Ou se preferir
 * <pre>
 *modelo = Sequencial(new Camada[]{
 *    new Densa(2, 3)),
 *    new Densa(2))
 *});
 * </pre>
 * <p>
 *    Para poder usar o modelo é necessário compilá-lo, informando parâmetros 
 *    função de perda, otimizador e inicializador para os kernels (inicializador
 *    de bias é opcional).
 * </p>
 *    Exemplo:
 * <pre>
 * modelo.compilar(new SGD(), new ErroMedioQuadrado(), new Xavier());
 * </pre>
 * O modelo sequencial não é limitado apenas a camadas densas, modelos 
 * de camadas convolucionais e de achatamento (flatten) também são suportados 
 * (mas ainda estão em testes).
 * <p>
 *    Exemplo:
 * </p>
 * <pre>
 *modelo = Sequencial(new Camada[]{
 *    new Convolucional(new int[]{28, 28, 1}, new int[]{3, 3}, 5),
 *    new Flatten(),
 *    new Densa(50)),
 *    new Densa(10)),
 *});
 * </pre>
 * No exemplo acima é criada uma camada convolucional com formato de entrada 
 * (28, 28, 1), o formato de entrada para as camadas convolucionais segue o 
 * formato (altura, largura, profundidade)
 * <p>
 *    Modelos sequenciais podem ser facilmente treinados usando o método {@code treinar},
 *    onde é apenas necessário informar os dados de entrada, saída e a quantidade de épocas 
 *    desejada para treinar. A entrada pode variar dependendo da primeira camada que for 
 *    adicionada ao modelo.
 * </p>
 * @author Thiago Barroso, acadêmico de Engenharia da Computação pela Universidade Federal do Pará, 
 * Campus Tucuruí. Dezembro/2023.
 */
public class Sequencial extends Modelo{

   /**
    * Lista de camadas do modelo.
    */
   private Camada[] camadas;

   /**
    * Auxiliar na verificação para o salvamento do histórico
    * de perda do modelo durante o treinamento.
    */
   private boolean calcularHistorico = false;

   /**
    * Inicializa um modelo sequencial vazio.
    * <p>
    *    As camadas do modelo deverão ser adicionadas manualmente
    *    usando o método {@code add()}.
    * </p>
    */
   public Sequencial(){
      this.camadas = new Camada[0];
      this.compilado = false;
   }

   /**
    * Inicializa um modelo sequencial a partir de um conjunto de camadas
    * definido
    * <p>
    *    É necessário especificar o formato de entrada da primeira camada
    *    do modelo.
    * </p>
    * @param camadas camadas que serão usadas pelo modelo.
    * @throws IllegalArgumentException caso o conjunto de camadas seja nulo
    * ou alguma camada contida seja.
    */
   public Sequencial(Camada[] camadas){
      if(camadas == null){
         throw new IllegalArgumentException(
            "O conjunto de camadas fornecido é nulo."
         );
      }
      for(int i = 0; i < camadas.length; i++){
         if(camadas[i] == null){
            throw new IllegalArgumentException(
               "O conjunto de camadas fornecido possui uma camada nula, id = " + i
            );
         }
      }

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
    * Ao adicionar novas camadas, o modelo precisará ser compilado novamente.
    * @param camada nova camada.
    * @throws IllegalArgumentException se a camada fornecida for nula,
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

   @Override
   public void configurarHistorico(boolean calcular){
      this.calcularHistorico = calcular;
      this.treinador.configurarHistoricoCusto(calcular);
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

   @Override
   public void compilar(Otimizador otimizador, Perda perda, Inicializador iniKernel){
      this.compilar(otimizador, perda, iniKernel, null);
   }

   @Override
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

      if(seedInicial != 0){
         iniKernel.configurarSeed(seedInicial);
         if(iniBias != null){
            iniBias.configurarSeed(seedInicial);
         } 
         this.treinador.configurarSeed(seedInicial);
      }
      
      for(int i = 1; i < this.camadas.length; i++){
         this.camadas[i].construir(this.camadas[i-1].formatoSaida());
         this.camadas[i].inicializar(iniKernel, iniBias, 1);
         this.camadas[i].configurarId(i);
      }

      if(perda == null){
         throw new IllegalArgumentException(
            "A função de perda não pode ser nula."
         );
      }
      this.perda = perda;

      if(otimizador == null){
         throw new IllegalArgumentException(
            "O otimizador não pode ser nulo,"
         );
      }
      this.otimizador = otimizador;
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

   @Override
   public void calcularSaida(Object entrada){
      verificarCompilacao();

      this.camadas[0].calcularSaida(entrada);
      for(int i = 1; i < this.camadas.length; i++){
         this.camadas[i].calcularSaida(this.camadas[i-1].saidaParaArray());
      }
   }

   @Override
   public Object[] calcularSaidas(Object[] entradas){
      verificarCompilacao();

      double[][] previsoes = new double[entradas.length][];

      for(int i = 0; i < previsoes.length; i++){
         this.calcularSaida(entradas[i]);
         previsoes[i] = this.saidaParaArray().clone();
      }

      return previsoes;
   }

   @Override
   public void treinar(Object[] entradas, Object[] saidas, int epochs){
      verificarCompilacao();
      
      if(entradas.length != saidas.length){
         throw new IllegalArgumentException(
            "Incompatibilidade na quantidade de amostras de entrada (" + entradas.length + ")" +
            "e saídas (" + saidas.length + ")."
         );
      }

      if(epochs < 1){
         throw new IllegalArgumentException(
            "O valor de épocas deve ser maior que zero, recebido = " + epochs
         );
      }

      treinador.treino(this, entradas, saidas, epochs);
   }
   
   @Override
   public Otimizador otimizador(){
      return this.otimizador;
   }

   @Override
   public Perda perda(){
      return this.perda;
   }

   @Override
   public Camada camada(int id){
      verificarCompilacao();
   
      if((id < 0) || (id >= this.camadas.length)){
         throw new IllegalArgumentException(
            "O índice fornecido (" + id + 
            ") é inválido ou fora de alcance."
         );
      }
   
      return this.camadas[id];
   }

   @Override
   public Camada[] camadas(){
      verificarCompilacao();
      return this.camadas;
   }

   @Override
   public Camada camadaSaida(){
      this.verificarCompilacao();
      return this.camadas[this.camadas.length-1];
   }

   @Override
   public double[] saidaParaArray(){
      verificarCompilacao();
      return this.camadaSaida().saidaParaArray();
   }

   @Override
   public String nome(){
      return this.nome;
   }

   @Override
   public int numParametros(){
      int parametros = 0;
      for(Camada camada : this.camadas){
         parametros += camada.numParametros();
      }
      return parametros;
   }

   @Override
   public int numCamadas(){
      this.verificarCompilacao();
      return this.camadas.length;
   }

   @Override
   public double[] historico(){
      if(this.calcularHistorico){
         return this.treinador.obterHistorico();
      
      }else{
         throw new UnsupportedOperationException(
            "O histórico de treino do modelo deve ser configurado previamente."
         );
      }
   }

   @Override
   public String info(){
      verificarCompilacao();

      String buffer = "";
      String espacamento = "    ";
      String espacamaentoDuplo = espacamento + espacamento;

      buffer += this.nome + " = [\n";
      
      for(Camada camada : this.camadas){
         int[] entrada = camada.formatoEntrada();
         int[] saida = camada.formatoSaida();

         buffer += espacamento + camada.id + ": " + camada.getClass().getSimpleName() + " = [\n";
         if(camada instanceof Densa){
            buffer += espacamaentoDuplo + "Entrada (" + entrada[1] + ") Saída (" + saida[1] + ")\n";
            buffer += espacamaentoDuplo + "Ativação = " + camada.obterAtivacao().getClass().getSimpleName() + "\n";
            
         }else if(camada instanceof Convolucional){
            buffer += espacamaentoDuplo + "Entrada = (" + entrada[0] + ", " + entrada[1] + ", " + entrada[2] +  ") ";
            buffer += "Saída = (" + saida[0] + ", " + saida[1] + ", " + saida[2] + ") \n";
            buffer += espacamaentoDuplo + "Ativação = " + camada.obterAtivacao().getClass().getSimpleName() + "\n";
            
         }if(camada instanceof Flatten){
            buffer += espacamaentoDuplo + "Entrada = (" + entrada[0] + ", " + entrada[1] + ", " + entrada[2] +") ";
            buffer += "Saída = (" + saida[0] + ", " + saida[1] + ") \n";

         }
         buffer += espacamaentoDuplo + "Parâmetros: " + camada.numParametros() + "\n";
         buffer += espacamento + "]\n";
      }
      
      buffer += "]\n";

      return buffer;
   }
}
