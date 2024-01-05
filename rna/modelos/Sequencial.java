package rna.modelos;

import rna.avaliacao.Avaliador;
import rna.avaliacao.perda.Perda;
import rna.estrutura.Camada;
import rna.inicializadores.Inicializador;
import rna.otimizadores.Otimizador;
import rna.serializacao.Dicionario;
import rna.treinamento.Treinador;

//implementar serialização do modelo

/**
 * <h3>
 *    Modelo sequencial de camadas.
 * </h3>
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
public class Sequencial extends Modelo implements Cloneable{

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
         this.camadas[i].inicializar(iniKernel, iniBias, 0.5);
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
    * Auxiliar na verificação da compilação do modelo.
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
         this.camadas[i].calcularSaida(this.camadas[i-1].saida());
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
   public void treinar(Object[] entradas, Object[] saidas, int epochs, boolean logs){
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

      treinador.treino(this, entradas, saidas, epochs, logs);
   }
   
   @Override
   public void treinar(Object[] entradas, Object[] saidas, int epochs, int tamLote){
     this.verificarCompilacao();

     if(epochs < 1){
        throw new IllegalArgumentException(
           "O valor de epochs (" + epochs + ") não pode ser menor que um"
        );
     }
     if(tamLote <= 0 || tamLote > entradas.length){
        throw new IllegalArgumentException(
           "O valor de tamanho do lote (" + tamLote + ") é inválido."
        );
     }

     this.treinador.treino(
        this,
        entradas,
        saidas,
        epochs,
        tamLote
     );
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
   public void copiarDaSaida(double[] arr){
      double[] saida = this.saidaParaArray();
      if(saida.length != arr.length){
         throw new IllegalArgumentException(
            "Incompatibilidade de dimensões entre o array fornecido (" + arr.length + 
            ") e o array gerado pela saída da última camada (" + saida.length + ")."
         );
      }

      for(int i = 0; i < saida.length; i++){
         arr[i] = saida[i];
      }
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

      String espacamento = "    ";
      String buffer = "";
      buffer += this.nome + " = [\n";

      buffer += otimizador.info();
      buffer += "\n";

      buffer += espacamento + "Perda: " + this.perda.nome();
      buffer += "\n\n";

      buffer += espacamento + String.format(
         "%-23s%-23s%-23s%-23s\n", "Camada", "Formato de Entrada", "Formato de Saída", "Função de Ativação"
      );

      for(Camada camada : this.camadas){
         int[] e = camada.formatoEntrada();
         int[] s = camada.formatoSaida();
         
         //nome
         String nomeCamada = camada.id + " - " + camada.getClass().getSimpleName();

         //formato de entrada
         String formEntrada = String.format("(%d", e[0]);
         for(int i = 1; i < e.length; i++){
            formEntrada += String.format(", %d", e[i]);
         }
         formEntrada += ")";

         //formato de saída
         String formSaida = String.format("(%d", s[0]);
         for(int i = 1; i < s.length; i++){
            formSaida += String.format(", %d", s[i]);
         }
         formSaida += ")";

         //função de ativação
         String ativacao = "";
         try{
            ativacao = camada.obterAtivacao().nome();
         }catch(Exception exception){
            ativacao = "n/a";
         }

         buffer += espacamento + String.format(
            "%-23s%-23s%-23s%-23s\n", nomeCamada, formEntrada, formSaida, "Ativação = " + ativacao
         );
      }

      buffer += "\n" + espacamento + "Parâmetros treináveis: " + this.numParametros() + "\n";

      buffer += "]\n";

      return buffer;
   }

   @Override
   public Sequencial clonar(){
      try{
         Sequencial clone = (Sequencial) super.clone();

         clone.avaliador = new Avaliador(this);
         clone.calcularHistorico = this.calcularHistorico;
         clone.nome = "Clone de " + this.nome;
         
         Dicionario dic = new Dicionario();
         clone.otimizador = dic.obterOtimizador(this.otimizador.getClass().getSimpleName());
         clone.perda = dic.obterPerda(this.perda.getClass().getSimpleName());
         clone.seedInicial = this.seedInicial;
         clone.treinador = new Treinador();
         
         clone.camadas = new Camada[this.numCamadas()];
         for(int i = 0; i < this.camadas.length; i++){
            clone.camadas[i] = this.camada(i).clonar();
         }
         clone.compilado = this.compilado;

         return clone;
      }catch(Exception e){
         throw new RuntimeException(e);
      }
   }
}
