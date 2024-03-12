package rna.modelos;

import rna.avaliacao.Avaliador;
import rna.avaliacao.perda.Perda;
import rna.camadas.Camada;
import rna.camadas.Entrada;
import rna.core.Dicionario;
import rna.otimizadores.Otimizador;
import rna.treinamento.Treinador;

//implementar serialização do modelo

/**
 * <h1>
 *    Modelo sequencial de camadas
 * </h1>
 * <p>
 *    Uma api simples para criação de modelos de redes neurais, funciona
 *    empilhando camadas em sequência que podem ser customizáveis.
 * </p>
 * <h2>
 *    Criação
 * </h2>
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
 * O modelo sequencial não é limitado apenas a camadas densas, podem empilhar camadas
 * compatívels com {@code rna.camadas}, algumas camadas dispoíveis incluem:
 * <ul>
 *    <li> Densa; </li>
 *    <li> Convolucional; </li>
 *    <li> MaxPooling; </li>
 *    <li> AvgPooling; </li>
 *    <li> Flatten; </li>
 *    <li> Dropout; </li>
 * </ul>
 * <p>
 *    Exemplo:
 * </p>
 * <pre>
 *modelo = Sequencial(new Camada[]{
 *    new Convolucional(new int[]{1, 28, 28}, new int[]{3, 3}, 5),
 *    new MaxPooling(new int[]{2, 2}),
 *    new Flatten(),
 *    new Densa(50)),
 *    new Dropout(0.3)),
 *    new Densa(10)),
 *});
 * </pre>
 * <h2>
 *    Compilação
 * </h2>
 * <p>
 *    Para poder usar o modelo é necessário compilá-lo, informando parâmetros 
 *    função de perda, otimizador e inicializador para os kernels (inicializador
 *    de bias é opcional).
 * </p>
 *    Exemplo:
 * <pre>
 *modelo.compilar("sgd", "mse");
 *modelo.compilar(new SGD(0.01, 0.9), "mse");
 * </pre>
 * <h2>
 *    Treinamento
 * </h2>
 * <p>
 *    Modelos sequenciais podem ser facilmente treinados usando o método {@code treinar},
 *    onde é apenas necessário informar os dados de entrada, saída e a quantidade de épocas 
 *    desejada para treinar. A entrada pode variar dependendo da primeira camada que for 
 *    adicionada ao modelo.
 * </p>
 * Exemplo:
 * <pre>
 *Object[] treinoX = ...; //dados de entrada
 *Object[] treinoY = ...; //dados de saída
 *int epochs = ... ; //iterações dentro do conjunto de dados
 *boolean logs = ...; //impressão de perda do modelo durante o treino.
 *modelo.treinar(treinoX, treinoY, epochs, logs);
 * </pre>
 * <h2>
 *    Serialização
 * </h2>
 * <p>
 *    Modelos sequenciais podem ser salvos em arquivos externos {@code .txt} para preservar
 *    suas configurações mais importantes, como otimizador, função de perda e mais importante
 *    ainda, as configurações de cada camada, isso inclue os valores para os kernels e bias
 *    contidos em cada camada treinável além dos formatos para entrada e saída específicos.
 * </p>
 * <p>
 *    Para salvar o modelo deve-se fazer uso da classe Serializador disponível em {@code rna.serializacao.Serializador}
 * </p>
 * Exemplo:
 * <pre>
 *Sequencial modelo = //modelo já configurado e compilado
 *Serializador s = new Serializador();
 *String caminho = "./modelo.txt";
 *s.salvar(modelo, caminho);
 * </pre>
 * <h2>
 *    Desserialização
 * </h2>
 * <p>
 *    Como esperado após salvar, também é possível ler um modelo sequencial a partir de um
 *    arquivo gerado pelo serializador. Esse arquivo deve ser compatível com as configurações
 *    salvas pelo serializador.
 * </p>
 * Exemplo:
 * <pre>
 *Serializador s = new Serializador();
 *String caminho = "./modelo.txt";
 *Sequencial modelo = s.lerSequencial(caminho);
 * </pre>
 * @author Thiago Barroso, acadêmico de Engenharia da Computação pela Universidade Federal do Pará, 
 * Campus Tucuruí. Dezembro/2023.
 */
public class Sequencial extends Modelo implements Cloneable{

   /**
    * Lista de camadas do modelo.
    */
   private Camada[] camadas;

   /**
    * Instancia um modelo sequencial com o conjunto de camadas vazio.
    * <p>
    *    É necessário especificar o formato de entrada da primeira camada
    *    do modelo.
    * </p>
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

      this.camadas = new Camada[0];
      add(camadas[0]);

      for(int i = 1; i < camadas.length; i++){
         if(camadas[i] instanceof Entrada == false){
            add(camadas[i]);
         }
      }

      compilado = false;
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
         throw new IllegalArgumentException("\nCamada fornecida é nula.");
      }

      Camada[] antigas = camadas;
      camadas = new Camada[antigas.length + 1];
      for(int i = 0; i < antigas.length; i++){
         camadas[i] = antigas[i];
      }

      camadas[camadas.length-1] = camada;

      compilado = false;
   }

   /**
    * Remove a última camada contida na lista de camadas do modelo.
    * @return camada removida.
    * @throws IllegalArgumentException caso o modelo já não possua nenhuma 
    * camada disponível.
    */
   public Camada sub(){
      if(camadas.length < 1){
         throw new IllegalArgumentException(
            "\nNão há camadas no modelo."
         );
      }
      Camada ultima = camadaSaida();

      Camada[] novas = camadas;
      camadas = new Camada[camadas.length-1];
      for(int i = 0; i < camadas.length; i++){
         camadas[i] = novas[i];
      }

      compilado = false;

      return ultima;
   }

   @Override
   public void compilar(Object otimizador, Object perda){
      int[] formato = {};

      if(camadas[0] instanceof Entrada){
         formato = camadas[0].formatoSaida();

         //remover camada de entrada do modelo
         Camada[] temp = camadas;
         camadas = new Camada[camadas.length-1];
         for(int i = 0; i < camadas.length; i++){
            camadas[i] = temp[i + 1];
         }
      
      }else{
         if(camadas[0].construida == false){
            throw new IllegalArgumentException(
               "\nÉ necessário que a primeira camada (" + camadas[0].nome() +
               ") seja construída."
            );
         }
      }

      if(camadas.length == 0){
         throw new IllegalStateException(
            "\nO modelo não possui camadas para compilar."
         );
      }

      camadas[0].construir(formato);
      if(seedInicial != 0) camadas[0].configurarSeed(seedInicial);
      camadas[0].inicializar();
      camadas[0].configurarId(0);

      for(int i = 1; i < this.camadas.length; i++){
         camadas[i].construir(camadas[i-1].formatoSaida());
         if(seedInicial != 0) camadas[i].configurarSeed(seedInicial);
         camadas[i].inicializar();
         camadas[i].configurarId(i);
      }
      
      Dicionario dicio = new Dicionario();
      this.perda = dicio.obterPerda(perda);
      this.otimizador = dicio.obterOtimizador(otimizador);

      this.otimizador.construir(camadas);
      
      compilado = true;//modelo pode ser usado.
   }

   @Override
   public void calcularSaida(Object entrada){
      verificarCompilacao();

      camadas[0].calcularSaida(entrada);
      for(int i = 1; i < camadas.length; i++){
         camadas[i].calcularSaida(camadas[i-1].saida());
      }
   }

   @Override
   public Object[] calcularSaidas(Object[] entradas){
      verificarCompilacao();

      //implementar uma melhor generalização futuramente
      double[][] previsoes = new double[entradas.length][];

      for(int i = 0; i < previsoes.length; i++){
         calcularSaida(entradas[i]);
         previsoes[i] = saidaParaArray().clone();
      }

      return previsoes;
   }
  
   @Override
   public void zerarGradientes(){
      for(int i = 0; i < camadas.length; i++){
         if(camadas[i].treinavel) camadas[i].zerarGradientes();
      }
   }

   @Override
   public Otimizador otimizador(){
      verificarCompilacao();
      return this.otimizador;
   }

   @Override
   public Perda perda(){
      verificarCompilacao();
      return this.perda;
   }

   @Override
   public Camada camada(int id){
      if((id < 0) || (id >= camadas.length)){
         throw new IllegalArgumentException(
            "O índice fornecido (" + id + 
            ") é inválido ou fora de alcance."
         );
      }
   
      return camadas[id];
   }

   @Override
   public Camada[] camadas(){
      return this.camadas;
   }

   @Override
   public Camada camadaSaida(){
      if(camadas.length == 0){
         throw new UnsupportedOperationException(
            "\nO modelo não possui camadas adiciondas."
         );
      }

      return camadas[camadas.length-1];
   }

   @Override
   public double[] saidaParaArray(){
      verificarCompilacao();
      return camadaSaida().saidaParaArray();
   }

   @Override
   public String nome(){
      return nome;
   }

   @Override
   public int numParametros(){
      int parametros = 0;
      for(Camada camada : camadas){
         parametros += camada.numParametros();
      }

      return parametros;
   }

   @Override
   public int numCamadas(){
      return camadas.length;
   }

   @Override
   protected String construirInfo(){
      String pad = " ".repeat(4);
      StringBuilder sb = new StringBuilder();
      sb.append(nome() + " = [\n");

      //otimizador
      sb.append(otimizador.info());
      sb.append("\n");

      //função de perda
      sb.append(pad + "Perda: " + perda.nome());
      sb.append("\n\n");

      //camadas
      sb.append(
         pad + String.format(
         "%-23s%-23s%-23s%-23s%-23s\n", "Camada", "Entrada", "Saída", "Ativação", "Parâmetros"
         )
      );

      for(Camada camada : this.camadas){
         int[] e = camada.formatoEntrada();
         int[] s = camada.formatoSaida();
         
         //identificador da camada
         String nomeCamada = camada.id + " - " + camada.nome();

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
         String ativacao = "n/a";
         try{
            ativacao = camada.ativacao().nome();
         }catch(Exception exception){}

         String parametros = String.valueOf(camada.numParametros());

         sb.append(
            pad + String.format(
               "%-23s%-23s%-23s%-23s%-23s\n", nomeCamada, formEntrada, formSaida, ativacao, parametros
            )
         );
      }

      String params = String.format("%,d", numParametros());
      sb.append("\n" + pad + "Parâmetros treináveis: " + params + "\n");
      sb.append("]\n");

      return sb.toString();
   }

   @Override
   public void info(){
      verificarCompilacao();
      System.out.println(construirInfo());
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
         throw new RuntimeException("\nErro ao clonar modelo: \n" + e);
      }
   }
}
