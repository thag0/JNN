package rna.serializacao;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;

import rna.avaliacao.perda.Perda;
import rna.camadas.Camada;
import rna.camadas.Convolucional;
import rna.camadas.Densa;
import rna.camadas.Dropout;
import rna.camadas.Flatten;
import rna.camadas.MaxPooling;
import rna.core.Dicionario;
import rna.modelos.RedeNeural;
import rna.modelos.Sequencial;
import rna.otimizadores.Otimizador;

/**
 * Classe responsável por tratar da serialização/desserialização de objetos
 * da {@Rede Neural}.
 * <p>
 *    Manipula os arquivos {@code .txt} baseados na rede para escrita e leitura, 
 *    possibilitando mais portabilidade de Redes Neurais treinadas.
 * </p>
 * Os pesos salvos são do tipo double (8 bytes), caso seja necessário mais economia
 * de memória pode ser recomendável converter os arquivos escritos para o tipo float 
 * (4 bytes).
 */
public class Serializador{

   /**
    * Auxiliar na serialização de camadas densas.
    */
   private SerialDensa auxDensa = new SerialDensa();
   
   /**
    * Auxiliar na serialização de camadas convolucionais.
    */
   private SerialConv auxConv = new SerialConv();

   /**
    * Auxiliar na serialização de camadas flatten.
    */
   private SerialFlatten auxFlat = new SerialFlatten();

   /**
    * Auxiliar na serialização de camadas max pooling.
    */
   private SerialMaxPool auxMaxPool = new SerialMaxPool();

   /**
    * Auxiliar na serialização de camadas de dropout.
    */
   private SerialDropout auxDropout = new SerialDropout();

   /**
    * Serializador e desserializador de modelos.
    */
   public Serializador(){}

   public void salvar(RedeNeural rede, String caminho){
      salvar(rede, caminho, "double");
   }

   /**
    * Salva as informações mais essenciais sobre a Rede Neural incluindo arquitetura,
    * funções de ativação de todas as camadas, bias configurado e o mais importante que
    * são os pesos de cada neurônio da rede.
    * <p>
    *    <strong> Reforçando</strong>: as informações sobre o otimizador e todas suas 
    *    configurações, treino, nome e outras pequenas coisas que não afetam diretamente 
    *    o funcionamento da rede serão perdidas.
    * </p>
    * <p>
    *    O arquivo deve ser salvo no formato {@code .txt}
    * </p>
    * @param rede instância de uma Rede Neural.
    * @param caminho caminho onde o arquivo da rede será salvo.
    * @param tipo classe contendo tipo de valor que será usado para salvar os pesos da Rede Neural.
    */
   public void salvar(RedeNeural rede, String caminho, String tipo){
      File arquivo = new File(caminho);
      if(!arquivo.getName().toLowerCase().endsWith(".txt")){
         throw new IllegalArgumentException("O caminho especificado não é um arquivo de texto válido.");
      }

      try(BufferedWriter writer = new BufferedWriter(new FileWriter(arquivo))){
         //arquitetura da rede
         int[] arq = rede.obterArquitetura();
         for(int i = 0; i < arq.length; i++){
            writer.write(arq[i] + " ");
         }
         writer.newLine();

         //bias
         writer.write(Boolean.toString(rede.temBias()));
         writer.newLine();

         //funções de ativação
         Densa[] camadas = rede.camadas();
         for(int i = 0; i < camadas.length; i++){
            writer.write(camadas[i].obterAtivacao().getClass().getSimpleName());
            writer.write(" ");
         }
         writer.newLine();

         //pesos dos neuronios
         for(Densa camada : rede.camadas()){
            auxDensa.serializar(camada, writer, tipo);
         }

      }catch(Exception e){
         System.out.println("Houve um erro ao salvar o arquivo da Rede Neural.");
         e.printStackTrace();
      }
   }

   /**
    * Salva um modelo Sequencial em um arquivo de texto no caminho especificado 
    * serializando as informações mais importantes do modelo, incluindo o número de 
    * camadas, otimizador usado, função de perda e os detalhes de cada camada no arquivo.
    * <p>
    *    Os valores salvos estarão no formato double.
    * </p>
    * @param modelo instância de um modelo sequencial.
    * @param caminho caminho com nome e extensão do arquivo {@code .txt}.
    * @throws IllegalArgumentException Se o caminho não termina com a extensão ".txt".
    */
   public void salvar(Sequencial modelo, String caminho){
      salvar(modelo, caminho, "double");
   }

   /**
    * Salva um modelo Sequencial em um arquivo de texto no caminho especificado 
    * serializando as informações mais importantes do modelo, incluindo o número de 
    * camadas, otimizador usado, função de perda e os detalhes de cada camada no arquivo.
    * @param modelo instância de um modelo sequencial.
    * @param caminho caminho com nome e extensão do arquivo (.txt).
    * @param tipo tipo de valor usado na serialização, exemplo: float ou double.
    * @throws IllegalArgumentException Se o caminho não termina com a extensão ".txt".
    */
   public void salvar(Sequencial modelo, String caminho, String tipo){
      File arquivo = new File(caminho);
      if(!arquivo.getName().toLowerCase().endsWith(".txt")){
         throw new IllegalArgumentException(
            "O caminho especificado não é um arquivo de texto válido."
         );
      }

      try(BufferedWriter bw = new BufferedWriter(new FileWriter(arquivo))){
         //quantidade de camadas
         bw.write(String.valueOf(modelo.numCamadas()));
         bw.newLine();

         //otimizador usado
         bw.write(modelo.otimizador().getClass().getSimpleName());
         bw.newLine();

         //função de perda
         bw.write(modelo.perda().getClass().getSimpleName());
         bw.newLine();

         for(Camada camada : modelo.camadas()){
            if(camada instanceof Densa){
               auxDensa.serializar((Densa) camada, bw, tipo);

            }else if(camada instanceof Convolucional){
               auxConv.serializar((Convolucional) camada, bw, tipo);
            
            }else if(camada instanceof Flatten){
               auxFlat.serializar((Flatten) camada, bw);
            
            }else if(camada instanceof MaxPooling){
               auxMaxPool.serializar((MaxPooling) camada, bw);

            }else if(camada instanceof Dropout){
               auxDropout.serializar((Dropout) camada, bw);

            }else{
               throw new IllegalArgumentException(
                  "Tipo de camada \"" + camada.getClass().getTypeName() + "\" não suportado."
               );
            }
         }
      
      }catch(Exception e){
         e.printStackTrace();
      }
   }

   /**
    * Lê o arquivo de uma {@code Rede Neural} serializada e converte numa
    * instância pré configurada.
    * <p>
    *    Configurações mantidas: 
    * </p> 
    * <ul>
    *    <li>
    *       Pesos de todos os neurônios da rede.
    *    </li>
    *    <li>
    *       Arquitetura.
    *    </li>
    *    <li>
    *       Funções de ativação de todas as camadas.
    *    </li>
    * </ul>
    * <strong>Demais configurações não são recuperadas</strong>.
    * @param caminho caminho onde está salvo o arquivo {@code .txt} da Rede Neural.
    * @return Instância de Rede Neural baseada nas configurações lidas pelo arquivo.
    */
   public RedeNeural lerRedeNeural(String caminho){
      RedeNeural rede = null;
      Dicionario dicionario = new Dicionario();

      try(BufferedReader br = new BufferedReader(new FileReader(caminho))){
         //arquitetura
         String[] arqStr = br.readLine().split(" ");
         int[] arq = new int[arqStr.length];

         try{
            for(int i = 0; i < arqStr.length; i++){
               arq[i] = Integer.parseInt(arqStr[i]);
            }
         }catch(Exception e){
            System.out.println("Ocorreu um erro ao tentar ler os valores de arquitetura");
            System.out.println("Verifique se estão corretamente formatados");
            System.out.println("Cada elemento de arquitetura deve ser separado por espaços");
            System.out.println("Ex: \"2 3 4\"");
            System.exit(0);
         }

         //bias
         boolean bias = Boolean.parseBoolean(br.readLine());

         //funções de ativação
         String[] ativacoesStr = br.readLine().split(" ");

         //inicialização e configurações da rede
         rede = new RedeNeural(arq);
         rede.configurarBias(bias);
         rede.compilar();

         for(int i = 0; i < rede.numCamadas(); i++){
            rede.configurarAtivacao(rede.camada(i), dicionario.obterAtivacao(ativacoesStr[i]));
         }

         for(int i = 0; i < rede.numCamadas(); i++){
            String nome = br.readLine();
            if(nome.equals("Densa")){
               br.readLine();//entrada
               br.readLine();//saida
               rede.camada(i).configurarAtivacao(br.readLine());
               rede.configurarBias(Boolean.valueOf(br.readLine()));
               auxDensa.lerPesos(rede.camada(i), br);
            }
         }

      }catch(Exception e){
         System.out.println("Houve um erro ao ler o arquivo de Rede Neural \""+ caminho + "\".");
         e.printStackTrace();
         System.exit(0);
      }

      return rede;
   }

   /**
    * Lê o arquivo de um modelo {@code Sequencial} serializado e converte numa
    * instância pré configurada.
    * @param caminho caminho onde está saldo o arquivo {@code .txt} do modelo;
    * @return instância de um modelo {@code Sequencial} a partir do arquivo lido.
    */
   public Sequencial lerSequencial(String caminho){
      Sequencial modelo = new Sequencial();
      Dicionario dic = new Dicionario();

      try(BufferedReader br = new BufferedReader(new FileReader(caminho))){
         int numCamadas = Integer.parseInt(br.readLine());
         Otimizador otimizador = dic.obterOtimizador(br.readLine().trim());
         Perda perda = dic.obterPerda(br.readLine().trim());
      
         modelo.configurarOtimizador(otimizador);
         modelo.configurarPerda(perda);
         for(int i = 0; i < numCamadas; i++){
            String nome = br.readLine();
            
            if(nome.equalsIgnoreCase("densa")){
               Densa densa = auxDensa.lerConfig(br);
               auxDensa.lerPesos(densa, br);
               modelo.add(densa);

            }else if(nome.equalsIgnoreCase("convolucional")){
               Convolucional convolucional = auxConv.lerConfig(br);
               auxConv.lerPesos(convolucional, br);
               modelo.add(convolucional);
            
            }else if(nome.equalsIgnoreCase("flatten")){
               Flatten flat = auxFlat.lerConfig(br);
               modelo.add(flat);
            
            }else if(nome.equalsIgnoreCase("maxpooling")){
               MaxPooling maxPooling = auxMaxPool.lerConfig(br);
               modelo.add(maxPooling);
            
            }else if(nome.equalsIgnoreCase("dropout")){
               Dropout dropout = auxDropout.lerConfig(br);
               modelo.add(dropout);

            }else{
               throw new IllegalArgumentException(
                  "Tipo de camada \""+ nome +"\" não suportado."
               );
            }
         }

         modelo.compilado = true;
         for(int i = 0; i < modelo.numCamadas(); i++){
            modelo.camada(i).configurarId(i);
         }
         otimizador.construir(modelo.camadas());

      }catch(Exception e){
         throw new RuntimeException(e);
      }

      return modelo;
   }
}
