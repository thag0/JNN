package rna.serializacao;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;

import rna.avaliacao.perda.Perda;
import rna.estrutura.Camada;
import rna.estrutura.Densa;
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

   private UtilsDensa auxDensa = new UtilsDensa();

   public Serializador(){}

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
    * O tipo de dado para salvar os pesos será o tipo padrão {@code Double}.
    * @param rede instância de uma Rede Neural.
    * @param caminho caminho onde o arquivo da rede será salvo.
    */
   public void salvar(RedeNeural rede, String caminho){
      salvar(rede, caminho, Double.TYPE);
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
    * @param tipo tipo de valor que será usado para salvar os pesos da Rede Neural. O tipo 
    * pode ser um objeto do tipo {@code String} contendo o nome (por exemplo "float") ou uma
    * instância de objeto do tipo {@code Number}.
    */
   public void salvar(RedeNeural rede, String caminho, Object tipo){
      if(tipo instanceof String){
         String t = (String) tipo;
         
         if(t.toLowerCase().equals("double")){
            salvar(rede, caminho, Double.TYPE);

         }else if(t.toLowerCase().endsWith("float")){
            salvar(rede, caminho, Float.TYPE);
         
         }else if(t.toLowerCase().endsWith("int") || t.toLowerCase().endsWith("integer")){
            salvar(rede, caminho, Integer.TYPE);
         
         }else if(t.toLowerCase().endsWith("short")){
            salvar(rede, caminho, Short.TYPE);
         
         }else if(t.toLowerCase().endsWith("byte")){
            salvar(rede, caminho, Byte.TYPE);
         }
      
      }else{
         throw new IllegalArgumentException("Tipo \"" + tipo.getClass().getSimpleName() + "\" não suportado.");
      }
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
   public void salvar(RedeNeural rede, String caminho, Class<?> tipo){
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
            auxDensa.serializar(camada, writer);
         }

      }catch(Exception e){
         System.out.println("Houve um erro ao salvar o arquivo da Rede Neural.");
         e.printStackTrace();
      }
   }

   public void salvar(Sequencial modelo, String caminho){
      File arquivo = new File(caminho);
      if(!arquivo.getName().toLowerCase().endsWith(".txt")){
         throw new IllegalArgumentException("O caminho especificado não é um arquivo de texto válido.");
      }

      try(BufferedWriter bw = new BufferedWriter(new FileWriter(arquivo))){
         //quantidade de camadas
         bw.write(String.valueOf(modelo.numCamadas()));
         bw.newLine();

         //otimizador usado
         bw.write(modelo.otimizador().getClass().getSimpleName());
         bw.newLine();

         //
         bw.write(modelo.perda().getClass().getSimpleName());
         bw.newLine();

         for(Camada camada : modelo.camadas()){
            if(camada instanceof Densa){
               auxDensa.serializar((Densa)camada, bw);
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
    * Lê o arquivo de uma {@code Rede Neural} serializada e converter numa
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
            if(nome.equals("Densa")){
               Densa densa = auxDensa.lerConfig(br);
               auxDensa.lerPesos(densa, br);
               modelo.add(densa);
            }
         }

         modelo.compilado = true;

      }catch(Exception e){
         throw new RuntimeException(e);
      }

      return modelo;
   }
}
