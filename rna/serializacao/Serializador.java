package rna.serializacao;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;

import rna.estrutura.Densa;
import rna.modelos.RedeNeural;

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
   public static void salvar(RedeNeural rede, String caminho){
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
   public static void salvar(RedeNeural rede, String caminho, Object tipo){
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
   public static void salvar(RedeNeural rede, String caminho, Class<?> tipo){
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
         Densa[] camadas = rede.obterCamadas();
         for(int i = 0; i < camadas.length; i++){
            writer.write(camadas[i].obterAtivacao().getClass().getSimpleName());
            writer.write(" ");
         }
         writer.newLine();

         //pesos dos neuronios
         for(Densa camada : rede.obterCamadas()){

            for(int i = 0; i < camada.pesos.lin; i++){
               for(int j = 0; j < camada.pesos.col; j++){
                  double peso = camada.pesos.dado(i, j);
                  
                  if(tipo.equals(Double.TYPE)){
                     writer.write(Double.toString(peso));
                  
                  }else if(tipo.equals(Float.TYPE)){
                     writer.write(Float.toString((float)peso));
                  
                  }else if(tipo.equals(Integer.TYPE)){
                     writer.write(Float.toString((int)peso));
                     
                  }else if(tipo.equals(Short.TYPE)){
                     writer.write(Float.toString((short)peso));

                  }else if(tipo.equals(Byte.TYPE)){
                     writer.write(Float.toString((byte)peso));
                  }

                  writer.newLine();
               }
            }

            if(camada.temBias()){
               for(int i = 0; i < camada.bias.col; i++){
                  double bias = camada.bias.dado(0, i);

                  if(tipo.equals(Double.TYPE)){
                     writer.write(Double.toString(bias));
                  
                  }else if(tipo.equals(Float.TYPE)){
                     writer.write(Float.toString((float)bias));
                  
                  }else if(tipo.equals(Integer.TYPE)){
                     writer.write(Float.toString((int)bias));
                     
                  }else if(tipo.equals(Short.TYPE)){
                     writer.write(Float.toString((short)bias));

                  }else if(tipo.equals(Byte.TYPE)){
                     writer.write(Float.toString((byte)bias));
                  }

                  writer.newLine();
               }
            }
 
         }

      }catch(Exception e){
         System.out.println("Houve um erro ao salvar o arquivo da Rede Neural.");
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
   public static RedeNeural ler(String caminho){
      RedeNeural rede = null;
      DicionarioAtivacoes dicionario = new DicionarioAtivacoes();

      try(BufferedReader reader = new BufferedReader(new FileReader(caminho))){
         //arquitetura
         String[] arqStr = reader.readLine().split(" ");
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
         boolean bias = Boolean.parseBoolean(reader.readLine());

         //funções de ativação
         String[] ativacoesStr = reader.readLine().split(" ");

         //inicialização e configurações da rede
         rede = new RedeNeural(arq);
         rede.configurarBias(bias);
         rede.compilar();

         for(int i = 0; i < rede.obterQuantidadeCamadas(); i++){
            rede.configurarAtivacao(rede.obterCamada(i), dicionario.obterAtivacao(ativacoesStr[i]));
         }

         int cont = 1;
         try{
            for(Densa camada : rede.obterCamadas()){
               for(int i = 0; i < camada.pesos.lin; i++){
                  for(int j = 0; j < camada.pesos.col; j++){
                     double d = Double.parseDouble(reader.readLine());
                     camada.pesos.editar(i, j, d);
                     cont++;
                  }
               }

               if(camada.temBias()){
                  for(int i = 0; i < camada.bias.col; i++){
                     double d = Double.parseDouble(reader.readLine());
                     camada.bias.editar(0, i, d);
                     cont++;
                  }
               }
            }
         }catch(Exception e){
            System.out.println("Ocorreu um erro ao ler o peso " + cont);
            e.printStackTrace();
            System.exit(0);
         }

      }catch(Exception e){
         System.out.println("Houve um erro ao ler o arquivo de Rede Neural \""+ caminho + "\".");
         e.printStackTrace();
         System.exit(0);
      }

      return rede;
   }
}
