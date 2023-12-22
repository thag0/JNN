package rna.serializacao;

import java.io.BufferedReader;
import java.io.BufferedWriter;

import rna.estrutura.Densa;

class SerialDensa{

   /**
    * Transforma os dados contidos na camada Densa numa sequência
    * de informações sequenciais. Essas informações contém:
    * <ul>
    *    <li> Nome da camada; </li>
    *    <li> Formato de entrada (altura, largura); </li>
    *    <li> Formato de saída (altura, largura); </li>
    *    <li> Função de ativação configurada; </li>
    *    <li> Uso de bias; </li>
    *    <li> Valores dos pesos; </li>
    *    <li> Valores dos bias (se houver); </li>
    * </ul>
    * @param camada camada densa que será serializada.
    * @param bw escritor de buffer usado para salvar os dados da camada.
    */
   public void serializar(Densa camada, BufferedWriter bw){
      try{
         //nome da camada pra facilitar
         bw.write(camada.getClass().getSimpleName());
         bw.newLine();

         //formato de entrada
         int[] entrada = camada.formatoEntrada();
         for(int i = 0; i < entrada.length; i++){
            bw.write(entrada[i] + " ");
         }
         bw.newLine();
         
         //formato de saída
         int[] saida = camada.formatoSaida();
         for(int i = 0; i < saida.length; i++){
            bw.write(saida[i] + " ");
         }
         bw.newLine();
         
         //função de ativação
         bw.write(String.valueOf(camada.obterAtivacao().getClass().getSimpleName()));
         bw.newLine();

         //bias
         bw.write(String.valueOf(camada.temBias()));
         bw.newLine();
         
         for(int i = 0; i < camada.pesos.lin(); i++){
            for(int j = 0; j < camada.pesos.col(); j++){
               double peso = camada.pesos.dado(i, j);
               bw.write(String.valueOf(peso));
               bw.newLine();
            }
         }
         
         if(camada.temBias()){
            for(int i = 0; i < camada.bias.lin(); i++){
               for(int j = 0; j < camada.bias.col(); j++){
                  double bias = camada.bias.dado(i, j);
                  bw.write(String.valueOf(bias));
                  bw.newLine();
               }
            }
         }
      }catch(Exception e){
         e.printStackTrace();
      }
   }

   /**
    * Lê as informações da camada contida no arquivo.
    * @param br leitor de buffer.
    * @return instância de uma camada densa, os valores de
    * pesos e bias ainda não são inicializados.
    */
   public Densa lerConfig(BufferedReader br){
      try{
         //formato de entrada
         String[] sEntrada = br.readLine().split(" ");
         int[] entrada = new int[sEntrada.length];
         for(int i = 0; i < sEntrada.length; i++){
            entrada[i] = Integer.parseInt(sEntrada[i]);
         }

         //formato de saída
         String[] sSaida = br.readLine().split(" ");
         int[] saida = new int[sSaida.length];
         for(int i = 0; i < sSaida.length; i++){
            saida[i] = Integer.parseInt(sSaida[i]);
         }
         
         //função de ativação
         String ativacao = br.readLine();

         //bias
         boolean bias = Boolean.valueOf(br.readLine());
         
         Densa camada = new Densa(saida[1], ativacao);
         camada.configurarBias(bias);
         camada.construir(entrada);
         return camada;
      }catch(Exception e){
         throw new RuntimeException(e);
      }
   }

   /**
    * Lê os valores dos pesos e bias para a camada.
    * @param camada camada densa que será editada.
    * @param br leitor de buffer.
    */
   public void lerPesos(Densa camada, BufferedReader br){
      try{         
         for(int i = 0; i < camada.pesos.lin(); i++){
            for(int j = 0; j < camada.pesos.col(); j++){
               double p = Double.parseDouble(br.readLine());
               camada.pesos.editar(i, j, p);
            }
         }
         
         if(camada.temBias()){
            for(int i = 0; i < camada.bias.lin(); i++){
               for(int j = 0; j < camada.bias.col(); j++){
                  double b = Double.parseDouble(br.readLine());
                  camada.bias.editar(i, j, b);
               }
            }
         }
      }catch(Exception e){
         throw new RuntimeException(e);
      }
   }
}
