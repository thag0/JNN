package rna.serializacao;

import java.io.BufferedReader;
import java.io.BufferedWriter;

import rna.camadas.Dropout;

class SerialDropout{
   
   /**
    * Transforma os dados contidos na camada Flatten numa sequência
    * de informações sequenciais. Essas informações contém:
    * <ul>
    *    <li> Nome da camada; </li>
    *    <li> Formato de entrada (altura, largura, profundidade); </li>
    * </ul>
    * @param camada camada flatten que será serializada.
    * @param bw escritor de buffer usado para salvar os dados da camada.
    */
   public void serializar(Dropout camada, BufferedWriter bw){
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
         
         //taxa
         double taxa = camada.taxa();
         bw.write(String.valueOf(taxa));
         bw.newLine();

      }catch(Exception e){
         e.printStackTrace();
      }
   }

   /**
    * Lê as informações da camada contida no arquivo.
    * @param br leitor de buffer.
    * @return instância de uma camada dropout.
    */
   public Dropout lerConfig(BufferedReader br){
      try{
         //formato de entrada
         String[] sEntrada = br.readLine().split(" ");
         int[] entrada = new int[sEntrada.length];
         for(int i = 0; i < sEntrada.length; i++){
            entrada[i] = Integer.parseInt(sEntrada[i]);
         }

         //taxa
         double taxa = Double.parseDouble(br.readLine());

         Dropout camada = new Dropout(taxa);
         camada.construir(entrada);
         return camada;
      }catch(Exception e){
         throw new RuntimeException(e);
      }
   }
}
