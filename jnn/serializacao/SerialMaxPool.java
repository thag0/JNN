package jnn.serializacao;

import java.io.BufferedReader;
import java.io.BufferedWriter;

import jnn.camadas.MaxPooling;

class SerialMaxPool{

   /**
    * Transforma os dados contidos na camada MaxPooling em
    * informações sequenciais. Essas informações contém:
    * <ul>
    *    <li> Nome da camada; </li>
    *    <li> Formato de entrada (altura, largura, profundidade); </li>
    *    <li> Formato de saída (altura, largura, profundidade); </li>
    *    <li> Formato do filtro (altura, largura); </li>
    *    <li> Formato dos strides (altura, largura); </li>
    * </ul>
    * @param camada camada de max pooling que será serializada.
    * @param bw escritor de buffer usado para salvar os dados da camada.
    */
   public void serializar(MaxPooling camada, BufferedWriter bw){
      try{
         //nome da camada pra facilitar
         bw.write(camada.nome());
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
         
         //formato do filtro
         int[] formFiltro = camada.formatoFiltro();
         for(int i = 0; i < formFiltro.length; i++){
            bw.write(formFiltro[i] + " ");
         }
         bw.newLine();
         
         //formato dos strides
         int[] formStride = camada.formatoStride();
         for(int i = 0; i < formStride.length; i++){
            bw.write(formStride[i] + " ");
         }
         bw.newLine();

      }catch(Exception e){
         e.printStackTrace();
      }
   }

   /**
    * Lê as informações da camada contida no arquivo.
    * @param br leitor de buffer.
    * @return instância de uma camada max pooling.
    */
   public MaxPooling lerConfig(BufferedReader br){
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

         //formato do filtro
         String[] sFiltro = br.readLine().split(" ");
         int[] filtro = new int[sFiltro.length];
         for(int i = 0; i < sFiltro.length; i++){
            filtro[i] = Integer.parseInt(sFiltro[i]);
         }

         //formato dos strides
         String[] sStrides = br.readLine().split(" ");
         int[] strides = new int[sStrides.length];
         for(int i = 0; i < sStrides.length; i++){
            strides[i] = Integer.parseInt(sStrides[i]);
         }

         MaxPooling camada = new MaxPooling(filtro, strides);
         camada.construir(entrada);
         return camada;
      }catch(Exception e){
         throw new RuntimeException(e);
      }
   }
}
