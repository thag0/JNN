package rna.serializacao;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;

import rna.camadas.Convolucional;

/**
 * Utilitário usado para serialização e desserialização de camadas Convolucionais.
 */
class SerialConv{

   public SerialConv(){}

   /**
    * Transforma os dados contidos na camada Convolucional numa sequência
    * de informações sequenciais. Essas informações contém:
    * <ul>
    *    <li> Nome da camada; </li>
    *    <li> Formato de entrada (altura, largura, profundidade); </li>
    *    <li> Formato de saída (altura, largura, profundidade); </li>
    *    <li> Função de ativação configurada; </li>
    *    <li> Uso de bias; </li>
    *    <li> Valores dos filtros; </li>
    *    <li> Valores dos bias (se houver); </li>
    * </ul>
    * @param camada camada convolucional que será serializada.
    * @param bw escritor de buffer usado para salvar os dados da camada.
    */
   public void serializar(Convolucional camada, BufferedWriter bw, String tipo){
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
         
         //formato dos filtros
         int[] filtros = camada.formatoFiltro();
         for(int i = 0; i < filtros.length; i++){
            bw.write(filtros[i] + " ");
         }
         bw.newLine();
         
         //função de ativação
         bw.write(String.valueOf(camada.obterAtivacao().getClass().getSimpleName()));
         bw.newLine();

         //bias
         bw.write(String.valueOf(camada.temBias()));
         bw.newLine();

         //filtros
         for(int i = 0; i < camada.filtros.length; i++){
            for(int j = 0; j < camada.filtros[i].length; j++){
               for(int k = 0; k < camada.filtros[i][j].lin(); k++){
                  for(int l = 0; l < camada.filtros[i][j].col(); l++){
                     escreverDado(camada.filtros[i][j].elemento(k, l), tipo, bw);
                     bw.newLine();
                  }
               }
            }
         }
         
         if(camada.temBias()){
            for(int i = 0; i < camada.bias.length; i++){
               for(int j = 0; j < camada.bias[i].lin(); j++){
                  for(int k = 0; k < camada.bias[i].col(); k++){
                     escreverDado(camada.bias[i].elemento(j, k), tipo, bw);
                     bw.newLine();
                  }
               }
            }
         }
      }catch(Exception e){
         e.printStackTrace();
      }
   }

   /**
    * Salva o valor de acordo com a configuração de tipo definida.
    * @param valor valor desejado.
    * @param tipo formatação do dado (float, double).
    * @param bw escritor de buffer usado.
    * @throws IOException
    */
   private void escreverDado(double valor, String tipo, BufferedWriter bw) throws IOException{
      tipo = tipo.toLowerCase();
      switch(tipo){
         case "float":
            bw.write(String.valueOf((float) valor));
         break;

         case "double":
            bw.write(String.valueOf(valor));
         break;
            
         default:
            throw new IllegalArgumentException("Tipo de dado (" + tipo + ") não suportado");
      }
   }

   /**
    * Lê as informações da camada contida no arquivo.
    * @param br leitor de buffer.
    * @return instância de uma camada convolucional, os valores de
    * filtros e bias ainda não são inicializados.
    */
   public Convolucional lerConfig(BufferedReader br){
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

         //formato dos filtros
         String[] sFiltros = br.readLine().split(" ");
         int[] formFiltro = new int[sFiltros.length];
         for(int i = 0; i < sFiltros.length; i++){
            formFiltro[i] = Integer.parseInt(sFiltros[i]);
         }
         
         //função de ativação
         String ativacao = br.readLine();

         //bias
         boolean bias = Boolean.valueOf(br.readLine());
         

         int numFiltros = saida[2];

         Convolucional camada = new Convolucional(formFiltro, numFiltros);
         camada.configurarAtivacao(ativacao);
         camada.configurarBias(bias);
         camada.construir(entrada);

         return camada;
      }catch(Exception e){
         throw new RuntimeException(e);
      }
   }

   /**
    * Lê os valores dos filtros e bias para a camada.
    * @param camada camada convolucional que será editada.
    * @param br leitor de buffer.
    */
   public void lerPesos(Convolucional camada, BufferedReader br){
      try{
         for(int i = 0; i < camada.filtros.length; i++){
            for(int j = 0; j < camada.filtros[i].length; j++){
               for(int k = 0; k < camada.filtros[i][j].lin(); k++){
                  for(int l = 0; l < camada.filtros[i][j].col(); l++){
                     double filtro = Double.parseDouble(br.readLine());
                     camada.filtros[i][j].editar(k, l, filtro);
                  }
               }
            }
         }
         
         if(camada.temBias()){
            for(int i = 0; i < camada.bias.length; i++){
               for(int j = 0; j < camada.bias[i].lin(); j++){
                  for(int k = 0; k < camada.bias[i].col(); k++){
                     double bias = Double.parseDouble(br.readLine());
                     camada.bias[i].editar(j, k, bias);
                  }
               }
            }
         }
      }catch(Exception e){
         throw new RuntimeException(e);
      }
   }
}
