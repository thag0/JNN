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
         int[] formFiltro = camada.formatoFiltro();
         for(int i = 0; i < formFiltro.length; i++){
            bw.write(formFiltro[i] + " ");
         }
         bw.newLine();
         
         //função de ativação
         bw.write(String.valueOf(camada.ativacao().getClass().getSimpleName()));
         bw.newLine();

         //bias
         bw.write(String.valueOf(camada.temBias()));
         bw.newLine();

         //filtros
         for(int i = 0; i < camada.filtros.dim1(); i++){
            for(int j = 0; j < camada.filtros.dim2(); j++){
               for(int k = 0; k < camada.filtros.dim3(); k++){
                  for(int l = 0; l < camada.filtros.dim4(); l++){
                     escreverDado(camada.filtros.get(i, j, k, l), tipo, bw);
                     bw.newLine();
                  }
               }
            }
         }
         
         if(camada.temBias()){
            double[] bias = camada.bias.paraArray();
            for(double valor : bias){
               escreverDado(valor, tipo, bw);
               bw.newLine();               
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
         

         int numFiltros = saida[0];

         Convolucional camada = new Convolucional(formFiltro, numFiltros);
         camada.setAtivacao(ativacao);
         camada.setBias(bias);
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
         int tamKernel = camada.filtros.tamanho();
         double[] arrKernel = new double[tamKernel];

         for(int i = 0; i < tamKernel; i++){
            arrKernel[i] = Double.parseDouble(br.readLine());
         }

         camada.editarKernel(arrKernel);
         
         if(camada.temBias()){
            int tamBias = camada.bias.tamanho();
            double[] arrBias = new double[tamBias];

            for(int i = 0; i < tamBias; i++){
               arrBias[i] = Double.parseDouble(br.readLine());
            }
            
            camada.editarBias(arrBias);
         }
      }catch(Exception e){
         throw new RuntimeException(e);
      }
   }
}
