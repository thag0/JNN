package rna.serializacao;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;

import rna.camadas.Densa;
import rna.core.Tensor4D;

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
   public void serializar(Densa camada, BufferedWriter bw, String tipo){
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
         
         //função de ativação
         bw.write(String.valueOf(camada.ativacao().getClass().getSimpleName()));
         bw.newLine();

         //bias
         bw.write(String.valueOf(camada.temBias()));
         bw.newLine();
         
         double[] pesos = camada.kernelParaArray();
         for(int i = 0; i < pesos.length; i++){
            escreverDado(pesos[i], tipo, bw);
            bw.newLine();
         }

         if(camada.temBias()){
            double[] bias = camada.biasParaArray();
            for(int i = 0; i < bias.length; i++){
               escreverDado(bias[i], tipo, bw);
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
         camada.setBias(bias);
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
         Tensor4D pesos = camada.kernel();
         int linPesos = pesos.dim3();
         int colPesos = pesos.dim4();        
         for(int i = 0; i < linPesos; i++){
            for(int j = 0; j < colPesos; j++){
               double p = Double.parseDouble(br.readLine());
               camada.kernel().set(p, 0, 0, i, j);
            }
         }
         
         if(camada.temBias()){
            int colBias = camada.bias().dim4();        
            for(int i = 0; i < colBias; i++){
               double b = Double.parseDouble(br.readLine());
               camada.bias().set(b, 0, 0, 0, i);
            }
         }
      }catch(Exception e){
         throw new RuntimeException(e);
      }
   }
}
