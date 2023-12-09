package rna.modelos;

import rna.avaliacao.perda.Perda;
import rna.estrutura.Camada;
import rna.inicializadores.Inicializador;
import rna.otimizadores.Otimizador;

/**
 * Base para os modelos dentro da biblioteca.
 */
public class Modelo{

   protected String nome = "Modelo";
   
   public Modelo(){

   }

   public void compilar(Perda perda, Otimizador otimizador, Inicializador iniPesos, Inicializador iniBias){
      throw new IllegalArgumentException(
         "Implementar compilação do modelo."
      );
   }

   public void treinar(Object[] entradas, Object[] saidas, int epochs){
      throw new IllegalArgumentException(
         "Implementar treinamento para o modelo."
      );
   }
   
   public void treinar(Object[] entradas, Object[] saidas, int epochs, int tamLote){
      throw new IllegalArgumentException(
         "Implementar treinamento em lotes para o modelo."
      );
   }

   public void calcularSaida(Object entrada){
      throw new IllegalArgumentException(
         "Implementar calculo de saída do modelo."
      );
   }

   public Object[] calcularSaidas(Object[] entradas){
      throw new IllegalArgumentException(
         "Implementar calculo de saída do modelo para múltiplas entradas."
      );
   }

   public Otimizador obterOtimizador(){
     throw new IllegalArgumentException(
         "Implementar retorno do otimizador do modelo."
      );       
   }

   public Perda obterPerda(){
     throw new IllegalArgumentException(
         "Implementar retorno da função de perda do modelo."
      );       
   }

   public Camada obterCamada(int id){
      throw new IllegalArgumentException(
         "Implementar retorno de camada baseado em índice do modelo."
      ); 
   }

   public Camada obterCamadaSaida(){
      throw new IllegalArgumentException(
         "Implementar retorno da camada de saída do modelo."
      ); 
   }

   public Camada[] obterCamadas(){
      throw new IllegalArgumentException(
         "Implementar retorno das camadas do modelo."
      ); 
   }
   
   public double[] saidaParaArray(){      
      throw new IllegalArgumentException(
         "Implementar retorno de saída para array do modelo."
      ); 
   }

   public String obterNome(){
      return this.nome;
   }

   public int obterQuantidadeParametros(){
      throw new IllegalArgumentException(
         "Implementar retorno da quantidade de parâmetros do modelo."
      );     
   }

   public int obterQuantidadeCamadas(){
      throw new IllegalArgumentException(
         "Implementar retorno da quantidade de camadas do modelo."
      ); 
   }

   public double[] obterHistorico(){
      throw new IllegalArgumentException(
         "Implementar retorno do histórico de perdas do modelo."
      ); 
   }

   public String info(){
      return "Modelo base.";
   }
}
