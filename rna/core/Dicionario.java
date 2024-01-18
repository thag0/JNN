package rna.core;

import rna.ativacoes.ArcTan;
import rna.ativacoes.Argmax;
import rna.ativacoes.ELU;
import rna.ativacoes.Ativacao;
import rna.ativacoes.GELU;
import rna.ativacoes.LeakyReLU;
import rna.ativacoes.Linear;
import rna.ativacoes.ReLU;
import rna.ativacoes.Seno;
import rna.ativacoes.Sigmoid;
import rna.ativacoes.SoftPlus;
import rna.ativacoes.Softmax;
import rna.ativacoes.Swish;
import rna.ativacoes.TanH;
import rna.avaliacao.perda.EntropiaCruzada;
import rna.avaliacao.perda.EntropiaCruzadaBinaria;
import rna.avaliacao.perda.MAE;
import rna.avaliacao.perda.MSE;
import rna.avaliacao.perda.MSLE;
import rna.avaliacao.perda.Perda;
import rna.avaliacao.perda.RMSE;
import rna.inicializadores.Aleatorio;
import rna.inicializadores.AleatorioPositivo;
import rna.inicializadores.Constante;
import rna.inicializadores.Gaussiano;
import rna.inicializadores.Glorot;
import rna.inicializadores.He;
import rna.inicializadores.Identidade;
import rna.inicializadores.Inicializador;
import rna.inicializadores.LeCun;
import rna.inicializadores.Xavier;
import rna.inicializadores.Zeros;
import rna.otimizadores.AMSGrad;
import rna.otimizadores.AdaGrad;
import rna.otimizadores.Adadelta;
import rna.otimizadores.Adam;
import rna.otimizadores.GD;
import rna.otimizadores.Nadam;
import rna.otimizadores.Otimizador;
import rna.otimizadores.RMSProp;
import rna.otimizadores.SGD;

/**
 * Tradutor para as funções de ativação, otimizadores e funções de pperda 
 * e inicializadores.
 */
public class Dicionario{

   /**
    * Tradutor para as funções de ativação, otimizadores, funções de perda 
    * e inicializadores.
    */
   public Dicionario(){}

   /**
    * Converte a ativação lida em uma instância de função
    * de ativação correspondente.
    * @param ativacao tipo função de ativação.
    * @return instância da função de ativação lida.
    */
   public Ativacao obterAtivacao(Object ativacao){
      if(ativacao == null){
         throw new IllegalArgumentException(
            "Ativação não pode ser nula."
         );
      
      }else if(ativacao instanceof Ativacao){
         return (Ativacao) ativacao;
      
      }else if(ativacao instanceof String){
         String nome = (String) ativacao;
         switch(nome.toLowerCase()){
            case "argmax"     : return new Argmax();
            case "elu"        : return new ELU();
            case "gelu"       : return new GELU();
            case "leakyrelu"  : return new LeakyReLU();
            case "linear"     : return new Linear();
            case "relu"       : return new ReLU();
            case "seno"       : return new Seno();
            case "sigmoid"    : return new Sigmoid();
            case "softmax"    : return new Softmax();
            case "softplus"   : return new SoftPlus();
            case "swish"      : return new Swish();
            case "tanh"       : return new TanH();
            case "arctan"     : return new ArcTan();
   
            default: throw new IllegalArgumentException(
               "Ativação \"" + ativacao + "\" não encontada."
            );
         }

      }else{
         throw new IllegalArgumentException(
            "Tipo de dado \"" + ativacao.getClass().getTypeName() + "\" não suportado."
         );
      }
   }

   /**
    * Converte o otimizador lido em uma instância de 
    * otimizador correspondente.
    * @param otimizador tipode de otimizador.
    * @return instância do otimizador lido.
    */
   public Otimizador obterOtimizador(Object otimizador){
      if(otimizador == null){
         throw new IllegalArgumentException(
            "Otimizador não pode ser nulo."
         );
      }else if(otimizador instanceof Otimizador){
         return (Otimizador) otimizador;

      }else if(otimizador instanceof String){
         String nome = (String) otimizador;
         switch(nome.toLowerCase()){
            case "adadelta":  return new Adadelta();
            case "adagrad":   return new AdaGrad();
            case "adam":      return new Adam();
            case "amsgrad":   return new AMSGrad();
            case "gd":        return new GD();
            case "nadam":     return new Nadam();
            case "rmsprop":   return new RMSProp();
            case "sgd":       return new SGD();
   
            default: throw new IllegalArgumentException(
               "Otimizador \"" + nome + "\" não encontado."
            );
         }

      }else{
         throw new IllegalArgumentException(
            "Tipo de dado \"" + otimizador.getClass().getTypeName() + "\" não suportado."
         );
      }
   }

   /**
    * Converte a função de perda recebida em uma instância de função
    * de perda.
    * @param perda tipo de função de perda.
    * @return instância da função de perda lida.
    */
   public Perda obterPerda(Object perda){
      if(perda == null){
         throw new IllegalArgumentException(
            "Função de perda não pode ser nula."
         );
      }else if(perda instanceof Perda){
         return (Perda) perda;

      }else if(perda instanceof String){
         String nome = (String) perda;
         switch(nome.toLowerCase()){
            case "mse"                    : return new MSE();
            case "mae"                    : return new MAE();
            case "msle"                   : return new MSLE();
            case "rmse"                   : return new RMSE();
            case "entropiacruzada"        : return new EntropiaCruzada();
            case "entropiacruzadabinaria" : return new EntropiaCruzadaBinaria();
   
            default: throw new IllegalArgumentException(
               "Função de perda \"" + perda + "\" não encontada."
            );
         }           
      }else{
         throw new IllegalArgumentException(
            "Tipo de dado \"" + perda.getClass().getTypeName() + "\" não suportado."
         );
      }
    
   }

   /**
    * Converte o inicializador recebido em uma instância de inicializador.
    * @param inicializador tipo de inicializador.
    * @return instância do inicializador lido.
    */
   public Inicializador obterInicializador(Object inicializador){
      if(inicializador == null){
         throw new IllegalArgumentException(
            "Inicializador não pode ser nulo."
         );

      }else if(inicializador instanceof Inicializador){
         return (Inicializador) inicializador;
      
      }else if(inicializador instanceof String){
         String nome = (String) inicializador;
         switch(nome.toLowerCase()){
            case "aleatorio"           : return new Aleatorio();
            case "aleatoriopositivo"   : return new AleatorioPositivo();
            case "constante"           : return new Constante();
            case "gaussiano"           : return new Gaussiano();
            case "glorot"              : return new Glorot();
            case "he"                  : return new He();
            case "identidade"          : return new Identidade();
            case "lecun"               : return new LeCun();
            case "xavier"              : return new Xavier();
            case "zeros"               : return new Zeros();
            
            default: throw new IllegalArgumentException(
               "Inicializador \"" + inicializador + "\" não encontado."
            );
         }
      
      }else{
         throw new IllegalArgumentException(
            "Tipo de dado \"" + inicializador.getClass().getTypeName() + "\" não suportado."
         );
      }

   }
}
