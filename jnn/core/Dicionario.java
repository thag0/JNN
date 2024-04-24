package jnn.core;

import jnn.ativacoes.Argmax;
import jnn.ativacoes.Atan;
import jnn.ativacoes.Ativacao;
import jnn.ativacoes.ELU;
import jnn.ativacoes.GELU;
import jnn.ativacoes.LeakyReLU;
import jnn.ativacoes.Linear;
import jnn.ativacoes.ReLU;
import jnn.ativacoes.SELU;
import jnn.ativacoes.Seno;
import jnn.ativacoes.Sigmoid;
import jnn.ativacoes.SoftPlus;
import jnn.ativacoes.Softmax;
import jnn.ativacoes.Swish;
import jnn.ativacoes.TanH;
import jnn.avaliacao.perda.EntropiaCruzada;
import jnn.avaliacao.perda.EntropiaCruzadaBinaria;
import jnn.avaliacao.perda.MAE;
import jnn.avaliacao.perda.MSE;
import jnn.avaliacao.perda.MSLE;
import jnn.avaliacao.perda.Perda;
import jnn.avaliacao.perda.RMSE;
import jnn.inicializadores.Aleatorio;
import jnn.inicializadores.AleatorioPositivo;
import jnn.inicializadores.Constante;
import jnn.inicializadores.Gaussiano;
import jnn.inicializadores.GlorotNormal;
import jnn.inicializadores.GlorotUniforme;
import jnn.inicializadores.He;
import jnn.inicializadores.Identidade;
import jnn.inicializadores.Inicializador;
import jnn.inicializadores.LeCun;
import jnn.inicializadores.Zeros;
import jnn.otimizadores.AMSGrad;
import jnn.otimizadores.AdaGrad;
import jnn.otimizadores.Adadelta;
import jnn.otimizadores.Adam;
import jnn.otimizadores.GD;
import jnn.otimizadores.Nadam;
import jnn.otimizadores.Otimizador;
import jnn.otimizadores.RMSProp;
import jnn.otimizadores.SGD;

/**
 * Tradutor para as funções de ativação, otimizadores e funções de pperda 
 * e inicializadores.
 */
public class Dicionario {

   /**
    * Tradutor para as funções de ativação, otimizadores, funções de perda 
    * e inicializadores.
    */
   public Dicionario() {}

   /**
    * Remove caracteres especiais.
    * @param nome nome de alguma instância que o dicionário lê.
    * @return valor do nome tratado.
    */
   private String tratarNome(String nome) {
      nome = nome.trim();
      nome = nome.replace("_", "");
      nome = nome.replace("-", "");
      nome = nome.replace(".", "");

      return nome;
   }

   /**
    * Converte a ativação lida em uma instância de função
    * de ativação correspondente.
    * @param ativacao tipo função de ativação.
    * @return instância da função de ativação lida.
    */
   public Ativacao getAtivacao(Object ativacao) {
      if (ativacao == null) {
         throw new IllegalArgumentException(
            "Ativação não pode ser nula."
         );
      
      } else if (ativacao instanceof Ativacao) {
         return (Ativacao) ativacao;
      
      } else if (ativacao instanceof String) {
         String nome = (String) ativacao;
         nome = tratarNome(nome);
         switch (nome.toLowerCase()) {
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
            case "atan"       : return new Atan();
            case "selu"       : return new SELU();
   
            default: throw new IllegalArgumentException(
               "Ativação \"" + ativacao + "\" não encontada."
            );
         }

      } else {
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
   public Otimizador getOtimizador(Object otimizador) {
      if (otimizador == null) {
         throw new IllegalArgumentException(
            "Otimizador não pode ser nulo."
         );

      } else if (otimizador instanceof Otimizador) {
         return (Otimizador) otimizador;

      } else if (otimizador instanceof String) {
         String nome = (String) otimizador;
         nome = tratarNome(nome);
         switch (nome.toLowerCase()){
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

      } else {
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
   public Perda getPerda(Object perda) {
      if (perda == null) {
         throw new IllegalArgumentException(
            "Função de perda não pode ser nula."
         );

      } else if (perda instanceof Perda) {
         return (Perda) perda;

      } else if (perda instanceof String) {
         String nome = (String) perda;
         nome = tratarNome(nome);
         switch (nome.toLowerCase()) {
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
      } else {
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
   public Inicializador getInicializador(Object inicializador) {
      if (inicializador == null) {
         throw new IllegalArgumentException(
            "Inicializador não pode ser nulo."
         );

      } else if (inicializador instanceof Inicializador) {
         return (Inicializador) inicializador;
      
      } else if (inicializador instanceof String) {
         String nome = (String) inicializador;
         nome = tratarNome(nome);
         switch (nome.toLowerCase()) {
            case "aleatorio"           : return new Aleatorio();
            case "aleatoriopositivo"   : return new AleatorioPositivo();
            case "constante"           : return new Constante();
            case "gaussiano"           : return new Gaussiano();
            case "glorotnormal"        : return new GlorotNormal();
            case "glorotuniforme"      : return new GlorotUniforme();
            case "he"                  : return new He();
            case "identidade"          : return new Identidade();
            case "lecun"               : return new LeCun();
            case "zeros"               : return new Zeros();
            
            default: throw new IllegalArgumentException(
               "Inicializador \"" + inicializador + "\" não encontado."
            );
         }
      
      } else {
         throw new IllegalArgumentException(
            "Tipo de dado \"" + inicializador.getClass().getTypeName() + "\" não suportado."
         );
      }

   }
}
