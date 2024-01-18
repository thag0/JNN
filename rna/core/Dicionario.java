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
 * Tradutor para as funções de ativação, otimizadores e funções de perda.
   */
public class Dicionario{

   /**
    * Tradutor para as funções de ativação, otimizadores e funções de perda.
    */
   public Dicionario(){}

   /**
    * Converte o texto lido em uma instância de função
    * de ativação correspondente.
    * @param nome nome da função de ativação.
    * @return instância da função de ativação lida.
    * @param IllegalArgumentException caso a ativação não for encontrada.
    */
   public Ativacao obterAtivacao(String nome){
      //essa provavelmente não é a melhor abordagem
      //mas é a mais fácil de implementar
      nome = nome.toLowerCase();
      switch(nome){
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
            "Ativação \"" + nome + "\" não encontada."
         );
      }
   }

   /**
    * Converte o texto lido em uma instância de 
    * otimizador correspondente.
    * @param nome nome do otimizador.
    * @return instância do otimizador lido.
    */
   public Otimizador obterOtimizador(String nome){
      nome = nome.toLowerCase();
      switch(nome){
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
   }

   /**
    * Converte o texto lido em uma instância de função
    * de perda correspondente.
    * @param nome nome da função de perda.
    * @return instância da função de perda lida.
    */
   public Perda obterPerda(String nome){
      nome = nome.toLowerCase();
      switch(nome){
         case "mse"                    : return new MSE();
         case "mae"                    : return new MAE();
         case "msle"                   : return new MSLE();
         case "rmse"                   : return new RMSE();
         case "entropiacruzada"        : return new EntropiaCruzada();
         case "entropiacruzadabinaria" : return new EntropiaCruzadaBinaria();

         default: throw new IllegalArgumentException(
            "Função de perda \"" + nome + "\" não encontada."
         );
      }      
   }

   /**
    * Converte o texto lido em uma instância de inicializador
    * @param nome nome do inicializador.
    * @return instância do inicializador lido.
    */
   public Inicializador obterInicializador(String nome){
      nome = nome.toLowerCase();
      switch(nome){
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
            "Inicializador \"" + nome + "\" não encontado."
         );
      }
   }
}