package testes;

import ged.Ged;
import rna.ativacoes.Sigmoid;
import rna.estrutura.CamadaDensa;
import rna.inicializadores.Xavier;

public class TesteCamada{
   public static void main(String[] args){
      Ged ged = new Ged();
      ged.limparConsole();
      
      double[][] entrada = {
         {0, 0},
         {0, 1},
         {1, 0},
         {1, 1}
      };
      
      //simples teste unitário de uma camada densa
      //criada para se comportar como uma porta lógica OR.
      CamadaDensa camada = new CamadaDensa(2, 1, true);
      camada.inicializar(new Xavier(), 0);
      camada.configurarAtivacao(new Sigmoid());

      //valores de uma camada pré treinada
      double[] pesos = {10.51693203318985, 10.516960411442009};
      double bias = -4.797469093074007;
      camada.configurarPesos(0, pesos);
      camada.configurarBias(0, bias);

      for(int i = 0; i < entrada.length; i++){
         double[] amostra = entrada[i];
         camada.calcularSaida(amostra);
         double[] saida = camada.obterSaida()[0];
         System.out.println(amostra[0] + " - " + amostra[1] + " = " + saida[0]);
      }

      System.out.println(camada.info());
   }
}
