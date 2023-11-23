package rna.otimizadores;

import rna.estrutura.CamadaDensa;

public class Adadelta extends Otimizador{

   private double rho;
   private double epsilon;
   private double[][][] ac;
   private double[][][] acb;

   private double[][][] acAt;
   private double[][][] acAtb;

   public Adadelta(double beta2, double epsilon){
      this.rho = beta2;
      this.epsilon = epsilon;
   }

   public Adadelta(){
      this(0.99, 1e-7);
   }

   @Override
   public void inicializar(CamadaDensa[] redec){
      this.ac = new double[redec.length][][];
      this.acAt = new double[redec.length][][];

      this.acb = new double[redec.length][][];
      this.acAtb = new double[redec.length][][];
   
      for(int i = 0; i < redec.length; i++){
         CamadaDensa camada = redec[i];

         this.ac[i] = new double[camada.pesos.length][camada.pesos[0].length];
         this.acAt[i] = new double[camada.pesos.length][camada.pesos[0].length];
         
         if(camada.temBias()){
            this.acb[i] = new double[camada.bias.length][camada.bias[0].length];
            this.acAtb[i] = new double[camada.bias.length][camada.bias[0].length];
         }
      }
   }

   @Override
   public void atualizar(CamadaDensa[] redec){
      double g;
      for(int i = 0; i < redec.length; i++){
         CamadaDensa camada = redec[i];

         for(int j = 0; j < camada.pesos.length; j++){
            for(int k = 0; k < camada.pesos[j].length; k++){
               g = camada.gradientes[j][k];
               ac[i][j][k] = (rho * ac[i][j][k]) + ((1 - rho) * (g*g));
               double delta = Math.sqrt(acAt[i][j][k] + epsilon) / Math.sqrt(ac[i][j][k] + epsilon) * g;
               acAt[i][j][k] = (rho * acAt[i][j][k]) + ((1 - rho) * (delta*delta));
               camada.pesos[j][k] += delta;
            }
         }

         if(camada.temBias()){
            for(int j = 0; j < camada.bias.length; j++){
               for(int k = 0; k < camada.bias[j].length; k++){
                  g = camada.erros[j][k];
                  acb[i][j][k] = (rho * acb[i][j][k]) + ((1 - rho) * (g*g));
                  double delta = Math.sqrt(acAtb[i][j][k] + epsilon) / Math.sqrt(acb[i][j][k] + epsilon) * g;
                  acAtb[i][j][k] = (rho * acAtb[i][j][k]) + ((1 - rho) * (delta*delta));
                  camada.pesos[j][k] += delta;
               }
            }      
         }
      }
   }

   @Override
   public String info(){
      String buffer = "";

      String espacamento = "    ";
      buffer += espacamento + "Rho: " + this.rho + "\n";
      buffer += espacamento + "Epsilon: " + this.epsilon + "\n";

      return buffer;
   }
}
