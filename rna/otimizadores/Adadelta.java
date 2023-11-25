package rna.otimizadores;

import rna.core.Mat;
import rna.estrutura.CamadaDensa;

public class Adadelta extends Otimizador{

   private double rho;
   private double epsilon;
   private Mat[] ac;
   private Mat[] acb;

   private Mat[] acAt;
   private Mat[] acAtb;

   public Adadelta(double beta2, double epsilon){
      this.rho = beta2;
      this.epsilon = epsilon;
   }

   public Adadelta(){
      this(0.99, 1e-7);
   }

   @Override
   public void inicializar(CamadaDensa[] redec){
      this.ac   = new Mat[redec.length];
      this.acAt = new Mat[redec.length];

      this.acb   = new Mat[redec.length];
      this.acAtb = new Mat[redec.length];
   
      for(int i = 0; i < redec.length; i++){
         CamadaDensa camada = redec[i];

         this.ac[i]   = new Mat(camada.pesos.lin, camada.pesos.col);
         this.acAt[i] = new Mat(camada.pesos.lin, camada.pesos.col);
         
         if(camada.temBias()){
            this.acb[i]   = new Mat(camada.bias.lin, camada.bias.col);
            this.acAtb[i] = new Mat(camada.bias.lin, camada.bias.col);
         }
      }
   }

   @Override
   public void atualizar(CamadaDensa[] redec){
      for(int i = 0; i < redec.length; i++){
         CamadaDensa camada = redec[i];
         Mat pesos = camada.pesos;
         Mat grads = camada.gradientes;

         for(int j = 0; j < pesos.lin; j++){
            for(int k = 0; k < pesos.col; k++){
               calcular(pesos, grads, ac[i], acAt[i], j, k);
            }
         }

         if(camada.temBias()){
            Mat bias = camada.bias;
            Mat gradsB = camada.erros;
            for(int j = 0; j < bias.lin; j++){
               for(int k = 0; k < bias.col; k++){
                  calcular(bias, gradsB, acb[i], acAtb[i], j, k);
               }
            }
         }
      }
   }

   private void calcular(Mat var, Mat grad, Mat ac, Mat acAt, int lin, int col){
      double g = grad.dado(lin, col);
      double ac2 = (rho * ac.dado(lin, col)) + ((1 - rho) * (g*g));
      ac.editar(lin, col, ac2);

      double delta = Math.sqrt(acAt.dado(lin, col) + epsilon) / Math.sqrt(ac.dado(lin, col) + epsilon) * g;
      double acAt2 = (rho * acAt.dado(lin, col)) + ((1 - rho) * (delta*delta));
      acAt.editar(lin, col, acAt2);
      
      var.add(lin, col, delta);
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
