package rna.core;

public class Tensor{
   private int[] formato;
   private double[] dados;

   public Tensor(int[] formato){
      this.formato = formato;

      int tamanho = 1;
      for(int dimensao : formato){
         tamanho *= dimensao;
      }
      this.dados = new double[tamanho];
   }

   public int tamanho(int dim){
      if(dim < 0 || dim >= this.formato.length){
         throw new IllegalArgumentException(
            "Índice de dimensão inválido."
         );
      }
      return this.formato[dim];
   }

   public int tamanho(){
      return this.dados.length;
   }

   public int[] formato(){
      return this.formato;
   }

   public double dado(int... indices){
      return this.dados[indice(indices)];
   }

   public void editar(double valor, int... indices){
      this.dados[indice(indices)] = valor;
   }

   private int indice(int... indices) {
      if (indices.length < this.formato.length) {
          // Preencher os índices faltantes com zeros
          int[] novosIndices = new int[this.formato.length];
          System.arraycopy(indices, 0, novosIndices, 0, indices.length);
          indices = novosIndices;
      }
  
      if (indices.length != this.formato.length) {
          throw new IllegalArgumentException(
                  "O número de índices deve ser igual ao número de dimensões ou preenchido com zeros."
          );
      }
  
      int indice = 0;
      int multiplicador = 1;
  
      for (int i = 0; i < indices.length; i++) {
          if (indices[i] < 0 || indices[i] >= this.formato[i]) {
              throw new IndexOutOfBoundsException("Índice fora de alcance das dimensões: " + i);
          }
  
          indice += indices[i] * multiplicador;
          multiplicador *= this.formato[i];
      }
  
      return indice;
  }
  

   public double[] paraArray(){
      return this.dados;
   }

   /**
    * Exibe o conteúdo contido no tensor.
    */
   public void print(){
      System.out.print("Tensor = ");
      printRecursivo(this.dados, this.formato, 0, new int[this.formato.length]);
   }

   private void printRecursivo(double[] dados, int[] formato, int nivel, int[] indices){
      if(nivel == formato.length - 1){
         System.out.print("[");
         for(int i = 0; i < formato[nivel]; i++){
            indices[nivel] = i;
            int id = indice(indices);
            System.out.print(dados[id]);

            if (i < formato[nivel] - 1) {
               System.out.print(", ");
            }
         }
         System.out.print("]");
      
      }else{
         System.out.print("[");
         for(int i = 0; i < formato[nivel]; i++){
            indices[nivel] = i;
            printRecursivo(dados, formato, nivel + 1, indices);

            if(i < formato[nivel] - 1){
               System.out.print(", ");
            }
         }
         System.out.print("]");
      }

      if(nivel == 0){
         System.out.println();
      }
   }

   public void add(double valor, int... indices){
      int id = indice(indices);
      this.dados[id] += valor;
   }

   public void addTudo(double valor){
      for(int i = 0; i < this.dados.length; i++){
         this.dados[i] += valor;
      }
   }

   public void sub(double valor, int... indices){
      int id = indice(indices);
      this.dados[id] -= valor;
   }

   public void subTudo(double valor){
      for(int i = 0; i < this.dados.length; i++){
         this.dados[i] -= valor;
      }
   }

   public void mult(double valor, int... indices){
      int id = indice(indices);
      this.dados[id] *= valor;
   }

   public void multTudo(double valor){
      for(int i = 0; i < this.dados.length; i++){
         this.dados[i] *= valor;
      }
   }

   public void div(double valor, int... indices){
      int id = indice(indices);
      this.dados[id] /= valor;
   }

   public void divTudo(double valor){
      for(int i = 0; i < this.dados.length; i++){
         this.dados[i] /= valor;
      }
   }

   public void preencher(double valor){
      for(int i = 0; i < this.dados.length; i++){
         this.dados[i] = valor;
      }
   }

   public void copiar(Tensor t){
      if(t.dados.length != this.dados.length){
         throw new IllegalArgumentException(
            "Os dados do tensor fornecido deve conter o mesmo tamanho do tensor."
         );
      }

      System.arraycopy(t.dados, 0, this.dados, 0, this.dados.length);
   }
}
