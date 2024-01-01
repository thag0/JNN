package lib.ged;

public class ImpressaoDados {
   public ImpressaoDados(){

   }


   public void imprimirInicio(Dados dados){
      if(dados.vazio()){
         dados.imprimir();
         return;
      }
      
      String espacamento = "   ";
      int[] shape = dados.shape();
      int linPadrao = (shape[0] < 5) ? shape[0] : 5;

      if(linPadrao < 5){
         dados.imprimir();
         return;
      }

      System.out.println("Início " + "\"" + dados.nome() + "\" (" + shape[0] + ", " + shape[1] + ")" +" = [");

      //comprimento máximo de cada coluna
      int[] comprimentoMaximo = new int[dados.conteudo().get(0).length];
      for(int i = 0; i < linPadrao; i++){
         String[] linha = dados.conteudo().get(i);

         for(int j = 0; j < linha.length; j++){
            int comprimento = linha[j].length();

            if(comprimento > comprimentoMaximo[j]){
               comprimentoMaximo[j] = comprimento;
            }
         }
      }

      for(int i = 0; i < linPadrao; i++){
         String[] linha = dados.conteudo().get(i);
         for(int j = 0; j < linha.length; j++){
            String valor = linha[j];
            int distancia = comprimentoMaximo[j] - (valor.length()-1);
            String espacos = " ".repeat(distancia);
            System.out.print(espacamento + valor + espacos);
         }
         
         System.out.println();
      }

      System.out.println("]");
   }
}
