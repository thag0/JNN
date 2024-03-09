import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;

//Apenas pra renomear as imagens do mnist
public class Renomear{

   public static void main(String[] args){
      int digitos = 10;
      for(int i = 0; i < digitos; i++){
         //renomear pra qualquer coisa porque as vezes buga
         //e alguns arquivos somem
         renomearArquivos(i, "Eimg");
         renomearArquivos(i, "img");
      }
   }

   private static void renomearArquivos(int digito, String nome){
      String caminho = "./dados/mnist/treino/" + digito;
      File diretorio = new File(caminho);

      if(diretorio.isDirectory()){
         File[] arquivos = diretorio.listFiles();
         
         if(arquivos != null){
            for(int i = 0; i < arquivos.length; i++){
               String novoNome = nome + "_" + i + ".jpg";
               renomearArquivo(arquivos[i].toPath(), novoNome);
            }
            System.out.println("Renomeação concluída.");
         }else{
            System.out.println("O diretório está vazio.");
         }
      }else{
         System.out.println("O caminho fornecido não é um diretório válido.");
      }
   }

   static void renomearArquivo(Path arquivo, String novoNome) {
      try {
         Files.move(arquivo, arquivo.resolveSibling(novoNome), StandardCopyOption.REPLACE_EXISTING);
      } catch (IOException e) {
         System.out.println("Erro ao renomear o arquivo " + arquivo.getFileName() + ": " + e.getMessage());
      }
   }
}
