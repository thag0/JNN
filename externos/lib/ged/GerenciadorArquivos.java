package externos.lib.ged;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;


/**
 * Gerenciador de arquivos do Ged
 */
class GerenciadorArquivos {
	
	/**
	 * Responsável por manuseio de arquivos externos
	 */
	public GerenciadorArquivos() {}


	public Dados lerCsv(String caminho) {
		//diretório não existe
		if (!(new File(caminho).exists())) {
			throw new IllegalArgumentException("O caminho especificado não existe ou não foi encontrado.");
		}

		//extensão não é .csv
		if (new File(caminho).getClass().getName().toLowerCase().endsWith(".csv")) {
			throw new IllegalArgumentException("O arquivo especificado não contém as extensão .csv");
		}

		ArrayList<String[]> linhas = new ArrayList<>();
		String separador = ",";
		String linha = "";

		try {
			BufferedReader br = new BufferedReader(new FileReader(caminho));
			while ((linha = br.readLine()) != null) {
				linha = linha.trim();//remover espaços vazios
				String linhaDados[] = linha.split(separador);//separar pelas vírgulas

				for (int i = 0; i < linhaDados.length; i++) {
					linhaDados[i] = linhaDados[i].replaceAll(" ", "");
				}
				
				linhas.add(linhaDados);
			}
			br.close();

		} catch(Exception e) {
			e.printStackTrace();
		}

		Dados dados = new Dados(linhas);
		return dados;
	}


	public Dados lerTxt(String caminho) {
		if (!(new File(caminho).exists())) {
			throw new IllegalArgumentException("O caminho especificado não existe ou não foi encontrado.");
		}

		//extensão não é .txt
		if (new File(caminho).getClass().getName().toLowerCase().endsWith(".txt")) {
			throw new IllegalArgumentException("O arquivo especificado não contém as extensão .txt");
		}

		ArrayList<String[]> linhas = new ArrayList<>();
		String linha = "";

		try {
			BufferedReader br = new BufferedReader(new FileReader(caminho));
			while ((linha = br.readLine()) != null) {
				String linhaDados[] = linha.trim().split("\\s+");//dividir por espaços em branco
				linhas.add(linhaDados);
			}
			br.close();

		} catch(IOException e) {
			e.printStackTrace();
		}

		Dados dados = new Dados(linhas);
		return dados;
	}


	public void exportarCsv(Dados dados, String caminho) {
		String separador = ",";

		ArrayList<String[]> conteudo = dados.conteudo();

		try {
			BufferedWriter bw = new BufferedWriter(new FileWriter(caminho + ".csv"));
			
			for (String[] linha : conteudo) {
				for (int i = 0; i < linha.length; i++) {

					bw.write(linha[i]);
					if (i < linha.length - 1) {
						bw.write(separador);
					}
				}
				bw.newLine();
			}

			bw.close();

		} catch(Exception e) {
			System.out.println("Houve um erro ao exportar o arquivo.");
			e.printStackTrace();
		}
	}


	public void exportarTxt(Dados dados, String caminho) {
		String separador = " ";

		ArrayList<String[]> conteudo = dados.conteudo();

		try {
			BufferedWriter bw = new BufferedWriter(new FileWriter(caminho + ".txt"));
			
			for (String[] linha : conteudo) {
				for (int i = 0; i < linha.length; i++) {

					bw.write(linha[i]);
					if (i < linha.length - 1) {
						bw.write(separador);
					}
				}
				bw.newLine();
			}

			bw.close();

		} catch(Exception e) {
			System.out.println("Houve um erro ao exportar o arquivo.");
			e.printStackTrace();
		}
	}

}
