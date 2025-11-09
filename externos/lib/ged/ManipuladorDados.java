package externos.lib.ged;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Set;


/**
 * Manipulador de dados do Ged
 */
class ManipuladorDados{

	/**
	 * Contém as implementações das manipulações no conjunto de dados.
	 */
	public ManipuladorDados() {}
 
	public void addCol(Dados dados) {
		ArrayList<String[]> conteudo = dados.conteudo();

		int nColunas = conteudo.get(0).length;
  
		for (int i = 0; i < conteudo.size(); i++) {
			String[] linhaAtual = conteudo.get(i);
			String[] novaLinha = new String[nColunas + 1];

			System.arraycopy(linhaAtual, 0, novaLinha, 0, nColunas);

			novaLinha[nColunas] = "";

			conteudo.set(i, novaLinha);
		}

		dados.atribuir(conteudo);
  	}  

	public void addCol(Dados dados, int indice) {
		if (!dados.simetrico()) {
			throw new IllegalArgumentException("O conteúdo dos dados deve ser simético.");
		}

		ArrayList<String[]> conteudo = dados.conteudo();
		if ((indice < 0) || (indice >= conteudo.get(0).length)) {
			throw new IllegalArgumentException("O índice fornecido é inválido.");
		}
	
		int nColunas = conteudo.get(0).length;
	
		for (int i = 0; i < conteudo.size(); i++) {
			String[] linhaAtual = conteudo.get(i);
			String[] novaLinha = new String[nColunas + 1];
	
			//copiando valores antigos e deslocando a partir
			//do novo indice
			System.arraycopy(linhaAtual, 0, novaLinha, 0, indice);
	
			novaLinha[indice] = "";
	
			//copiando valores restantes
			System.arraycopy(linhaAtual, indice, novaLinha, indice + 1, nColunas - indice);
	
			conteudo.set(i, novaLinha);
		}

		dados.atribuir(conteudo);
	}

	public void addLin(Dados dados) {
		if (!dados.simetrico()) {
			throw new IllegalArgumentException("O conteúdo dos dados deve ser simético.");
		}

		ArrayList<String[]> conteudo = dados.conteudo();
		int colunas = conteudo.get(0).length;

		String[] novaLinha = new String[colunas];
		for (int i = 0; i < novaLinha.length; i++) {
			novaLinha[i] = "";
		}
		conteudo.add(novaLinha);

		dados.atribuir(conteudo);
	}

	public void addLin(Dados dados, int indice) {
		if (!dados.simetrico()) {
			throw new IllegalArgumentException("O conteúdo dos dados deve ser simético.");
		}
		
		ArrayList<String[]> conteudo = dados.conteudo();
		if (indice < 0 || indice >= conteudo.size()) {
			throw new IllegalArgumentException("O índice fornecido é inválido.");
		}

		int colunas = conteudo.get(0).length;

		String[] novaLinha = new String[colunas];
		for (int i = 0; i < novaLinha.length; i++) {
			novaLinha[i] = "";
		}
		conteudo.add(indice, novaLinha);

		dados.atribuir(conteudo);
	}

	public void remLim(Dados dados, int indice) {
		ArrayList<String[]> conteudo = dados.conteudo();

		if (conteudo == null) throw new IllegalArgumentException("O conteúdo dos dados é nulo.");
		if ((indice < 0) || (indice > conteudo.get(0).length-1)) {
			throw new IllegalArgumentException("Índice fornecido para remoção é inválido");
		}

		conteudo.remove(indice);
		dados.atribuir(conteudo);
	}

	public void remCol(Dados dados, int indice) {
		ArrayList<String[]> conteudo = dados.conteudo();

		if (conteudo == null) throw new IllegalArgumentException("O conteúdo dos dados é nulo.");
		if ((indice < 0) || (indice > conteudo.get(0).length-1)) {
			throw new IllegalArgumentException("Índice fornecido para remoção é inválido");
		}

		for (int i = 0; i < conteudo.size(); i++) {
			String[] linha = conteudo.get(i);
			//remover coluna original e substituir pelas novas categorias
			String[] novaLinha = new String[linha.length - 1];

			int contador1 = 0;
			int contador2 = 0;

			while(contador2 < linha.length) {
				if (contador2 == indice) {
					contador2++;
					
				} else {
					novaLinha[contador1] = linha[contador2];
					contador1++;
					contador2++;
				}
			}

			conteudo.set(i, novaLinha);
		}

		dados.atribuir(conteudo);
	}

	public void setValor(Dados dados, int idLinha, int idColuna, String novoValor) {
		ArrayList<String[]> conteudo = dados.conteudo();

		if (conteudo == null) throw new IllegalArgumentException("O conteúdo dos dados é nulo.");
		if (!dadosSimetricos(dados)) throw new IllegalArgumentException("O conteúdo dos dados deve ser simétrico para a operação.");

		if (idLinha < 0 || (idLinha >= conteudo.size())) {
			throw new IllegalArgumentException("Valor do índice de linha é inválido");
		}
		if (idColuna < 0 || (idColuna >= conteudo.get(0).length)) {
			throw new IllegalArgumentException("Valor do índice de coluna é inválido.");
		}

		if (novoValor == null) throw new IllegalArgumentException("O novo valor para substituição é nulo");

		dados.setItem(idLinha, idColuna, novoValor);
	}

	public void setValor(Dados dados, int idColuna, String busca, String novoValor) {
		ArrayList<String[]> conteudo = dados.conteudo();

		if (conteudo == null) throw new IllegalArgumentException("O conteúdo dos dados é nulo.");
		if (!dadosSimetricos(dados)) throw new IllegalArgumentException("O conteúdo dos dados deve ser simétrico para a operação.");

		if (idColuna < 0 || (idColuna >= conteudo.get(0).length)) {
			throw new IllegalArgumentException("Valor do índice de coluna é inválido.");
		}
		if (busca == null) throw new IllegalArgumentException("O valor de busca é nulo");
		if (novoValor == null) throw new IllegalArgumentException("O novo valor para substituição é nulo");

		dados.setItem(idColuna, busca, novoValor);
	}

	public void trocarColunas(Dados dados, int idColuna1, int idColuna2) {
		ArrayList<String[]> conteudo = dados.conteudo();

		if (conteudo == null) throw new IllegalArgumentException("O conteúdo dos dados fornicido está nulo.");
		if (conteudo.size() < 2) throw new IllegalArgumentException("É necessário que o conteúdo dos dados tenha pelo menos duas colunas.");
		if (!(dadosSimetricos(dados))) throw new IllegalArgumentException("O conteúdo dos dados deve ser simétrica.");

		if (idColuna1 < 0 || idColuna1 >= conteudo.get(0).length) throw new IllegalArgumentException("O índice fornecido da coluna 1 é inválido.");
		if (idColuna2 < 0 || idColuna2 >= conteudo.get(0).length) throw new IllegalArgumentException("O índice fornecido da coluna 2 é inválido.");
		if (idColuna1 == idColuna2) throw new IllegalArgumentException("Os índices fornecidos devem ser diferentes.");

		for (String[] linha : conteudo) {
			String intermediario = linha[idColuna1];
			linha[idColuna1] = linha[idColuna2];
			linha[idColuna2] = intermediario;
		}

		dados.atribuir(conteudo);
	}

	public void remNaoNumericos(Dados dados) {
		ArrayList<String[]> conteudo = dados.conteudo();

		int indiceInicial = 0;
		boolean removerLinha = false;
	
		while(indiceInicial < conteudo.size()) {
			removerLinha = false;

			for (int j = 0; j < conteudo.get(indiceInicial).length; j++) {
				//verificar se existe algum valor que não possa ser convertido para número.
				if ((valorInt(conteudo.get(indiceInicial)[j]) == false) || 
					(valorFloat(conteudo.get(indiceInicial)[j]) == false) || 
					(valorDouble(conteudo.get(indiceInicial)[j]) == false)
				) {
					removerLinha = true;
					break;
				}
			}

			if (removerLinha) conteudo.remove(indiceInicial);
			else indiceInicial++; 
		}

		dados.atribuir(conteudo);
	}

	public void categorizar(Dados dados, int indice) {
		ArrayList<String[]> conteudo = dados.conteudo();

		if (!dadosSimetricos(dados)) {
			throw new IllegalArgumentException("O conteúdo dos dados deve ser simétrico para categorizar.");
		}
		if ((indice < 0) || (indice >= conteudo.get(0).length)) {
			throw new IllegalArgumentException("O índice fornecido é inválido.");
		}

		//objeto que recebe apenas valores únicos
		HashSet<String> categoriasUnicas = new HashSet<>();
		for (String[] linha : conteudo) {
			categoriasUnicas.add(linha[indice]);
		}

		//definindo indice de cada categoria
		int nCategorias = categoriasUnicas.size();
		String[] listaCategorias = categoriasUnicas.toArray(new String[0]);
		int[] indiceCategorias = new int[nCategorias];
		for (int i = 0; i < indiceCategorias.length; i++) {
			indiceCategorias[i] = i;
		}

		//nova lista para armazenar as linhas atualizadas
		ArrayList<String[]> novaLista = new ArrayList<>();

		//reestruturando a lista com as novas categorias
		for (String[] linha : conteudo) {
			//remover coluna original e substituir pelas novas categorias
			String[] novaLinha = new String[(linha.length - 1) + nCategorias];

			//copiando valores existentes até o índice original
			for (int j = 0; j < indice; j++) {
				novaLinha[j] = linha[j];
			}

			//novos valores para as categorias
			for (int j = 0; j < nCategorias; j++) {
				if (listaCategorias[j].equals(linha[indice])) novaLinha[indice + j] = "1";
				else novaLinha[indice + j] = "0";
			}

			//copiando valores existentes após o índice original
			for (int j = indice + 1; j < linha.length; j++) {
				novaLinha[j + nCategorias - 1] = linha[j];
			}

			novaLista.add(novaLinha);
		}

		//substituindo a lista original pela nova lista modificada
		conteudo.clear();
		conteudo.addAll(novaLista);
		dados.atribuir(novaLista);
	}

	public Dados unir(Dados a, Dados b) {
		//simetria
		if (!a.simetrico()) {
			throw new IllegalArgumentException("O conteúdo de A deve ser simétrico.");
		}
		if (!b.simetrico()) {
			throw new IllegalArgumentException("O conteúdo de B deve ser simétrico.");
		}

		//dimensionalidade
		int[] shapeA = a.shape();
		int[] shapeB = b.shape();
		if (shapeA[1] != shapeB[1]) {
			throw new IllegalArgumentException("O contéudo de A e B deve conter a mesma quantidade de colunas.");
		}

		//é seguro trabalhar com os dados agora
		ArrayList<String[]> conteudoA = a.conteudo();
		ArrayList<String[]> conteudoB = b.conteudo();

		ArrayList<String[]> conteudoNovo = new ArrayList<>();
		conteudoNovo.addAll(conteudoA);
		conteudoNovo.addAll(conteudoB);

		Dados dados = new Dados();
		dados.atribuir(conteudoNovo);

		return dados;
	}

	public Dados unirColuna(Dados a, Dados b) {
		int[] shapeA = a.shape();
		int[] shapeB = b.shape();

		if (shapeA[0] != shapeB[0]) {
			throw new IllegalArgumentException(
				"A quantidade de linhas de A (" + shapeA[0] + 
				") deve ser igual a quantidade de linhas de B (" + shapeB[0] +")"
			);
		}

		int qLinhas = shapeA[0];

		ArrayList<String[]> conteudoA = a.conteudo();
		ArrayList<String[]> conteudoB = b.conteudo();
		
		ArrayList<String[]> conteudoNovo = new ArrayList<>();
		for (int i = 0; i < qLinhas; i++) {
			String[] linhaA = conteudoA.get(i);
			String[] linhaB = conteudoB.get(i);
			String[] novaLinha = new String[linhaA.length + linhaB.length];

			System.arraycopy(linhaA, 0, novaLinha, 0, linhaA.length);
			System.arraycopy(linhaB, 0, novaLinha, linhaA.length, linhaB.length);

			conteudoNovo.add(novaLinha);
		}
		
		Dados dados = new Dados();
		dados.atribuir(conteudoNovo);
		
		return dados;
	}

	public void removerDuplicadas(Dados dados) {
		if (!dados.simetrico()) {
			throw new IllegalArgumentException("O conjunto de dados deve ser simétrico.");
		}

		ArrayList<String[]> conteudo = dados.conteudo();
		Set<String> linhasVistas = new HashSet<>();
		ArrayList<String[]> novoConteudo = new ArrayList<>();

		for (String[] linha : conteudo) {
			
			String linhaComoString = String.join(",", linha);//transforma a linha em uma única string
			if (!linhasVistas.contains(linhaComoString)) {
				novoConteudo.add(linha);
				linhasVistas.add(linhaComoString);
			}
		}

		dados.atribuir(novoConteudo);
	}

	public void capitalizar(Dados dados) {
		if (dados.vazio()) {
			throw new IllegalArgumentException("O conteúdo dos dados está vazio.");
		}
		if (!dados.simetrico()) {
			throw new IllegalArgumentException("O conteúdo dos dados deve ser simétrico.");
		}

		int nColunas = dados.conteudo().get(0).length;
		for (int i = 0; i < nColunas; i++) {
			dados.capitalizar(i);
		}
	}

	public void substituir(Dados dados, String busca, String valor) {
		if (dados.vazio()) {
			throw new IllegalArgumentException("O conteúdo dos dados está vazio.");
		}
		if (!dados.simetrico()) {
			throw new IllegalArgumentException("O conteúdo dos dados deve ser simétrico.");
		}

		int[] shape = dados.shape();
		for (int i = 0; i < shape[1]; i++) {
			dados.substituir(i, busca, valor);
		}
	}
 
	public void normalizar(Dados dados) {
		if (dados.vazio()) {
			throw new IllegalArgumentException("O conteúdo dos dados está vazio.");
		}
		if (!dados.simetrico()) {
			throw new IllegalArgumentException("O conteúdo dos dados deve ser simétrico.");
		}

		int nColunas = dados.shape()[1];

		for (int i = 0; i < nColunas; i++) {
			dados.normalizar(nColunas);
		}
	}

	public Dados filtrar(Dados dados, int idCol, String busca) {
		dadosSimetricos(dados);
		if (dados.vazio()) {
			throw new IllegalArgumentException("O conteúdo dos dados está vazio.");
		}

		ArrayList<String[]> conteudo = dados.conteudo();
		if (idCol < 0 || idCol > conteudo.get(0).length) {
			throw new IllegalArgumentException("Índice da coluna fornecido é invalido.");
		}

		ArrayList<String[]> conteudoFiltrado = new ArrayList<>();
		//percorre a lista e procura o valor desejado
		for (String[] linha : conteudo) {
			if (linha[idCol].equals(busca)) {
				conteudoFiltrado.add(linha);
			}
		}

		//salvar numa estrutura separada
		Dados dadosFiltados = new Dados();
		dadosFiltados.atribuir(conteudoFiltrado);
		dadosFiltados.setNome(dados.nome());

		return dadosFiltados;
	}

	public Dados filtrar(Dados dados, int idCol, String operador, String valor) {
		dadosSimetricos(dados);
		if (dados.vazio()) {
			throw new IllegalArgumentException("O conteúdo dos dados está vazio.");
		}

		ArrayList<String[]> conteudo = dados.conteudo();
		if (idCol < 0 || idCol > conteudo.get(0).length) {
			throw new IllegalArgumentException("Índice da coluna fornecido é invalido.");
		}

		ArrayList<String[]> conteudoFiltrado = new ArrayList<>();
		for (String[] linha : conteudo) {
			if (compararValores(linha[idCol], operador, valor)) {
				conteudoFiltrado.add(linha);
			}
		}

		Dados dadosFiltados = new Dados();
		dadosFiltados.atribuir(conteudoFiltrado);
		dadosFiltados.setNome(dados.nome());

		return dadosFiltados;
	}

	public void preencherAusentes(Dados dados, double valor) {
		ArrayList<String[]> conteudo = dados.conteudo();

		for (String[] linha : conteudo) {
			for (int j = 0; j < linha.length; j++) {
				if (linha[j].equals("?") || linha[j].isBlank() || linha[j].isEmpty()) {
					linha[j] = Double.toString(valor);
				}
			}
		}

		dados.atribuir(conteudo);
	}

	public void preencherAusentes(Dados dados, int idCol, double valor) {
		ArrayList<String[]> conteudo = dados.conteudo();

		for (String[] linha : conteudo) {
			if (linha[idCol].equals("?") || linha[idCol].isBlank() || linha[idCol].isEmpty()) {
				linha[idCol] = Double.toString(valor);
			}
		}

		dados.atribuir(conteudo);
	}

	public Dados clonarDados(Dados dados) {
		return dados.clonar();
	}

	/**
	 * Tenta converter o valor para um numérico do tipo int
	 * @param valor valor que será testado.
	 * @return resultado da verificação, verdadeiro se foi convertido ou false se não
	 */
	private boolean valorInt(String valor) {
		try{
			Integer.parseInt(valor);
			return true;
		
		}catch(Exception e) {
			return false;
		}
	}

	/**
	 * Tenta converter o valor para um numérico do tipo float.
	 * @param valor valor que será testado.
	 * @return resultado da verificação, verdadeiro se foi convertido ou false se não
	 */
	private boolean valorFloat(String valor) {
		try{
			Float.parseFloat(valor);
			return true;
		
		}catch(Exception e) {
			return false;
		}
	}

	/**
	 * Tenta converter o valor para um numérico do tipo double
	 * @param valor valor que será testado.
	 * @return resultado da verificação, verdadeiro se foi convertido ou false se não
	 */
	private boolean valorDouble(String valor) {
		try{
			Double.parseDouble(valor);
			return true;
		
		}catch(Exception e) {
			return false;
		}
	}

	/**
	 * Verifica se a operação entre os dois valores é válida de acordo com a expressão.
	 * v1 (operador) v2.
	 * @param v1 primeiro valor.
	 * @param operador operador esperado.
	 * @param v2 segundo valor.
	 * @return resultado da operação, valores que não possam ser convertidos serão desconsiderados.
	 */
	private boolean compararValores(String v1, String operador, String v2) {
		double valor1;
		double valor2;

		try{
			valor1 = Double.parseDouble(v1);
			valor2 = Double.parseDouble(v2);

		}catch(Exception e) {
			return false;
		}

		switch(operador) {
			case ">":  return (valor1 > valor2);
			case ">=": return (valor1 >= valor2);
			case "<":  return (valor1 < valor2);
			case "<=": return (valor1 <= valor2);
			case "==": return (valor1 == valor2);
			case "!=": return (valor1 != valor2);
			default: throw new IllegalArgumentException("Operador não suportado.");
		}

	}

	public boolean dadosSimetricos(Dados dados) {
		return dados.simetrico();
	}
}