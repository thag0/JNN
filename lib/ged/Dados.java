package lib.ged;

import java.text.Collator;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Locale;

/**
 * Classe criada para centralizar um tipo de dado 
 * específico para uso dentro do Ged.
 * <p>
 *    O objeto de dados possui um conjunto de pequenas ferramentas para manipulação 
 *    e visualização de conteudo. Algumas operações mais elaboradas podem ser executadas
 *    usando o Gerenciador de Dados {@code Ged}.
 * </p>
 * @see https://github.com/thag0/Treinando-Rede-Neural-Artificial/tree/main/utilitarios/ged
 * @author Thiago Barroso, acadêmico de Engenharia da Computação pela Universidade Federal do Pará, 
 * Campus Tucuruí. Maio/2023.
 */
public class Dados{

	/**
	 * <p>
	 *    Região crítica.
	 * </p>
	 * Estrutura que armazena o conteúdo de dados lidos
	 */
	private ArrayList<String[]> conteudo;

	/**
	 * Quantidade de alterações feitas no conjunto de dados.
	 */
	private int alteracoes = 0;

	/**
	 * Nome personalizável.
	 */
	private String nome = getClass().getSimpleName();

	/**
	 * Inicializa um objeto do tipo Dados com seu conteúdo vazio.
	 * <p>
	 *    {@code Dados} é um objeto criado para armazenar informações de conjuntos de 
	 *    dados, possui algumas funcionalidades mais básicas e é uma dependência para 
	 *    algumas funcionalidades presentes dentro do {@code Ged}.
	 * </p>
	 */
	public Dados() {
		conteudo = new ArrayList<>();
	}

	/**
	 * Inicializa um objeto do tipo Dados de acordo com o conteúdo especificado.
	 * <p>
	 *    {@code Dados} é um objeto criado para armazenar informações de conjuntos de 
	 *    dados, possui algumas funcionalidades mais básicas e é uma dependência para 
	 *    algumas funcionalidades presentes dentro do {@code Ged}.
	 * </p>
	 * <p>
	 *    Tipos suportados:
	 * </p>
	 * <ul>
	 *    <li>
	 *       {@code int[], int[][]};
	 *    </li>
	 *    <li>
	 *       {@code float[], float[][]};
	 *    </li>
	 *    <li>
	 *       {@code double[], double[][]};
	 *    </li>
	 *    <li>
	 *       {@code String[], String[][]};
	 *    </li>
	 * </ul>
	 * @param conteudo matriz contendo o conteúdo de dados.
	 * @throws IllegalArgumentException se o tipo de dado fornecido não for suportado.
	 */
	public Dados(Object conteudo) {
		if (conteudo instanceof int[][]) {
			atribuir((int[][]) conteudo);
		
		}else if (conteudo instanceof float[][]) {
			atribuir((float[][]) conteudo);
		
		}else if (conteudo instanceof double[][]) {
			atribuir((double[][]) conteudo);
		
		}else if (conteudo instanceof String[][]) {
			atribuir((String[][]) conteudo);
		
			//arrays
		}else if (conteudo instanceof int[]) {
			int[] c = (int[]) conteudo;
			int[][] cont = new int[1][];
			cont[0] = c;
			atribuir(cont);
		
		}else if (conteudo instanceof float[]) {
			float[] c = (float[]) conteudo;
			float[][] cont = new float[1][];
			cont[0] = c;
			atribuir(cont);
		
		}else if (conteudo instanceof double[]) {
			double[] c = (double[]) conteudo;
			double[][] cont = new double[1][];
			cont[0] = c;
			atribuir(cont);

		}else if (conteudo instanceof String[]) {
			String[] c = (String[]) conteudo;
			String[][] cont = new String[1][];
			cont[0] = c;
			atribuir(cont);

		}else{
			throw new IllegalArgumentException(
				"Tipo de dado (" + conteudo.getClass().getSimpleName() +
				") não suportado."
			);
		}
	}

	/**
	 * Inicializa um objeto do tipo Dados de acordo com o conteúdo especificado.
	 * <p>
	 *    {@code Dados} é um objeto criado para armazenar informações de conjuntos de 
	 *    dados, possui algumas funcionalidades mais básicas e é uma dependência para 
	 *    algumas funcionalidades presentes dentro do {@code Ged}.
	 * </p>
	 */
	public Dados(ArrayList<String[]> conteudo) {
		atribuir(conteudo);
	}

	/**
	 * Configura um novo nome personalizado para o conjunto de dados.
	 * @param nome novo nome do conjunto de dados.
	 * @throws IllegalArgumentException se novo nome for nulo.
	 * @throws IllegalArgumentException se novo estiver vazio ou em branco.
	 */
	public void setNome(String nome) {
		if (nome == null) {
			throw new IllegalArgumentException("O novo nome não pode ser nulo.");
		}
		if (nome.isBlank() || nome.isEmpty()) {
			throw new IllegalArgumentException(
				"O novo nome não pode estar vazio ou em branco."
			);
		}

		this.nome = nome;
		addAlteracao();
	}

	/**
	 * Retorna o item correspondente pela linha e coluna fornecidos.
	 * @param lin linha para busca.
	 * @param col coluna para busca.
	 * @return valor contido com base na linha e coluna.
	 * @throws IllegalArgumentException se o conteúdo estiver vazio.
	 * @throws IllegalArgumentException se os índices fornecidos estiverem fora de alcance.
	 */
	public String getItem(int lin, int col) {
		if (conteudo.isEmpty()) {
			throw new IllegalArgumentException("O conteúdo está vazio.");
		}
		if (lin < 0 || lin >= conteudo.size()) {
			throw new IllegalArgumentException(
				"Índice de busca de linha (" + lin + ") inválido"
			);
		}
		if (col < 0 || col >= conteudo.get(lin).length) {
			throw new IllegalArgumentException(
				"Índice de busca de coluna (" + col + ") inválido"
			);
		}

		return conteudo.get(lin)[col];
	}

	/**
	 * Edita o valor contido no espaço indicado pela linha e coluna.
	 * @param lin linha para edição.
	 * @param col coluna para edição.
	 * @param valor novo valor.
	 * @throws IllegalArgumentException se os índices fornecidos forem inválidos.
	 */
	public void setItem(int lin, int col, String valor) {
		if (lin < 0 || lin >= conteudo.size()) {
			throw new IllegalArgumentException(
				"Índice de linha fornecido (" + lin + ") é inválido."
			);
		}
		if (col < 0 || col >= conteudo.get(lin).length) {
			throw new IllegalArgumentException(
				"Índice de coluna fornecido (" + col + ") é inválido."
			);
		}

		String[] linha = conteudo.get(lin);
		linha[col] = valor;
		conteudo.set(lin, linha);
		addAlteracao();
	}

	/**
	 * Edita o valor em todas as linhas de acordo com a coluna especificada.
	 * @param col coluna para edição.
	 * @param busca valor alvo.
	 * @param valor novo valor.
	 * @throws IllegalArgumentException se o conteúdo dos dados não for simétrico.
	 * @throws IllegalArgumentException se o índice de coluna for inválido.
	 */
	public void setItem(int col, String busca, String valor) {
		if (!simetrico()) {
			throw new IllegalArgumentException("O conteúdo dos dados deve ser simétrico.");
		}
		if (col < 0 || col >= conteudo.get(0).length) {
			throw new IllegalArgumentException(
				"O índice de coluna fornecido (" + col + ") é inválido."
			);
		}

		for (String[] linha : conteudo) {
			if (linha[col].contains(busca)) linha[col] = valor;
		}

		addAlteracao();
	}

	/**
	 * Substitui todo o conteúdo atual de Dados pela nova lista.
	 * @param conteudo lista com os novos dados.
	 */
	public void atribuir(ArrayList<String[]> conteudo) {
		if (conteudo != null) {
			this.conteudo = conteudo;
			addAlteracao();
		}
	}

	/**
	 * Atribui os valores contidos na matriz fonercida ao 
	 * conteúdo de Dados.
	 * <p>
	 *    Os dados contidos na matriz devem ser simétricos, o que quer dizer 
	 *    que eles devem possuir a mesma quantidade de colunas para todas as 
	 *    linhas presentes.
	 * </p>
	 * @param matriz matriz com os dados.
	 * @throws IllegalArgumentException se a matriz for vazia.
	 * @throws IllegalArgumentException se a matriz não for simétrica.
	 */
	public void atribuir(int[][] matriz) {
		int linhas = matriz.length;
		if (linhas < 1) {
			throw new IllegalArgumentException("A matriz fornecida está vazia.");
		}

		ArrayList<String[]> conteudo = new ArrayList<>(linhas);
		
		int colunas = matriz[0].length;
		for (int i = 0; i < linhas; i++) {
			if (matriz[i] == null) {
				throw new IllegalArgumentException(
					"Linha " + i + " == null." 
				);
			}

			if (matriz[i].length != colunas) {
				throw new IllegalArgumentException(
					"A matriz deve conter o mesmo número de colunas para todas as linhas."
				);
			}

			String[] linha = new String[colunas];
			for (int j = 0; j < colunas; j++) {
				linha[j] = Integer.toString(matriz[i][j]);
			}
			conteudo.add(i, linha);
		}

		this.conteudo = conteudo;
		addAlteracao();
	}

	/**
	 * Atribui os valores contidos na matriz fonercida ao 
	 * conteúdo de Dados.
	 * <p>
	 *    Os dados contidos na matriz devem ser simétricos, o que quer dizer 
	 *    que eles devem possuir a mesma quantidade de colunas para todas as 
	 *    linhas presentes.
	 * </p>
	 * @param matriz matriz com os dados.
	 * @throws IllegalArgumentException se a matriz for vazia.
	 * @throws IllegalArgumentException se a matriz não for simétrica.
	 */
	public void atribuir(float[][] matriz) {
		int linhas = matriz.length;
		if (linhas < 1) {
			throw new IllegalArgumentException("A matriz fornecida está vazia.");
		}
		
		ArrayList<String[]> conteudo = new ArrayList<>(linhas);
		
		int colunas = matriz[0].length;
		for (int i = 0; i < linhas; i++) {
			if (matriz[i] == null) {
				throw new IllegalArgumentException(
					"Linha " + i + " == null." 
				);
			}

			if (matriz[i].length != colunas) {
				throw new IllegalArgumentException(
					"A matriz deve conter o mesmo número de colunas para todas as linhas."
				);
			}

			String[] linha = new String[colunas];
			for (int j = 0; j < colunas; j++) {
				linha[j] = Float.toString(matriz[i][j]);
			}
			conteudo.add(i, linha);
		}

		this.conteudo = conteudo;
		addAlteracao();
	}

	/**
	 * Atribui os valores contidos na matriz fonercida ao 
	 * conteúdo de Dados.
	 * <p>
	 *    Os dados contidos na matriz devem ser simétricos, o que quer dizer 
	 *    que eles devem possuir a mesma quantidade de colunas para todas as 
	 *    linhas presentes.
	 * </p>
	 * @param matriz matriz com os dados.
	 * @throws IllegalArgumentException se a matriz for vazia.
	 * @throws IllegalArgumentException se a matriz não for simétrica.
	 */
	public void atribuir(double[][] matriz) {
		int linhas = matriz.length;
		if (linhas < 1) {
			throw new IllegalArgumentException("A matriz fornecida está vazia.");
		}

		ArrayList<String[]> conteudo = new ArrayList<>(linhas);
		
		int colunas = matriz[0].length;
		for (int i = 0; i < linhas; i++) {
			if (matriz[i] == null) {
				throw new IllegalArgumentException(
					"Linha " + i + " == null." 
				);
			}

			if (matriz[i].length != colunas) {
				throw new IllegalArgumentException(
					"A matriz deve conter o mesmo número de colunas para todas as linhas."
				);
			}

			String[] linha = new String[colunas];
			for (int j = 0; j < colunas; j++) {
				linha[j] = Double.toString(matriz[i][j]);
			}
			conteudo.add(i, linha);
		}

		this.conteudo = conteudo;
		addAlteracao();
	}

	/**
	 * Atribui os valores contidos na matriz fonercida ao 
	 * conteúdo de Dados.
	 * <p>
	 *    Os dados contidos na matriz devem ser simétricos, o que quer dizer 
	 *    que eles devem possuir a mesma quantidade de colunas para todas as 
	 *    linhas presentes.
	 * </p>
	 * @param matriz matriz com os dados.
	 * @throws IllegalArgumentException se a matriz for vazia.
	 * @throws IllegalArgumentException se a matriz não for simétrica.
	 */
	public void atribuir(String[][] matriz) {
		int linhas = matriz.length;
		if (linhas < 1) {
			throw new IllegalArgumentException("A matriz fornecida está vazia.");
		}
		
		ArrayList<String[]> conteudo = new ArrayList<>(linhas);
		
		int colunas = matriz[0].length;
		for (int i = 0; i < linhas; i++) {
			if (matriz[i] == null) {
				throw new IllegalArgumentException(
					"Linha " + i + " == null." 
				);
			}

			if (matriz[i].length != colunas) {
				throw new IllegalArgumentException(
					"A matriz deve conter o mesmo número de colunas para todas as linhas."
				);
			}

			String[] linha = new String[colunas];
			for (int j = 0; j < colunas; j++) {
				linha[j] = matriz[i][j] == null ? "?" : matriz[i][j];
			}
			conteudo.add(i, linha);
		}

		this.conteudo = conteudo;
		addAlteracao();
	}

	/**
	 * Retorna todo o conteúdo presente nos dados.
	 * @return estrutura {@code ArrayList<String[]>} que armazena o conteúdo dos dados.
	 */
	public ArrayList<String[]> conteudo() {
		return this.conteudo;
	}

	/**
	 * Retorna um novo conjunto de dados simples contendo apenas o conteúdo da coluna
	 * indicada.
	 * <p>
	 *    Exemplo:
	 * </p>
	 * <pre>
	 * d = [
	 *    1, 2
	 *    3, 4
	 *    5, 6 
	 * ]
	 *
	 * col = d.coluna(0)
	 *
	 * col = [
	 *    1
	 *    3
	 *    5 
	 * ]
	 * </pre>
	 * @param col índice da coluna desejada.
	 * @return conjunto de dados contendo o conteúdo da coluna indicada.
	 * @throws IllegalArgumentException se o conteúdo dos dados não for simétrico. 
	 * @throws IllegalArgumentException se o índice fornecido for inválido.
	 */
	public Dados coluna(int col) {
		if (!simetrico()) {
			throw new IllegalArgumentException("O conteúdo dos dados deve ser simétrico.");
		}
		if (col < 0 || col >= this.conteudo.get(0).length) {
			throw new IllegalArgumentException(
				"Índice da coluna fornecido (" + col + ") é inválido."
			);
		}

		ArrayList<String[]> conteudo = new ArrayList<>();

		//criando uma nova coluna com os valores
		//da coluna desejada
		for (String[] linha : this.conteudo) {
			String[] novaLinha = new String[1];
			novaLinha[0] = linha[col];
			conteudo.add(novaLinha);
		}

		Dados d = new Dados(conteudo);
		d.nome = "Coluna " + col;
		d.alteracoes = 0;
		return d;
	}

	/**
	 * Aumenta em uma unidade a quantidade de alterações
	 * feitas dentro do conjunto de dados.
	 */
	private void addAlteracao() {
		alteracoes++;
	}

	/**
	 * Calula a média do conteúdo que pode ser transformado para valor numérico 
	 * presente na coluna especificada.
	 * @param col índice da coluna desejada.
	 * @return média dos elementos numéricos presentes na coluna, 0 caso não seja possível 
	 * converter nenhum valor.
	 * @throws IllegalArgumentException se o índice fornecido for inválido.
	 */
	public double media(int col) {
		if (!simetrico()) {
			throw new IllegalArgumentException("O conteúdo dos dados deve ser simétrico.");
		}
		if (col < 0 || col > this.conteudo.get(0).length) {
			throw new IllegalArgumentException(
				"Índice fornecido (" + col + ") inválido."
			);
		}

		double soma = 0;
		int contador = 0;//contador de transformadas feitas

		for (String[] linha : this.conteudo) {
			String valor = linha[col];
			
			try {
				double valorTransformado = Double.parseDouble(valor);
				soma += valorTransformado;
				contador++;

			} catch(Exception e) {

			}
		}

		if (contador == 0) return 0;//evitar divisão por zero

		return soma / contador;
	}

	/**
	 * Calcula a mediana do conteúdo que pode ser transformado para valor numérico
	 * presente na coluna especificada.
	 * @param col índice da coluna desejada.
	 * @return valor da mediana dos elementos numéricos presentes na coluna, 0 caso não seja 
	 * possível converter nenhum valor.
	 * @throws IllegalArgumentException se o índice fornecido for inválido.
	 */
	public double mediana(int col) {
		if (!simetrico()) {
			throw new IllegalArgumentException("O conteúdo dos dados deve ser simétrico.");
		}
		if (col < 0 || col > this.conteudo.get(0).length) {
			throw new IllegalArgumentException(
				"Índice fornecido (" + col + ") inválido."
			);
		}

		ArrayList<Double> valoresNumericos = new ArrayList<>();

		for (String[] linha : conteudo) {
			String valor = linha[col];
			try {
				double valorTransformado = Double.parseDouble(valor);
				valoresNumericos.add(valorTransformado);
			
			} catch(NumberFormatException e) {}
		}  

		if (valoresNumericos.isEmpty()) return 0;

		Collections.sort(valoresNumericos);
		int meio = valoresNumericos.size() / 2;

		if (valoresNumericos.size() % 2 == 0) {
			return (valoresNumericos.get(meio-1) + valoresNumericos.get(meio))/2;
		
		}else{
			return valoresNumericos.get(meio);
		}
	}

	/**
	 * Calcula a moda do conteúdo que pode ser transformado para valor numérico
	 * presente na coluna especificada.
	 * @param col índice da coluna desejada.
	 * @return valor da moda dos elementos numéricos presentes na coluna, 0 caso não seja 
	 * possível converter nenhum valor.
	 * @throws IllegalArgumentException se o índice fornecido for inválido.
	 */
	public double moda(int col) {
		if (!simetrico()) {
			throw new IllegalArgumentException("O conteúdo dos dados deve ser simétrico.");
		}
		if (col < 0 || col > this.conteudo.get(0).length) {
			throw new IllegalArgumentException(
				"Índice fornecido (" + col + ") inválido."
			);
		}

		HashMap<String, Integer> frequenciaElementos = new HashMap<>();

		//frequência de cada valor na coluna
		for (String[] linha : this.conteudo) {
			
			String valor = linha[col].trim();
			if (!valor.isEmpty()) {
				try {
					Double.parseDouble(valor);
					if (frequenciaElementos.containsKey(valor)) {//elemento existente
						frequenciaElementos.put(valor, frequenciaElementos.get(valor) + 1);
						
					}else{//elemento novo
						frequenciaElementos.put(valor, 1);
					}

				} catch(Exception e) {

				}
			}
		}

		if (frequenciaElementos.isEmpty()) return 0;

		String moda = "";
		int maxFrequencia = 0;

		// Encontra o valor com maior frequência
		for (String valor : frequenciaElementos.keySet()) {
			
			int frequencia = frequenciaElementos.get(valor);
			if (frequencia > maxFrequencia) {
				maxFrequencia = frequencia;
				moda = valor;
			}
		}

		if (moda.isEmpty()) return 0;

		try {
			return Double.parseDouble(moda);
		
		} catch(NumberFormatException e) {
			return 0;
		}
	}

	/**
	 * Calcula o desvio padrão dos valores numéricos presentes na coluna especificada.
	 * @param col índice da coluna desejada.
	 * @return desvio padrão dos valores numéricos na coluna.
	 * @throws IllegalArgumentException Se o índice fornecido for inválido.
	 */
	public double desvp(int col) {
		if (!simetrico()) {
			throw new IllegalArgumentException("O conteúdo dos dados deve ser simétrico.");
		}
		if (col < 0 || col > this.conteudo.get(0).length) {
			throw new IllegalArgumentException(
				"Índice fornecido (" + col + ") inválido."
			);
		}

		ArrayList<Double> valoresNumericos = new ArrayList<>();

		//valores numéricos da coluna
		for (String[] linha : this.conteudo) {
			String valor = linha[col];
			try {
				double valorTransformado = Double.parseDouble(valor);
				valoresNumericos.add(valorTransformado);
			} catch(Exception e) {}
		}

		//média dos valores
		double media = 0;
		int contador = 0;
		for (double valor : valoresNumericos) {
			media += valor;
			contador++;
		}
		if (contador > 0) media /= contador;
		else return 0;//não houver valores numéricos.

		//somatório dos quadrados das diferenças entre os valores e a média.
		double somaDiferencasQuadrado = 0;
		for (double valor : valoresNumericos) {
			double diferenca = valor - media;
			somaDiferencasQuadrado += diferenca * diferenca;
		}

		//desvio padrão.
		double desvioPadrao = Math.sqrt(somaDiferencasQuadrado / contador);
		return desvioPadrao;
	}

	/**
	 * Calcula o maior valor do conteúdo que pode ser transformado para valor numérico
	 * presente na coluna especificada.
	 * @param col índice da coluna desejada.
	 * @return valor máximo entre os elementos numéricos presentes na coluna, 0 caso não 
	 * seja possível converter nenhum valor.
	 * @throws IllegalArgumentException se o índice fornecido for inválido.
	 */
	public double maximo(int col) {
		if (!simetrico()) {
			throw new IllegalArgumentException("O conteúdo dos dados deve ser simétrico.");
		}
		if (col < 0 || col > this.conteudo.get(0).length) {
			throw new IllegalArgumentException(
				"Índice fornecido (" + col + ") inválido."
			);
		}

		ArrayList<Double> valoresNumericos = new ArrayList<>();
  
		//transformando o que dá em valor
		for (String[] linha : this.conteudo) {
			String valor = linha[col];
			try {
				double num = Double.parseDouble(valor);
				valoresNumericos.add(num);
			
			} catch(NumberFormatException e) {
				
			}
		}
  
		if (valoresNumericos.isEmpty()) return 0;
		double valorMaximo = valoresNumericos.get(0);

		//procurando valor maior
		for (int i = 1; i < valoresNumericos.size(); i++) {
			if (valoresNumericos.get(i) > valorMaximo) {
				valorMaximo = valoresNumericos.get(i);
			}
		}
  
		return valorMaximo;
	}

	/**
	 * Calcula o menor valor do conteúdo que pode ser transformado para valor numérico
	 * presente na coluna especificada.
	 * @param col índice da coluna desejada.
	 * @return valor mínimo entre os elementos numéricos presentes na coluna, 0 caso não 
	 * seja possível converter nenhum valor.
	 * @throws IllegalArgumentException se o índice fornecido for inválido.
	 */
	public double minimo(int col) {
		if (!simetrico()) {
			throw new IllegalArgumentException("O conteúdo dos dados deve ser simétrico.");
		}
		if (col < 0 || col > this.conteudo.get(0).length) {
			throw new IllegalArgumentException(
				"Índice fornecido (" + col + ") inválido."
			);
		}

		ArrayList<Double> valoresNumericos = new ArrayList<>();
  
		//transformando o que dá em valor
		for (String[] linha : this.conteudo) {
			String valor = linha[col];
			try {
				double num = Double.parseDouble(valor);
				valoresNumericos.add(num);
			
			} catch(NumberFormatException e) {
				
			}
		}
  
		if (valoresNumericos.isEmpty()) return 0;
		double valorMaximo = valoresNumericos.get(0);

		//procurando valor menor
		for (int i = 1; i < valoresNumericos.size(); i++) {
			if (valoresNumericos.get(i) < valorMaximo) {
				valorMaximo = valoresNumericos.get(i);
			}
		}
  
		return valorMaximo;
	}

	/**
	 * Normaliza os valores numéricos contido na coluna fornecida.
	 * <p>
	 *    Caso a coluna possua algum valor que não possa ser convertido o 
	 *    processo é cancelado.
	 * </p>
	 * Exemplo:
	 * <pre>
	 * d = [
	 *    1, 5 
	 *    2, a
	 *    3, 7
	 *    4, 8
	 *    5, 9
	 * ]
	 *
	 * d.normalizar(0);
	 *
	 * d = [
	 *    0.00, 5 
	 *    0.25, a
	 *    0.50, 7
	 *    0.75, 8
	 *    1.00, 9
	 * ]
	 * </pre>
	 * @param col índice da coluna desejada.
	 * @throws IllegalArgumentException se o conteúdo dos dados estiver vazio.
	 * @throws IllegalArgumentException se o conteúdo não for simétrico.
	 */
	public void normalizar(int col) {
		if (vazio()) {
			throw new IllegalArgumentException("O conteúdo dos dados está vazio.");
		}
		if (!simetrico()) {
			throw new IllegalArgumentException("Os dados devem ser simétricos para normalização.");
		}

		if (this.naoNumericos(col)) return;
			 
		double min = this.minimo(col);
		double max = this.maximo(col);

		for (String[] linha : this.conteudo) {
			double valor = Double.parseDouble(linha[col]);
			double valorNormalizado = (valor - min) / (max - min);
			linha[col] = Double.toString(valorNormalizado);
		}
		addAlteracao();
	}

	/**
	 * Capitaliza todo o conteúdo alfabético contido na coluna fornecida.
	 * @param col índice da coluna desejada.
	 * @throws IllegalArgumentException se o conteúdo dos dados estiver vazio.
	 * @throws IllegalArgumentException se os dados não forem simétricos.
	 * @throws IllegalArgumentException e o índice da coluna for inválido.
	 */
	public void capitalizar(int col) {
		if (vazio()) {
			throw new IllegalArgumentException("O conteúdo dos dados está vazio.");
		}
		if (!simetrico()) {
			throw new IllegalArgumentException("Os dados devem ser simétricos para normalização.");
		}
		if (col < 0 || col > this.conteudo.get(0).length) {
			throw new IllegalArgumentException(
				"Índice fornecido (" + col + ") inválido."
			);
		}

		for (String[] linha : this.conteudo) {
			linha[col] = cap(linha[col]);
		}
		addAlteracao();
	}

	/**
	 * Captaliza a palavra.
	 * @param palavra palavra desejada.
	 * @return nova palavra captalizada com base na forncedida.
	 */
	private String cap(String palavra) {
		if (palavra == null || palavra.isEmpty()) return palavra;

		char[] caracteres = palavra.toCharArray();

		//primeiro caractere é maiúsculo e os demais são minúsculos.
		caracteres[0] = Character.toUpperCase(caracteres[0]);
		for (int i = 1; i < caracteres.length; i++) {
			caracteres[i] = Character.toLowerCase(caracteres[i]);
		}

		return new String(caracteres);
	}

	/**
	 * Substitui pelo novo valor todo o conteúdo encontrado na linha de acordo com a busca.
	 * <p>
	 *    Exemplo:
	 * </p>
	 * <pre>
	 * d = [
	 *    a.b, c.d
	 *    e.f, g.h  
	 * ]
	 * 
	 * substituir(0, ".", "");
	 * 
	 * d = [
	 *    ab, c.d
	 *    ef, g.h  
	 * ]
	 * </pre>
	 * @param col índice da coluna desejada.
	 * @param busca valor desejado para substituição.
	 * @param valor novo valor que ficará no lugar do valor buscado.
	 * @throws IllegalArgumentException se o conteúdo dos dados estiver vazio.
	 * @throws IllegalArgumentException se o conteúdo dos dados não forem simétricos.
	 * @throws IllegalArgumentException se o índice da coluna for inválido.
	 */
	public void substituir(int col, String busca, String valor) {
		if (vazio()) {
			throw new IllegalArgumentException("O conteúdo dos dados está vazio.");
		}
		if (!simetrico()) {
			throw new IllegalArgumentException("O conteúdo dos dados deve ser simétrico.");
		}
		if (col < 0 || col > this.conteudo.get(0).length) {
			throw new IllegalArgumentException(
				"Índice fornecido (" + col + ") inválido."
			);
		}

		for (String[] linha : this.conteudo) {
			if (linha[col].contains(busca)) {
				linha[col] = linha[col].replace(busca, valor);
			}
		}
		addAlteracao();
	}

	/**
	 * Ordena o conteúdo contido nos dados de acordo com a coluna desejada.
	 * <p>
	 *    A ordenação consequentemente irá mudar a ordem de organização
	 *    dos outros elementos.
	 * </p>
	 * Exemplo:
	 * <pre>
	 * d = [
	 *    d
	 *    b
	 *    a
	 *    e
	 *    c
	 * ]
	 * 
	 * d.ordenar(0, true).
	 * 
	 * d = [
	 *    a
	 *    b
	 *    c
	 *    d
	 *    e
	 * ]
	 * </pre>
	 * @param col índice da coluna desejada.
	 * @param cres true caso a ordenação deva ser crescente, false caso contrário.
	 * @throws IllegalArgumentException se o conteúdo dos dados estiver vazio.
	 * @throws IllegalArgumentException se o conteúdo dos dados não forem simétricos.
	 * @throws IllegalArgumentException se o índice da coluna for inválido.
	 */
	public void ordenar(int col, boolean cres) {
		if (vazio()) {
			throw new IllegalArgumentException("O conteúdo dos dados está vazio.");
		}
		if (!simetrico()) {
			throw new IllegalArgumentException("O conteúdo dos dados deve ser simétrico.");
		}
		if (col < 0 || col > this.conteudo.get(0).length) {
			throw new IllegalArgumentException(
				"Índice fornecido (" + col + ") inválido."
			);
		}

		Comparator<String[]> comp = (linha1, linha2) -> {
			String valor1 = linha1[col];
			String valor2 = linha2[col];

			//configurando língua e região
			Locale regiao = Locale.of("pt");
			Collator collator = Collator.getInstance(regiao);
			collator.setStrength(Collator.TERTIARY);

			return cres ? collator.compare(valor1, valor2) : collator.compare(valor2, valor1);
		};

		Collections.sort(this.conteudo, comp);
		addAlteracao();
	}

	/**
	 * Verifica se o conteúdo do conjunto de dados está vazio.
	 * <p>
	 *    Ele é considerado vazio se não conter nenhuma linha ou 
	 *    se todas as linhas estiverem vazias.
	 * </p>
	 * @return true se o conjunto de dados estiver vazio, false caso contrário.
	 */
	public boolean vazio() {
		if (conteudo.isEmpty()) return true;
		
		for (int i = 0; i < conteudo.size(); i++) {
			if (conteudo.get(i).length > 0) return false;
		}

		return true;
	}

	/**
	 * Retorna um array contendo as linhas e colunas do conteúdo dos dados.
	 * <p>
	 *    {@code shape[0] = linhas}
	 * </p>
	 * <p>
	 *    {@code shape[1] = colunas}
	 * </p>
	 * @return estrutura contendo o formato da lista, considerando que ela é simétrica.
	 * @throws IllegalArgumentException se o conteúdo estiver vazio.
	 * @throws IllegalArgumentException se o conteúdo não for simétrico.
	 */
	public int[] shape() {
		if (vazio()) {
			throw new IllegalArgumentException("O conteúdo dos dados está vazio.");
		}

		if (!simetrico()) {
			throw new IllegalArgumentException("O conteúdo dos dados não é simétrico.");
		}

		return new int[] {
			conteudo.size(), 
			conteudo.get(0).length
		};
	}

	/**
	 * Retorna um buffer contendo as informações do conteúdo dos dados, onde:
	 * <p>
	 *    {@code shape = (linhas, colunas)}
	 * </p>
	 * @return buffer contendo o formato da lista, considerando que ela é simétrica.
	 */
	public String shapeStr() {
		if (vazio()) return "()";

		int[] shape = shape();

		return "(" + shape[0] + ", " + shape[1] + ")";
	}

	/**
	 * Monsta as informações de exibição dos dados.
	 * @return string formatada.
	 */
	public String construirPrint() {
		String pad = "    ";

		StringBuilder sb = new StringBuilder();

		if (vazio()) {
			sb.append(nome()).append(" = [").append("\n");
			sb.append("   (Vazio)").append("\n");
		
		} else {
			if (simetrico()) {
				sb.append(nome).append(" ").append(shapeStr()).append(" = [").append("\n");

			} else {
				sb.append(nome).append(" = [").append("\n");
			}

			// comprimento máximo de cada coluna
			int[] compMax = new int[conteudo.get(0).length];
			for (String[] linha : conteudo) {
				for (int i = 0; i < linha.length; i++) {
					int comp = linha[i].length();

					if (comp > compMax[i]) compMax[i] = comp;
				}
			}

			// colunas dinâmicas
			for (String[] linha : conteudo) {
				for (int i = 0; i < linha.length; i++) {
					String valor = linha[i];
					int distancia = compMax[i] - valor.length() + 1;
					String espacos = " ".repeat(distancia);
					sb.append(pad).append(valor).append(espacos);
				}
				sb.append("\n");
			}
		}

		sb.append("]").append("\n");

		return sb.toString();
	}

	/**
	 * Método de impressão básico, via console, de todo o 
	 * conteúdo contido em formato de tabela.
	 * <p>
	 *    Caso os dados sejam simétricos, também é exibido o formato do conteúdo.
	 * </p>
	 */
	public void print() {
		System.out.println(construirPrint());
	}

	/**
	 * Exibe algumas informações sobre o conjunto de dados.
	 * <p>
	 *    As informações incluem:
	 * </p>
	 * <ul>
	 *    <li>Formato dos dados;</li>
	 *    <li>Existência de valores não numéricos;</li>
	 *    <li>Quantidade de valores ausentes;</li>
	 * </ul>
	 */
	public void info() {
		String pad = "    ";
		String format = "\t\t";
		StringBuilder sb = new StringBuilder();

		sb.append("Info ").append("\"").append(nome).append("\"").append(" = [").append("\n");

		if (vazio()) {
			sb.append(pad).append("Conteúdo vazio").append("\n]");
		
		} else {
			if (simetrico()) {
				int[] shape = this.shape();
				sb.append(pad).append("Linhas").append(format).append(shape[0]).append("\n");
				sb.append(pad).append("Colunas").append(format).append(shape[0]).append("\n");
			
			} else {
				sb.append(pad).append("Tamanho inconsistente").append("\n");
			}
		}

		sb.append(pad).append("Alterações:").append(format).append(alteracoes).append("\n");

		sb.append("\n");

		boolean apenasNumericos = true;
		for (int i = 0; i < this.conteudo.get(0).length; i++) {
			if (naoNumericos(i)) {
				apenasNumericos = false;
				break;
			}
		}

		sb.append(pad).append("Numéricos: ")
		.append(format).append((apenasNumericos) ? "sim" : "não")
		.append("\n");

		boolean temValoresAusentes = false;
		for (int i = 0; i < this.conteudo.get(0).length; i++) {
			int valoresAusentes = ausentes(i);
			if (valoresAusentes > 0) {
				sb.append(pad).append("Ausentes col ").append(i).append(": \t")
				.append(valoresAusentes).append("\n");
				temValoresAusentes = true;
			}
		}
  
		if (!temValoresAusentes) {
			sb.append(pad).append("Valores ausentes: ").append("\tnão").append("\n");
		}
		
		sb.append("]\n");

		System.out.println(sb.toString());
	}

	/**
	 * Exibe uma visão geral das informações da coluna especificada.
	 * <p>
	 *    As informações incluem:
	 * </p>
	 * <ul>
	 *    <li>Média;</li>
	 *    <li>Mediana;</li>
	 *    <li>Valor máximo;</li>
	 *    <li>Valor mínimo;</li>
	 *    <li>Moda;</li>
	 *    <li>Desvio Padrão;</li>
	 *    <li>Coluna composta apenas por valores numéricos;</li>
	 *    <li>Coluna contém valores ausentes ou em branco;</li>
	 * </ul>
	 * @param col índice da coluna desejada.
	 */
	public void info(int col) {
		if (col < 0 || col > this.conteudo.get(0).length) {
			throw new IllegalArgumentException(
				"Índice fornecido (" + col + ") inválido."
			);
		}

		String pad = "   ";

		StringBuilder sb = new StringBuilder();

		sb.append("Coluna " ).append(col).append(" = [").append("\n");
		sb.append(pad).append("Média:").append("\t\t").append(media(col)).append("\n");
		sb.append(pad).append("Mediana:").append("\t\t").append(mediana(col)).append("\n");
		sb.append(pad).append("Máximo:").append("\t\t").append(maximo(col)).append("\n");
		sb.append(pad).append("Mínimo:").append("\t\t").append(minimo(col)).append("\n");
		sb.append(pad).append("Moda:").append("\t\t").append(moda(col)).append("\n");
		sb.append(pad).append("Desv Padrão:").append("\t\t").append(desvp(col)).append("\n");
		sb.append(pad).append("Numéricos:").append("\t\t").append(!naoNumericos(col) ? "sim" : "não").append("\n");
		sb.append(pad).append("Ausentes:").append("\t\t").append(ausentes(col)).append("\n");
		sb.append("]").append("\n");

		System.out.println(sb.toString());
	}

	/**
	 * Retorna o nome personalizado do conjunto de dados.
	 * @return nome personalizado do conjunto de dados.
	 */
	public String nome() {
		return this.nome;
	}

	/**
	 * Verifica se a coluna indicada possui algum valor que não possa
	 * ser convertido para um valor numérico.
	 * @param col índice da coluna desejada.
	 * @return verdadeiro caso a coluna possua valores que não possam ser convertidos, falso caso contrário.
	 * @throws IllegalArgumentException se o índice for inválido.
	 */
	public boolean naoNumericos(int col) {
		if (col < 0 || col >= this.conteudo.get(0).length) {
			throw new IllegalArgumentException(
				"Índice fornecido (" + col + ") inválido."
			);
		}
  
		for (String[] linha : this.conteudo) {
			String valor = linha[col];
			try {
				Double.parseDouble(valor);//conversão "universal"
			
			} catch(NumberFormatException e) {
				return true;
			}
		}
  
		return false;
	}

	/**
	 * Verifica a quantidade de valores ausentes presentes na coluna especificada.
	 * <p>
	 *    São considerados dados ausentes quaisquer valores que sejam {@code vazio},
	 *    {@code em branco} ou {@code com "?"}
	 * </p>
	 * @param col índice da coluna desejada.
	 * @return quantidade de valores considerados ausentes.
	 * @throws IllegalArgumentException se o índice for inválido.
	 */
	public int ausentes(int col) {
		if (col < 0 || col >= this.conteudo.get(0).length) {
			throw new IllegalArgumentException(
				"Índice fornecido (" + col +") inválido."
			);
		}
  
		int cont = 0;
		for (String[] linha : conteudo) {
			String valor = linha[col].trim();
			if (valor.equals("?") || valor.isEmpty()) {
				cont++;
			}
		}
  
		return cont;
	}

	/**
	 * Verifica se o conteúdo dos dados é simetrico. A simetria leva em conta 
	 * se todas as colunas têm o mesmo tamanho.
	 * <p>
	 *    A simetria também leva em conta se o conteúdo dos dados possui elementos, 
	 *    caso o tamanho seja zero será considerada como não simétrica.
	 * <p>
	 * <p>
	 *    Dados nulos não classificados.
	 * <p>
	 * Exemplo:
	 * <pre>
	 * a =  [
	 *    1, 2, 3
	 *    4, 5, 6, 7
	 *    8, 9
	 * ]
	 * 
	 * a.simetrico() == false
	 *
	 * b =  [
	 *    1, 2, 3
	 *    4, 5, 6
	 *    7, 8, 9
	 * ]
	 * 
	 * b.simetrico() == true
	 * </pre>
	 * @param dados conjunto de dados.
	 * @return true caso os dados sejam simétricos, false caso contrário.
	 * @throws IllegalArgumentException se o conteúdo dos dados for nulo.
	 */
	public boolean simetrico() {
		if (this.conteudo == null) {
		  throw new IllegalArgumentException("O conteúdo dos dados é nulo."); 
		} 
		
		//lista sem dados é considerada como não simétrica
		if (this.conteudo.size() == 0) return false;

		int colunas = this.conteudo.get(0).length;// tamanho base
		for (String[] linha : this.conteudo) {
			if (linha.length != colunas) return false;
		}

		return true;
	}

	/**
	 * Clona o conteúdo em uma nova estrutura, devolvendo um novo objeto
	 * de {@code Dados} com o mesmo conteúdo.
	 * @return novo objeto do tipo {@code Dados} com a cópia do conteúdo.
	 */
	public Dados clonar() {
		ArrayList<String[]> cloneConteudo = new ArrayList<>();
		for (String[] linha : this.conteudo) {
			cloneConteudo.add(linha);
		}

		Dados cloneDados = new Dados(cloneConteudo);
		cloneDados.nome = "Clone de " + this.nome;
		cloneDados.alteracoes = this.alteracoes;

		return cloneDados;
	}


	@Override
	public String toString() {
		return construirPrint();
	}
}
