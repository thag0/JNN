package lib.ged;

/**
 * <p>
 *    Gerenciador de Dados.
 * </p>
 * Conjunto de ferramentas criadas para lidar com dados. As funcionalidades
 * incluem:
 * <ul>
 *    <li>Leitura de arquivos;</li>
 *    <li>Conversão de dados;</li>
 *    <li>Manipulação de estrutura de dados;</li>
 *    <li>Gerenciamento de treino e teste para rede neural;</li>
 *    <li>Operações matriciais;</li>
 * </ul>
 *    Algumas operação necessitam de um objeto do tipo {@code Dados} para serem realizadas que 
 *    deve ser importado através de <pre>import ged.Dados;</pre>
 * <p>
 *    Aos poucos são adicionadas novas funcionalidades de acordo com as necessidades que surgem então o
 *    Ged sempre pode sofrer alterações e melhorias com o passar do tempo.
 * </p>
 * @see https://github.com/thag0/Treinando-Rede-Neural-Artificial/tree/main/utilitarios/ged
 * @author Thiago Barroso, acadêmico de Engenharia da Computação pela Universidade Federal do Pará, Campus Tucuruí. Maio/2023.
*/
public class Ged{

	ImpressaoMatriz im;//exibição
	ImpressaoArray ia;//exibição
	ImpressaoDados id;//exibição
	ManipuladorDados md;//manipulador de dados
	GerenciadorArquivos ga;//leitor de arquivos
	ConversorDados cd;//conversor de dados 
	TreinoTeste gtt;//gerenciador de treino e teste da rede
	OperadorMatriz om;//operador de matrizes
	OperadorMatrizMultithread omt;

	/**
	 * Objeto responsável pelo manuseio de um conjunto de dados contendo 
	 * diversas implementações de métodos para gerenciamento e manipulação.
	 * <p>
	 *    Algumas operação necessitam de um objeto do tipo {@code Dados} para serem realizadas que 
	 *    deve ser importado através de <pre>import ged.Dados;</pre>
	 * </p>
	 */
	public Ged() {
		im = new ImpressaoMatriz();
		ia = new ImpressaoArray();
		id = new ImpressaoDados();
		md = new ManipuladorDados();
		ga = new GerenciadorArquivos();
		gtt = new TreinoTeste();
		cd = new ConversorDados();
		om = new OperadorMatriz();

		omt = new OperadorMatrizMultithread(Runtime.getRuntime().availableProcessors()/2);
	}

	/**
	 * Função auxiliar destinada ao uso no windows.
	 * <p>
	 *    Limpa o conteúdo do console onde o programa está sendo executado.
	 * </p>
	 */
	public void limparConsole() {
		try {
			String nomeSistema = System.getProperty("os.name");

			if (nomeSistema.contains("Windows")) {
				new ProcessBuilder("cmd", "/c", "cls").inheritIO().start().waitFor();
				return;
			
			} else {
				for (int i = 0; i < 100; i++) {
					System.out.println();
				}
			}

		} catch(Exception e) {
			return;
		}
	}

	/**
	 * Exibe pelo console as informações contidas no conteúdo dos dados.
	 * @param dados conjunto de dados.
	 */
	public void printDados(Dados dados) {
		dados.print();
	}

	/**
	 * Imprime o início do conteúdo do conjunto de dados para facilitar a visualização 
	 * em dados muito grandes.
	 * @param dados conjunto de dados.
	 */
	public void printInicio(Dados dados) {
		id.printInicio(dados);
	}

	/**
	 * Exibe as informações contidas no array fornecido.
	 * <p>
	 *    Tipos suportados:
	 * </p>
	 * <pre>
	 *int[];
	 *float[];
	 *double[];
	 *String[];
	 * </pre>
	 * @param array array com os dados.
	 */
	public void prinArray(Object array) {
		ia.imprimirArray(array);
	}

	/**
	 * Exibe as informações contidas no array fornecido.
	 * <p>
	 *    Tipos suportados:
	 * </p>
	 * <pre>
	 *int[];
	 *float[];
	 *double[];
	 *String[];
	 * </pre>
	 * @param array array com os dados.
	 * @param nome nome personalizado para o array impresso.
	 */
	public void printArray(Object array, String nome) {
		ia.printArray(array, nome);
	}

	/**
	 * Exibe as informações contidas na matriz fornecida.
	 * <p>
	 *    Tipos suportados:
	 * </p>
	 * <pre>
	 *int[];
	 *float[];
	 *double[];
	 *String[];
	 * </pre>
	 * @param matriz matriz com os dados.
	 */
	public void printMatriz(Object matriz) {
		im.printMatiz(matriz);
	}

	/**
	 * Exibe as informações contidas na matriz fornecida.
	 * <p>
	 *    Tipos suportados:
	 * </p>
	 * <pre>
	 *int[];
	 *float[];
	 *double[];
	 *String[];
	 * </pre>
	 * @param matriz matriz com os dados.
	 * @param nome nome personalizado para a matriz impressa.
	 */
	public void printMatriz(Object matriz, String nome) {
		im.printMatriz(matriz, nome);
	}

	
	//MANIPULADOR DE DADOS ---------------------


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
	 * dados =  [
	 *    1, 2, 3
	 *    4, 5, 6, 7
	 *    8, 9, 
	 * ]
	 * 
	 * dadosSimetricos(dados) == false
	 * </pre>
	 * @param dados conjunto de dados.
	 * @return true caso os dados sejam simétricos, false caso contrário.
	 * @throws IllegalArgumentException se o conteúdo dos dados for nulo.
	 */
	public boolean dadosSimetricos(Dados dados) {
		return md.dadosSimetricos(dados);
	}

	/**
	 * Adiciona uma coluna com conteúdo {@code vazio} ao final de todas as 
	 * linhas do conteúdo dos dados.
	 * <p>
	 *    Exemplo:
	 * </p>
	 * <pre>
	 * dados = [
	 *    1, 2, 3 
	 *    4, 5, 6 
	 *    7, 8, 9 
	 * ]
	 *
	 * novaColuna = [
	 *    1, 2, 3, 
	 *    4, 5, 6,  
	 *    7, 8, 9, 
	 * ]
	 * </pre>
	 * @param dados conjunto de dados.
	 */
	public void addCol(Dados dados) {
		md.addCol(dados);
	}

	/**
	 * Adiciona uma coluna com conteúdo {@code vazio} no índice fornecido. 
	 * Todos os itens depois do índice serão deslocados para a direita.
	 * <p>
	 *    Exemplo:
	 * </p>
	 * <pre>
	 * dados = [
	 *    1, 2, 3 
	 *    4, 5, 6 
	 *    7, 8, 9 
	 * ]
	 *
	 * int indice = 1;
	 *
	 * novaColuna = [
	 *    1,  , 2, 3
	 *    4,  , 5, 6 
	 *    7,  , 8, 9
	 * ]
	 * </pre>
	 * @param dados conjunto de dados.
	 * @param id índice onde a nova coluna será adicionada.
	 * @throws IllegalArgumentException se o conteúdo dos dados não for simétrico.
	 * @throws IllegalArgumentException se o índice fornecido for inválido.
	 */
	public void addCol(Dados dados, int id) {
		md.addCol(dados, id);
	}

	/**
	 * Adiciona uma linha com conteúdo {@code vazio} ao final do conteúdo dos dados.
	 * <p>
	 *    Exemplo:
	 * </p>
	 * <pre>
	 * dados = [
	 *    1, 2, 3 
	 *    4, 5, 6 
	 *    7, 8, 9 
	 * ]
	 *
	 * novaLinha = [
	 *    1, 2, 3
	 *    4, 5, 6 
	 *    7, 8, 9 
	 *     ,  , 
	 * ]
	 * </pre>
	 * @param dados conjunto de dados.
	 * @throws IllegalArgumentException se o conteúdo dos dados não for simétrico.
	 */
	public void addLin(Dados dados) {
		md.addLin(dados);
	}

	/**
	 * Adiciona uma linha com conteúdo {@code vazio} de acordo com o índice especificado. Todos 
	 * os elementos depois da linha fornecida serão deslocados para baixo.
	 * <p>
	 *    Exemplo:
	 * </p>
	 * <pre>
	 * dados = [
	 *    1, 2, 3 
	 *    4, 5, 6 
	 *    7, 8, 9 
	 * ]
	 *
	 * int indice = 1;
	 *
	 * novaLinha = [
	 *    1, 2, 3
	 *     ,  , 
	 *    4, 5, 6 
	 *    7, 8, 9 
	 * ]
	 * </pre>
	 * @param dados conjunto de dados.
	 * @param id índice onde a nova linha será adicionada.
	 * @throws IllegalArgumentException se o conteúdo dos dados não for simétrico.
	 * @throws IllegalArgumentException se o índice fornecido for inválido.
	 */
	public void addLin(Dados dados, int id) {
		md.addLin(dados, id);
	}

	/**
	 * Remove uma linha inteira do conjunto de dados
	 * @param dados conjunto de dados.
	 * @param id índice da linha que será removida.
	 * @throws IllegalArgumentException o conteúdo dos dados for nulo.
	 * @throws IllegalArgumentException se o indice for inválido.
	 */
	public void remLin(Dados dados, int id) {
		md.remLim(dados, id);
	}

	/**
	 * Remove todas as colunas dos dados de acordo com o índice fornecido.
	 * @param dados conjunto de dados.
	 * @param id índice da coluna que será removida.
	 * @throws IllegalArgumentException o conteúdo dos dados for nulo.
	 * @throws IllegalArgumentException se o indice for inválido.
	 */
	public void remCol(Dados dados, int id) {
		md.remCol(dados, id);
	}

	/**
	 * Substitui o valor de busca pelo novo valor fornecido, de acordo com a linha e coluna especificadas.
	 * @param dados conjunto de dados.
	 * @param idLinha índice da linha alvo para a alteração dos dados.
	 * @param idColuna índice da coluna alvo para a alteração dos dados.
	 * @param valor novo valor que será colocado.
	 * @throws IllegalArgumentException se o conteúdo dos dados for nulo.
	 * @throws IllegalArgumentException se o conteúdo dos dados não for simétrica.
	 * @throws IllegalArgumentException se o valor do índice da linha fornecido estiver fora de alcance.
	 * @throws IllegalArgumentException se o valor do índice da coluna fornecido estiver fora de alcance.
	 * @throws IllegalArgumentException se o valor de busca for nulo.
	 * @throws IllegalArgumentException se o novo valor de substituição for nulo;
	 */
	public void setValo(Dados dados, int idLinha, int idColuna, String valor) {
		md.setValor(dados, idLinha, idColuna, valor);
	}

	/**
	 * Substitui todas as linhas dos dados pelo valor fornecido, caso na coluna fornecida tenha o valor buscado.
	 * @param dados conjunto de dados.
	 * @param idColuna índice da coluna alvo para a alteração dos dados.
	 * @param busca valor que será procurado para ser substituído.
	 * @param valor novo valor que será colocado.
	 * @throws IllegalArgumentException se o conteúdo dos dados estiver nulo.
	 * @throws IllegalArgumentException se o conteúdo dos dados não for simétrico.
	 * @throws IllegalArgumentException se o valor do índice fornecido estiver fora de alcance.
	 * @throws IllegalArgumentException se o valor de busca for nulo.
	 * @throws IllegalArgumentException se o novo valor de substituição for nulo;
	 */
	public void setValor(Dados dados, int idColuna, String busca, String valor) {
		md.setValor(dados, idColuna, busca, valor);
	}

	/**
	 * Troca os valores das colunas no conteúdo dos dados de acordo com os índices fornecidos.
	 * <p>
	 *    Exemplo:
	 * </p>
	 * <pre>
	 * dados = [
	 *    1, 2, 3
	 *    4, 5, 6
	 *    7, 8, 9    
	 * ]
	 *
	 * int col1 = 0;
	 * int col2 = 2;
	 *
	 * dados = [
	 *    3, 2, 1
	 *    6, 5, 2
	 *    9, 8, 3    
	 * ]
	 * </pre>
	 * @param dados conjunto de dados.
	 * @param id1 índice da primeira coluna que será trocada.
	 * @param id2 índice da segunda coluna que será trocada.
	 * @throws IllegalArgumentException se o conteúdo dos dados estiver nulo.
	 * @throws IllegalArgumentException se o conteúdo dos dados não for simétrico.
	 * @throws IllegalArgumentException se a o conteúdo dos dados não tiver pelo menos duas colunas.
	 * @throws IllegalArgumentException se os índices fornecidos estiverem fora de alcance do tamanho das colunas.
	 * @throws IllegalArgumentException se as colunas fornecidas forem iguais.
	 */
	public void trocarColunas(Dados dados, int id1, int id2) {
		md.trocarColunas(dados, id1, id2);
	}

	/**
	 * Remove a linha inteira dos dados caso exista algum valor nas colunas que não consiga ser convertido para
	 * um valor numérico.
	 * <p>
	 *    É importante verificar e ter certeza se os dados não possuem nenhuma coluna com caracteres, caso isso seja verdade
	 *    o método irá remover todas as colunas como consequência e o conteúdo dos dados estará vazio.
	 * </p>
	 * Exemplo:
	 * <pre>
	 * dados = [
	 *    1, "a", 3
	 *    4,  5 , 6
	 *    7,    , 9    
	 * ]
	 * resultado = [
	 *    4, 5, 6
	 * ]
	 * </pre>
	 * @param dados conjunto de dados.
	 */
	public void remNaoNumericos(Dados dados) {
		md.remNaoNumericos(dados);
	}

	/**
	 * Categoriza o conteúdo de dados na coluna relativa ao índice fornecido,
	 * usando a técnica de One-Hot Encoding. 
	 * <p>
	 *    As novas colunas adicionadas representarão as categorias únicas encontradas 
	 *    na coluna especificada, e os valores das novas colunas serão definidos como "1" 
	 *    quando a categoria correspondente estiver presente na linha e "0" caso contrário.
	 * <p>
	 * Exemplo:
	 * <pre>
	 * dados = [
	 *    a, b, c
	 *    d, e, f
	 *    g, h, i
	 * ]
	 *
	 * categorizar(dados, 2);
	 *
	 * dados = [
	 *    a, b, 1, 0, 0
	 *    d, e, 0, 1, 0
	 *    g, h, 0, 0, 1
	 * ]
	 * </pre>
	 * @param dados conjunto de dados.
	 * @param id O índice da coluna desejada para categorização.
	 * @throws IllegalArgumentException se o conteúdo dos dados não for simétrico.
	 * @throws IllegalArgumentException se o índice fornecido for inválido.
	 */
	public void categorizar(Dados dados, int id) {
		md.categorizar(dados, id);
	}

	/**
	 * Une os dois conjuntos de dados fornecidos
	 * <p>
	 *    A união adiciona as linhas do conteúdo de B ao final das linhas
	 *    do conteúdo de A. 
	 * </p>
	 * Exemplo:
	 * <pre>
	 * a = [
	 *    "a", "b", "c" 
	 *    "d", "e", "f" 
	 *    "g", "h", "i" 
	 * ]
	 *
	 * b = [
	 *    "j", "k", "l" 
	 *    "m", "n", "o" 
	 *    "p", "q", "r" 
	 * ]
	 *
	 * união = [
	 *    "a", "b", "c" 
	 *    "d", "e", "f" 
	 *    "g", "h", "i" 
	 *    "j", "k", "l" 
	 *    "m", "n", "o" 
	 *    "p", "q", "r" 
	 * ]
	 * </pre>
	 * @param a primeiro conjunto de dados.
	 * @param b segnudo conjunto de dados.
	 * @return novo conjunto de dados contendo a união entre A e B.
	 * @throws IllegalArgumentException se o conteúdo de A não for simétrico.
	 * @throws IllegalArgumentException se o conteúdo de B não for simétrico.
	 * @throws IllegalArgumentException se a quantidade de colunas de A e B forem diferentes.
	 */
	public Dados unir(Dados a, Dados b) {
		return md.unir(a, b);
	}

	/**
	 * Une o conteúdo de cada coluna dentro de A e B num
	 * novo conjunto de dados.
	 * <p>
	 *    A lógica de união envolve primeiro adicionar o conteúdo da linha de A e
	 *    depois adicionar o conteúdo da linha de B.
	 * </p>
	 *    Exemplo:
	 * <pre>
	 * a = [
	 *    1, 2
	 *    4, 5
	 * ]
	 *
	 * b = [
	 *    3
	 *    6
	 * ]
	 *
	 * união = [
	 *    1, 2, 3
	 *    4, 5, 6
	 *    7, 8, 9
	 * ]
	 * </pre>
	 * @param a primeiro conjunto de dados.
	 * @param b segundo conjunto de dados.
	 * @return novo objeto do tipo {@code Dados} conténdo a união por colunas de A com B.
	 * @throws IllegalArgumentException caso a quantiade de linhas de A e B sejam diferentes.
	 */
	public Dados unirCols(Dados a, Dados b) {
		return md.unirColuna(a, b);
	}

	/**
	 * Remove linhas repetidas dentro do conjunto de dados.
	 * <p>
	 *    Exemplo:
	 * </p>
	 * <pre>
	 * a = [
	 *    1, 2, 3
	 *    4, 5, 6
	 *    4, 5, 6
	 *    7, 8, 9
	 * ]
	 *
	 * sem duplicadas = [
	 *    1, 2, 3
	 *    4, 5, 6
	 *    7, 8, 9
	 * ]
	 * </pre>
	 * @param dados conjunto de dados.
	 */
	public void remDuplicadas(Dados dados) {
		md.removerDuplicadas(dados);
	}

	/**
	 * Normaliza todos os valores numéricos contidos no conjunto de dados.
	 * <p>
	 *    Caso alguma coluna possua algum valor que não possa ser convertido o 
	 *    processo é cancelado e a coluna não sofrerá alterações.
	 * </p>
	 * Exemplo:
	 * <pre>
	 * dados = [
	 *    1, 5 
	 *    2, a
	 *    3, 7
	 *    4, 8
	 *    5, 9
	 * ]
	 *
	 * normalizado = [
	 *    0.00, 5 
	 *    0.25, a
	 *    0.50, 7
	 *    0.75, 8
	 *    1.00, 9
	 * ]
	 * </pre>
	 * @param dados conjunto de dados.
	 * @throws IllegalArgumentException se o conteúdo dos dados estiver vazio.
	 * @throws IllegalArgumentException se o conteúdo dos dados não for simétrico.
	 */
	public void normalizar(Dados dados) {
		md.normalizar(dados);
	}

	/**
	 * Normaliza os valores numéricos contido na coluna fornecida.
	 * <p>
	 *    Caso a coluna possua algum valor que não possa ser convertido o 
	 *    processo é cancelado.
	 * </p>
	 * Exemplo:
	 * <pre>
	 * dados = [
	 *    1, 5 
	 *    2, a
	 *    3, 7
	 *    4, 8
	 *    5, 9
	 * ]
	 *
	 * normalizar(dados, 0);
	 *
	 * dados = [
	 *    0.00, 5 
	 *    0.25, a
	 *    0.50, 7
	 *    0.75, 8
	 *    1.00, 9
	 * ]
	 * </pre>
	 * @param dados conjunto de dados.
	 * @throws IllegalArgumentException se o conteúdo dos dados não for simétrico.
	 */
	public void normalizar(Dados dados, int id) {
		dados.normalizar(id);
	}

	/**
	 * Captaliza todo valor alfabético contido no conteúdo dos dados.
	 * <p>
	 *    Exemplo
	 * </p>
	 * <pre>
	 * a = [
	 *    UM,   doIS
	 *    trÊs, QuAtRo 
	 * ]
	 *
	 * capitalizado = [
	 *    Um,   Dois
	 *    Três, Quatro 
	 * ]
	 * </pre>
	 * @param dados conjunto de dados.
	 * @throws IllegalArgumentException se o conteúdo dos dados for vazio.
	 * @throws IllegalArgumentException se o conteúdo dos dados não for simétrico.
	 */
	public void capitalizar(Dados dados) {
		md.capitalizar(dados);
	}

	/**
	 * Captaliza todo valor alfabético contido no conteúdo dos dados.
	 * <p>
	 *    Exemplo
	 * </p>
	 * <pre>
	 * a = [
	 *    UM,   doIS
	 *    trÊs, QuAtRo 
	 * ]
	 *
	 * capitalizar(dados, 0);
	 *
	 * capitalizado = [
	 *    Um,   doIS
	 *    Três, QuAtRo 
	 * ]
	 * </pre>
	 * @param dados conjunto de dados.
	 * @param id índice da coluna desejada.
	 * @throws IllegalArgumentException se o conteúdo dos dados for vazio.
	 * @throws IllegalArgumentException se o conteúdo dos dados não for simétrico.
	 * @throws IllegalArgumentException se o índice da coluna fornecida for inválido.
	 */
	public void capitalizar(Dados dados, int id) {
		dados.capitalizar(id);
	}

	/**
	 * Substitui pelo novo valor todo o conteúdo encontrado na coluna de acordo com a busca.
	 * <p>
	 *    Exemplo:
	 * </p>
	 * <pre>
	 * d = [
	 *    a.b, c.d
	 *    e.f, g.h  
	 * ]
	 * 
	 * substituir(dados, 0, ".", "");
	 * 
	 * d = [
	 *    ab, c.d
	 *    ef, g.h  
	 * ]
	 * </pre>
	 * @param idCol índice da coluna desejada.
	 * @param busca valor que será substituído.
	 * @param valor novo valor que será colocado no lugar.
	 * @throws IllegalArgumentException se o conteúdo dos dados estiver vazio.
	 * @throws IllegalArgumentException se os dados não forem simétricos.
	 * @throws IllegalArgumentException e o índice da coluna for inválido.
	 */
	public void substituir(Dados dados, int idCol, String busca, String valor) {
		dados.substituir(idCol, null, null);
	}

	/**
	 * Substitui pelo novo valor todo o conteúdo encontrado dentro do conteúdo dos dados.
	 * <p>
	 *    Exemplo:
	 * </p>
	 * <pre>
	 * d = [
	 *    a.b, c.d
	 *    e.f, g.h  
	 * ]
	 * 
	 * substituir(dados, ".", "");
	 * 
	 * d = [
	 *    ab, cd
	 *    ef, gh  
	 * ]
	 * </pre>
	 * @param idCol índice da coluna desejada.
	 * @param busca valor que será substituído.
	 * @param valor novo valor que será colocado no lugar.
	 * @throws IllegalArgumentException se o conteúdo dos dados estiver vazio.
	 * @throws IllegalArgumentException se os dados não forem simétricos.
	 */
	public void substituir(Dados dados, String busca, String valor) {
		md.substituir(dados, busca, valor);
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
	 *    30
	 *    40
	 *    10
	 *    50
	 *    20
	 * ]
	 * 
	 * ordenar(d, 0, true).
	 * 
	 * d = [
	 *    10
	 *    20
	 *    30
	 *    40
	 *    50
	 * ]
	 * </pre>
	 * @param dados conjunto de dados.
	 * @param id índice da coluna desejada.
	 * @param cres true caso a ordenação deva ser crescente, false caso contrário.
	 * @throws IllegalArgumentException se o conteúdo dos dados estiver vazio.
	 * @throws IllegalArgumentException se o conteúdo dos dados não forem simétricos.
	 * @throws IllegalArgumentException se o índice da coluna for inválido.
	 * @throws IllegalArgumentException se a coluna conter valores que não possam ser convertidos para números.
	 */
	public void ordenar(Dados dados, int id, boolean cres) {
		dados.ordenar(id, cres);
	}

	/**
	 * Filtra o conteúdo contido nos dados fornecidos de 
	 * acordo com o valor de busca.
	 * <p>
	 *    A busca retorna toda a linha caso o valor desejado seja encontrado. É importante 
	 *    que o valor seja exatamente igual ao que está no conteúdo dos dados.
	 * </p>
	 * Exemplo:
	 * <pre>
	 * dados = [
	 *    a, 1, 2
	 *    a, 3, 4
	 *    b, 5, 6
	 *    c, 7, 8  
	 * ]
	 *
	 * Dados filtro = filtrar(dados, 0, "a");
	 *
	 * filtro = [
	 *    a, 1, 2
	 *    a, 3, 4
	 * ]
	 * </pre>
	 * @param dados conjunto de dados.
	 * @param id índice da coluna para busca.
	 * @param busca valor de busca desejado.
	 * @return novo conjunto de dados contendo apenas as informações filtradas, caso 
	 *    não seja encontrado nenhum valor desejado, o conteúdo dos novos dados
	 *    estará {@code vazio}.
	 */
	public Dados filtrar(Dados dados, int id, String busca) {
		return md.filtrar(dados, id, busca);
	}

	/**
	 * <p>
	 *    Filtra o conteúdo numérico contido nos dados fornecidos de 
	 *    acordo com o operador fornecido.
	 * </p>
	 * A ordem da filtragem segue o seguinte critério:
	 * <p>
	 *    {@code valorContidoNosDados (operador) valorDesejado}
	 * </p>
	 * Dados que não possam ser convertidos serão desconsiderados e consequentemente 
	 * não incluídos no resultado filtrado.
	 * <p>
	 *    Operadores suportados:
	 * </p>
	 * <ul>
	 *    <li> {@code >} </li>
	 *    <li> {@code >=} </li>
	 *    <li> {@code <} </li>
	 *    <li> {@code <=} </li>
	 *    <li> {@code ==} </li>
	 *    <li> {@code !=} </li>
	 * </ul>
	 * @param dados conjunto de dados.
	 * @param id índice da coluna para busca.
	 * @param op operador desejado.
	 * @param valor valor de busca desejado.
	 * @return novo conjunto de dados contendo apenas as informações filtradas, caso 
	 *    não seja encontrado nenhum valor desejado, o conteúdo dos novos dados
	 *    estará {@code vazio}.
	 * @throws IllegalArgumentException se o conteúdo dos dados estiver vazio.
	 * @throws IllegalArgumentException se o índice da coluna for inválido.
	 * @throws IllegalArgumentException se o operador fornecido não for suportado.
	 */
	public Dados filtrar(Dados dados, int id, String op, String valor) {
		return md.filtrar(dados, id, op, valor);
	}

	/**
	 * Substitui todos os valores ausentes no conteúdos dos dados fornecidos.
	 * <p>
	 *    São considerados dados ausentes quaisquer valores {@code vazio},
	 *    {@code em branco} e {@code com "?"}
	 * </p>
	 * Valores de preenchimento como {@code média}, {@code mediana} entre outros, podem 
	 * ser obtidos diretamente pelo objeto {@code Dados}.
	 * @param dados conjunto de dados.
	 * @param valor valor de preenchimento em elementos ausentes.
	 */
	public void preencherAusentes(Dados dados, double valor) {
		md.preencherAusentes(dados, valor);
	}

	/**
	 * Clona o conteúdo dos dados fornecidos em uma nova estrutura e devolve um novo objeto
	 * de dados conteúdo o mesmo conteúdo do original.
	 * @param dados conjunto de dados originais.
	 * @return novo objeto do tipo {@code Dados} com o mesmo conteúdo do original.
	 */
	public Dados clonarDados(Dados dados) {
		return md.clonarDados(dados);
	}

	/**
	 * Substitui todos os valores ausentes no conteúdos dos dados fornecidos de 
	 * acordo com a coluna informada.
	 * <p>
	 *    São considerados dados ausentes quaisquer valores {@code vazio},
	 *    {@code em branco} e {@code com "?"}
	 * </p>
	 * Valores de preenchimento como {@code média}, {@code mediana} entre outros, podem 
	 * ser obtidos diretamente pelo objeto {@code Dados}.
	 * @param dados conjunto de dados.
	 * @param id índice da coluna desejada.
	 * @param valor valor de preenchimento em elementos ausentes.
	 */
	public void preencherAusentes(Dados dados, int id, double valor) {
		md.preencherAusentes(dados, id, valor);
	}

	/**
	 * Descreve as dimensões do conteúdo dos dados, tanto em questão de quantidade de linhas 
	 * quanto quantidade de colunas.
	 * @param dados conjunto de dados.
	 * @return array contendo as informações das dimensões do conteúdo do conjunto de dados, o primeiro elemento 
	 * corresponde a quantidade de linhas e o segundo elemento corresponde a quantidade de colunas seguindo o 
	 * formato {@code [linhas, colunas]}.
	 * @throws IllegalArgumentException se o conteúdo estiver vazio.
	 * @throws IllegalArgumentException se os dados não forem simétricos, tendo colunas com tamanhos diferentes.
	 */
	public int[] shapeDados(Dados dados) {
		return dados.shape();
	}

	//GERENCIADOR DE ARQUIVOS ---------------------

	/**
	 * Lê o arquivo .csv de acordo com o caminho especificado.
	 * Espaços contidos serão removidos.
	 * <p>
	 *    O formato da estrutura de dados será um objeto do tipo 
	 *    {@code Dados}, contendo as linhas e colunas das informações lidas.
	 * </p>
	 * @param caminho caminho do arquivo, com extensão.
	 * @return {@code Dados} contendo as informações do arquivo lido.
	 * @throws IllegalArgumentException caso não encontre o diretório fornecido.
	 * @throws IllegalArgumentException caso arquivo não possua a extensão .csv.
	 */
	public Dados lerCsv(String caminho) {
		return ga.lerCsv(caminho);
	}

	/**
	 * Lê o arquivo .txt de acordo com o caminho especificado.
	 * Espaços contidos serão removidos.
	 * <p>
	 *    A separação do arquivo txt é considerada usando espaços
	 * </p>
	 * <p>
	 *    O formato da estrutura de dados será um objeto do tipo 
	 *    {@code Dados}, contendo as linhas e colunas das informações lidas.
	 * </p>
	 * @param caminho caminho do arquivo, com extensão.
	 * @return {@code Dados} contendo as informações do arquivo lido.
	 * @throws IllegalArgumentException caso não encontre o diretório fornecido.
	 * @throws IllegalArgumentException caso arquivo não possua a extensão .txt.
	 */
	public Dados lerTxt(String caminho) {
		return ga.lerTxt(caminho);
	}

	/**
	 * Grava o conteúdo do conjunto de dados em um arquivo {@code .csv}.
	 * @param dados conjunto de dados.
	 * @param caminho caminho do arquivo onde os dados serão gravados, excluindo a extensão .csv.
	 */
	public void exportarCsv(Dados dados, String caminho) {
		ga.exportarCsv(dados, caminho);
	}

	/**
	 * Grava o conteúdo do conjunto de dados em um arquivo {@code .txt}.
	 * @param dados conjunto de dados.
	 * @param caminho caminho do arquivo onde os dados serão gravados, excluindo a extensão .txt.
	 */
	public void exportarTxt(Dados dados, String caminho) {
		ga.exportarTxt(dados, caminho);
	}

	// GERENCIADOR TREINO TESTE ---------------------

	/**
	 * Embaralha o conjunto de dados aleatoriamente.
	 * <p>
	 *    A alteração irá afetar o conteúdo dos dados recebidos.
	 *    Caso queira manter os dados originais, é recomendado fazer uma cópia previamente.
	 * </p>
	 * Dados suportados: 
	 * <pre>
	 *int[][];
	 *float[][];
	 *double[][];
	 *</pre>
	 * @param dados matriz de conjunto de dados.
	 */
	public void embaralharDados(Object dados) {
		gtt.embaralharDados(dados);
	}

	/**
	 * <p>
	 *    Método para treino da rede neural.
	 * </p>
	 * Separa os dados que serão usados como entrada de acordo com os valores fornecidos.
	 * <p>
	 *    A lógica de separação dos dados de entrada envolve iniciar a coleta das colunas em ordem crescente,
	 *    exemplo: 
	 * </p>
	 * <pre>
	 * dados = [
	 *    1, 2, 3
	 *    4, 5, 6
	 *    7, 8, 9    
	 * ]
	 *
	 * entrada = (int[][]) separarDadosEntrdada(dados, 2);
	 *
	 * entrada = [
	 *    1, 2
	 *    4, 5
	 *    7, 8 
	 * ]
	 * </pre>
	 * <pre>
	 *int[][];
	 *float[][];
	 *double[][];
	 *</pre>
	 * @param dados conjunto de dados completo.
	 * @param colunas quantidade de colunas que serão preservadas, começando pela primeira até o valor fornecido.
	 * @return nova matriz de dados apenas com as colunas desejadas.
	 * @throws IllegalArgumentException Se o número de colunas for maior que o número de colunas disponíveis nos dados.
	 * @throws IllegalArgumentException Se o número de colunas for menor que um.
	 */
	public Object separarDadosEntrada(Object dados, int colunas) {
		return gtt.separarDadosEntrada(dados, colunas);
	}

	/**
	 * <p>
	 *    Método para treino da rede neural.
	 * </p>
	 * Extrai os dados de saída do conjunto de dados e devolve um novo conjunto de dados contendo apenas as 
	 * colunas de dados de saída especificadas.
	 * <p>
	 *    A lógica de separação dos dados de saída envolve iniciar a coleta das colunas em ordem decrescente,
	 *    exemplo: 
	 * </p>
	 * <pre>
	 * dados = [
	 *    1, 2, 3
	 *    4, 5, 6
	 *    7, 8, 9
	 * ]
	 *
	 * entrada = (int[][]) separarDadosSaida(dados, 2);
	 *
	 * saida = [
	 *    2, 3
	 *    5, 6
	 *    8, 9
	 * ]
	 * </pre>
	 * <pre>
	 *int[][];
	 *float[][];
	 *double[][];
	 *</pre>
	 * @param dados O conjunto de dados com as informações completas.
	 * @param colunas O número de colunas de dados de saída que serão extraídas.
	 * @return novo conjunto de dados com apenas as colunas de dados de saída.
	 * @throws IllegalArgumentException Se o número de colunas for maior que o número de colunas disponíveis nos dados.
	 * @throws IllegalArgumentException Se o número de colunas for menor que um.
	 */
	public Object separarDadosSaida(Object dados, int colunas) {
		return gtt.separarDadosSaida(dados, colunas);
	}

	/**
	 * Separa o conjunto de dados em dados de treino e dados de teste, de acordo com o tamanho do teste fornecido.
	 * 
	 * <p>
	 *    A função recebe um conjunto de dados completo e separa ele em duas matrizes, uma para treino e outra para teste.
	 *    A quantidade de dados para o conjunto de teste é determinada pelo parâmetro tamanhoTeste.
	 * </p>
	 * 
	 * <p>
	 *    Exemplo de uso:
	 * </p>
	 * <pre>{@code 
	 *int[][][] treinoTeste = (int[][][]) separarTreinoTeste(dados, 0.25f);
	 *int[][] treino = treinoTeste[0];
	 *int[][] teste = treinoTeste[1];}
	 * </pre>
	 * Dados suportados:
	 * <pre>
	 *int[][];
	 *float[][];
	 *double[][];
	 *</pre>
	 * @param dados O conjunto de dados completo.
	 * @param tamanhoTeste O tamanho relativo do conjunto de teste (entre 0 e 1).
	 * @return Um array de duas matrizes contendo os dados de treino e teste, respectivamente.
	 * @throws IllegalArgumentException caso o conjunto de dados for nulo.
	 * @throws IllegalArgumentException caso o tamanho de teste estiver fora do intervalo (0, 1).
	 */
	public Object separarTreinoTeste(Object dados, float tamanhoTeste) {
		return gtt.separarTreinoTeste(dados, tamanhoTeste);
	}

	//CONVERSOR DE DADOS ----------------------

	/**
	 * Converte o conteúdo do conjunto de dados para uma matriz bidimensional 
	 * com os valores numéricos.
	 * @param dados conjunto de dados.
	 * @return matriz convertida para valores tipo {@code int}.
	 */
	public int[][] dadosParaInt(Dados dados) {
		return cd.dadosParaInt(dados);
	}

	/**
	 * Converte o conteúdo do conjunto de dados para uma matriz bidimensional 
	 * com os valores numéricos.
	 * @param dados conjunto de dados.
	 * @return matriz convertida para valores tipo {@code float}.
	 */
	public float[][] dadosParaFloat(Dados dados) {
		return cd.dadosParaFloat(dados);
	}

	/**
	 * Converte o conteúdo do conjunto de dados para uma matriz bidimensional 
	 * com os valores numéricos.
	 * @param dados conjunto de dados.
	 * @return matriz convertida para valores tipo {@code double}.
	 */
	public double[][] dadosParaDouble(Dados dados) {
		return cd.dadosParaDouble(dados);
	}

	/**
	 * Converte o conjunto de dados para uma matriz bidimensional 
	 * com os conteúdo contido nos dados.
	 * @param dados conjunto de dados.
	 * @return matriz convertida para valores tipo {@code String}.
	 */
	public String[][] dadosParaString(Dados dados) {
		return cd.dadosParaString(dados);
	}

	//OPERADOR MATRIZ --------------

	/**
	 * Tranforma todo o conteúdo da matriz fornecida numa forma contínua
	 * de dados.
	 * <p>
	 *    Exemplo:
	 * </p>
	 * <pre>
	 * m = [
	 *    1, 2, 3 
	 *    4, 5, 6 
	 *    7, 8, 9 
	 * ]
	 *
	 * v = (int[]) vetorizar(m);
	 *
	 * v = [1, 2, 3, 4, 5, 6, 7, 8, 9]
	 * </pre>
	 * Dados suportados:
	 * <pre>
	 *int[][];
	 *float[][];
	 *double[][];
	 *</pre>
	 * @param matriz matriz com os dados desejados.
	 * @return arrays contendo os dados serializados da matriz.
	 */
	public Object matParaArray(Object matriz) {
		return om.paraArray(matriz);
	}

	/**
	 * Retorna uma nova matriz que possui o conteúdo das linhas 
	 * de acordo com os índices fornecidos.
	 * <p>
	 *    Exemplo:
	 * </p>
	 * <pre>
	 * dados = [
	 *    1, 2, 3
	 *    4, 5, 6
	 *    7, 8, 9
	 * ]
	 *
	 * subLinhas = (int[][]) obterSubLinhas(dados, 0, 2);
	 *
	 * subLinhas = [
	 *    1, 2, 3
	 *    4, 5, 6
	 * ]
	 * </pre>
	 * Dados suportados:
	 * <pre>
	 *int[][];
	 *float[][];
	 *double[][];
	 *</pre>
	 * @param dados matriz contendo os dados completos.
	 * @param inicio índice inicial do corte (inclusivo).
	 * @param fim índice final do corte (exclusivo).
	 * @return submatriz contendo o conteúdo da matriz original, com os dados selecionados.
	 * @throws IllegalArgumentException se os índices fornecidos forem inválidos.
	 */
	public Object matSublins(Object dados, int inicio, int fim) {
		return om.obterSubLins(dados, inicio, fim);
	}

	/**
	 * Retorna uma nova matriz que possui o conteúdo das colunas 
	 * de acordo com os índices fornecidos.
	 * <p>
	 *    Exemplo:
	 * </p>
	 * <pre>
	 * dados = [
	 *    1, 2, 3
	 *    4, 5, 6
	 *    7, 8, 9
	 * ]
	 *
	 * subColunas = (int[][]) obterColunas(dados, 0, 2);
	 *
	 * subColunas = [
	 *    1, 2
	 *    4, 5
	 *    7, 8
	 * ]
	 * </pre>
	 * Dados suportados:
	 * <pre>
	 *int[][];
	 *float[][];
	 *double[][];
	 *</pre>
	 * @param dados matriz contendo os dados completos.
	 * @param inicio índice inicial do corte (inclusivo).
	 * @param fim índice final do corte (exclusivo).
	 * @return submatriz contendo o conteúdo da matriz original, com os dados filtrados de acordo
	 *    com as colunas indicadas.
	 */
	public Object matSubcols(Object dados, int inicio, int fim) {
		return om.obterSubCols(dados, inicio, fim);
	}

	/**
	 * Preenche cada elemento da matriz de acordo com o valor fornecido.
	 * <p>
	 *    O valor de preenchimento é automaticamente convertido para o mesmo 
	 *    tipo da matriz fornecida.
	 * </p>
	 * Dados suportados:
	 * <pre>
	 *int[][];
	 *float[][];
	 *double[][];
	 *</pre>
	 * @param matriz matriz com os dados.
	 * @param valor valor de preenchimento.
	 */
	public void matPreencher(Object matriz, Number valor) {
		om.preencherMatriz(matriz, valor);
	}

	/**
	 * Preenche o conteúdo da matriz para que fique no formato identidade, onde
	 * apenas os elementos da diagonal principal têm valores iguais a 1.
	 * <p>Exemplo:<pre>
	 * m = [
	 *  1, 0, 0
	 *  0, 1, 0
	 *  0, 0, 1
	 * ]
	 * </pre></p>
	 * Dados suportados:
	 * <pre>
	 *int[][];
	 *float[][];
	 *double[][];
	 *</pre>
	 * @param matriz matriz base.
	 */
	public void matId(Object matriz) {
		om.matId(matriz);
	}

	/**
	 * Realiza a transposição da matriz fornecida. A transposição consiste em 
	 * inverter as linhas e colunas da matriz.
	 * <p>Exemplo:<pre>
	 * m = [
	 *  1, 2, 3
	 *  4, 5, 6
	 *  7, 8, 9
	 * ]
	 * t = [
	 *  1, 4, 7
	 *  2, 5, 8
	 *  3, 6, 9
	 * ]
	 * </pre></p>
	 * @param matriz matriz original para transposição
	 * @return matriz transposta.
	 */
	public Object matTransp(Object matriz) {
		return om.matTransp(matriz);
	}

	/**
	 * Realiza a soma entre as duas matrizes fornecidas de acordo com a expressão 
	 * <pre>R = A + B</pre>
	 * <p>Exemplo:<pre>
	 * a = [
	 *   1, 1
	 *   1, 1
	 * ]
	 * b = [
	 *   2, 2
	 *   2, 2
	 * ]
	 * r = [
	 *   3, 3
	 *   3, 3
	 * ]
	 * </pre></p>
	 * Dados suportados:
	 * <pre>
	 *int[][];
	 *float[][];
	 *double[][];
	 *</pre>
	 * @param a primeira matriz.
	 * @param b segunda matriz.
	 * @param r matriz que conterá o resultado.
	 * @throws IllegalArgumentException se as dimensões de A, B e R forem incompatíveis.
	 */
	public void matAdd(Object a, Object b, Object r) {
		om.matAdd(a, b, r);
	}

	/**
	 * Realiza a subtração entre as duas matrizes fornecidas de acordo com a expressão 
	 * <pre>R = A - B</pre>
	 * <p>Exemplo:<pre>
	 * a = [
	 *   1, 1
	 *   1, 1
	 * ]
	 * b = [
	 *   2, 2
	 *   2, 2
	 * ]
	 * r = [
	 *   -1, -1
	 *   -1, -1
	 * ]
	 * </pre></p>
	 * Dados suportados:
	 * <pre>
	 *int[][];
	 *float[][];
	 *double[][];
	 *</pre>
	 * @param a primeira matriz.
	 * @param b segunda matriz.
	 * @param r matriz que conterá o resultado.
	 * @throws IllegalArgumentException se as dimensões de A, B e R forem incompatíveis.
	 */
	public void matSub(Object a, Object b, Object r) {
		om.matSub(a, b, r);
	}

	/**
	 * Realiza a multiplicação da matriz A pela matriz B.
	 * <p>Exemplo:<pre>
	 * a = [
	 *   1, 2
	 *   3, 4
	 * ]
	 * b = [
	 *   5, 6
	 *   7, 8
	 * ]
	 * r = [
	 *   19.0, 22.0
	 *   43.0, 50.0
	 * ]
	 * </pre></p>
	 * Dados suportados:
	 * <pre>
	 *int[][];
	 *float[][];
	 *double[][];
	 *</pre>
	 * @param a primeira matriz.
	 * @param b segunda matriz.
	 * @param r matriz que conterá o resultado.
	 * @throws IllegalArgumentException se as dimensões de A, B e R forem incompatíveis.
	 */
	public void matMult(Object a, Object b, Object r) {
		om.matMult(a, b, r);
	}

	/**
	 * Multiplica cada elemento da matriz por um valor escalar fornecido.
	 * <p>Exemplo:<pre>
	 * a = [
	 *   1, 2
	 *   3, 4
	 * ]
	 * 
	 * escalar = 2;
	 * 
	 * r = [
	 *   2, 4
	 *   6, 8
	 * ]
	 * </pre></p>
	 * <p>
	 *    O valor do escalar é automaticamente convertido para o mesmo 
	 *    tipo da matriz fornecida.
	 * </p>
	 * Dados suportados:
	 * <pre>
	 *int[][];
	 *float[][];
	 *double[][];
	 *</pre>
	 * @param matriz matriz contendo os dados.
	 * @param escalar escalar para a ultiplicação.
	 */
	public void matMultEscalar(Object matriz, Number escalar) {
		om.matMultEscalar(matriz, escalar);
	}

	/**
	 * Multiplica cada elemento da matriz A pelo mesmo elementos correspondente
	 * na matriz B.
	 * <p>Exemplo:<pre>
	 * a = [
	 *   1, 2
	 *   3, 4
	 * ]
	 * b = [
	 *   1, 2
	 *   3, 4
	 * ]
	 * r = [
	 *   1, 4
	 *   16, 25
	 * ]
	 * </pre></p>
	 * Dados suportados:
	 * <pre>
	 *int[][];
	 *float[][];
	 *double[][];
	 *</pre>
	 * @param a primeira matriz.
	 * @param b segunda matriz.
	 * @param r matriz que conterá o resultado.
	 * @throws IllegalArgumentException se as dimensões de A, B e R forem incompatíveis.
	 */
	public void matHadamard(Object a, Object b, Object r) {
		om.hadamard(a, b, r);
	}
}
