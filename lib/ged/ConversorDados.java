package lib.ged;


/**
 * Conversor de dados para o Ged
 */
class ConversorDados {
	
	/**
	 * Objeto responsável converter o conteúdo dos dados em
	 * arrays do tipo int, float e double.
	 */
	public ConversorDados() {}


	public int[][] dadosParaInt(Dados dados) {
		int[] shape = dados.shape();

		int[][] dadosConvertidos = new int[shape[0]][shape[1]];

		for (int i = 0; i < dadosConvertidos.length; i++) {
			for (int j = 0; j < dadosConvertidos[0].length; j++) { 
				try {
					dadosConvertidos[i][j] = Integer.parseInt(dados.obterItem(i, j));
				} catch (Exception e) {
					e.printStackTrace();
				}
			}
		}
	
		return dadosConvertidos;
	}

	
	public float[][] dadosParaFloat(Dados dados) {
		int[] shape = dados.shape();

		float[][] dadosConvertidos = new float[shape[0]][shape[1]];

		for (int i = 0; i < dadosConvertidos.length; i++) {
			for (int j = 0; j < dadosConvertidos[0].length; j++) {
				try {
					dadosConvertidos[i][j] = Float.parseFloat(dados.obterItem(i, j));
				} catch(Exception e) {
					e.printStackTrace();
				}
			}
		}
	
		return dadosConvertidos;
	}

	
	public double[][] dadosParaDouble(Dados dados) {
		int[] shape = dados.shape();

		double[][] dadosConvertidos = new double[shape[0]][shape[1]];

		for (int i = 0; i < dadosConvertidos.length; i++) {
			for (int j = 0; j < dadosConvertidos[0].length; j++) {
				try {
					dadosConvertidos[i][j] = Double.parseDouble(dados.obterItem(i, j));
				} catch(Exception e) {
					e.printStackTrace();
				}
			}
		}
	
		return dadosConvertidos;
	}

	
	public String[][] dadosParaString(Dados dados) {
		int[] shape = dados.shape();

		String[][] dadosConvertidos = new String[shape[0]][shape[1]];

		for (int i = 0; i < dadosConvertidos.length; i++) {
			for (int j = 0; j < dadosConvertidos[0].length; j++) {
				try {
					dadosConvertidos[i][j] = dados.obterItem(i, j);
				} catch(Exception e) {
					e.printStackTrace();
				}
			}
		}
	
		return dadosConvertidos;
	}

}
