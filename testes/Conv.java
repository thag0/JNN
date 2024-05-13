package testes;

import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.concurrent.TimeUnit;

import jnn.camadas.*;
import jnn.core.tensor.OpTensor;
import jnn.core.tensor.Tensor;
import jnn.core.tensor.Variavel;
import jnn.modelos.Sequencial;
import jnn.otimizadores.Otimizador;
import jnn.serializacao.Serializador;
import lib.ged.Dados;
import lib.ged.Ged;
import lib.geim.Geim;

public class Conv{
	static Ged ged = new Ged();
	static Geim geim = new Geim();
	static OpTensor optensor = new OpTensor();
	static Serializador serializador = new Serializador();
	static final int amostras = 100;
	static final int digitos = 10;

	static final String CAMINHO_MODELOS = "./dados/modelos/";
	
	public static void main(String[] args){
		ged.limparConsole();

		// String nomeModelo = "mlp-mnist-89";
		// String nomeModelo = "conv-mnist-95-8";
		String nomeModelo = "modelo-treinado";
		Sequencial modelo = serializador.lerSequencial(CAMINHO_MODELOS + nomeModelo + ".nn");
		// modelo.info();

		testarPrevisao(modelo, "treino/3/img_1", true);
		testarPrevisao(modelo, "3_deslocado", true);

		// testarAcertosMNIST(modelo);
		// testarTodosDados(modelo);

		// Dados forward = tempoForward(modelo);//media 83/120 ms
		// Dados backward = tempoBackward(modelo);//media 118/133 ms
		// forward = ged.filtrar(forward, 1, "Convolucional");
		// backward = ged.filtrar(backward, 1, "Convolucional");
		// forward.print();
		// backward.print();

		// testarForward();
		// testarBackward();

		// tempoOtimizador(modelo);
	}

	static void testarAcertosMNIST(Sequencial modelo){
		final String caminho = "/dados/mnist/teste/";
		
		double media = 0;
		for(int digito = 0; digito < digitos; digito++){

			double acertos = 0;
			for(int amostra = 0; amostra < amostras; amostra++){
				String caminhoImagem = caminho + digito + "/img_" + amostra + ".jpg";
				Tensor img = new Tensor(imagemParaMatriz(caminhoImagem));
				
				double[] previsoes = modelo.forward(img).paraArrayDouble();
				if(maiorIndice(previsoes) == digito){
					acertos++;
				}
			}

			double porcentagem = acertos / (double)amostras;
			media += porcentagem;
			System.out.println("Acertos " + digito + " -> " + porcentagem);
		}
		System.out.println("média acertos: " + String.format("%.2f", (media/digitos)*100) + "%");
	}

	static void testarTodosDados(Sequencial modelo){
		for(int i = 0; i < digitos; i++){
			for(int j = 0; j < amostras; j++){
				testarPrevisao(modelo, ("teste/" + i + "/img_" + j), false);
			}
			System.out.println();
		}
	}

	static long medirTempo(Runnable func){
		long t1 = System.nanoTime();
		func.run();
		return System.nanoTime() - t1;
	}

	static Dados tempoForward(Sequencial modelo){
		//arbritário
		double[][] img = imagemParaMatriz("./dados/mnist/teste/1/img_0.jpg");
		double[][][] entrada = new double[1][][];
		entrada[0] = img;

		int n = modelo.numCamadas();
		long t, total = 0;

		Dados dados = new Dados();
		dados.editarNome("Tempos Forward");
		ArrayList<String[]> conteudo = new ArrayList<>();

		t = medirTempo(() -> modelo.camada(0).forward(entrada));
		conteudo.add(new String[]{
			"0",
			modelo.camada(0).nome(),
			String.valueOf(TimeUnit.NANOSECONDS.toMillis(t)) + " ms"        
		});
		total += t;
		
		for(int i = 1; i < n; i++){
			Camada atual = modelo.camada(i);
			Camada anterior = modelo.camada(i-1);
			t = medirTempo(() -> {
				atual.forward(anterior.saida());
			});
			total += t;

			conteudo.add(new String[]{
				String.valueOf(i),
				modelo.camada(i).nome(),
				String.valueOf(TimeUnit.NANOSECONDS.toMillis(t)) + " ms"        
			});
		}
		conteudo.add(new String[]{
			"-",
			"Tempo total", 
			String.valueOf(TimeUnit.NANOSECONDS.toMillis(total)) + " ms"
		});

		dados.atribuir(conteudo);
		dados.editarNome("tempos forward");
		return dados;
	}

	static Dados tempoBackward(Sequencial modelo){
		//arbritário
		double[] grad = new double[modelo.saidaParaArray().length];
		grad[0] = 1;
		for(int i = 1; i < grad.length; i++){
			grad[i] = 0.02;
		}

		int n = modelo.numCamadas();
		long t, total = 0;

		Dados dados = new Dados();
		dados.editarNome("Tempos Backward");
		ArrayList<String[]> conteudo = new ArrayList<>();

		Tensor g = new Tensor(grad.length);
		g.copiarElementos(grad);
		t = medirTempo(() -> modelo.camada(n-1).backward(g));
		conteudo.add(new String[]{
			String.valueOf(n-1),
			modelo.camada(n-1).nome(),
			String.valueOf(TimeUnit.NANOSECONDS.toMillis(t)) + " ms"        
		});
		total += t;
		
		for(int i = n-2; i >= 0; i--){
			final int id = i;
			Camada atual = modelo.camada(id);
			Camada proxima = modelo.camada(id+1);
			t = medirTempo(() -> {
				atual.backward(proxima.gradEntrada());
			});
			total += t;

			conteudo.add(new String[]{
				String.valueOf(i),
				modelo.camada(i).nome(),
				String.valueOf(TimeUnit.NANOSECONDS.toMillis(t)) + " ms"        
			});
		}
		conteudo.add(new String[]{
			"-",
			"Tempo total",
			String.valueOf(TimeUnit.NANOSECONDS.toMillis(total)) + " ms"        
		});

		dados.atribuir(conteudo);
		dados.editarNome("tempos backward");
		return dados;
	}

	static void tempoOtimizador(Sequencial modelo){
		Otimizador otm = modelo.otimizador();

		//arbritário
		int tamGrad = modelo.saidaParaArray().length;
		double[] grad = new double[tamGrad];
		grad[0] = 1;
		for(int i = 1; i < grad.length; i++){
			grad[i] = 0.02;
		}

		//backward simples
		modelo.camadaSaida().backward(new Tensor(grad, tamGrad));
		for(int i = modelo.numCamadas()-2; i >= 0; i--){
			modelo.camada(i).backward(modelo.camada(i+1).gradEntrada());
		}

		long t = System.nanoTime();
		otm.atualizar(modelo.camadas());
		t = System.nanoTime() - t;

		System.out.println(
			"Tempo otimizador (" + otm.nome() + "): " + TimeUnit.NANOSECONDS.toMillis(t) + "ms"
		);
	}

	static void testarPrevisao(Sequencial modelo, String caminhoImagem, boolean prob){
		String extensao = ".jpg";
		Tensor entrada = new Tensor(imagemParaMatriz("./dados/mnist/" + caminhoImagem + extensao));
		entrada.unsqueeze(0);
		Variavel[] previsao = modelo.forward(entrada).paraArray();
		
		System.out.print("\nTestando: " + caminhoImagem + extensao);
		if(prob){
			System.out.println();
			for(int i = 0; i < previsao.length; i++){
				System.out.println("Prob: " + i + ": " + (int)(previsao[i].get()*100) + "%");
			}
		}else{
			System.out.println(" -> Prev: " + maiorIndice((previsao)));
		}

	}

	static int maiorIndice(double[] arr){
		int id = 0;
		double maior = arr[0];

		for(int i = 1; i < arr.length; i++){
			if(arr[i] > maior){
				id = i;
				maior = arr[i];
			}
		}

		return id;
	}

	static int maiorIndice(Variavel[] arr){
		int id = 0;
		double maior = arr[0].get();

		for(int i = 1; i < arr.length; i++){
			if(arr[i].get() > maior){
				id = i;
				maior = arr[i].get();
			}
		}

		return id;
	}

	static double[][] imagemParaMatriz(String caminho){
		BufferedImage img = geim.lerImagem(caminho);
		double[][] imagem = new double[img.getHeight()][img.getWidth()];

		int[][] cinza = geim.obterCinza(img);

		for(int y = 0; y < imagem.length; y++){
			for(int x = 0; x < imagem[y].length; x++){
				imagem[y][x] = (double)cinza[y][x] / 255;
			}
		}
		return imagem;
	}

	public static double[][][][] carregarDadosMNIST(String caminho, int amostras, int digitos){
		double[][][][] entradas = new double[digitos * amostras][1][][];

		int id = 0;
		for(int i = 0; i < digitos; i++){
			for(int j = 0; j < amostras; j++){
				String caminhoCompleto = caminho + i + "/img_" + j + ".jpg";
				double[][] imagem = imagemParaMatriz(caminhoCompleto);
				entradas[id++][0] = imagem;
			}
		}

		System.out.println("Imagens carregadas. (" + entradas.length + ")");
		return entradas;
	}

	public static double[][] criarRotulosMNIST(int amostras, int digitos){
		double[][] rotulos = new double[digitos * amostras][digitos];
		for(int numero = 0; numero < digitos; numero++){
			for(int i = 0; i < amostras; i++){
				int indice = numero * amostras + i;
				rotulos[indice][numero] = 1;
			}
		}
		
		System.out.println("Rótulos gerados de 0 a " + (digitos-1) + ".");
		return rotulos;
	}
}
