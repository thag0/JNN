package testes;

import java.awt.image.BufferedImage;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
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

		// String nomeModelo = "mlp-mnist-89-1";
		String nomeModelo = "conv-mnist-97-1";
		// String nomeModelo = "modelo-treinado";
		Sequencial modelo = serializador.lerSequencial(CAMINHO_MODELOS + nomeModelo + ".nn");
		// modelo.print();

		// testarPrevisao(modelo, "treino/3/img_1", true);
		// testarPrevisao(modelo, "3_deslocado", true);

		// testarAcertosMNIST(modelo);
		// testarTodosDados(modelo);

		Dados forward = tempoForward(modelo);//media 32/44ms
		Dados backward = tempoBackward(modelo);//media 58/125 ms
		// forward = ged.filtrar(forward, 1, "MaxPool2D");
		// backward = ged.filtrar(backward, 1, "MaxPool2D");
		forward.print();
		backward.print();

		// tempoOtimizador(modelo);
	}

	static void testarAcertosMNIST(Sequencial modelo){
		final String caminho = "./dados/mnist/teste/";
		
		double media = 0;
		for(int digito = 0; digito < digitos; digito++){

			Tensor[] imagens = new Tensor[amostras];

			for(int amostra = 0; amostra < amostras; amostra++){
				String caminhoImagem = caminho + digito + "/img_" + amostra + ".jpg";
				Tensor img = new Tensor(imagemParaMatriz(caminhoImagem));
				img.unsqueeze(0);// 2d -> 3d
				imagens[amostra] = img;
			}

			double acertos = 0;
			Tensor[] prevs = modelo.forwards(imagens);
			for(Tensor t : prevs) {
				double[] previsoes = t.paraArrayDouble();
				if(maiorIndice(previsoes) == digito){
					acertos++;
				}
			}

			double porcentagem = acertos / (double)amostras;
			media += porcentagem;
			System.out.println("Acertos " + digito + " -> " + porcentagem + "%");
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
		Tensor entrada = new Tensor(modelo.camada(0).formatoEntrada());
		entrada.aplicar(x -> Math.random());

		int n = modelo.numCamadas();
		long t, total = 0;

		Dados dados = new Dados();
		dados.setNome("Tempos Forward");
		ArrayList<String[]> conteudo = new ArrayList<>();

		DecimalFormat df = new DecimalFormat();

		t = medirTempo(() -> modelo.camada(0).forward(entrada));
		conteudo.add(new String[]{
			"0",
			modelo.camada(0).nome(),
			df.format(t) + " ns"        
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
				df.format(t) + " ns"        
			});
		}
		conteudo.add(new String[]{
			"-",
			"Tempo total", 
			df.format(total) + " ns"
		});

		dados.atribuir(conteudo);
		dados.setNome("tempos forward");
		return dados;
	}

	static Dados tempoBackward(Sequencial modelo){
		//arbritário
		Tensor grad = new Tensor(modelo.camadaSaida().formatoSaida());
		grad.aplicar(x -> Math.random());

		int n = modelo.numCamadas();
		long t, total = 0;

		DecimalFormat df = new DecimalFormat();

		Dados dados = new Dados();
		dados.setNome("Tempos Backward");
		ArrayList<String[]> conteudo = new ArrayList<>();

		t = medirTempo(() -> modelo.camada(n-1).backward(grad));
		conteudo.add(new String[]{
			String.valueOf(n-1),
			modelo.camada(n-1).nome(),
			df.format(t) + " ns"        
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
				df.format(t) + " ns"        
			});
		}
		conteudo.add(new String[]{
			"-",
			"Tempo total",
			df.format(total) + " ns"        
		});

		dados.atribuir(conteudo);
		dados.setNome("tempos backward");
		return dados;
	}

	static void tempoOtimizador(Sequencial modelo){
		Otimizador otm = modelo.otimizador();

		//arbritário
		Tensor grad = new Tensor(modelo.saidaParaArray().length);
		grad.aplicar(x -> Math.random());

		//backward simples
		modelo.camadaSaida().backward(grad);
		for(int i = modelo.numCamadas()-2; i >= 0; i--){
			modelo.camada(i).backward(modelo.camada(i+1).gradEntrada());
		}

		long t = medirTempo(() -> otm.atualizar(modelo.camadas()));

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

	static double[][][][] carregarDadosMNIST(String caminho, int amostras, int digitos) {
		final double[][][][] imagens = new double[digitos * amostras][1][][];
		final int numThreads = Runtime.getRuntime().availableProcessors() / 2;
  
		try (ExecutorService exec = Executors.newFixedThreadPool(numThreads)) {
			int id = 0;
			for (int i = 0; i < digitos; i++) {
				for (int j = 0; j < amostras; j++) {
					final String caminhoCompleto = caminho + i + "/img_" + j + ".jpg";
					final int indice = id;
					
					exec.submit(() -> {
						try {
							double[][] imagem = imagemParaMatriz(caminhoCompleto);
							imagens[indice][0] = imagem;
						} catch (Exception e) {
							System.out.println(e.getMessage());
							System.exit(1);
						}
					});

					id++;
				}
			}
  
		} catch (Exception e) {
			System.out.println(e.getMessage());
		}
  
		System.out.println("Imagens carregadas (" + imagens.length + ").");
  
		return imagens;
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
