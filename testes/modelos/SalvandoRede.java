package testes.modelos;

import java.io.Serializable;

import jnn.ativacoes.Sigmoid;
import jnn.avaliacao.perda.MSE;
import jnn.camadas.Camada;
import jnn.camadas.Conv2D;
import jnn.camadas.Densa;
import jnn.inicializadores.GlorotUniforme;
import jnn.inicializadores.Zeros;
import jnn.modelos.*;
import jnn.otimizadores.*;
import jnn.serializacao.Serializador;
import lib.ged.Dados;
import lib.ged.Ged;

@SuppressWarnings("unused")
public class SalvandoRede{
	public static void main(String[] args){
		Ged ged = new Ged();
		ged.limparConsole();
		String caminho = "./dados/modelos/modelo-treinado.nn";
		Serializador serializador = new Serializador();
		
		Sequencial modelo = ler(caminho);
		// Sequencial modelo = criar();

		modelo.print();

		// serializador.salvar(modelo, caminho);
	}

	static Sequencial criar(){
		Sequencial modelo = new Sequencial(new Camada[]{
			new Conv2D(new int[]{2, 10, 10}, 2, new int[]{3, 3}),
		});

		modelo.compilar("sgd", "mse");
		return modelo;
	}

	static Sequencial ler(String caminho){
		return new Serializador().lerSequencial(caminho); 
	}
}
