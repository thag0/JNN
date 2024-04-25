package render;

import java.awt.image.BufferedImage;
import java.io.File;

import javax.imageio.ImageIO;
import javax.swing.JFrame;

import jnn.modelos.Modelo;

public class JanelaTreino extends JFrame {

	public PainelTreino painelTreino;
	private int numThreads = 1;

	public JanelaTreino(int larguraImagem, int alturaImagem, double escala, int numThreads) {
		try {
			BufferedImage icone = ImageIO.read(new File("./render/rede-neural.png"));
			setIconImage(icone);

		} catch (Exception e) {}

		this.painelTreino = new PainelTreino(larguraImagem, alturaImagem, escala);
		
		setTitle("Treinamento rede");
		add(painelTreino);
		setVisible(true);
		pack();
		setResizable(false);
		setLocationRelativeTo(null);

		if (numThreads < 1) {
			throw new IllegalArgumentException(
				"O número de threads deve ser maior que zero."
			);
		}
		this.numThreads = numThreads;
	}


	public void desenharTreino(Modelo modelo, int epocasPorFrame) {
		if (numThreads == 1) {
			painelTreino.desenhar(modelo, epocasPorFrame);
		
		} else {
			painelTreino.desenhar(modelo, epocasPorFrame, this.numThreads);
		}
	}
}
