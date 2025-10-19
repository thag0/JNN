package externos.render;

import java.awt.image.BufferedImage;
import java.io.File;

import javax.imageio.ImageIO;
import javax.swing.JFrame;

import externos.render.widgets.PainelTreino;
import jnn.modelos.Modelo;

public class JanelaTreino extends JFrame {

	private PainelTreino pt;
	private int numThreads = 1;

	public JanelaTreino(int largura, int altura, double escala, int numThreads) {
		try {
			BufferedImage icone = ImageIO.read(new File("./render/rede-neural.png"));
			setIconImage(icone);

		} catch (Exception e) {}

		pt = new PainelTreino(largura, altura, escala);
		
		setTitle("Treino");
		setVisible(true);
		add(pt);
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
			pt.desenhar(modelo, epocasPorFrame);
		
		} else {
			pt.desenhar(modelo, epocasPorFrame, this.numThreads);
		}
	}
}
