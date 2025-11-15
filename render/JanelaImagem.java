package render;

import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;

import javax.swing.JFrame;

import jnn.core.tensor.Tensor;
import render.widgets.TensorImg;

public class JanelaImagem extends JFrame {
	
	TensorImg painel;
	LeitorTeclado leitorTeclado;

	public JanelaImagem(int altura, int largura, int escala, String titulo) {
		if(titulo == null) titulo = "Janela";
		else setTitle(titulo);

		this.painel = new TensorImg(altura*escala, largura*escala);
		add(painel);
		pack();

		setResizable(false);
		setVisible(true);
		setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		setLocationRelativeTo(null);
		
		leitorTeclado = new LeitorTeclado();
		addKeyListener(leitorTeclado); 
		setFocusable(true);
	}

	public JanelaImagem(int altura, int largura, String titulo){
		this(altura, largura, 1, titulo);
	}

	public JanelaImagem(int altura, int largura, int escala){
		this(altura, largura, escala, "Janela");
	}

	public JanelaImagem(int altura, int largura){
		this(altura, largura, 1, "Janela");
	}

	public void desenharImagem(Tensor tensor){
		desenharImagem(tensor, getTitle());
	}

	public void desenharImagem(Tensor tensor, String titulo){
		if(tensor == null){
			throw new IllegalArgumentException(
				"\nTensor não pode ser nulo."
			);
		}
		
		setTitle(titulo);
		painel.update(tensor);
	}

	/**
	 * Desenha as imagens contidas no array.
	 * @param arr array de tensores.
	 */
	public void desenharImagens(Tensor[] arr){
		new Thread(() -> {
			int indice = 0;
			int tamanho = arr.length;
	
			while(isEnabled()){
				if(leitorTeclado.d){
					if(indice+1 < tamanho) indice++;
					else indice = 0;
					leitorTeclado.d = false;
				
				}else if(leitorTeclado.a){
					if(indice-1 >= 0) indice--;
					else indice = tamanho-1;
					leitorTeclado.a = false;
				}
	
				setTitle("Img " + indice);
				painel.update(arr[indice]);
	
				try{
					Thread.sleep(50);
				}catch(Exception e){
					e.printStackTrace();
				}
			}
		}).start();
	}

}

class LeitorTeclado implements KeyListener{

	boolean w = false, a = false, s = false, d = false;

	public void keyTyped(KeyEvent e){}

	@Override
	public void keyPressed(KeyEvent e){
		switch(e.getKeyCode()){
			case KeyEvent.VK_W: w = true; break;
			case KeyEvent.VK_A: a = true; break;
			case KeyEvent.VK_S: s = true; break;
			case KeyEvent.VK_D: d = true; break;
		}
	}

	@Override
	public void keyReleased(KeyEvent e){
		switch(e.getKeyCode()){
			case KeyEvent.VK_W: w = false; break;
			case KeyEvent.VK_A: a = false; break;
			case KeyEvent.VK_S: s = false; break;
			case KeyEvent.VK_D: d = false; break;
		}
	}
}
