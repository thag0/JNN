package render.widgets;

import java.awt.Color;
import java.awt.Graphics;
import java.awt.Graphics2D;

import jnn.core.tensor.Tensor;

public class TensorImg extends Widget {

	/**
	 * Tensor usado como imagem para desenho (2D ou 3D).
	 */
    Tensor img = new Tensor(1, 1);

    public TensorImg(int altura, int largura, Tensor img) {
        super(altura, largura);
        update(img);
    }

    public TensorImg(int altura, int largura) {
        super(altura, largura);
    }

    public void update(Tensor img) {
        if (img == null) {
            throw new IllegalArgumentException(
                "\nTensor nulo."
            );
        }

        int n = img.numDim();
        if (n != 2 && n != 3) {
            throw new IllegalArgumentException(
                "\nTensor deve ser 2D ou 3D, recebido " + n + "D"
            );
        }

        this.img = img;
		repaint();
	}

	@Override
	protected void paintComponent(Graphics g) {
		super.paintComponent(g);
		Graphics2D g2 = (Graphics2D) g;

		boolean rgb = img.numDim() == 3;
		int[] shape = img.shape();

		int alt  = rgb ? shape[1] : shape[0];
		int larg = rgb ? shape[2] : shape[1];

		int largPixel = largura / larg;
		int altPixel  = altura / alt;

		if (img.max().item() > 1) {
			img = img.clone().norm(0, 1);
		}

		for (int i = 0; i < alt; i++) {
			for (int j = 0; j < larg; j++) {
				if (rgb) {
					double valR = img.get(0, i, j);
					double valG = img.get(1, i, j);
					double valB = img.get(2, i, j);
					
					int corR = (int)(valR * 255);
					if(corR > 255) corR = 255;
					if(corR < 0) corR = 0;
					
					int corG = (int)(valG * 255);
					if(corG > 255) corG = 255;
					if(corG < 0) corG = 0;
					
					int corB = (int)(valB * 255);
					if(corB > 255) corB = 255;
					if(corB < 0) corR = 0;
					
					g2.setColor(new Color(corR, corG, corB));
				
				} else {
					double valCinza = img.get(i, j);
					int cinza = (int)(valCinza * 255);
					if (cinza > 255) cinza = 255;
					if (cinza < 0) cinza = 0;
					g2.setColor(new Color(cinza, cinza, cinza));
				}

				int x = j * largPixel;
				int y = i * altPixel;

				g2.fillRect(x, y, largPixel, altPixel);
			}
		}
	}
    
}
