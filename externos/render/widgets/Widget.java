package externos.render.widgets;

import java.awt.Dimension;

import javax.swing.JPanel;

/**
 * Widget básico para reaproveitamento.
 */
public abstract class Widget extends JPanel {

    /**
     * Altura da janela
     */
    protected final int altura;
	
    /**
     * Largura da janela
     */
    protected final int largura;
    
    /**
     * Inicializa o widget de acordo com as dimensões desejadas.
     * @param altura altura do painel.
     * @param largura largura do painel.
     */
    protected Widget(int altura, int largura) {
        this.altura = altura;
        this.largura = largura;
		setPreferredSize(new Dimension(this.largura, this.altura));
    }

}
