package jnn.layers;

import jnn.core.JNNutils;
import jnn.core.Parameter;
import jnn.core.tensor.Tensor;

/**
 * Experimental
 */
public class Residual extends Layer {
    
    private Layer[] main;
    private Layer[] skip;

    private int[] shapeIn;
    private int[] shapeOut;

    public Tensor _output;
    public Tensor _gradInput;

    public Residual(Layer[] main, Layer[] shortcut) {
        JNNutils.validateNonNull(main, "main == null");
        JNNutils.validateNonNull(shortcut, "shortcut == null");

        if (main.length == 0) {
            throw new IllegalArgumentException(
                "\nO caminho principal do bloco residual não pode ser vazio."
            );
        }

        this.main = main;
        this.skip = shortcut;
    }

    public Residual(Layer... main) {
        this(main, new Layer[0]);
    }

    @Override
    public void build(int[] shape) {
        JNNutils.validateNonNull(shape, "shape == null");

        int[] shapeMain = buildBranch(main, shape);
        int[] shapeSkip = buildBranch(skip, shape);

        if (!JNNutils.arrayComp(shapeMain, shapeSkip)) {
            throw new IllegalArgumentException(
                "\nOs ramos do bloco residual produzem saídas de formato diferente: " +
                " principal: " + JNNutils.arrayStr(shapeMain) + ", atalho " + JNNutils.arrayStr(shapeSkip)
            );
        }

        shapeIn = shape.clone();
        shapeOut = shapeMain.clone();

        _output = addBuffer("output", shapeOut);
        _gradInput = addBuffer("grad input", shapeIn);

        _treinable = true;
        _built     = true;
    }

    private int[] buildBranch(Layer[] branch, int[] shape) {
        int[] s = shape;

        for (var layer : branch) {
            layer.build(s);
            s = layer.shapeOut();
        }

        return s;
    }

    @Override
    public void initParams() {
        validateBuid();

        for (var layer : main)     layer.initParams();
        for (var layer : skip) layer.initParams();
    }

    @Override
    public void fitToBatch(int batchSize) {
        for (var layer : main)     layer.fitToBatch(batchSize);
        for (var layer : skip) layer.fitToBatch(batchSize);

        _output    = addBuffer("output",     withBatch(shapeOut, batchSize));
        _gradInput = addBuffer("grad input", withBatch(shapeIn,  batchSize));

        this.batchSize = batchSize;
    }

    private int[] withBatch(int[] shape, int batchSize) {
        if (batchSize == 0) return shape;

        int n = shape.length;
        int[] s = new int[n + 1];
        s[0] = batchSize;
        System.arraycopy(shape, 0, s, 1, n);
    
        return s;
    }

    private void zeroAllOutputs() {
        for (Layer layer : main) layer.output().zero();
        for (Layer layer : skip) layer.output().zero();
        _output.zero();
    }

    @Override
    public Tensor forward(Tensor x) {
        validateBuid();
        validateShapes(x.shape(), shapeIn);

        int lotes = (x.numDim() == shapeIn.length + 1) ? x.lenDim(0) : 0;
        if (lotes != batchSize) fitToBatch(lotes);

        zeroAllOutputs();

        Tensor outMain = x;
        for (var layer : main) {
            outMain = layer.forward(outMain);
        }

        Tensor outSkip = x.contiguous();
        for (var layer : skip) {
            outSkip = layer.forward(outSkip);
        }

        _output.add(outMain);
        _output.add(outSkip);

        return _output;
    }

    private void zeroAllGradInputs() {
        for (Layer layer : main) layer.gradInput().zero();
        for (Layer layer : skip) layer.gradInput().zero();
        _gradInput.zero();
    }

    @Override
    public Tensor backward(Tensor g) {
        validateBuid();

        zeroAllGradInputs();

        Tensor gradMain = g;
        for (int i = main.length-1; i >= 0; i--) {
            gradMain = main[i].backward(gradMain);
        }
        
        Tensor gradSkip = g;
        for (int i = skip.length-1; i >= 0; i--) {
            gradSkip = skip[i].backward(gradSkip);
        }

        _gradInput.add(gradMain);
        _gradInput.add(gradSkip);

        return _gradInput;
    }

    @Override
    public void zeroGrad() {
        validateBuid();
        
        for (Parameter param : params()) {
            param.grad.zero();
        }

        // desse jeito não tava dando certo
        // ainda não sei o motivo

        // for (Layer layer : main) if (layer.isTreinable()) layer.zeroGrad();
        // for (Layer layer : skip) if (layer.isTreinable()) layer.zeroGrad();
    }

    @Override
    public Tensor output() {
        validateBuid();
        return _output;
    }

    @Override
    public int[] shapeIn() {
        validateBuid();
        return shapeIn.clone();
    }

    @Override
    public int[] shapeOut() {
        validateBuid();
        return shapeOut.clone();
    }

    @Override
    public Parameter[] params() {
        validateBuid();

        Parameter[] params = {};

        for (Layer layer : main) {
            if (layer.isTreinable()) {
                Parameter[] parameter = layer.params();
                params = JNNutils.addToArray(params, parameter[0]);

                if (layer.hasBias()) {
                    params = JNNutils.addToArray(params, parameter[1]);
                }
            }
        }

        for (Layer layer : skip) {
            if (layer.isTreinable()) {
                Parameter[] parameter = layer.params();
                params = JNNutils.addToArray(params, parameter[0]);
                
                if (layer.hasBias()) {
                    params = JNNutils.addToArray(params, parameter[1]);
                }
            }
        }

        return params;
    }

    @Override
    public int numParams() {
        validateBuid();

        int p = 0;
        
        for (Parameter param : params()) {
            p += param.weight.len();
        }

        return p;
    }

    @Override
    public boolean hasBias() {
        return false;
    }

    @Override
    public void train() {
        for (Layer layer : main) layer.train();
        for (Layer layer : skip) layer.train();
    }
    
    @Override
    public void eval() {
        for (Layer layer : main) layer.eval();
        for (Layer layer : skip) layer.eval();
    }

    @Override
    public long sizeBytes() {
        long tamVars = super.sizeBytes();

        tamVars += _output.sizeBytes();
        tamVars += _gradInput.sizeBytes();

        for (Layer layer : main) tamVars += layer.sizeBytes();
        for (Layer layer : skip) tamVars += layer.sizeBytes();

        return tamVars;
    }

}
