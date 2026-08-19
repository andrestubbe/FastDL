package fastdl.layer;

import fastdl.tensor.Tensor;
import java.util.List;

/**
 * Fully Connected / Dense Layer (y = xW + b).
 */
public class Dense implements Layer {

    private final int inFeatures;
    private final int outFeatures;
    private final Tensor weights;
    private final Tensor bias;
    private Tensor lastInput;

    public Dense(int inFeatures, int outFeatures) {
        this.inFeatures = inFeatures;
        this.outFeatures = outFeatures;
        this.weights = Tensor.randn(inFeatures, outFeatures);
        this.bias = Tensor.zeros(outFeatures);
    }

    @Override
    public Tensor forward(Tensor input) {
        this.lastInput = input;
        int batch = input.shape().length > 1 ? input.shape()[0] : 1;
        Tensor out = new Tensor(batch, outFeatures);

        float[] inData = input.data();
        float[] wData = weights.data();
        float[] bData = bias.data();
        float[] outData = out.data();

        for (int b = 0; b < batch; b++) {
            for (int j = 0; j < outFeatures; j++) {
                float sum = bData[j];
                for (int i = 0; i < inFeatures; i++) {
                    sum += inData[b * inFeatures + i] * wData[i * outFeatures + j];
                }
                outData[b * outFeatures + j] = sum;
            }
        }
        return out;
    }

    @Override
    public Tensor backward(Tensor gradOutput) {
        int batch = lastInput.shape().length > 1 ? lastInput.shape()[0] : 1;
        Tensor gradInput = new Tensor(batch, inFeatures);

        float[] gradOut = gradOutput.data();
        float[] inData = lastInput.data();
        float[] wData = weights.data();
        float[] wGrad = weights.grad();
        float[] bGrad = bias.grad();
        float[] inGrad = gradInput.data();

        for (int b = 0; b < batch; b++) {
            for (int j = 0; j < outFeatures; j++) {
                float go = gradOut[b * outFeatures + j];
                bGrad[j] += go;
                for (int i = 0; i < inFeatures; i++) {
                    wGrad[i * outFeatures + j] += inData[b * inFeatures + i] * go;
                    inGrad[b * inFeatures + i] += wData[i * outFeatures + j] * go;
                }
            }
        }
        return gradInput;
    }

    @Override
    public List<Tensor> parameters() {
        return List.of(weights, bias);
    }

    public Tensor weights() {
        return weights;
    }

    public Tensor bias() {
        return bias;
    }
}
