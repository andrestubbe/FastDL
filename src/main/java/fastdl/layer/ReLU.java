package fastdl.layer;

import fastdl.tensor.Tensor;
import java.util.Collections;
import java.util.List;

/**
 * Rectified Linear Unit Activation (ReLU: max(0, x)).
 */
public class ReLU implements Layer {

    private Tensor lastInput;

    @Override
    public Tensor forward(Tensor input) {
        this.lastInput = input;
        Tensor out = new Tensor(input.shape());
        float[] in = input.data();
        float[] o = out.data();
        for (int i = 0; i < in.length; i++) {
            o[i] = Math.max(0f, in[i]);
        }
        return out;
    }

    @Override
    public Tensor backward(Tensor gradOutput) {
        Tensor gradInput = new Tensor(lastInput.shape());
        float[] in = lastInput.data();
        float[] go = gradOutput.data();
        float[] gi = gradInput.data();
        for (int i = 0; i < in.length; i++) {
            gi[i] = in[i] > 0f ? go[i] : 0f;
        }
        return gradInput;
    }

    @Override
    public List<Tensor> parameters() {
        return Collections.emptyList();
    }
}
