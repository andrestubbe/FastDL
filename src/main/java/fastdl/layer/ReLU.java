package fastdl.layer;

import fastdl.tensor.Tensor;
import java.util.Collections;
import java.util.List;

/**
 * Rectified Linear Unit activation: {@code max(0, x)}.
 *
 * <p>ReLU is a simple nonlinear activation used in early MLP-style networks and in some
 * classical neural-network baselines. It is computationally cheap and easy to reason about,
 * but modern language models generally prefer GELU because it provides smoother gradients.
 *
 * <h2>Forward pass</h2>
 * <pre>
 *   out[i] = max(0, x[i])
 * </pre>
 *
 * <h2>Backward pass</h2>
 * The derivative is 1 for positive inputs and 0 for non-positive inputs.
 *
 * @see fastdl.layer.GELU
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
