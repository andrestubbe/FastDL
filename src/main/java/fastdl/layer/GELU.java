package fastdl.layer;

import fastdl.ops.TensorOps;
import fastdl.tensor.Tensor;

import java.util.Collections;
import java.util.List;

/**
 * GELU (Gaussian Error Linear Unit) activation layer.
 *
 * <p>GELU is the standard non-linearity used in modern Transformer architectures
 * (GPT-2, GPT-3, BERT, etc.) inside the FeedForward blocks. It outperforms ReLU on
 * language tasks because it provides a smooth, stochastic gating behaviour — inputs
 * are scaled by the probability that a Gaussian-distributed random variable is
 * smaller than the input.
 *
 * <h2>Formula (tanh approximation — used here)</h2>
 * <pre>
 *   GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
 * </pre>
 *
 * <p>This is the same approximation used in the original GPT papers and by PyTorch's
 * {@code torch.nn.functional.gelu(approximate='tanh')}.
 *
 * <h2>Derivative (for backward pass)</h2>
 * The analytic derivative is computed in {@link fastdl.ops.TensorOps#geluBackward}
 * using the chain rule through the tanh:
 * <pre>
 *   dGELU/dx = 0.5*(1 + tanh(u)) + 0.5*x*(1 - tanh²(u))*u'
 *   where u  = sqrt(2/π) * (x + 0.044715*x³)
 *         u' = sqrt(2/π) * (1 + 3*0.044715*x²)
 * </pre>
 *
 * <h2>Why not ReLU?</h2>
 * ReLU has zero gradient for all negative inputs (dead neurons). GELU keeps a small,
 * smooth gradient even for negative values, which improves training stability on
 * language modelling tasks.
 *
 * <h2>Usage</h2>
 * <pre>{@code
 * GELU gelu = new GELU();
 * Tensor out  = gelu.forward(hidden);      // [batch, seqLen, 4*dModel]
 * Tensor grad = gelu.backward(gradOut);    // propagates gradient back
 * }</pre>
 *
 * @see fastdl.model.FeedForward
 * @see fastdl.ops.TensorOps#gelu(Tensor)
 * @see fastdl.ops.TensorOps#geluBackward(Tensor, Tensor)
 */
public class GELU implements Layer {

    private Tensor lastInput;

    @Override
    public Tensor forward(Tensor input) {
        this.lastInput = input;
        return TensorOps.gelu(input);
    }

    @Override
    public Tensor backward(Tensor gradOutput) {
        return TensorOps.geluBackward(lastInput, gradOutput);
    }

    @Override
    public List<Tensor> parameters() {
        return Collections.emptyList();
    }
}
