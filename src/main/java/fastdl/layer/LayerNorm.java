package fastdl.layer;

import fastdl.ops.TensorOps;
import fastdl.tensor.Tensor;

import java.util.List;

/**
 * Layer Normalization as introduced by Ba et al., 2016 ("Layer Normalization").
 *
 * <p>Normalizes the activations across the <em>last dimension</em> (the feature/embedding
 * dimension) independently for each token or sample. This is the standard normalization
 * used inside Transformer blocks and is fundamentally different from Batch Normalization,
 * which normalizes across the batch dimension.
 *
 * <h2>Formula</h2>
 * <pre>
 *   mean    = mean(x, dim=-1)
 *   var     = var(x, dim=-1)
 *   x_norm  = (x - mean) / sqrt(var + eps)
 *   output  = weight * x_norm + bias          (element-wise affine transform)
 * </pre>
 *
 * <h2>Learnable parameters</h2>
 * <ul>
 *   <li><b>weight</b> (gamma) — initialized to ones, shape [dModel]</li>
 *   <li><b>bias</b>   (beta)  — initialized to zeros, shape [dModel]</li>
 * </ul>
 *
 * <h2>Backward pass</h2>
 * Computes the full analytic gradient including the dependency of the mean and variance
 * on all input elements. Gradients are accumulated into {@code weight.grad()} and
 * {@code bias.grad()} so that the optimizer can update them via {@code step()}.
 *
 * <h2>Supported input shapes</h2>
 * Any shape {@code [..., dModel]} — e.g., {@code [batch, seqLen, dModel]} for sequences.
 * The last dimension is always the normalization target.
 *
 * <h2>Usage in TransformerBlock (Pre-LN)</h2>
 * <pre>{@code
 * LayerNorm norm = new LayerNorm(128);
 * Tensor normed = norm.forward(x);      // [batch, seqLen, 128]
 * Tensor grad   = norm.backward(dOut);  // propagates back through norm
 * }</pre>
 *
 * @see fastdl.model.TransformerBlock
 * @see fastdl.ops.TensorOps#layerNorm(Tensor, Tensor, Tensor, float)
 */
public class LayerNorm implements Layer {

    private final int dModel;
    private final float eps;
    private final Tensor weight;  // gamma
    private final Tensor bias;    // beta

    private Tensor lastInput;
    private Tensor lastNorm;      // normalized x before affine
    private float[] lastMean;
    private float[] lastInvStd;

    public LayerNorm(int dModel) {
        this(dModel, 1e-5f);
    }

    public LayerNorm(int dModel, float eps) {
        this.dModel = dModel;
        this.eps = eps;
        this.weight = Tensor.ones(dModel);
        this.bias = Tensor.zeros(dModel);
    }

    @Override
    public Tensor forward(Tensor input) {
        this.lastInput = input;
        int[] shape = input.shape();
        int D = shape[shape.length - 1];
        int rows = input.size() / D;

        float[] xd = input.data();
        float[] wd = weight.data();
        float[] bd = bias.data();

        lastNorm = new Tensor(shape);
        float[] nd = lastNorm.data();
        lastMean = new float[rows];
        lastInvStd = new float[rows];

        Tensor out = new Tensor(shape);
        float[] od = out.data();

        for (int r = 0; r < rows; r++) {
            int base = r * D;
            double mean = 0.0;
            for (int i = 0; i < D; i++) mean += xd[base + i];
            mean /= D;

            double var = 0.0;
            for (int i = 0; i < D; i++) {
                double diff = xd[base + i] - mean;
                var += diff * diff;
            }
            var = var / D + eps;

            float invStd = (float) (1.0 / Math.sqrt(var));
            lastMean[r] = (float) mean;
            lastInvStd[r] = invStd;

            for (int i = 0; i < D; i++) {
                float norm = (xd[base + i] - (float) mean) * invStd;
                nd[base + i] = norm;
                od[base + i] = norm * wd[i] + bd[i];
            }
        }
        return out;
    }

    @Override
    public Tensor backward(Tensor gradOutput) {
        int[] shape = lastInput.shape();
        int D = shape[shape.length - 1];
        int rows = lastInput.size() / D;

        float[] go = gradOutput.data();
        float[] nd = lastNorm.data();
        float[] wd = weight.data();
        float[] wGrad = weight.grad();
        float[] bGrad = bias.grad();

        Tensor gradInput = new Tensor(shape);
        float[] gi = gradInput.data();

        for (int r = 0; r < rows; r++) {
            int base = r * D;
            float invStd = lastInvStd[r];

            // accumulate weight/bias gradients
            for (int i = 0; i < D; i++) {
                wGrad[i] += go[base + i] * nd[base + i];
                bGrad[i] += go[base + i];
            }

            // grad w.r.t. input
            // dL/dx = (1/D) * invStd * (D * dL/dy_norm - sum(dL/dy_norm) - norm * sum(dL/dy_norm * norm))
            double sumGo = 0.0, sumGoNorm = 0.0;
            for (int i = 0; i < D; i++) {
                float goNorm = go[base + i] * wd[i];
                sumGo += goNorm;
                sumGoNorm += goNorm * nd[base + i];
            }

            for (int i = 0; i < D; i++) {
                float goNorm = go[base + i] * wd[i];
                gi[base + i] = invStd / D * (D * goNorm - (float) sumGo - nd[base + i] * (float) sumGoNorm);
            }
        }
        return gradInput;
    }

    @Override
    public List<Tensor> parameters() {
        return List.of(weight, bias);
    }

    public Tensor weight() { return weight; }
    public Tensor bias()   { return bias; }
}
