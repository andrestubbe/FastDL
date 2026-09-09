package fastdl.optim;

import fastdl.tensor.Tensor;

import java.util.List;

/**
 * AdamW optimizer — Adam with decoupled weight decay (Loshchilov &amp; Hutter, 2019).
 *
 * <p>AdamW is the standard optimizer for training Transformer-based language models.
 * It extends the original Adam optimizer by fixing a subtle flaw: in Adam, weight
 * decay is incorrectly mixed into the gradient update and therefore interacts with
 * the adaptive learning rate. AdamW decouples weight decay from the gradient-based
 * update, applying it directly to the parameters <em>after</em> the Adam step.
 *
 * <h2>Update rule per parameter (per step t)</h2>
 * <pre>
 *   m_t = β₁ · m_{t-1} + (1 - β₁) · g_t          first moment (mean)
 *   v_t = β₂ · v_{t-1} + (1 - β₂) · g_t²          second moment (uncentred variance)
 *
 *   m̂ = m_t / (1 - β₁ᵗ)                            bias correction
 *   v̂ = v_t / (1 - β₂ᵗ)
 *
 *   θ_t = θ_{t-1} - lr · (m̂ / (√v̂ + ε) + λ · θ_{t-1})
 *                                     ↑ decoupled weight decay
 * </pre>
 *
 * <h2>Default hyperparameters</h2>
 * <ul>
 *   <li>β₁ = 0.9   — first moment decay (controls gradient smoothing)</li>
 *   <li>β₂ = 0.999 — second moment decay (controls adaptive scaling)</li>
 *   <li>ε  = 1e-8  — denominator stability constant</li>
 *   <li>λ  = 0.01  — weight decay (L2 regularisation coefficient)</li>
 * </ul>
 * These are the same defaults used by Hugging Face Transformers and the original GPT papers.
 *
 * <h2>Important: bias parameters</h2>
 * Weight decay on bias terms is debated — many implementations set {@code weightDecay=0}
 * specifically for bias and LayerNorm parameters. This implementation applies the same
 * weight decay to all parameters for simplicity. To exclude specific parameters, pass
 * only the desired subset in the {@code parameters} list.
 *
 * <h2>Usage</h2>
 * <pre>{@code
 * GPTModel model = new GPTModel(config);
 * AdamW optimizer = new AdamW(model.parameters(), 3e-4f, 0.01f);
 *
 * // Training loop:
 * optimizer.zeroGrad();
 * model.backwardPass(gradLogits);
 * optimizer.step();
 * }</pre>
 *
 * @see fastdl.training.Trainer
 * @see fastdl.optim.SGD
 */
public class AdamW {

    private final List<Tensor> parameters;
    private final float lr;
    private final float beta1;
    private final float beta2;
    private final float eps;
    private final float weightDecay;

    private final float[][] m;   // first moment
    private final float[][] v;   // second moment
    private int stepCount = 0;

    public AdamW(List<Tensor> parameters, float lr) {
        this(parameters, lr, 0.9f, 0.999f, 1e-8f, 0.01f);
    }

    public AdamW(List<Tensor> parameters, float lr, float weightDecay) {
        this(parameters, lr, 0.9f, 0.999f, 1e-8f, weightDecay);
    }

    public AdamW(List<Tensor> parameters, float lr,
                 float beta1, float beta2, float eps, float weightDecay) {
        this.parameters = parameters;
        this.lr = lr;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.eps = eps;
        this.weightDecay = weightDecay;
        this.m = new float[parameters.size()][];
        this.v = new float[parameters.size()][];
        for (int i = 0; i < parameters.size(); i++) {
            int n = parameters.get(i).size();
            this.m[i] = new float[n];
            this.v[i] = new float[n];
        }
    }

    public void step() {
        stepCount++;
        float biasCorrect1 = 1f - (float) Math.pow(beta1, stepCount);
        float biasCorrect2 = 1f - (float) Math.pow(beta2, stepCount);

        for (int p = 0; p < parameters.size(); p++) {
            Tensor param = parameters.get(p);
            float[] data = param.data();
            float[] grad = param.grad();
            float[] mi = m[p];
            float[] vi = v[p];

            for (int i = 0; i < data.length; i++) {
                float g = grad[i];
                mi[i] = beta1 * mi[i] + (1f - beta1) * g;
                vi[i] = beta2 * vi[i] + (1f - beta2) * g * g;

                float mHat = mi[i] / biasCorrect1;
                float vHat = vi[i] / biasCorrect2;

                // decoupled weight decay applied directly to param (not through grad)
                data[i] -= lr * (mHat / ((float) Math.sqrt(vHat) + eps) + weightDecay * data[i]);
            }
        }
    }

    public void zeroGrad() {
        for (Tensor param : parameters) {
            param.zeroGrad();
        }
    }

    public int getStepCount() {
        return stepCount;
    }
}
