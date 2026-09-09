package fastdl.loss;

import fastdl.tensor.Tensor;

/**
 * Mean Squared Error loss for regression-style targets.
 *
 * <p>For a prediction vector {@code p} and target vector {@code t}, the loss is
 * computed as the average squared error:
 * <pre>
 *   L = (1 / N) * Σ_i (p_i - t_i)^2
 * </pre>
 *
 * <h2>Backward pass</h2>
 * The gradient is the standard derivative of the MSE:
 * <pre>
 *   dL/dp_i = (2 / N) * (p_i - t_i)
 * </pre>
 *
 * <h2>Typical use</h2>
 * This loss is useful for simple regression tasks and small numeric toy models,
 * but modern language models typically use cross-entropy instead because it is
 * better suited to discrete next-token prediction.
 */
public class MSELoss {

    public float forward(Tensor pred, Tensor target) {
        float[] p = pred.data();
        float[] t = target.data();
        float sum = 0f;
        for (int i = 0; i < p.length; i++) {
            float diff = p[i] - t[i];
            sum += diff * diff;
        }
        return sum / p.length;
    }

    public Tensor backward(Tensor pred, Tensor target) {
        Tensor grad = new Tensor(pred.shape());
        float[] p = pred.data();
        float[] t = target.data();
        float[] g = grad.data();
        float n = p.length;
        for (int i = 0; i < p.length; i++) {
            g[i] = (2f / n) * (p[i] - t[i]);
        }
        return grad;
    }
}
