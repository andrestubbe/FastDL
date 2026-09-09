package fastdl.optim;

import fastdl.tensor.Tensor;
import java.util.List;

/**
 * Stochastic Gradient Descent with optional momentum.
 *
 * <p>SGD is the simplest parameter update rule for training neural networks. It updates
 * each parameter by subtracting a scaled gradient, optionally smoothing the update with
 * a momentum term to accelerate convergence on noisy loss landscapes.
 *
 * <h2>Update rule</h2>
 * <pre>
 *   v_t = momentum * v_{t-1} + lr * grad
 *   θ_t = θ_{t-1} - v_t
 * </pre>
 *
 * <h2>When to use it</h2>
 * This implementation is useful for small experiments and as a baseline optimizer.
 * For modern Transformers, AdamW is usually more stable and faster in practice.
 *
 * @see fastdl.optim.AdamW
 */
public class SGD {

    private final List<Tensor> parameters;
    private final float lr;
    private final float momentum;
    private final float[][] velocities;

    public SGD(List<Tensor> parameters, float lr) {
        this(parameters, lr, 0f);
    }

    public SGD(List<Tensor> parameters, float lr, float momentum) {
        this.parameters = parameters;
        this.lr = lr;
        this.momentum = momentum;
        this.velocities = new float[parameters.size()][];
        for (int i = 0; i < parameters.size(); i++) {
            this.velocities[i] = new float[parameters.get(i).size()];
        }
    }

    public void step() {
        for (int p = 0; p < parameters.size(); p++) {
            Tensor param = parameters.get(p);
            float[] data = param.data();
            float[] grad = param.grad();
            float[] v = velocities[p];

            for (int i = 0; i < data.length; i++) {
                v[i] = momentum * v[i] + lr * grad[i];
                data[i] -= v[i];
            }
        }
    }

    public void zeroGrad() {
        for (Tensor param : parameters) {
            param.zeroGrad();
        }
    }
}
