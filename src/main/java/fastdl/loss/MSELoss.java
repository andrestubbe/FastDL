package fastdl.loss;

import fastdl.tensor.Tensor;

/**
 * Mean Squared Error Loss (MSE).
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
