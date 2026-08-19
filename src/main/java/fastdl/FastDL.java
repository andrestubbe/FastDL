package fastdl;

import fastdl.layer.Dense;
import fastdl.layer.Layer;
import fastdl.layer.ReLU;
import fastdl.loss.MSELoss;
import fastdl.optim.SGD;
import fastdl.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;

/**
 * FastDL — Deep Learning, Tensor Computing & Neural Backprop for Java.
 */
public final class FastDL {

    private FastDL() {}

    public static Tensor tensor(int... shape) {
        return new Tensor(shape);
    }

    public static Tensor tensor(float[] data, int... shape) {
        return new Tensor(data, shape);
    }

    public static Tensor randn(int... shape) {
        return Tensor.randn(shape);
    }

    public static Dense dense(int inFeatures, int outFeatures) {
        return new Dense(inFeatures, outFeatures);
    }

    public static ReLU relu() {
        return new ReLU();
    }

    public static MSELoss mse() {
        return new MSELoss();
    }

    public static SGD sgd(List<Tensor> params, float lr) {
        return new SGD(params, lr);
    }

    public static SGD sgd(List<Tensor> params, float lr, float momentum) {
        return new SGD(params, lr, momentum);
    }

    /**
     * Sequential container chaining layers together.
     */
    public static class Sequential implements Layer {
        private final List<Layer> layers = new ArrayList<>();

        public Sequential add(Layer layer) {
            layers.add(layer);
            return this;
        }

        @Override
        public Tensor forward(Tensor input) {
            Tensor current = input;
            for (Layer layer : layers) {
                current = layer.forward(current);
            }
            return current;
        }

        @Override
        public Tensor backward(Tensor gradOutput) {
            Tensor current = gradOutput;
            for (int i = layers.size() - 1; i >= 0; i--) {
                current = layers.get(i).backward(current);
            }
            return current;
        }

        @Override
        public List<Tensor> parameters() {
            List<Tensor> params = new ArrayList<>();
            for (Layer layer : layers) {
                params.addAll(layer.parameters());
            }
            return params;
        }
    }

    public static Sequential sequential(Layer... layers) {
        Sequential seq = new Sequential();
        for (Layer l : layers) seq.add(l);
        return seq;
    }
}
