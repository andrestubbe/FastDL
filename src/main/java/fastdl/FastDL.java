package fastdl;

import fastdl.layer.Dense;
import fastdl.layer.GELU;
import fastdl.layer.Layer;
import fastdl.layer.LayerNorm;
import fastdl.layer.ReLU;
import fastdl.loss.CrossEntropyLoss;
import fastdl.loss.MSELoss;
import fastdl.model.GPTConfig;
import fastdl.model.GPTModel;
import fastdl.optim.AdamW;
import fastdl.optim.SGD;
import fastdl.tensor.Tensor;
import fastdl.tokenizer.CharTokenizer;
import fastdl.training.TrainerConfig;

import java.util.ArrayList;
import java.util.List;


/**
 * Public convenience facade for the FastDL library.
 *
 * <p>This class exposes the most common constructors and factory helpers used by the
 * repo's examples, demos, and model code. Rather than requiring callers to instantiate
 * every layer and optimizer manually, FastDL acts as a compact entry point for the core
 * building blocks:
 *
 * <ul>
 *   <li>tensor creation helpers</li>
 *   <li>layer constructors such as {@link fastdl.layer.Dense} and {@link fastdl.layer.GELU}</li>
 *   <li>loss functions and optimizers</li>
 *   <li>small Transformer factory methods</li>
 * </ul>
 *
 * <h2>Typical use</h2>
 * <pre>{@code
 * GPTConfig config = FastDL.gptConfigSmall(1024);
 * GPTModel model = FastDL.gpt(config);
 * }</pre>
 *
 * @see fastdl.model.GPTModel
 * @see fastdl.training.Trainer
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

    // -------------------------------------------------------------------------
    // Transformer / LLM factory methods
    // -------------------------------------------------------------------------

    public static GELU gelu() { return new GELU(); }

    public static LayerNorm layerNorm(int dModel) { return new LayerNorm(dModel); }

    public static CrossEntropyLoss crossEntropy() { return new CrossEntropyLoss(); }

    public static AdamW adamW(List<Tensor> params, float lr) {
        return new AdamW(params, lr);
    }

    public static AdamW adamW(List<Tensor> params, float lr, float weightDecay) {
        return new AdamW(params, lr, weightDecay);
    }

    public static GPTModel gpt(GPTConfig config) {
        return new GPTModel(config);
    }

    public static GPTConfig gptConfigSmall(int vocabSize) {
        return GPTConfig.small(vocabSize);
    }

    public static GPTConfig gptConfigMedium(int vocabSize) {
        return GPTConfig.medium(vocabSize);
    }

    public static CharTokenizer charTokenizer(String corpus) {
        return CharTokenizer.build(corpus);
    }

    public static TrainerConfig trainerConfigDemo() {
        return TrainerConfig.demo();
    }

    public static TrainerConfig trainerConfigStandard() {
        return TrainerConfig.standard();
    }
}

