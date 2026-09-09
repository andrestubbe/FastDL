package fastdl.layer;

import fastdl.tensor.Tensor;
import java.util.List;

/**
 * Shared contract for all trainable or differentiable components in FastDL.
 *
 * <p>Every layer implements a forward and backward pass, and exposes its learnable
 * parameters so that optimizers can update them in a uniform way across the library.
 * This interface is deliberately minimal, allowing both simple primitives such as
 * {@link ReLU} and larger transformer blocks such as {@link fastdl.model.TransformerBlock}
 * to share the same training pipeline.
 *
 * <h2>Contract</h2>
 * <ul>
 *   <li>{@link #forward(fastdl.tensor.Tensor)} computes activations</li>
 *   <li>{@link #backward(fastdl.tensor.Tensor)} propagates the gradient</li>
 *   <li>{@link #parameters()} returns the learnable tensors</li>
 * </ul>
 *
 * @see fastdl.layer.Dense
 * @see fastdl.model.GPTModel
 */
public interface Layer {

    /**
     * Performs forward computation.
     */
    Tensor forward(Tensor input);

    /**
     * Performs backward computation / backpropagation.
     */
    Tensor backward(Tensor gradOutput);

    /**
     * @return learnable parameter tensors (weights, biases)
     */
    List<Tensor> parameters();
}
