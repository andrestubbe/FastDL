package fastdl.layer;

import fastdl.tensor.Tensor;
import java.util.List;

/**
 * Fundamental building block for Deep Learning operations & layers in FastDL.
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
