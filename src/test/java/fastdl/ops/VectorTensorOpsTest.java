package fastdl.ops;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

import fastdl.tensor.Tensor;
import org.junit.jupiter.api.Test;

class VectorTensorOpsTest {

    @Test
    void matmulShouldMatchReferenceImplementation() {
        Tensor a = new Tensor(new float[] {
            1f, 2f,
            3f, 4f
        }, 2, 2);

        Tensor b = new Tensor(new float[] {
            5f, 6f,
            7f, 8f
        }, 2, 2);

        Tensor result = VectorTensorOps.matmul(a, b);

        assertArrayEquals(new float[] {
            19f, 22f,
            43f, 50f
        }, result.data(), 1e-5f);
    }
}
