package fastdl.ops;

import static jdk.incubator.vector.FloatVector.SPECIES_256;
import static jdk.incubator.vector.FloatVector.SPECIES_128;

import fastdl.tensor.Tensor;
import jdk.incubator.vector.FloatVector;

/**
 * Vectorized tensor operations built on Java 17's Vector API.
 *
 * <p>This implementation keeps the scalar fallback semantics of {@link TensorOps}
 * but accelerates the core matrix multiplication path with SIMD-friendly loops.
 * The code intentionally remains small and readable so it can be used as the
 * first performance step before adding JNI or GPU backends.
 */
public final class VectorTensorOps {

    private VectorTensorOps() {
    }

    public static Tensor matmul(Tensor a, Tensor b) {
        int[] sa = a.shape();
        int[] sb = b.shape();
        if (sa.length != 2 || sb.length != 2) {
            return TensorOps.matmul(a, b);
        }

        int rows = sa[0];
        int cols = sb[1];
        int inner = sb[0];
        Tensor out = new Tensor(rows, cols);
        float[] ad = a.data();
        float[] bd = b.data();
        float[] od = out.data();

        for (int r = 0; r < rows; r++) {
            int rowBase = r * inner;
            for (int c = 0; c < cols; c++) {
                float sum = 0f;
                int k = 0;
                FloatVector acc = FloatVector.zero(SPECIES_256);

                int vecLength = Math.min(inner - k, SPECIES_256.length());
                while (k + SPECIES_256.length() <= inner) {
                    FloatVector va = FloatVector.fromArray(SPECIES_256, ad, rowBase + k);
                    FloatVector vb = FloatVector.fromArray(SPECIES_256, bd, k * cols + c);
                    acc = va.fma(vb, acc);
                    k += SPECIES_256.length();
                }

                if (k < inner) {
                    int remainder = inner - k;
                    if (remainder >= SPECIES_128.length()) {
                        FloatVector va = FloatVector.fromArray(SPECIES_128, ad, rowBase + k);
                        FloatVector vb = FloatVector.fromArray(SPECIES_128, bd, k * cols + c);
                        acc = acc.add(va.mul(vb));
                        k += SPECIES_128.length();
                    }
                }

                for (int i = 0; i < SPECIES_256.length(); i++) {
                    sum += acc.toArray()[i];
                }

                while (k < inner) {
                    sum += ad[rowBase + k] * bd[k * cols + c];
                    k++;
                }

                od[r * cols + c] = sum;
            }
        }
        return out;
    }

    public static Tensor batchedMatmul(Tensor a, Tensor b) {
        int[] sa = a.shape();
        int[] sb = b.shape();
        if (sa.length != 3 || sb.length != 3) {
            return TensorOps.batchedMatmul(a, b);
        }

        int batch = sa[0];
        int rows = sa[1];
        int inner = sa[2];
        int cols = sb[2];
        Tensor out = new Tensor(batch, rows, cols);
        float[] ad = a.data();
        float[] bd = b.data();
        float[] od = out.data();

        for (int bi = 0; bi < batch; bi++) {
            for (int r = 0; r < rows; r++) {
                int rowBase = bi * rows * inner + r * inner;
                for (int c = 0; c < cols; c++) {
                    float sum = 0f;
                    for (int k = 0; k < inner; k++) {
                        sum += ad[rowBase + k] * bd[bi * inner * cols + k * cols + c];
                    }
                    od[bi * rows * cols + r * cols + c] = sum;
                }
            }
        }
        return out;
    }
}
