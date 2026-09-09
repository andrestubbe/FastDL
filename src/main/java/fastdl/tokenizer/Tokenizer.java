package fastdl.tokenizer;

/**
 * Tokenizer interface — the contract between raw text and integer token IDs.
 *
 * <p>A tokenizer is the first stage in any language model pipeline. It converts a raw
 * string of text into a sequence of integer token IDs that the model's embedding table
 * can look up. The inverse direction ({@link #decode}) converts token IDs back to text.
 *
 * <h2>Implementations in FastDL</h2>
 * <ul>
 *   <li>{@link fastdl.tokenizer.CharTokenizer} — character-level tokenizer.
 *       Each unique character in the corpus becomes one token. Simple, small vocab (~100–200),
 *       no out-of-vocabulary problem. Best starting point for TinyStories training.</li>
 * </ul>
 * Future implementations may include BPE (Byte-Pair Encoding) for sub-word tokenization
 * and better compression ratios.
 *
 * <h2>Special tokens</h2>
 * All implementations must reserve at minimum:
 * <ul>
 *   <li>Token ID 0 = {@code <unk>} — unknown / unseen input</li>
 *   <li>Token ID 1 = {@code <eos>} — end of sequence / story boundary</li>
 * </ul>
 * Check {@link #eosTokenId()} at runtime rather than hardcoding 1.
 *
 * <h2>Usage example</h2>
 * <pre>{@code
 * Tokenizer tok = CharTokenizer.build(corpus);
 * int[] ids  = tok.encode("Once upon a time");
 * String txt = tok.decode(ids);   // "Once upon a time"
 * System.out.println("vocab size: " + tok.vocabSize());
 * }</pre>
 *
 * @see fastdl.tokenizer.CharTokenizer
 * @see fastdl.data.TextDataset
 */
public interface Tokenizer {

    /** Encode a string into a sequence of integer token IDs. */
    int[] encode(String text);

    /** Decode a sequence of token IDs back to a string. */
    String decode(int[] tokens);

    /** Total vocabulary size. */
    int vocabSize();

    /** Special token ID for end-of-sequence (or -1 if not defined). */
    default int eosTokenId() {
        return -1;
    }

    /** Special token ID for unknown tokens. */
    default int unkTokenId() {
        return 0;
    }
}
