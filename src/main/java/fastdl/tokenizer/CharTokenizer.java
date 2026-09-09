package fastdl.tokenizer;

import java.util.ArrayList;
import java.util.List;
import java.util.TreeMap;

/**
 * Character-level tokenizer that builds its vocabulary directly from a training corpus.
 *
 * <p>This is the simplest possible tokenizer: every unique character in the corpus
 * becomes a single token. There are no sub-word splits, no byte-pair encoding, and no
 * out-of-vocabulary problem — any character already seen during {@link #build} will
 * be encoded correctly. Unknown characters (not seen during build) are mapped to the
 * {@code <unk>} token (ID 0).
 *
 * <h2>Vocabulary layout</h2>
 * <pre>
 *   ID 0  = &lt;unk&gt;  — unknown / unseen character placeholder
 *   ID 1  = &lt;eos&gt;  — end of sequence / story boundary
 *   ID 2  = first character in sorted order (e.g. ' ' space)
 *   ID 3  = second character ...
 *   ...
 *   ID N  = last character in sorted order
 * </pre>
 * Characters are sorted lexicographically so that the mapping is <strong>deterministic</strong>
 * across JVM runs — the same corpus always produces the same ID assignments.
 *
 * <h2>Encoding speed</h2>
 * Encoding uses a direct array lookup (char codepoint as array index) in O(1) per character,
 * making it extremely fast on large corpora like TinyStories (1.9 GB, ~1.9 billion chars).
 *
 * <h2>Limitations</h2>
 * <ul>
 *   <li>Only supports BMP (Basic Multilingual Plane) Unicode characters (codepoints ≤ 65535).
 *       Emoji and supplementary characters are mapped to {@code <unk>}.</li>
 *   <li>Vocabulary size is limited to the number of distinct chars in the corpus.
 *       For English text this is typically 70–130 tokens.</li>
 *   <li>Longer token sequences than BPE/SentencePiece for the same text — a typical
 *       English word requires 4–8 tokens instead of 1–2. This increases the sequence
 *       length the model must process.</li>
 * </ul>
 *
 * <h2>Usage</h2>
 * <pre>{@code
 * String corpus = Files.readString(Path.of("data/TinyStories-train.txt"));
 * CharTokenizer tok = CharTokenizer.build(corpus);
 * System.out.println(tok);  // CharTokenizer(vocabSize=102, charCount=100)
 *
 * int[] ids = tok.encode("Once upon a time");
 * int[] withEos = tok.encodeWithEos("The end.");
 * String decoded = tok.decode(ids);
 * }</pre>
 *
 * @see fastdl.tokenizer.Tokenizer
 * @see fastdl.data.TextDataset
 * @see fastdl.model.Embedding
 */
public class CharTokenizer implements Tokenizer {

    private static final int UNK = 0;
    private static final int EOS = 1;

    private final char[] idToChar;          // id -> char  (index 0 = unk, 1 = eos)
    private final int[] charToId;           // direct lookup by char codepoint (BMP only)
    private final int vocab;

    private CharTokenizer(char[] idToChar, int[] charToId) {
        this.idToChar = idToChar;
        this.charToId = charToId;
        this.vocab = idToChar.length;
    }

    /**
     * Build a CharTokenizer from a corpus string.
     * Scans all unique characters and assigns sorted IDs starting at 2
     * (0 = unk, 1 = eos).
     */
    public static CharTokenizer build(String corpus) {
        TreeMap<Character, Integer> seen = new TreeMap<>();
        for (int i = 0; i < corpus.length(); i++) {
            char c = corpus.charAt(i);
            seen.putIfAbsent(c, 0);
        }

        List<Character> chars = new ArrayList<>(seen.keySet());
        int vocabSize = 2 + chars.size();   // 0=unk, 1=eos, 2..N=chars

        char[] idToChar = new char[vocabSize];
        idToChar[0] = '\0';   // unk placeholder
        idToChar[1] = '\0';   // eos placeholder

        // find max codepoint to size the reverse lookup array
        int maxCode = 0;
        for (char c : chars) if (c > maxCode) maxCode = c;

        int[] charToId = new int[maxCode + 1];
        java.util.Arrays.fill(charToId, UNK);

        for (int i = 0; i < chars.size(); i++) {
            char c = chars.get(i);
            int id = 2 + i;
            idToChar[id] = c;
            charToId[c] = id;
        }

        return new CharTokenizer(idToChar, charToId);
    }

    @Override
    public int[] encode(String text) {
        int[] out = new int[text.length()];
        for (int i = 0; i < text.length(); i++) {
            char c = text.charAt(i);
            out[i] = (c < charToId.length) ? charToId[c] : UNK;
        }
        return out;
    }

    /** Encode and append EOS token. */
    public int[] encodeWithEos(String text) {
        int[] base = encode(text);
        int[] out = new int[base.length + 1];
        System.arraycopy(base, 0, out, 0, base.length);
        out[base.length] = EOS;
        return out;
    }

    @Override
    public String decode(int[] tokens) {
        StringBuilder sb = new StringBuilder(tokens.length);
        for (int id : tokens) {
            if (id >= 2 && id < idToChar.length) {
                sb.append(idToChar[id]);
            }
            // skip unk (0) and eos (1) in plain decode
        }
        return sb.toString();
    }

    @Override
    public int vocabSize() {
        return vocab;
    }

    @Override
    public int eosTokenId() {
        return EOS;
    }

    @Override
    public int unkTokenId() {
        return UNK;
    }

    /** Number of unique printable characters (excluding unk/eos). */
    public int charCount() {
        return vocab - 2;
    }

    @Override
    public String toString() {
        return "CharTokenizer(vocabSize=" + vocab + ", charCount=" + charCount() + ")";
    }
}
