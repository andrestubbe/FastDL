package fastdl.demo;

import fastdl.layer.Dense;
import fastdl.model.*;
import fastdl.ops.TensorOps;
import fastdl.tensor.Tensor;
import fastdl.tokenizer.CharTokenizer;

import java.io.*;
import java.nio.file.*;
import java.util.*;

/**
 * Step-by-step Walkthrough Demo — visualises every layer of one GPT forward pass.
 *
 * <p>Reads the first story from TinyStories-train.txt (or the fallback tinystories.txt),
 * builds a tiny model, and runs inference printing the intermediate tensor shape
 * and sample values after every single class in the pipeline.
 *
 * <p>Run via: run-walkthrough.bat
 */
public class TinyStoriesWalkthroughDemo {

    private static final String RESET  = "\u001B[0m";
    private static final String GREY   = "\u001B[38;5;245m";
    private static final String WHITE  = "\u001B[38;5;252m";
    private static final String ORANGE = "\u001B[38;5;208m";
    private static final String GREEN  = "\u001B[38;5;114m";
    private static final String BLUE   = "\u001B[38;5;117m";
    private static final String BOLD   = "\u001B[1m";

    private static final int D_MODEL    = 64;
    private static final int NUM_HEADS  = 4;
    private static final int NUM_LAYERS = 2;
    private static final int SEQ_LEN    = 32;
    private static final int INPUT_LEN  = 12;

    public static void main(String[] args) throws Exception {

        banner();

        // ── 0. Datensatz laden (Streaming) ───────────────────────────────────
        step(0, "TinyStories-Datei", "Erste Geschichte per Stream lesen — kein RAM-Limit");
        String firstStory = loadFirstStory();
        String corpus     = loadCorpusSample(200_000);
        printSnippet("Erste Geschichte", firstStory, 250);
        ok();

        // ── 1. CharTokenizer ─────────────────────────────────────────────────
        step(1, "CharTokenizer", "Jedes einzigartige Zeichen bekommt eine ID (Vokabular aufbauen)");
        CharTokenizer tok = CharTokenizer.build(corpus);
        print(GREY + "  Vokabular-Groesse : " + WHITE + tok.vocabSize() + " Token" + RESET);
        print(GREY + "  EOS-Token ID      : " + WHITE + tok.eosTokenId() + RESET);
        int[] ids = tok.encode(firstStory.substring(0, Math.min(INPUT_LEN * 6, firstStory.length())));
        int[] inputIds = Arrays.copyOf(ids, Math.min(INPUT_LEN, ids.length));
        print(GREY + "  Eingabe-Text      : " + WHITE + "\"" + escapeToken(tok.decode(inputIds)) + "\"" + RESET);
        print(GREY + "  Token-IDs         : " + WHITE + Arrays.toString(inputIds) + RESET);
        ok();

        // ── 2. GPTConfig ─────────────────────────────────────────────────────
        step(2, "GPTConfig", "Architektur-Parameter definieren und validieren");
        GPTConfig cfg = new GPTConfig(tok.vocabSize(), SEQ_LEN, D_MODEL, NUM_HEADS, NUM_LAYERS, 0f);
        print(GREY + "  vocabSize  = " + WHITE + cfg.vocabSize + RESET);
        print(GREY + "  seqLen     = " + WHITE + cfg.seqLen   + RESET);
        print(GREY + "  dModel     = " + WHITE + cfg.dModel   + "  (Embedding-Dimension)" + RESET);
        print(GREY + "  numHeads   = " + WHITE + cfg.numHeads + "  (Attention-Koepfe, dHead=" + D_MODEL/NUM_HEADS + ")" + RESET);
        print(GREY + "  numLayers  = " + WHITE + cfg.numLayers + RESET);
        ok();

        // ── 3. GPTModel (Initialisierung) ────────────────────────────────────
        step(3, "GPTModel", "Alle Gewichte zufaellig initialisieren  N(0, 0.02)");
        GPTModel model = new GPTModel(cfg);
        print(GREY + "  Gesamt-Parameter  : " + WHITE + String.format("%,d", model.paramCount()) + RESET);
        print(GREY + "  Speicher (float32): " + WHITE + String.format("%.1f KB", model.paramCount() * 4.0 / 1024) + RESET);
        ok();

        // ── 4. Token Embedding ───────────────────────────────────────────────
        step(4, "Embedding", "Token-IDs -> dichte Vektoren  int[T] -> float[B, T, dModel]");
        int[][] tokenIds = { inputIds };
        Tensor tokEmbed  = model.tokenEmbed().forward(tokenIds);
        printTensor("Token-Embedding Ausgabe", tokEmbed);
        ok();

        // ── 5. Positional Encoding ───────────────────────────────────────────
        step(5, "PositionalEncoding", "Lernbare Positions-Vektoren addieren  [B,T,D] + [B,T,D]");
        Tensor posEmbed  = model.posEncode().forward(tokEmbed);
        printTensor("Nach Positional Encoding", posEmbed);
        ok();

        // ── 6. Transformer Blocks ────────────────────────────────────────────
        Tensor x = posEmbed;
        List<TransformerBlock> blocks = model.blocks();
        for (int b = 0; b < blocks.size(); b++) {
            step(6 + b, "TransformerBlock[" + b + "]",
                 "Pre-LN -> CausalAttention + Residual -> Pre-LN -> FeedForward + Residual");
            print(GREY + "  Eingabe-Shape : " + WHITE + shapeStr(x) + RESET);
            x = blocks.get(b).forward(x);
            printTensor("Block[" + b + "] Ausgabe", x);
            ok();
        }

        // ── 7. Final LayerNorm ───────────────────────────────────────────────
        step(6 + blocks.size(), "LayerNorm (Final)", "Stabilisierung vor dem LM-Head");
        Tensor normed = model.finalNorm().forward(x);
        printTensor("Nach Final-LayerNorm", normed);
        ok();

        // ── 8. LM Head ───────────────────────────────────────────────────────
        step(7 + blocks.size(), "Dense (LM Head)", "Hidden -> Logits  [B*T, D] -> [B*T, vocabSize]");
        int B = normed.shape()[0];
        int T = normed.shape()[1];
        int D = normed.shape()[2];
        Tensor flat   = TensorOps.reshape(normed, B * T, D);
        Tensor logits = model.lmHead().forward(flat);
        printTensor("Logits (roh, unnormiert)", logits);
        ok();

        // ── 9. Softmax + Top-5 ───────────────────────────────────────────────
        step(8 + blocks.size(), "Softmax + Top-5",
             "Welches Token kommt als naechstes? (nach Position " + (inputIds.length - 1) + ")");

        int lastPos = inputIds.length - 1;
        float[] row = new float[tok.vocabSize()];
        System.arraycopy(logits.data(), lastPos * tok.vocabSize(), row, 0, tok.vocabSize());

        float maxL = Float.NEGATIVE_INFINITY;
        for (float v : row) if (v > maxL) maxL = v;
        float sum   = 0;
        float[] probs = new float[row.length];
        for (int i = 0; i < row.length; i++) { probs[i] = (float) Math.exp(row[i] - maxL); sum += probs[i]; }
        for (int i = 0; i < row.length; i++) probs[i] /= sum;

        Integer[] idx = new Integer[probs.length];
        for (int i = 0; i < idx.length; i++) idx[i] = i;
        Arrays.sort(idx, (a2, b2) -> Float.compare(probs[b2], probs[a2]));

        print("\n" + BOLD + WHITE + "  Top-5 Vorhersagen fuer das naechste Token:" + RESET);
        print(GREY + "  Kontext: " + WHITE + "\"" + escapeToken(tok.decode(inputIds)) + "\"" + RESET);
        print(GREY + "  +-------+------------------------+------------------+" + RESET);
        print(GREY + "  | Rang  | Token                  | Wahrscheinlichkeit|" + RESET);
        print(GREY + "  +-------+------------------------+------------------+" + RESET);
        for (int r = 0; r < Math.min(5, idx.length); r++) {
            int    id    = idx[r];
            float  prob  = probs[id];
            String token = id == tok.eosTokenId() ? "<eos>" : tok.decode(new int[]{id});
            String bar   = greenBar(prob, 12);
            System.out.printf(GREY + "  | %5d | " + GREEN + "%-22s" + GREY + " | %5.1f%% %s |" + RESET + "%n",
                r + 1, escapeToken(token), prob * 100, bar);
        }
        print(GREY + "  +-------+------------------------+------------------+" + RESET);

        // ── 10. Zusammenfassung ──────────────────────────────────────────────
        summary(model, tok, blocks.size());
    }

    // ── Hilfsmethoden ────────────────────────────────────────────────────────

    private static void banner() {
        print("\n" + ORANGE + BOLD + "  +=======================================================+" + RESET);
        print(ORANGE + BOLD        + "  |  FastDL -- GPT Forward-Pass Walkthrough               |" + RESET);
        print(ORANGE + BOLD        + "  |  Jeder Schritt, jede Klasse, live im Terminal         |" + RESET);
        print(ORANGE + BOLD        + "  +=======================================================+" + RESET);
        print(GREY + "  TinyStories-Datensatz  *  Mini-GPT  *  Java 17\n" + RESET);
    }

    private static void step(int n, String cls, String what) {
        print("\n" + BLUE + BOLD + "  -- Schritt " + (n + 1) + ": " + WHITE + cls + RESET);
        print(GREY + "     " + what + RESET);
    }

    private static void ok() { print(GREEN + "     OK" + RESET); }

    private static void printTensor(String label, Tensor t) {
        float[] d = t.data();
        float min = Float.MAX_VALUE, max = -Float.MAX_VALUE;
        double mean = 0;
        for (float v : d) { if (v < min) min = v; if (v > max) max = v; mean += v; }
        mean /= d.length;

        int show = Math.min(6, d.length);
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < show; i++) { sb.append(String.format("%.3f", d[i])); if (i < show-1) sb.append(", "); }
        if (d.length > show) sb.append(", ...");
        sb.append("]");

        print(GREY + "  " + label + RESET);
        print(GREY + "    Shape    : " + WHITE + shapeStr(t) + RESET);
        print(String.format(GREY + "    Werte    : min=%.4f  max=%.4f  mean=%.4f" + RESET, min, max, (float) mean));
        print(GREY + "    Erste 6  : " + WHITE + sb + RESET);
    }

    private static void printSnippet(String label, String text, int maxLen) {
        String s = text.replace("\n", " ").trim();
        if (s.length() > maxLen) s = s.substring(0, maxLen) + "...";
        print(GREY + "  " + label + ": " + WHITE + "\"" + s + "\"" + RESET);
    }

    private static String shapeStr(Tensor t) {
        int[] s = t.shape();
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < s.length; i++) { sb.append(s[i]); if (i < s.length-1) sb.append(", "); }
        return sb.append("]").toString();
    }

    private static String escapeToken(String t) {
        return t.replace("\n","\\n").replace("\r","\\r").replace("\t","\\t");
    }

    private static String greenBar(float prob, int width) {
        int filled = (int)(prob * width * 5);
        return GREEN + "#".repeat(Math.min(filled, width)) + GREY;
    }

    private static void summary(GPTModel model, CharTokenizer tok, int numBlocks) {
        print("\n" + ORANGE + BOLD + "  == Pipeline-Zusammenfassung ==" + RESET);
        print(GREY + "  Klasse                  Aufgabe" + RESET);
        print(GREY + "  -------------------------------------------------------------" + RESET);
        row("CharTokenizer",       "Text -> IDs  (Vokabular: " + tok.vocabSize() + " Token)");
        row("GPTConfig",           "Architektur-Parameter buendeln und validieren");
        row("GPTModel",            "Alle Gewichte + forward/backward koordinieren");
        row("Embedding",           "[T] -> [T, " + D_MODEL + "]  Token-ID auf Vektor");
        row("PositionalEncoding",  "[B,T,D] += Positions-Vektor");
        for (int i = 0; i < numBlocks; i++)
            row("TransformerBlock[" + i + "]", "Norm->Attention->Res + Norm->FFN->Res");
        row("LayerNorm (Final)",   "Stabilisierung vor dem Ausgang");
        row("Dense (LM Head)",     "[B*T, D] -> [B*T, " + tok.vocabSize() + "]  Logits");
        row("Softmax + Top-5",     "Wahrscheinlichste naechste Token anzeigen");
        print(GREY + "  -------------------------------------------------------------" + RESET);
        print(GREEN + "\n  FastDL  " + String.format("%,d", model.paramCount()) + " Parameter  Java 17\n" + RESET);
    }

    private static void row(String cls, String desc) {
        print(GREY + "  " + String.format("%-24s", WHITE + cls + GREY) + " " + desc + RESET);
    }

    private static void print(String s) { System.out.println(s); }

    // ── Datei-Lader ──────────────────────────────────────────────────────────

    private static String loadFirstStory() throws Exception {
        Path p = resolveDataset();
        try (BufferedReader br = new BufferedReader(new FileReader(p.toFile()))) {
            StringBuilder sb = new StringBuilder();
            String line;
            while ((line = br.readLine()) != null) {
                if (line.equals("<|endoftext|>") && sb.length() > 50) break;
                sb.append(line).append('\n');
            }
            return sb.toString().trim();
        }
    }

    private static String loadCorpusSample(int maxChars) throws Exception {
        Path p = resolveDataset();
        try (BufferedReader br = new BufferedReader(new FileReader(p.toFile()))) {
            StringBuilder sb = new StringBuilder();
            char[] buf = new char[8192];
            int read;
            while (sb.length() < maxChars && (read = br.read(buf)) != -1) sb.append(buf, 0, read);
            return sb.length() > maxChars ? sb.substring(0, maxChars) : sb.toString();
        }
    }

    private static Path resolveDataset() {
        for (String c : new String[]{
                "data/TinyStories-train.txt", "data/tinystories.txt",
                "../../data/TinyStories-train.txt", "../../data/tinystories.txt"}) {
            Path p = Path.of(c);
            if (Files.exists(p)) return p;
        }
        throw new RuntimeException("Datensatz nicht gefunden. Lege TinyStories-train.txt unter data/ ab.");
    }
}
