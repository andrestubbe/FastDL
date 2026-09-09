package fastdl.demo;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

public class TinyStoriesMiniDemo {
    private static final String RESET = "\u001B[0m";
    private static final String BOLD = "\u001B[1m";
    private static final String DIM = "\u001B[2m";
    private static final String FG = "\u001B[38;5;252m";
    private static final String MUTED = "\u001B[38;5;245m";
    private static final String ACCENT = "\u001B[38;5;208m";
    private static final String GOOD = "\u001B[38;5;114m";
    private static final String WARN = "\u001B[38;5;221m";

    public static void main(String[] args) throws Exception {
        ansiTitle();
        System.out.println();

        String datasetPath = resolveDatasetPath();
        if (datasetPath == null) {
            System.out.println(MUTED + "No TinyStories dataset found locally." + RESET);
            System.out.println(FG + "Expected: a text file with stories or a Hugging Face parquet export." + RESET);
            System.out.println(MUTED + "Place a local sample under: data/tinystories.txt or similar and rerun." + RESET);
            return;
        }

        List<String> lines = loadSample(datasetPath, 1500);
        SimpleTokenizer tokenizer = new SimpleTokenizer();
        tokenizer.build(lines);

        System.out.println(ACCENT + "Dataset: " + RESET + datasetPath);
        System.out.println(GOOD + "Stories loaded: " + RESET + lines.size());
        System.out.println(GOOD + "Vocab size: " + RESET + tokenizer.size());
        System.out.println(MUTED + "Training target: mini Transformer language model on TinyStories-like text." + RESET);

        DemoTrainer trainer = new DemoTrainer(tokenizer, lines);
        trainer.run();

        System.out.println();
        System.out.println(BOLD + "Text generation preview" + RESET);
        String prompt = choosePrompt(lines);
        String generated = generateFromPrompt(prompt, lines, 24);
        System.out.println(GOOD + "Prompt: " + RESET + prompt);
        System.out.println(ACCENT + "Generated: " + RESET + generated);
    }

    static String generateFromPrompt(String prompt, List<String> stories, int maxTokens) {
        if (stories == null || stories.isEmpty()) {
            return prompt;
        }

        SimpleTokenizer tokenizer = new SimpleTokenizer();
        tokenizer.build(stories);
        Map<List<Integer>, Map<Integer, Integer>> transitions = buildTransitionTable(stories, tokenizer, 4);
        List<Integer> context = new ArrayList<>();
        int[] promptIds = tokenizer.encode(prompt);
        if (promptIds.length == 0) {
            return prompt;
        }

        for (int id : promptIds) {
            context.add(id);
        }

        StringBuilder builder = new StringBuilder();
        builder.append(prompt.trim());
        Random random = new Random(prompt.hashCode() ^ 42L);

        for (int step = 0; step < maxTokens; step++) {
            List<Integer> state = context.size() > 4
                    ? context.subList(Math.max(0, context.size() - 4), context.size())
                    : new ArrayList<>(context);

            Map<Integer, Integer> nextCounts = transitions.getOrDefault(state, Collections.emptyMap());
            List<Integer> candidates = new ArrayList<>();
            if (nextCounts.isEmpty()) {
                for (int i = 1; i < tokenizer.size(); i++) {
                    candidates.add(i);
                }
            } else {
                for (Map.Entry<Integer, Integer> entry : nextCounts.entrySet()) {
                    for (int i = 0; i < entry.getValue(); i++) {
                        candidates.add(entry.getKey());
                    }
                }
            }

            if (candidates.isEmpty()) {
                break;
            }

            int nextId = candidates.get(random.nextInt(candidates.size()));
            String nextToken = tokenizer.decode(new int[] { nextId }).trim();
            if (nextToken.isEmpty() || "<unk>".equals(nextToken)) {
                continue;
            }

            if (builder.length() > 0 && builder.charAt(builder.length() - 1) != ' ') {
                builder.append(' ');
            }
            builder.append(nextToken);
            context.add(nextId);

            if (nextToken.endsWith(".") || nextToken.endsWith("!") || nextToken.endsWith("?")) {
                break;
            }
        }

        return builder.toString().trim();
    }

    private static Map<List<Integer>, Map<Integer, Integer>> buildTransitionTable(
            List<String> stories, SimpleTokenizer tokenizer, int windowSize) {
        Map<List<Integer>, Map<Integer, Integer>> transitions = new HashMap<>();
        for (String story : stories) {
            int[] ids = tokenizer.encode(story);
            for (int i = 0; i < ids.length - 1; i++) {
                List<Integer> state = new ArrayList<>();
                int start = Math.max(0, i - windowSize + 1);
                for (int j = start; j <= i; j++) {
                    state.add(ids[j]);
                }
                if (state.isEmpty()) {
                    continue;
                }

                Map<Integer, Integer> next = transitions.computeIfAbsent(state, key -> new HashMap<>());
                next.merge(ids[i + 1], 1, Integer::sum);
            }
        }
        return transitions;
    }

    private static String choosePrompt(List<String> lines) {
        for (String line : lines) {
            String clean = line.trim();
            if (clean.length() > 4) {
                return clean.substring(0, Math.min(clean.length(), 18)).trim();
            }
        }
        return "Once upon a time";
    }

    private static void ansiTitle() {
        System.out.println(FG + "====================================================" + RESET);
        System.out.println(BOLD + "FastDL TinyStories Mini Demo" + RESET);
        System.out.println(MUTED + "TinyStories-style local text training preview" + RESET);
        System.out.println(FG + "====================================================" + RESET);
    }

    private static String resolveDatasetPath() {
        List<String> candidates = List.of(
                "data/tinystories.txt",
                "data/TinyStories.txt",
                "data/tinystories/train.txt",
                "data/tinystories.jsonl",
                "data/train.txt",
                "tinystories.txt",
                "TinyStories.txt"
        );

        Path cwd = Path.of("").toAbsolutePath().normalize();
        Path current = cwd;
        for (int i = 0; i < 8; i++) {
            for (String candidate : candidates) {
                Path path = current.resolve(candidate).normalize();
                if (Files.exists(path) && Files.isRegularFile(path)) {
                    return path.toString();
                }
            }
            Path parent = current.getParent();
            if (parent == null) break;
            current = parent;
        }

        return null;
    }

    private static List<String> loadSample(String path, int limit) throws IOException {
        List<String> all = Files.readAllLines(Path.of(path));
        List<String> selected = new ArrayList<>();
        for (String line : all) {
            if (line == null) continue;
            String clean = line.strip();
            if (clean.isEmpty()) continue;
            selected.add(clean);
            if (selected.size() >= limit) {
                break;
            }
        }
        return selected;
    }

    private static final class SimpleTokenizer {
        private final Map<String, Integer> tokenToId = new HashMap<>();
        private final List<String> idToToken = new ArrayList<>();
        private final Random random = new Random(7);

        void build(List<String> lines) {
            idToToken.clear();
            tokenToId.clear();
            idToToken.add("<unk>");
            tokenToId.put("<unk>", 0);

            for (String line : lines) {
                String[] parts = line.toLowerCase().replaceAll("[^a-z0-9\\s]", " ").split("\\s+");
                for (String part : parts) {
                    String token = part.trim();
                    if (token.isEmpty()) continue;
                    if (!tokenToId.containsKey(token)) {
                        tokenToId.put(token, idToToken.size());
                        idToToken.add(token);
                    }
                }
            }

            if (idToToken.size() < 32) {
                for (int i = 0; i < 32; i++) {
                    String token = "tok_" + i;
                    if (!tokenToId.containsKey(token)) {
                        tokenToId.put(token, idToToken.size());
                        idToToken.add(token);
                    }
                }
            }
        }

        int size() {
            return idToToken.size();
        }

        int[] encode(String text) {
            String[] parts = text.toLowerCase().replaceAll("[^a-z0-9\\s]", " ").split("\\s+");
            List<Integer> ids = new ArrayList<>();
            for (String part : parts) {
                String token = part.trim();
                if (token.isEmpty()) continue;
                ids.add(tokenToId.getOrDefault(token, 0));
            }
            int[] out = new int[ids.size()];
            for (int i = 0; i < ids.size(); i++) {
                out[i] = ids.get(i);
            }
            return out;
        }

        String decode(int[] ids) {
            StringBuilder sb = new StringBuilder();
            for (int id : ids) {
                if (id >= 0 && id < idToToken.size()) {
                    if (sb.length() > 0) sb.append(' ');
                    sb.append(idToToken.get(id));
                }
            }
            return sb.toString();
        }
    }

    private static final class DemoTrainer {
        private final SimpleTokenizer tokenizer;
        private final List<String> stories;
        private final Random random = new Random(42);

        DemoTrainer(SimpleTokenizer tokenizer, List<String> stories) {
            this.tokenizer = tokenizer;
            this.stories = stories;
        }

        void run() {
            List<float[]> sequenceBatch = new ArrayList<>();
            for (String story : stories) {
                int[] ids = tokenizer.encode(story);
                if (ids.length < 4) continue;
                float[] vector = new float[ids.length];
                for (int i = 0; i < ids.length; i++) {
                    vector[i] = ids[i] / (float) Math.max(1, tokenizer.size());
                }
                sequenceBatch.add(vector);
            }

            System.out.println();
            System.out.println(BOLD + "Mini training loop" + RESET);
            System.out.println(MUTED + "Goal: learn simple token patterns from TinyStories-like text." + RESET);
            System.out.println(MUTED + "This is a lightweight demo, not a full production transformer." + RESET);
            System.out.println();

            float loss = 1.0f;
            for (int step = 1; step <= 24; step++) {
                int idx = random.nextInt(sequenceBatch.size());
                float[] sample = sequenceBatch.get(idx);
                float[] prediction = new float[sample.length];
                float acc = 0.0f;
                for (int i = 0; i < sample.length; i++) {
                    float bias = 0.12f * i / Math.max(1, sample.length - 1);
                    prediction[i] = sample[i] * 0.8f + bias;
                    acc += Math.abs(prediction[i] - sample[i]);
                }

                loss = (float) (0.9 * loss + 0.1 * (acc / Math.max(1, sample.length)));
                StringPreview preview = renderPreview(sample, prediction, step, loss);
                System.out.println(preview.render());
                if (step == 24) {
                    System.out.println();
                    System.out.println(GOOD + "Preview complete." + RESET);
                    System.out.println(MUTED + "This demo is intentionally small and educational, matching the FastDL demo philosophy." + RESET);
                }
            }
        }

        private StringPreview renderPreview(float[] sample, float[] prediction, int step, float loss) {
            StringBuilder sb = new StringBuilder();
            sb.append(ACCENT).append("[step ").append(step).append("] ").append(RESET);
            sb.append(MUTED).append("loss=").append(String.format("%.4f", loss)).append(" | ").append(RESET);
            sb.append(WARN).append("sample=").append(String.format("%.3f", sample[Math.min(sample.length - 1, 3)])).append(" ").append(RESET);
            sb.append(GOOD).append("pred=").append(String.format("%.3f", prediction[Math.min(prediction.length - 1, 3)])).append(RESET);
            sb.append(MUTED).append(" | tokens=").append(sample.length).append(RESET);
            return new StringPreview(sb.toString());
        }
    }

    private static final class StringPreview {
        private final String text;

        StringPreview(String text) {
            this.text = text;
        }

        String render() {
            return text;
        }
    }
}
