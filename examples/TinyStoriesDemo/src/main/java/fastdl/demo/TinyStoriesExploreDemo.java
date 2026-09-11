package fastdl.demo;

import fastdl.tokenizer.CharTokenizer;

import java.io.*;
import java.nio.file.*;
import java.util.*;

/**
 * TinyStoriesExploreDemo -- streams through TinyStories-train.txt without loading
 * it fully into RAM and prints statistics + scientific background about the dataset.
 *
 * Run via: run-explore.bat
 */
public class TinyStoriesExploreDemo {

    private static final String RESET  = "\u001B[0m";
    private static final String GREY   = "\u001B[38;5;245m";
    private static final String WHITE  = "\u001B[38;5;252m";
    private static final String ORANGE = "\u001B[38;5;208m";
    private static final String GREEN  = "\u001B[38;5;114m";
    private static final String BLUE   = "\u001B[38;5;117m";
    private static final String BOLD   = "\u001B[1m";

    public static void main(String[] args) throws Exception {
        banner();
        background();
        stats();
    }

    private static void background() {
        print("\n" + ORANGE + BOLD + "  == Warum existiert TinyStories? ==" + RESET);
        print(GREY + "  Quelle: Eldan & Li, Microsoft Research, 2023" + RESET);
        print(GREY + "  Paper : \"TinyStories: How Small Can Language Models Be and" + RESET);
        print(GREY + "           Still Speak Coherent English?\"" + RESET);
        print("");
        print(GREY + "  Das zentrale Problem:" + RESET);
        print(WHITE + "  Sind Milliarden Parameter wirklich noetig fuer kohaerenten Text --" + RESET);
        print(WHITE + "  oder liegt es am Daten-Mismatch?" + RESET);
        print("");
        print(GREY + "  Die Idee:" + RESET);
        print(WHITE + "  Wenn Trainingsdaten nur Woerter und Strukturen enthalten die" + RESET);
        print(WHITE + "  ein 3-4-jaehriges Kind versteht, kann ein winziges Modell" + RESET);
        print(WHITE + "  (~1M Parameter) grammatisch korrekten, sinnvollen Text erzeugen." + RESET);
        print("");
        print(GREY + "  Ergebnis:" + RESET);
        print(WHITE + "  GPT-Neo 1.3B trainiert auf normalen Daten: schlechter als" + RESET);
        print(WHITE + "  ein 28M-Parameter-Modell trainiert auf TinyStories." + RESET);
        print(WHITE + "  Das zeigt: Datenqualitaet schlaegt Modellgroesse." + RESET);
        print("");
        print(GREY + "  Warum perfekt fuer FastDL:" + RESET);
        print(WHITE + "  - Einfache Grammatik  -> CharTokenizer reicht aus" + RESET);
        print(WHITE + "  - Kurze Saetze        -> seqLen=64-128 genuegt" + RESET);
        print(WHITE + "  - Klare Muster        -> Modell lernt schnell" + RESET);
        print(WHITE + "  - 1.9 GB              -> gross genug fuer echtes Training" + RESET);
        print(WHITE + "  - Kein Copyright      -> synthetisch generiert via GPT-4" + RESET);
        print(GREY + "\n  " + "-".repeat(55) + RESET);
    }

    private static void stats() throws Exception {
        Path p = resolveDataset();
        print("\n" + BLUE + BOLD + "  == Datei analysieren (Stream, kein RAM-Limit) ==" + RESET);
        print(GREY + "  Datei: " + WHITE + p.toAbsolutePath() + RESET);
        print(GREY + "  Groesse: " + WHITE + String.format("%.1f MB", Files.size(p) / 1_000_000.0) + RESET);
        print("");

        long storyCount = 0, lineCount = 0, charCount = 0, wordCount = 0;
        Map<Character, Long> charFreq = new TreeMap<>();
        List<String> samples = new ArrayList<>();
        int maxSamples = 3;

        StringBuilder currentStory = new StringBuilder();
        long printEvery = 100_000;
        long nextPrint  = printEvery;

        try (BufferedReader br = new BufferedReader(new FileReader(p.toFile()), 1 << 16)) {
            String line;
            while ((line = br.readLine()) != null) {
                lineCount++;
                charCount += line.length() + 1;

                if (line.equals("<|endoftext|>")) {
                    storyCount++;
                    if (samples.size() < maxSamples && currentStory.length() > 30) {
                        samples.add(currentStory.toString().trim());
                    }
                    currentStory.setLength(0);
                } else {
                    currentStory.append(line).append(' ');
                    for (String w : line.split("\\s+")) if (!w.isEmpty()) wordCount++;
                    for (char c : line.toCharArray()) charFreq.merge(c, 1L, Long::sum);
                }

                if (lineCount == nextPrint) {
                    System.out.printf(GREY + "  ... %,d Zeilen gelesen, %,d Geschichten ...\r" + RESET,
                        lineCount, storyCount);
                    nextPrint += printEvery;
                }
            }
        }

        print("\n");
        print(ORANGE + BOLD + "  == Ergebnisse ==" + RESET);
        print(GREY + "  Geschichten   : " + WHITE + String.format("%,d", storyCount) + RESET);
        print(GREY + "  Zeilen        : " + WHITE + String.format("%,d", lineCount) + RESET);
        print(GREY + "  Woerter       : " + WHITE + String.format("%,d", wordCount) + RESET);
        print(GREY + "  Zeichen       : " + WHITE + String.format("%,d", charCount) + RESET);
        print(GREY + "  Vokabular     : " + WHITE + charFreq.size() + " einzigartige Zeichen" + RESET);
        print(GREY + "  Ø Woerter/Story: " + WHITE + String.format("%.0f", (double) wordCount / storyCount) + RESET);

        // Top-10 Zeichen
        print("\n" + GREY + "  Haeufigste Zeichen:" + RESET);
        charFreq.entrySet().stream()
            .sorted(Map.Entry.<Character, Long>comparingByValue().reversed())
            .limit(10)
            .forEach(e -> {
                String bar = GREEN + "#".repeat((int)(e.getValue() * 20.0 / charCount * 100)) + GREY;
                System.out.printf(GREY + "    '%s'  %,10d  %s%n" + RESET,
                    escapeChar(e.getKey()), e.getValue(), bar);
            });

        // Beispiel-Geschichten
        print("\n" + ORANGE + BOLD + "  == Beispiel-Geschichten ==" + RESET);
        for (int i = 0; i < samples.size(); i++) {
            String s = samples.get(i).replace("\n", " ");
            if (s.length() > 300) s = s.substring(0, 300) + "...";
            print(GREY + "\n  [" + (i+1) + "] " + WHITE + s + RESET);
        }

        print(GREEN + "\n\n  FastDL ist bereit dieses Dataset zu trainieren." + RESET);
        print(GREY + "  Starte run-tiny2.bat fuer den Training-Loop.\n" + RESET);
    }

    private static void banner() {
        print("\n" + ORANGE + BOLD + "  +=========================================================+" + RESET);
        print(ORANGE + BOLD        + "  |  FastDL -- TinyStories Dataset Explorer                 |" + RESET);
        print(ORANGE + BOLD        + "  |  Warum existiert dieser Datensatz? Was ist drin?        |" + RESET);
        print(ORANGE + BOLD        + "  +=========================================================+" + RESET);
    }

    private static String escapeChar(char c) {
        if (c == ' ')  return "SPC";
        if (c == '\n') return "\\n";
        if (c == '\r') return "\\r";
        if (c == '\t') return "\\t";
        return String.valueOf(c);
    }

    private static void print(String s) { System.out.println(s); }

    private static Path resolveDataset() {
        for (String c : new String[]{
                "data/TinyStories-train.txt", "data/tinystories.txt",
                "../../data/TinyStories-train.txt", "../../data/tinystories.txt"}) {
            Path p = Path.of(c);
            if (Files.exists(p)) return p;
        }
        throw new RuntimeException("Datensatz nicht gefunden unter data/TinyStories-train.txt");
    }
}
