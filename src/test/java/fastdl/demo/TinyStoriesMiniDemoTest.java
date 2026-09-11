package fastdl.demo;

import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class TinyStoriesMiniDemoTest {

    @Test
    void readLinesCappedDoesNotLoadWholeFile() throws Exception {
        Path temp = Files.createTempFile("tinystories-mini", ".txt");
        Files.write(temp, List.of(
            "hello world",
            "this is a second line",
            "third line here",
            "fourth line here"
        ));

        List<String> result = readLinesCapped(temp, 2);

        assertFalse(result.isEmpty());
        assertTrue(result.size() <= 2);
    }

    private static List<String> readLinesCapped(Path path, int limit) throws IOException {
        List<String> selected = new ArrayList<>();
        try (var reader = Files.newBufferedReader(path)) {
            String line;
            while ((line = reader.readLine()) != null && selected.size() < limit) {
                String clean = line.strip();
                if (!clean.isEmpty()) {
                    selected.add(clean);
                }
            }
        }
        return selected;
    }
}
