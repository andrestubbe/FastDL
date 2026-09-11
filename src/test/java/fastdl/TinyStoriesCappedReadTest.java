package fastdl;

import org.junit.jupiter.api.Test;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertTrue;

class TinyStoriesCappedReadTest {

    @Test
    void cappedReadStopsBeforeMaxChars() throws Exception {
        Path temp = Files.createTempFile("tiny-capped", ".txt");
        Files.write(temp, List.of(
            "one two three four five",
            "six seven eight nine ten",
            "eleven twelve thirteen"
        ));

        String text = fastdl.data.TextDataset.readTextCapped(temp, 20);
        assertTrue(text.length() <= 20);
        assertTrue(!text.isEmpty());
    }
}
