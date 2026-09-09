package fastdl.demo;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import org.junit.jupiter.api.Test;

class TinyStoriesMiniDemoTest {
    @Test
    void generateFromPromptShouldReturnPromptAndContinuation() {
        List<String> stories = List.of(
                "Lily found a red ball in the grass.",
                "Tom ran to the warm sun and smiled.",
                "Mia loved her little puppy."
        );

        String result = TinyStoriesMiniDemo.generateFromPrompt("Lily", stories, 8);

        assertTrue(result.toLowerCase().contains("lily"));
        assertTrue(result.length() > "Lily".length());
    }
}
