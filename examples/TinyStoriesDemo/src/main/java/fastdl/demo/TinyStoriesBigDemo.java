package fastdl.demo;

/**
 * Explicit entrypoint for the long-running TinyStories big-data run.
 *
 * This is intentionally separate from the smoke/demo entrypoints so the root batch
 * files can select the correct training profile without relying on fragile Maven
 * property overrides.
 */
public class TinyStoriesBigDemo {
    public static void main(String[] args) throws Exception {
        TinyStoriesTransformerDemo.main(new String[] { "--mode", "big" });
    }
}
