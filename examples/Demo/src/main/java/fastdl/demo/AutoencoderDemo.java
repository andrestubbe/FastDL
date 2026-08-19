package fastdl.demo;

import fastdl.FastDL;
import fastdl.FastDL.Sequential;
import fastdl.loss.MSELoss;
import fastdl.optim.SGD;
import fastdl.tensor.Tensor;

import javax.swing.*;
import java.awt.*;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.util.ArrayList;
import java.util.List;

/**
 * FastDL Demo 3: Autoencoder 2D Latent Space & Reconstruction Visualizer.
 */
public class AutoencoderDemo extends JFrame {

    private static final int PIXEL_DIM = 64; // 8x8 image

    private final Sequential encoder;
    private final Sequential decoder;
    private final SGD optimizer;
    private final MSELoss loss = FastDL.mse();

    private final List<Tensor> syntheticDigits = new ArrayList<>();
    private final LatentCanvas latentCanvas = new LatentCanvas();
    private final ImageReconstructPanel reconstructPanel = new ImageReconstructPanel();
    private final Timer trainTimer;

    private float latentX = 0.0f;
    private float latentY = 0.0f;

    public AutoencoderDemo() {
        super("FastDL — Autoencoder 2D Latent Space Visualizer");
        setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        setSize(950, 600);
        setLocationRelativeTo(null);

        // Encoder: 64 inputs -> 16 hidden -> 2 latent dimensions (X, Y)
        encoder = FastDL.sequential(
                FastDL.dense(64, 16),
                FastDL.relu(),
                FastDL.dense(16, 2)
        );

        // Decoder: 2 latent dimensions -> 16 hidden -> 64 reconstructed outputs
        decoder = FastDL.sequential(
                FastDL.dense(2, 16),
                FastDL.relu(),
                FastDL.dense(16, 64)
        );

        List<Tensor> allParams = new ArrayList<>();
        allParams.addAll(encoder.parameters());
        allParams.addAll(decoder.parameters());
        optimizer = FastDL.sgd(allParams, 0.02f, 0.9f);

        generateSyntheticPatterns();

        setLayout(new BorderLayout(15, 10));

        JPanel center = new JPanel(new GridLayout(1, 2, 15, 10));
        center.setBorder(BorderFactory.createEmptyBorder(10, 15, 10, 15));

        JPanel leftBox = new JPanel(new BorderLayout());
        leftBox.setBorder(BorderFactory.createTitledBorder("2D Latent-Space (Klicke & Bewege Cursor)"));
        leftBox.add(latentCanvas, BorderLayout.CENTER);

        JPanel rightBox = new JPanel(new BorderLayout());
        rightBox.setBorder(BorderFactory.createTitledBorder("Live Rekonstruktion (8x8 Pixel Grid)"));
        rightBox.add(reconstructPanel, BorderLayout.CENTER);

        center.add(leftBox);
        center.add(rightBox);
        add(center, BorderLayout.CENTER);

        JLabel status = new JLabel("Autoencoder lernt Datenverteilung & Kompression live...", JLabel.CENTER);
        status.setBorder(BorderFactory.createEmptyBorder(6, 10, 10, 10));
        add(status, BorderLayout.SOUTH);

        trainTimer = new Timer(25, e -> {
            trainBatch();
            updateReconstruction();
            latentCanvas.repaint();
            reconstructPanel.repaint();
        });
        trainTimer.start();
    }

    private void generateSyntheticPatterns() {
        // Generate patterns: horizontal lines, vertical lines, diagonal, cross
        for (int i = 0; i < 40; i++) {
            Tensor t = new Tensor(1, 64);
            // Horizontal bar
            for (int x = 0; x < 8; x++) t.set(1.0f, 0, 3 * 8 + x);
            syntheticDigits.add(t);

            Tensor v = new Tensor(1, 64);
            // Vertical bar
            for (int y = 0; y < 8; y++) v.set(1.0f, 0, y * 8 + 3);
            syntheticDigits.add(v);

            Tensor c = new Tensor(1, 64);
            // Cross
            for (int j = 0; j < 8; j++) {
                c.set(1.0f, 0, j * 8 + j);
                c.set(1.0f, 0, j * 8 + (7 - j));
            }
            syntheticDigits.add(c);
        }
    }

    private void trainBatch() {
        for (Tensor input : syntheticDigits) {
            optimizer.zeroGrad();
            Tensor latent = encoder.forward(input);
            Tensor output = decoder.forward(latent);

            float l = loss.forward(output, input);
            Tensor decGrad = loss.backward(output, input);
            Tensor encGrad = decoder.backward(decGrad);
            encoder.backward(encGrad);

            optimizer.step();
        }
    }

    private void updateReconstruction() {
        Tensor q = new Tensor(new float[]{latentX, latentY}, 1, 2);
        Tensor out = decoder.forward(q);
        reconstructPanel.setPixels(out.data());
    }

    class LatentCanvas extends JPanel {
        LatentCanvas() {
            setBackground(new Color(245, 245, 245));
            MouseAdapter m = new MouseAdapter() {
                @Override
                public void mouseDragged(MouseEvent e) {
                    updateCoords(e);
                }

                @Override
                public void mousePressed(MouseEvent e) {
                    updateCoords(e);
                }

                private void updateCoords(MouseEvent e) {
                    latentX = (float) (e.getX() - getWidth() / 2) / (getWidth() / 4f);
                    latentY = (float) (e.getY() - getHeight() / 2) / (getHeight() / 4f);
                    updateReconstruction();
                    repaint();
                    reconstructPanel.repaint();
                }
            };
            addMouseListener(m);
            addMouseMotionListener(m);
        }

        @Override
        protected void paintComponent(Graphics g) {
            super.paintComponent(g);
            int w = getWidth();
            int h = getHeight();

            // Center Axis
            g.setColor(new Color(210, 210, 210));
            g.drawLine(w / 2, 0, w / 2, h);
            g.drawLine(0, h / 2, w, h / 2);

            // Draw Projected Training Points in Latent Space
            Graphics2D g2 = (Graphics2D) g;
            g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

            for (Tensor t : syntheticDigits) {
                Tensor lat = encoder.forward(t);
                float lx = lat.get(0, 0);
                float ly = lat.get(0, 1);

                int px = (int) (lx * (w / 4f) + w / 2f);
                int py = (int) (ly * (h / 4f) + h / 2f);

                g2.setColor(new Color(70, 130, 240, 180));
                g2.fillOval(px - 4, py - 4, 8, 8);
            }

            // Current Latent Probe Position
            int cx = (int) (latentX * (w / 4f) + w / 2f);
            int cy = (int) (latentY * (h / 4f) + h / 2f);

            g2.setColor(Color.RED);
            g2.setStroke(new BasicStroke(2));
            g2.drawOval(cx - 8, cy - 8, 16, 16);
            g2.fillOval(cx - 3, cy - 3, 6, 6);
        }
    }

    class ImageReconstructPanel extends JPanel {
        private float[] pixels = new float[64];

        void setPixels(float[] p) {
            this.pixels = p;
        }

        @Override
        protected void paintComponent(Graphics g) {
            super.paintComponent(g);
            int size = Math.min(getWidth(), getHeight()) - 20;
            int startX = (getWidth() - size) / 2;
            int startY = (getHeight() - size) / 2;
            int cell = size / 8;

            for (int y = 0; y < 8; y++) {
                for (int x = 0; x < 8; x++) {
                    float v = Math.min(1.0f, Math.max(0.0f, pixels[y * 8 + x]));
                    int gray = (int) (255 * (1.0f - v));
                    g.setColor(new Color(gray, gray, gray));
                    g.fillRect(startX + x * cell, startY + y * cell, cell, cell);
                    g.setColor(new Color(200, 200, 200, 100));
                    g.drawRect(startX + x * cell, startY + y * cell, cell, cell);
                }
            }
        }
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> new AutoencoderDemo().setVisible(true));
    }
}
