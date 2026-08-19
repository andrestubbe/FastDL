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
 * FastDL Demo 2: Neural Classification Boundary (Multi-Layer Perceptron / MLP).
 */
public class MLPBoundaryDemo extends JFrame {

    public record Sample(float x, float y, float label) {}

    private final List<Sample> dataset = new ArrayList<>();
    private final Sequential model;
    private final SGD optimizer;
    private final MSELoss loss = FastDL.mse();

    private float currentClass = 1.0f;
    private final BoundaryPanel canvas = new BoundaryPanel();
    private final Timer trainTimer;
    private float currentLoss = 0f;

    public MLPBoundaryDemo() {
        super("FastDL — Neural Classification Boundary (MLP)");
        setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        setSize(900, 700);
        setLocationRelativeTo(null);

        // Architecture: 2 inputs -> 16 hidden -> ReLU -> 16 hidden -> ReLU -> 1 output
        model = FastDL.sequential(
                FastDL.dense(2, 16),
                FastDL.relu(),
                FastDL.dense(16, 16),
                FastDL.relu(),
                FastDL.dense(16, 1)
        );
        optimizer = FastDL.sgd(model.parameters(), 0.05f, 0.9f);

        setLayout(new BorderLayout());
        add(canvas, BorderLayout.CENTER);

        JPanel controls = new JPanel(new FlowLayout(FlowLayout.CENTER, 12, 8));
        JButton btnClassRed = new JButton("Rot setzen (Klasse 0)");
        JButton btnClassBlue = new JButton("Blau setzen (Klasse 1)");
        JButton btnSpiral = new JButton("Spiral-Muster laden");
        JButton btnClear = new JButton("Löschen");
        JLabel lblLoss = new JLabel("Loss: 0.0000");

        controls.add(btnClassRed);
        controls.add(btnClassBlue);
        controls.add(btnSpiral);
        controls.add(btnClear);
        controls.add(lblLoss);
        add(controls, BorderLayout.SOUTH);

        btnClassRed.addActionListener(e -> currentClass = 0.0f);
        btnClassBlue.addActionListener(e -> currentClass = 1.0f);
        btnSpiral.addActionListener(e -> {
            loadSpiralData();
            canvas.repaint();
        });
        btnClear.addActionListener(e -> {
            dataset.clear();
            canvas.repaint();
        });

        trainTimer = new Timer(20, e -> {
            if (!dataset.isEmpty()) {
                trainEpoch();
                lblLoss.setText(String.format("Loss: %.4f | Samples: %d", currentLoss, dataset.size()));
                canvas.repaint();
            }
        });

        loadSpiralData();
        trainTimer.start();
    }

    private void trainEpoch() {
        int n = dataset.size();
        Tensor x = new Tensor(n, 2);
        Tensor y = new Tensor(n, 1);

        for (int i = 0; i < n; i++) {
            Sample s = dataset.get(i);
            x.set(s.x(), i, 0);
            x.set(s.y(), i, 1);
            y.set(s.label(), i, 0);
        }

        optimizer.zeroGrad();
        Tensor pred = model.forward(x);
        currentLoss = loss.forward(pred, y);
        model.backward(loss.backward(pred, y));
        optimizer.step();
    }

    private void loadSpiralData() {
        dataset.clear();
        int points = 40;
        for (int i = 0; i < points; i++) {
            double r = (double) i / points * 0.8;
            double t = 1.75 * i / points * 2 * Math.PI;

            // Class 0 (Red)
            dataset.add(new Sample((float) (r * Math.sin(t)), (float) (r * Math.cos(t)), 0.0f));

            // Class 1 (Blue)
            dataset.add(new Sample((float) (r * Math.sin(t + Math.PI)), (float) (r * Math.cos(t + Math.PI)), 1.0f));
        }
    }

    class BoundaryPanel extends JPanel {
        BoundaryPanel() {
            addMouseListener(new MouseAdapter() {
                @Override
                public void mousePressed(MouseEvent e) {
                    float normX = (float) (e.getX() - getWidth() / 2) / (getWidth() / 2f);
                    float normY = (float) (e.getY() - getHeight() / 2) / (getHeight() / 2f);
                    dataset.add(new Sample(normX, normY, currentClass));
                }
            });
        }

        @Override
        protected void paintComponent(Graphics g) {
            super.paintComponent(g);
            int w = getWidth();
            int h = getHeight();
            int step = 10;

            // Render Neural Network Decision Heatmap
            int gridW = w / step + 1;
            int gridH = h / step + 1;
            int total = gridW * gridH;

            Tensor gridX = new Tensor(total, 2);
            int idx = 0;
            for (int py = 0; py < h; py += step) {
                float ny = (float) (py - h / 2) / (h / 2f);
                for (int px = 0; px < w; px += step) {
                    float nx = (float) (px - w / 2) / (w / 2f);
                    gridX.set(nx, idx, 0);
                    gridX.set(ny, idx, 1);
                    idx++;
                }
            }

            Tensor gridPred = model.forward(gridX);
            float[] out = gridPred.data();

            idx = 0;
            for (int py = 0; py < h; py += step) {
                for (int px = 0; px < w; px += step) {
                    float val = Math.min(1.0f, Math.max(0.0f, out[idx]));
                    // Red (0.0) to Blue (1.0)
                    g.setColor(new Color(1.0f - val, 0.2f, val, 0.45f));
                    g.fillRect(px, py, step, step);
                    idx++;
                }
            }

            // Draw Samples
            Graphics2D g2 = (Graphics2D) g;
            g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

            for (Sample s : dataset) {
                int px = (int) (s.x() * (w / 2f) + w / 2f);
                int py = (int) (s.y() * (h / 2f) + h / 2f);

                g2.setColor(Color.BLACK);
                g2.fillOval(px - 6, py - 6, 12, 12);
                g2.setColor(s.label() > 0.5f ? new Color(40, 120, 255) : new Color(255, 40, 40));
                g2.fillOval(px - 5, py - 5, 10, 10);
            }
        }
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> new MLPBoundaryDemo().setVisible(true));
    }
}
