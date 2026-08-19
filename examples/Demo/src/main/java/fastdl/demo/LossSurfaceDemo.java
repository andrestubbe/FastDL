package fastdl.demo;

import javax.swing.*;
import java.awt.*;
import java.util.ArrayList;
import java.util.List;

/**
 * FastDL Loss Surface & Minima Valley Visualizer Demo.
 *
 * <p>Demonstrates finding local and global minima on non-convex landscapes
 * using gradient descent with momentum and learning rate controls.
 */
public class LossSurfaceDemo extends JFrame {

    // 2D Non-convex multi-valley function: f(x, y) = sin(x)*cos(y) + (x^2 + y^2)/20 + sin(2x)/2
    private static double lossFunction(double x, double y) {
        return Math.sin(x) * Math.cos(y) + (x * x + y * y) / 20.0 + Math.sin(2.0 * x) * 0.5;
    }

    private static double[] gradient(double x, double y) {
        double eps = 1e-5;
        double df_dx = (lossFunction(x + eps, y) - lossFunction(x - eps, y)) / (2 * eps);
        double df_dy = (lossFunction(x, y + eps) - lossFunction(x, y - eps)) / (2 * eps);
        return new double[]{df_dx, df_dy};
    }

    private double posX = -3.5;
    private double posY = 3.0;
    private double velX = 0.0;
    private double velY = 0.0;

    private double lr = 0.05;
    private double momentum = 0.85;

    private final List<Point.Double> trajectory = new ArrayList<>();
    private final SurfacePanel surfacePanel = new SurfacePanel();
    private final Timer loopTimer;

    public LossSurfaceDemo() {
        super("FastDL — Non-Convex Loss Surface & Minima Valley Visualizer");
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setSize(950, 700);
        setLocationRelativeTo(null);

        setLayout(new BorderLayout());
        add(surfacePanel, BorderLayout.CENTER);

        // Control Panel
        JPanel controls = new JPanel(new FlowLayout(FlowLayout.CENTER, 15, 8));
        JButton btnReset = new JButton("Neuer Zufalls-Start");
        JButton btnSGD = new JButton("Nur SGD (Momentum = 0)");
        JButton btnMomentum = new JButton("SGD + Momentum (0.85)");
        JLabel lblStatus = new JLabel(String.format("Loss: %.4f | Pos: (%.2f, %.2f)", lossFunction(posX, posY), posX, posY));

        controls.add(btnReset);
        controls.add(btnSGD);
        controls.add(btnMomentum);
        controls.add(lblStatus);
        add(controls, BorderLayout.SOUTH);

        trajectory.add(new Point.Double(posX, posY));

        btnReset.addActionListener(e -> {
            posX = (Math.random() - 0.5) * 8.0;
            posY = (Math.random() - 0.5) * 8.0;
            velX = 0;
            velY = 0;
            trajectory.clear();
            trajectory.add(new Point.Double(posX, posY));
            surfacePanel.repaint();
        });

        btnSGD.addActionListener(e -> momentum = 0.0);
        btnMomentum.addActionListener(e -> momentum = 0.85);

        loopTimer = new Timer(30, e -> {
            double[] grad = gradient(posX, posY);

            velX = momentum * velX + lr * grad[0];
            velY = momentum * velY + lr * grad[1];

            posX -= velX;
            posY -= velY;

            // Bounds clamp
            posX = Math.max(-4.8, Math.min(4.8, posX));
            posY = Math.max(-4.8, Math.min(4.8, posY));

            trajectory.add(new Point.Double(posX, posY));
            if (trajectory.size() > 500) trajectory.remove(0);

            lblStatus.setText(String.format("Loss: %.4f | Pos: (%.2f, %.2f) | Momentum: %.2f", lossFunction(posX, posY), posX, posY, momentum));
            surfacePanel.repaint();
        });

        loopTimer.start();
    }

    class SurfacePanel extends JPanel {
        @Override
        protected void paintComponent(Graphics g) {
            super.paintComponent(g);
            int w = getWidth();
            int h = getHeight();

            double range = 10.0;
            int step = 4;

            // Render Heatmap of Loss Surface
            for (int py = 0; py < h; py += step) {
                double y = range * ((double) py / h - 0.5);
                for (int px = 0; px < w; px += step) {
                    double x = range * ((double) px / w - 0.5);
                    double l = lossFunction(x, y);

                    // Map loss to color gradient (blue = deep valley/minima, red/yellow = mountain/high loss)
                    float norm = (float) Math.min(1.0, Math.max(0.0, (l + 1.8) / 4.0));
                    g.setColor(new Color(norm, 0.2f, 1.0f - norm));
                    g.fillRect(px, py, step, step);
                }
            }

            // Draw Trajectory
            g.setColor(Color.WHITE);
            Graphics2D g2 = (Graphics2D) g;
            g2.setStroke(new BasicStroke(2.5f));
            for (int i = 1; i < trajectory.size(); i++) {
                Point.Double p1 = trajectory.get(i - 1);
                Point.Double p2 = trajectory.get(i);

                int x1 = (int) ((p1.x / range + 0.5) * w);
                int y1 = (int) ((p1.y / range + 0.5) * h);
                int x2 = (int) ((p2.x / range + 0.5) * w);
                int y2 = (int) ((p2.y / range + 0.5) * h);

                g2.drawLine(x1, y1, x2, y2);
            }

            // Draw Ball (Current Optimizer State)
            int bx = (int) ((posX / range + 0.5) * w);
            int by = (int) ((posY / range + 0.5) * h);
            g2.setColor(Color.YELLOW);
            g2.fillOval(bx - 8, by - 8, 16, 16);
            g2.setColor(Color.BLACK);
            g2.drawOval(bx - 8, by - 8, 16, 16);
        }
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> new LossSurfaceDemo().setVisible(true));
    }
}
