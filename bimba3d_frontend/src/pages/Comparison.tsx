import { useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { api } from "../api/client";

interface EvaluationMetrics {
  lpips_score?: number;
  sharpness?: number;
  convergence_speed?: number;
  final_loss?: number;
  gaussian_count?: number;
}

interface ComparisonData {
  status: string;
  baseline?: {
    status: string;
    progress: number;
    metrics?: EvaluationMetrics;
  };
  optimized?: {
    status: string;
    progress: number;
    metrics?: EvaluationMetrics;
  };
  baseline_project_id?: string;
  optimized_project_id?: string;
}

export default function Comparison() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [data, setData] = useState<ComparisonData | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let interval: ReturnType<typeof setInterval>;

    const pollStatus = async () => {
      try {
        const res = await api.get(`/projects/comparison/${id}/status`);
        setData(res.data);

        // Stop polling if both complete
        if (res.data.status === "completed") {
          clearInterval(interval);
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to fetch comparison status");
        clearInterval(interval);
      }
    };

    pollStatus();
    interval = setInterval(pollStatus, 3000);

    return () => clearInterval(interval);
  }, [id]);

  if (error) {
    return (
      <div style={styles.container}>
        <div style={styles.errorBox}>
          <h3>❌ Error</h3>
          <p>{error}</p>
          <button onClick={() => navigate("/")} style={styles.button}>
            Back to Home
          </button>
        </div>
      </div>
    );
  }

  if (!data) {
    return (
      <div style={styles.container}>
        <h2>Loading comparison...</h2>
      </div>
    );
  }

  const isComplete = data.status === "completed";

  return (
    <div style={styles.container}>
      <h1>🔬 Baseline vs Optimized Comparison</h1>
      <p style={styles.subtitle}>Comparison ID: {id}</p>

      {/* Overall Status */}
      <div style={styles.statusCard}>
        <h3>Status: {data.status.toUpperCase()}</h3>
        {!isComplete && (
          <div style={styles.progressSection}>
            {data.baseline?.status === "running" && (
              <p>🔵 Running baseline training... ({data.baseline.progress}%)</p>
            )}
            {data.baseline?.status === "completed" && data.optimized?.status === "running" && (
              <p>🟢 Baseline complete! Now running optimized... ({data.optimized.progress}%)</p>
            )}
          </div>
        )}
      </div>

      {/* Side-by-side Progress */}
      <div style={styles.grid}>
        {/* Baseline */}
        <div style={styles.card}>
          <h2>📊 Baseline</h2>
          <div style={styles.statusBadge}>
            Status: {data.baseline?.status || "pending"}
          </div>
          {data.baseline && (
            <>
              <div style={styles.progressBar}>
                <div
                  style={{
                    ...styles.progressFill,
                    width: `${data.baseline.progress}%`,
                    backgroundColor: "#6c757d",
                  }}
                />
              </div>
              <p>{data.baseline.progress}%</p>
            </>
          )}
        </div>

        {/* Optimized */}
        <div style={styles.card}>
          <h2>✨ Optimized (Adaptive Tuning)</h2>
          <div style={styles.statusBadge}>
            Status: {data.optimized?.status || "pending"}
          </div>
          {data.optimized && (
            <>
              <div style={styles.progressBar}>
                <div
                  style={{
                    ...styles.progressFill,
                    width: `${data.optimized.progress}%`,
                    backgroundColor: "#28a745",
                  }}
                />
              </div>
              <p>{data.optimized.progress}%</p>
            </>
          )}
        </div>
      </div>

      {/* Metrics Comparison (when both complete) */}
      {isComplete && data.baseline?.metrics && data.optimized?.metrics && (
        <div style={styles.metricsSection}>
          <h2>📈 Evaluation Metrics Comparison</h2>
          <table style={styles.table}>
            <thead>
              <tr>
                <th style={styles.th}>Metric</th>
                <th style={styles.th}>Baseline</th>
                <th style={styles.th}>Optimized</th>
                <th style={styles.th}>Improvement</th>
              </tr>
            </thead>
            <tbody>
              <MetricRow
                name="LPIPS (Perceptual Quality)"
                baseline={data.baseline.metrics.lpips_score}
                optimized={data.optimized.metrics.lpips_score}
                lowerIsBetter
                unit=""
              />
              <MetricRow
                name="Image Sharpness"
                baseline={data.baseline.metrics.sharpness}
                optimized={data.optimized.metrics.sharpness}
                lowerIsBetter={false}
                unit=""
              />
              <MetricRow
                name="Convergence Speed"
                baseline={data.baseline.metrics.convergence_speed}
                optimized={data.optimized.metrics.convergence_speed}
                lowerIsBetter={false}
                unit="steps/sec"
              />
              <MetricRow
                name="Final Loss"
                baseline={data.baseline.metrics.final_loss}
                optimized={data.optimized.metrics.final_loss}
                lowerIsBetter
                unit=""
              />
              <MetricRow
                name="Gaussian Count"
                baseline={data.baseline.metrics.gaussian_count}
                optimized={data.optimized.metrics.gaussian_count}
                lowerIsBetter={false}
                unit=""
              />
            </tbody>
          </table>

          <div style={styles.actions}>
            <button
              onClick={() => navigate(`/viewer/${data.baseline_project_id}`)}
              style={{ ...styles.button, backgroundColor: "#6c757d" }}
            >
              View Baseline Render
            </button>
            <button
              onClick={() => navigate(`/viewer/${data.optimized_project_id}`)}
              style={{ ...styles.button, backgroundColor: "#28a745" }}
            >
              View Optimized Render
            </button>
          </div>
        </div>
      )}

      <button onClick={() => navigate("/")} style={styles.backButton}>
        ← Back to Home
      </button>
    </div>
  );
}

// Helper component for metric rows
interface MetricRowProps {
  name: string;
  baseline?: number;
  optimized?: number;
  lowerIsBetter: boolean;
  unit: string;
}

function MetricRow({ name, baseline, optimized, lowerIsBetter, unit }: MetricRowProps) {
  if (baseline === undefined || optimized === undefined) {
    return (
      <tr>
        <td style={styles.td}>{name}</td>
        <td style={styles.td}>N/A</td>
        <td style={styles.td}>N/A</td>
        <td style={styles.td}>-</td>
      </tr>
    );
  }

  const diff = optimized - baseline;
  const percentChange = baseline !== 0 ? (diff / baseline) * 100 : 0;
  const isImprovement = lowerIsBetter ? diff < 0 : diff > 0;

  return (
    <tr style={isImprovement ? styles.improvedRow : {}}>
      <td style={styles.td}>{name}</td>
      <td style={styles.td}>
        {baseline.toFixed(4)} {unit}
      </td>
      <td style={styles.td}>
        {optimized.toFixed(4)} {unit}
      </td>
      <td style={{ ...styles.td, color: isImprovement ? "#28a745" : "#dc3545" }}>
        {isImprovement ? "✅" : "⚠️"} {percentChange > 0 ? "+" : ""}
        {percentChange.toFixed(1)}%
      </td>
    </tr>
  );
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    padding: "2rem",
    maxWidth: "1200px",
    margin: "0 auto",
  },
  subtitle: {
    color: "#666",
    fontSize: "0.9rem",
    marginTop: "-0.5rem",
    marginBottom: "1.5rem",
  },
  statusCard: {
    border: "2px solid #007bff",
    borderRadius: "8px",
    padding: "1.5rem",
    marginBottom: "2rem",
    backgroundColor: "#f0f8ff",
  },
  progressSection: {
    marginTop: "1rem",
  },
  grid: {
    display: "grid",
    gridTemplateColumns: "1fr 1fr",
    gap: "1.5rem",
    marginBottom: "2rem",
  },
  card: {
    border: "1px solid #ddd",
    borderRadius: "8px",
    padding: "1.5rem",
    backgroundColor: "#fafafa",
  },
  statusBadge: {
    display: "inline-block",
    padding: "0.3rem 0.8rem",
    borderRadius: "4px",
    backgroundColor: "#e7f3ff",
    fontSize: "0.9rem",
    marginTop: "0.5rem",
    marginBottom: "1rem",
  },
  progressBar: {
    width: "100%",
    height: "24px",
    backgroundColor: "#e0e0e0",
    borderRadius: "12px",
    overflow: "hidden",
    marginTop: "1rem",
  },
  progressFill: {
    height: "100%",
    transition: "width 0.3s ease",
  },
  metricsSection: {
    marginTop: "2rem",
    border: "2px solid #28a745",
    borderRadius: "8px",
    padding: "1.5rem",
    backgroundColor: "#f0fff4",
  },
  table: {
    width: "100%",
    borderCollapse: "collapse",
    marginTop: "1rem",
    backgroundColor: "white",
  },
  th: {
    padding: "0.75rem",
    textAlign: "left",
    borderBottom: "2px solid #ddd",
    backgroundColor: "#f8f9fa",
    fontWeight: "bold",
  },
  td: {
    padding: "0.75rem",
    borderBottom: "1px solid #eee",
  },
  improvedRow: {
    backgroundColor: "#f0fff4",
  },
  actions: {
    display: "flex",
    gap: "1rem",
    marginTop: "1.5rem",
  },
  button: {
    flex: 1,
    padding: "0.75rem 1.5rem",
    fontSize: "1rem",
    color: "white",
    border: "none",
    borderRadius: "6px",
    cursor: "pointer",
    fontWeight: "bold",
  },
  backButton: {
    marginTop: "2rem",
    padding: "0.6rem 1.2rem",
    backgroundColor: "#6c757d",
    color: "white",
    border: "none",
    borderRadius: "6px",
    cursor: "pointer",
  },
  errorBox: {
    backgroundColor: "#fee",
    borderLeft: "4px solid #f00",
    padding: "1.5rem",
    borderRadius: "4px",
    color: "#c33",
  },
};
