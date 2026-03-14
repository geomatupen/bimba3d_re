import { useParams, useNavigate } from "react-router-dom";
import { useEffect, useState } from "react";
import { api } from "../api/client";

interface ProjectStatus {
  project_id: string;
  status: string;
  progress: number;
  error?: string | null;
  name?: string | null;
  created_at?: string | null;
  mode?: string;
  tuning_active?: boolean;
  currentStep?: number;
  maxSteps?: number;
  last_tuning?: {
    step: number;
    action: string;
    reason: string;
  };
  stop_requested?: boolean;
  stage?: string | null;
  message?: string | null;
  can_resume?: boolean;
  last_completed_step?: number | null;
}

interface EvaluationMetrics {
  lpips_score?: number;
  sharpness?: number;
  convergence_speed?: number;
  final_loss?: number;
  gaussian_count?: number;
}

export default function Project() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [projectStatus, setProjectStatus] = useState<ProjectStatus | null>(null);
  const [metrics, setMetrics] = useState<EvaluationMetrics | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [stopSubmitting, setStopSubmitting] = useState(false);

  useEffect(() => {
    if (!id) return;

    const fetchStatus = async () => {
      try {
        const res = await api.get(`/projects/${id}/status`);
        setProjectStatus(res.data);
        console.log("Status updated:", res.data);

        // Fetch metrics if completed
        if (res.data.status === "completed" || res.data.status === "done" || res.data.status === "stopped") {
          try {
            const metricsRes = await api.get(`/projects/${id}/metrics`);
            setMetrics(metricsRes.data);
          } catch (metricsErr) {
            console.log("Metrics not yet available");
          }
        }

        if (res.data.status === "failed") {
          setError(res.data.error || "Processing failed");
        }
      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : "Failed to fetch status";
        setError(errorMessage);
        console.error("Status fetch error:", err);
      }
    };

    // Fetch immediately
    fetchStatus();

    // Then poll every 3 seconds
    const interval = setInterval(fetchStatus, 3000);

    return () => clearInterval(interval);
  }, [id]);

  useEffect(() => {
    if (!id) return;

    let urlHandle: string | null = null;
    let mounted = true;

    const fetchPreview = async () => {
      try {
        const res = await api.get(`/projects/${id}/preview`, {
          responseType: "blob",
          headers: { "Cache-Control": "no-store" },
        });
        if (!mounted) return;
        const objectUrl = URL.createObjectURL(res.data);
        if (urlHandle) {
          URL.revokeObjectURL(urlHandle);
        }
        urlHandle = objectUrl;
        setPreviewUrl(objectUrl + `#t=${Date.now()}`);
      } catch {
        // Preview not ready yet; ignore
      }
    };

    fetchPreview();
    const interval = setInterval(fetchPreview, 5000);

    return () => {
      mounted = false;
      clearInterval(interval);
      if (urlHandle) {
        URL.revokeObjectURL(urlHandle);
      }
    };
  }, [id]);

  const requestStop = async () => {
    if (!id) return;
    setStopSubmitting(true);
    try {
      await api.post(`/projects/${id}/stop`);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : "Failed to request stop";
      setError(errorMessage);
    } finally {
      setStopSubmitting(false);
    }
  };

  const getStatusColor = (status: string): string => {
    switch (status) {
      case "completed":
      case "done":
        return "#28a745";
      case "failed":
        return "#dc3545";
      case "stopping":
      case "stopped":
        return "#ffc107";
      case "processing":
        return "#007bff";
      case "pending":
      default:
        return "#6c757d";
    }
  };

  return (
    <div style={styles.container}>
      <h1>Project Status</h1>
      
      {projectStatus && (
        <div style={styles.statusBox}>
          <p style={styles.projectId}>Project: <strong>{projectStatus.name || "Untitled"}</strong></p>
          <p style={styles.projectId}>Project ID: <code>{id}</code></p>
          <p style={styles.projectId}>Created: {projectStatus.created_at ? new Date(projectStatus.created_at).toLocaleString() : "–"}</p>
          
          {projectStatus.mode && (
            <div style={styles.modeBadge}>
              Mode: {projectStatus.mode === "modified" ? "✨ Optimized (Adaptive Tuning)" : "📊 Baseline"}
            </div>
          )}

          <div style={styles.statusSection}>
            <p style={styles.statusLabel}>Status:</p>
            <p style={{ ...styles.statusValue, color: getStatusColor(projectStatus.status) }}>
              {projectStatus.status.toUpperCase()}
            </p>
            
            {/* Comprehensive status message display */}
            {projectStatus.message && (
              <div style={styles.statusMessageBox}>
                <p style={styles.statusMessage}>{projectStatus.message}</p>
              </div>
            )}
            
            {/* Additional status details */}
            {projectStatus.stage && (
              <p style={styles.stageInfo}>📍 Stage: <strong>{projectStatus.stage}</strong></p>
            )}
            {projectStatus.currentStep && projectStatus.maxSteps && (
              <p style={styles.stepInfo}>
                🔄 Step {projectStatus.currentStep.toLocaleString()} of {projectStatus.maxSteps.toLocaleString()}
              </p>
            )}
          </div>

          <div style={styles.statusSection}>
            <p style={styles.statusLabel}>Progress: {projectStatus.progress}%</p>
            <div style={styles.progressBar}>
              <div
                style={{
                  ...styles.progressFill,
                  width: `${projectStatus.progress}%`,
                }}
              />
            </div>
            {projectStatus.currentStep !== undefined && projectStatus.maxSteps !== undefined && (
              <p style={styles.stepInfo}>
                Step: {projectStatus.currentStep} / {projectStatus.maxSteps}
              </p>
            )}
          </div>

          {/* Show last tuning action if in modified mode */}
          {projectStatus.mode === "modified" && projectStatus.tuning_active && projectStatus.last_tuning && (
            <div style={styles.tuningBox}>
              <h3 style={styles.tuningTitle}>🔧 Latest Adaptive Tuning</h3>
              <p style={styles.tuningAction}>
                <strong>Step {projectStatus.last_tuning.step}:</strong> {projectStatus.last_tuning.action}
              </p>
              <p style={styles.tuningReason}>{projectStatus.last_tuning.reason}</p>
            </div>
          )}

          {error && (
            <div style={styles.errorBox}>
              <p>{error}</p>
            </div>
          )}

          {projectStatus.status === "processing" && (
            <p style={styles.message}>
              {projectStatus.mode === "modified" 
                ? "Processing with adaptive tuning... Monitoring convergence every 10 steps."
                : "Processing... This may take a few minutes."}
            </p>
          )}

          {(projectStatus.status === "processing" || projectStatus.status === "stopping") && (
            <div style={styles.controlsRow}>
              <button
                onClick={requestStop}
                disabled={stopSubmitting || projectStatus.stop_requested}
                style={styles.stopButton}
              >
                {stopSubmitting ? "Requesting stop..." : "⏹ Manual stop"}
              </button>
            </div>
          )}

          {previewUrl && (
            <div style={styles.previewBox}>
              <h3 style={styles.previewTitle}>Live Preview</h3>
              <img src={previewUrl} alt="Preview" style={styles.previewImg} />
              <p style={styles.previewHint}>Auto-refreshes every 5s when training exports previews.</p>
            </div>
          )}

          {(projectStatus.status === "completed" || projectStatus.status === "done" || projectStatus.status === "stopped") && (
            <>
              <div style={styles.successBox}>
                <p>{projectStatus.status === "stopped" ? "⏸ Training stopped early" : "✅ Processing complete!"}</p>
                {projectStatus.can_resume && projectStatus.last_completed_step && (
                  <p style={{ marginTop: "0.5rem", fontSize: "0.9rem", color: "#555" }}>
                    💾 Checkpoint available at step {projectStatus.last_completed_step}. You can resume training by uploading to the same project with "Resume from checkpoint" enabled.
                  </p>
                )}
              </div>

              {/* Show evaluation metrics */}
              {metrics && (
                <div style={styles.metricsCard}>
                  <h3 style={styles.metricsTitle}>📊 Evaluation Metrics</h3>
                  <div style={styles.metricsGrid}>
                    {metrics.lpips_score !== undefined && (
                      <div style={styles.metricItem}>
                        <span style={styles.metricLabel}>LPIPS Score</span>
                        <span style={styles.metricValue}>{metrics.lpips_score.toFixed(4)}</span>
                        <span style={styles.metricHint}>Perceptual quality (lower is better)</span>
                      </div>
                    )}
                    {metrics.sharpness !== undefined && (
                      <div style={styles.metricItem}>
                        <span style={styles.metricLabel}>Image Sharpness</span>
                        <span style={styles.metricValue}>{metrics.sharpness.toFixed(2)}</span>
                        <span style={styles.metricHint}>Laplacian variance (higher is better)</span>
                      </div>
                    )}
                    {metrics.convergence_speed !== undefined && (
                      <div style={styles.metricItem}>
                        <span style={styles.metricLabel}>Convergence Speed</span>
                        <span style={styles.metricValue}>{metrics.convergence_speed.toFixed(2)}</span>
                        <span style={styles.metricHint}>Steps/sec</span>
                      </div>
                    )}
                    {metrics.final_loss !== undefined && (
                      <div style={styles.metricItem}>
                        <span style={styles.metricLabel}>Final Loss</span>
                        <span style={styles.metricValue}>{metrics.final_loss.toFixed(6)}</span>
                        <span style={styles.metricHint}>Training loss at completion</span>
                      </div>
                    )}
                    {metrics.gaussian_count !== undefined && (
                      <div style={styles.metricItem}>
                        <span style={styles.metricLabel}>Gaussian Count</span>
                        <span style={styles.metricValue}>{metrics.gaussian_count.toLocaleString()}</span>
                        <span style={styles.metricHint}>Number of 3D Gaussians</span>
                      </div>
                    )}
                  </div>
                </div>
              )}

              <button onClick={() => navigate(`/viewer/${id}`)} style={styles.viewButton}>
                🎨 View 3D Render
              </button>
            </>
          )}
        </div>
      )}

      <button onClick={() => navigate("/")} style={styles.backButton}>
        ← Back to Home
      </button>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    padding: "2rem",
    maxWidth: "800px",
    margin: "0 auto",
  },
  statusBox: {
    border: "2px solid #007bff",
    borderRadius: "8px",
    padding: "2rem",
    backgroundColor: "#f0f8ff",
    marginBottom: "1.5rem",
  },
  projectId: {
    marginBottom: "0.8rem",
    fontSize: "0.9rem",
    color: "#666",
  },
  modeBadge: {
    display: "inline-block",
    padding: "0.5rem 1rem",
    borderRadius: "6px",
    backgroundColor: "#e7f3ff",
    fontSize: "0.95rem",
    marginBottom: "1rem",
    border: "2px solid #007bff",
    fontWeight: "500",
  },
  statusSection: {
    marginBottom: "1.5rem",
  },
  statusLabel: {
    marginBottom: "0.5rem",
    fontWeight: "bold",
    color: "#333",
    fontSize: "0.95rem",
  },
  statusValue: {
    fontSize: "1.5rem",
    fontWeight: "bold",
    margin: 0,
  },
  statusHint: {
    color: "#b26a00",
    fontSize: "0.9rem",
    marginTop: "0.2rem",
  },
  progressBar: {
    width: "100%",
    height: "28px",
    backgroundColor: "#e0e0e0",
    borderRadius: "14px",
    overflow: "hidden",
  },
  progressFill: {
    height: "100%",
    backgroundColor: "#007bff",
    transition: "width 0.3s ease",
  },
  stepInfo: {
    marginTop: "0.5rem",
    fontSize: "0.9rem",
    color: "#666",
  },
  statusMessageBox: {
    marginTop: "1rem",
    padding: "1rem 1.2rem",
    backgroundColor: "#e7f3ff",
    border: "2px solid #007bff",
    borderRadius: "8px",
    boxShadow: "0 2px 4px rgba(0,0,0,0.05)",
  },
  statusMessage: {
    margin: 0,
    fontSize: "1rem",
    color: "#004085",
    lineHeight: "1.5",
    fontWeight: "500",
  },
  stageInfo: {
    marginTop: "0.8rem",
    fontSize: "0.95rem",
    color: "#495057",
  },
  tuningBox: {
    border: "2px solid #28a745",
    borderRadius: "8px",
    padding: "1.2rem",
    marginTop: "1.5rem",
    marginBottom: "1.5rem",
    backgroundColor: "#f0fff4",
  },
  tuningTitle: {
    margin: "0 0 0.8rem 0",
    fontSize: "1.1rem",
    color: "#28a745",
  },
  tuningAction: {
    margin: "0.5rem 0",
    fontSize: "1rem",
  },
  tuningReason: {
    fontSize: "0.9rem",
    color: "#666",
    fontStyle: "italic",
    marginTop: "0.5rem",
  },
  errorBox: {
    backgroundColor: "#fee",
    borderLeft: "4px solid #f00",
    padding: "1rem",
    marginBottom: "1rem",
    borderRadius: "4px",
    color: "#c33",
  },
  message: {
    marginTop: "1rem",
    padding: "1rem",
    backgroundColor: "#fff3cd",
    borderRadius: "6px",
    color: "#856404",
    fontStyle: "italic",
  },
  successBox: {
    backgroundColor: "#d4edda",
    border: "2px solid #c3e6cb",
    borderRadius: "8px",
    padding: "1.2rem",
    marginTop: "1.5rem",
    marginBottom: "1.5rem",
    color: "#155724",
    fontWeight: "bold",
    textAlign: "center",
    fontSize: "1.1rem",
  },
  metricsCard: {
    border: "1px solid #ddd",
    borderRadius: "8px",
    padding: "1.5rem",
    marginTop: "1.5rem",
    marginBottom: "1.5rem",
    backgroundColor: "#ffffff",
  },
  metricsTitle: {
    margin: "0 0 1rem 0",
    fontSize: "1.2rem",
    color: "#333",
  },
  metricsGrid: {
    display: "grid",
    gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))",
    gap: "1rem",
  },
  metricItem: {
    display: "flex",
    flexDirection: "column",
    gap: "0.4rem",
    padding: "1rem",
    backgroundColor: "#f8f9fa",
    borderRadius: "6px",
    border: "1px solid #dee2e6",
  },
  metricLabel: {
    fontSize: "0.85rem",
    color: "#666",
    fontWeight: "600",
    textTransform: "uppercase",
    letterSpacing: "0.5px",
  },
  metricValue: {
    fontSize: "1.4rem",
    fontWeight: "bold",
    color: "#007bff",
  },
  metricHint: {
    fontSize: "0.75rem",
    color: "#888",
    fontStyle: "italic",
  },
  controlsRow: {
    marginTop: "1rem",
    display: "flex",
    gap: "0.75rem",
  },
  stopButton: {
    padding: "0.65rem 1.2rem",
    backgroundColor: "#ffc107",
    border: "none",
    borderRadius: "6px",
    cursor: "pointer",
    fontWeight: "bold",
    color: "#1a1a1a",
  },
  previewBox: {
    marginTop: "1rem",
    padding: "1rem",
    border: "1px solid #e0e0e0",
    borderRadius: "8px",
    backgroundColor: "#fff",
  },
  previewTitle: {
    margin: "0 0 0.5rem 0",
  },
  previewImg: {
    width: "100%",
    maxHeight: "320px",
    objectFit: "contain",
    borderRadius: "6px",
    backgroundColor: "#000",
  },
  previewHint: {
    marginTop: "0.4rem",
    fontSize: "0.85rem",
    color: "#555",
  },
  viewButton: {
    width: "100%",
    padding: "1rem 2rem",
    fontSize: "1.1rem",
    backgroundColor: "#28a745",
    color: "white",
    border: "none",
    borderRadius: "8px",
    cursor: "pointer",
    fontWeight: "bold",
    transition: "background-color 0.2s",
  },
  backButton: {
    padding: "0.6rem 1.2rem",
    backgroundColor: "#6c757d",
    color: "white",
    border: "none",
    borderRadius: "6px",
    cursor: "pointer",
  },
};
