import { useParams, useNavigate } from "react-router-dom";
import { useState, useEffect } from "react";
import { api } from "../api/client";
import ThreeDViewer from "../components/ThreeDViewer";

export default function Viewer() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [files, setFiles] = useState<Record<string, any> | null>(null);
  const [metadata, setMetadata] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!id) return;

    const fetchData = async () => {
      try {
        // Fetch files list
        const filesRes = await api.get(`/projects/${id}/files`);
        setFiles(filesRes.data.files);
        console.log("Files:", filesRes.data.files);

        // Fetch metadata
        try {
          const metadataRes = await api.get(`/projects/${id}/metadata`);
          setMetadata(metadataRes.data);
          console.log("Metadata:", metadataRes.data);
        } catch (err) {
          console.warn("Could not load metadata:", err);
        }

        setLoading(false);
      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : "Failed to load project files";
        setError(errorMessage);
        console.error("Error loading files:", err);
        setLoading(false);
      }
    };

    fetchData();
  }, [id]);

  const splatsUrl = id ? `${api.defaults.baseURL}/projects/${id}/download/splats` : undefined;

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + " " + sizes[i];
  };
  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <h1>3D Gaussian Splat Viewer</h1>
        <button onClick={() => navigate("/")} style={styles.backButton}>
          ← Back to Upload
        </button>
      </div>

      {loading && (
        <div style={styles.loadingBox}>
          <p>Loading project data...</p>
        </div>
      )}

      {error && (
        <div style={styles.errorBox}>
          <p>{error}</p>
        </div>
      )}

      {!loading && !error && (
        <>
          <div style={styles.projectInfo}>
            <p>
              <strong>Project ID:</strong> <code style={styles.code}>{id}</code>
            </p>
          </div>

          <div style={styles.viewerSection}>
            <h2>3D Visualization</h2>
            <ThreeDViewer
              splatsUrl={splatsUrl}
              onLoaded={() => console.log("Viewer loaded")}
              onError={(error) => console.error("Viewer error:", error)}
            />
          </div>

          {metadata && (
            <div style={styles.metadataSection}>
              <h2>Model Information</h2>
              <div style={styles.metadataBox}>
                <p>
                  <strong>Type:</strong> {metadata.type}
                </p>
                <p>
                  <strong>Points:</strong> {metadata.num_points || "N/A"}
                </p>
                <p>
                  <strong>Version:</strong> {metadata.version}
                </p>
                {metadata.training_config && (
                  <div style={styles.configBox}>
                    <p>
                      <strong>Training Parameters:</strong>
                    </p>
                    <div style={styles.paramGrid}>
                      {Object.entries(metadata.training_config).map(([k, v]) => (
                        <div key={k} style={styles.paramItem}>
                          <span style={styles.paramKey}>{k}</span>
                          <span style={styles.paramVal}>{String(v) }</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

          {files && (
            <div style={styles.filesSection}>
              <h2>Project Files</h2>

              {files.splats && (
                <div style={styles.fileGroup}>
                  <h3>Splats</h3>
                  <div style={styles.fileBox}>
                    <p>
                      <strong>splats.bin</strong>
                    </p>
                    <p>Size: {formatFileSize(files.splats.size)}</p>
                    <a
                      href={`${api.defaults.baseURL}/projects/${id}/download/splats`}
                      style={styles.downloadButton}
                      download
                    >
                      ⬇ Download
                    </a>
                  </div>
                </div>
              )}

              {files.metadata && (
                <div style={styles.fileGroup}>
                  <h3>Metadata</h3>
                  <div style={styles.fileBox}>
                    <p>
                      <strong>metadata.json</strong>
                    </p>
                    <p>Size: {formatFileSize(files.metadata.size)}</p>
                    <a
                      href={`${api.defaults.baseURL}/projects/${id}/metadata`}
                      style={styles.downloadButton}
                      download
                    >
                      ⬇ Download
                    </a>
                  </div>
                </div>
              )}

              {files.checkpoints && files.checkpoints.length > 0 && (
                <div style={styles.fileGroup}>
                  <h3>Training Checkpoints ({files.checkpoints.length})</h3>
                  {files.checkpoints.map((ckpt: any) => (
                    <div key={ckpt.name} style={styles.fileBox}>
                      <p>
                        <strong>{ckpt.name}</strong>
                      </p>
                      <p>Size: {formatFileSize(ckpt.size)}</p>
                    </div>
                  ))}
                </div>
              )}

              {files.images && files.images.length > 0 && (
                <div style={styles.fileGroup}>
                  <h3>Input Images ({files.images.length})</h3>
                  <div style={styles.imageGrid}>
                    {files.images.slice(0, 6).map((img: any) => (
                      <div key={img.name} style={styles.imageThumb}>
                        <img
                          src={`${api.defaults.baseURL}${img.url}`}
                          alt={img.name}
                          style={styles.thumbImg}
                          onError={(e) => {
                            (e.target as HTMLImageElement).src =
                              "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='100' height='100'%3E%3Crect fill='%23ccc' width='100' height='100'/%3E%3C/svg%3E";
                          }}
                        />
                        <p style={styles.thumbName}>{img.name}</p>
                      </div>
                    ))}
                  </div>
                  {files.images.length > 6 && (
                    <p style={styles.moreImages}>
                      + {files.images.length - 6} more images
                    </p>
                  )}
                </div>
              )}
            </div>
          )}

          <div style={styles.controls}>
            <p>
              <strong>Controls:</strong>
            </p>
            <ul>
              <li>🖱 Drag to rotate</li>
              <li>🖱 Scroll to zoom</li>
              <li>Right-click for context menu</li>
            </ul>
          </div>
        </>
      )}
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    padding: "2rem",
    minHeight: "100vh",
    backgroundColor: "#f5f5f5",
  },
  header: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: "2rem",
  },
  backButton: {
    padding: "0.5rem 1rem",
    backgroundColor: "#6c757d",
    color: "white",
    border: "none",
    borderRadius: "4px",
    cursor: "pointer",
    fontSize: "0.9rem",
  },
  projectInfo: {
    marginBottom: "1.5rem",
    padding: "1rem",
    backgroundColor: "white",
    borderRadius: "4px",
  },
  code: {
    backgroundColor: "#f0f0f0",
    padding: "0.25rem 0.5rem",
    borderRadius: "3px",
    fontFamily: "monospace",
  },
  loadingBox: {
    padding: "2rem",
    backgroundColor: "white",
    borderRadius: "8px",
    textAlign: "center",
  },
  errorBox: {
    backgroundColor: "#fee",
    borderLeft: "4px solid #f00",
    padding: "1rem",
    marginBottom: "1rem",
    borderRadius: "4px",
    color: "#c33",
  },
  viewerSection: {
    marginBottom: "2rem",
  },
  metadataSection: {
    marginBottom: "2rem",
  },
  metadataBox: {
    backgroundColor: "white",
    padding: "1.5rem",
    borderRadius: "8px",
    boxShadow: "0 2px 4px rgba(0,0,0,0.1)",
  },
  configBox: {
    backgroundColor: "#f9f9f9",
    padding: "1rem",
    borderRadius: "4px",
    marginTop: "1rem",
  },
  paramGrid: {
    display: "grid",
    gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))",
    gap: "0.5rem",
    marginTop: "0.5rem",
  },
  paramItem: {
    display: "flex",
    flexDirection: "column",
    padding: "0.5rem",
    backgroundColor: "white",
    borderRadius: "4px",
    border: "1px solid #eee",
  },
  paramKey: {
    fontSize: "0.85rem",
    color: "#666",
  },
  paramVal: {
    fontWeight: 600,
  },
  filesSection: {
    marginBottom: "2rem",
  },
  fileGroup: {
    marginBottom: "1.5rem",
  },
  fileBox: {
    backgroundColor: "white",
    padding: "1rem",
    borderRadius: "4px",
    marginBottom: "0.5rem",
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
  },
  downloadButton: {
    padding: "0.5rem 1rem",
    backgroundColor: "#007bff",
    color: "white",
    textDecoration: "none",
    borderRadius: "4px",
    fontSize: "0.9rem",
  },
  imageGrid: {
    display: "grid",
    gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))",
    gap: "1rem",
    marginTop: "1rem",
  },
  imageThumb: {
    backgroundColor: "white",
    padding: "0.5rem",
    borderRadius: "4px",
    textAlign: "center",
  },
  thumbImg: {
    width: "100%",
    height: "150px",
    objectFit: "cover",
    borderRadius: "3px",
  },
  thumbName: {
    fontSize: "0.8rem",
    marginTop: "0.5rem",
    overflow: "hidden",
    textOverflow: "ellipsis",
    whiteSpace: "nowrap",
  },
  moreImages: {
    marginTop: "1rem",
    color: "#666",
    fontStyle: "italic",
  },
  controls: {
    backgroundColor: "white",
    padding: "1rem",
    borderRadius: "4px",
    marginTop: "2rem",
  },
};
