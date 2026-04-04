import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "../api/client";
import ConfirmModal from "../components/ConfirmModal";

type ProjectListItem = {
  project_id: string;
  name?: string | null;
  status: string;
  progress: number;
  created_at?: string | null;
  has_outputs: boolean;
  session_count: number;
};

export default function Projects() {
  const navigate = useNavigate();
  const [projects, setProjects] = useState<ProjectListItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [editingProject, setEditingProject] = useState<ProjectListItem | null>(null);
  const [editName, setEditName] = useState("");
  const [editSaving, setEditSaving] = useState(false);
  const [editError, setEditError] = useState<string | null>(null);
  const [pendingDeleteProject, setPendingDeleteProject] = useState<ProjectListItem | null>(null);
  const [deleteBusy, setDeleteBusy] = useState(false);

  const loadProjects = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await api.get<ProjectListItem[]>("/projects");
      setProjects(res.data);
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Failed to load projects";
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadProjects();
  }, []);

  const sortedProjects = useMemo(() => {
    return [...projects].sort((a, b) => {
      const aDate = a.created_at ? Date.parse(a.created_at) : 0;
      const bDate = b.created_at ? Date.parse(b.created_at) : 0;
      return bDate - aDate;
    });
  }, [projects]);

  const handleDelete = async () => {
    if (!pendingDeleteProject) return;
    const id = pendingDeleteProject.project_id;
    setDeleteBusy(true);
    try {
      await api.delete(`/projects/${id}`);
      setProjects(prev => prev.filter(p => p.project_id !== id));
      setPendingDeleteProject(null);
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Delete failed";
      setError(msg);
    } finally {
      setDeleteBusy(false);
    }
  };

  const statusColor = (status: string) => {
    switch (status) {
      case "done":
        return "#28a745";
      case "processing":
        return "#007bff";
      case "failed":
        return "#dc3545";
      default:
        return "#6c757d";
    }
  };

  const formatDate = (iso?: string | null) => {
    if (!iso) return "–";
    const d = new Date(iso);
    return isNaN(d.getTime()) ? "–" : d.toLocaleString();
  };

  const openEdit = (project: ProjectListItem) => {
    setEditingProject(project);
    setEditName(project.name || "");
    setEditError(null);
  };

  const closeEdit = () => {
    setEditingProject(null);
    setEditName("");
    setEditError(null);
    setEditSaving(false);
  };

  const saveEdit = async () => {
    if (!editingProject) return;
    const trimmed = editName.trim();
    if (!trimmed) {
      setEditError("Name cannot be empty");
      return;
    }
    setEditSaving(true);
    setEditError(null);
    try {
      await api.patch(`/projects/${editingProject.project_id}`, { name: trimmed });
      setProjects(prev => prev.map(p => p.project_id === editingProject.project_id ? { ...p, name: trimmed } : p));
      closeEdit();
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Failed to update project";
      setEditError(msg);
    } finally {
      setEditSaving(false);
    }
  };

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <div>
          <h1 style={{ margin: 0 }}>Projects</h1>
          <p style={{ margin: "0.25rem 0", color: "#666" }}>Manage, view, or delete processed runs.</p>
        </div>
        <div style={styles.headerActions}>
          <button style={styles.button} onClick={() => navigate("/upload")}>+ New Upload</button>
          <button style={{ ...styles.button, backgroundColor: "#6c757d" }} onClick={loadProjects}>↻ Refresh</button>
        </div>
      </div>

      {loading && <div style={styles.card}>Loading projects…</div>}
      {error && <div style={{ ...styles.card, ...styles.errorBox }}>{error}</div>}

      {!loading && !error && (
        <div style={styles.list}>
          {sortedProjects.length === 0 && (
            <div style={styles.card}>No projects yet. Start by uploading images.</div>
          )}
          {sortedProjects.map(project => (
            <div key={project.project_id} style={styles.card}>
              <div style={styles.rowTop}>
                <div>
                  <div style={styles.name}>{project.name || "Untitled Project"}</div>
                  <div style={styles.subtle}>ID: {project.project_id}</div>
                </div>
                <div style={styles.actions}>
                  <button
                    style={{ ...styles.button, backgroundColor: "#0d6efd" }}
                    onClick={() => navigate(`/project/${project.project_id}`)}
                  >Status</button>
                  <button
                    style={{ ...styles.button, backgroundColor: "#6f42c1" }}
                    onClick={() => openEdit(project)}
                  >Edit</button>
                  <button
                    style={{ ...styles.button, backgroundColor: project.has_outputs ? "#198754" : "#adb5bd" }}
                    onClick={() => navigate(`/viewer/${project.project_id}`)}
                    disabled={!project.has_outputs}
                    title={project.has_outputs ? "Open viewer" : "No outputs yet"}
                  >Viewer</button>
                  <button
                    style={{ ...styles.button, backgroundColor: "#dc3545" }}
                    onClick={() => setPendingDeleteProject(project)}
                  >Delete</button>
                </div>
              </div>

              <div style={styles.rowBottom}>
                <div style={styles.statusPill(project.status)}>{project.status}</div>
                <div style={styles.progressBar}>
                  <div style={{ ...styles.progressFill, backgroundColor: statusColor(project.status), width: `${project.progress}%` }} />
                </div>
                <div style={styles.metaField}>Progress: {project.progress}%</div>
                <div style={styles.metaField}>Created: {formatDate(project.created_at)}</div>
                <div style={styles.metaField}>Outputs: {project.has_outputs ? "Yes" : "No"}</div>
                <div style={styles.metaField}>Sessions: {project.session_count ?? 0}</div>
              </div>
            </div>
          ))}
        </div>
      )}

      {editingProject && (
        <div style={styles.modalBackdrop}>
          <div style={styles.modal}>
            <h3 style={{ marginTop: 0 }}>Edit Project</h3>
            <p style={{ margin: "0 0 0.75rem 0", color: "#555" }}>Update the project name.</p>
            <input
              style={styles.input}
              value={editName}
              onChange={(e) => setEditName(e.target.value)}
              maxLength={120}
              disabled={editSaving}
              autoFocus
            />
            {editError && <div style={styles.errorBox}>{editError}</div>}
            <div style={styles.modalActions}>
              <button style={{ ...styles.button, backgroundColor: "#6c757d" }} onClick={closeEdit} disabled={editSaving}>Cancel</button>
              <button style={{ ...styles.button, backgroundColor: "#0d6efd" }} onClick={saveEdit} disabled={editSaving}>
                {editSaving ? "Saving..." : "Save"}
              </button>
            </div>
          </div>
        </div>
      )}

      <ConfirmModal
        open={Boolean(pendingDeleteProject)}
        title="Delete Project"
        message={
          <>
            Are you sure you want to delete <span className="font-semibold text-slate-800">{pendingDeleteProject?.name || pendingDeleteProject?.project_id}</span>? This cannot be undone.
          </>
        }
        confirmLabel="Delete"
        cancelLabel="Cancel"
        tone="danger"
        busy={deleteBusy}
        onCancel={() => {
          if (!deleteBusy) setPendingDeleteProject(null);
        }}
        onConfirm={handleDelete}
      />
    </div>
  );
}

const styles: Record<string, any> = {
  container: {
    padding: "2rem",
    maxWidth: "960px",
    margin: "0 auto",
  },
  header: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: "1.5rem",
  },
  headerActions: {
    display: "flex",
    gap: "0.5rem",
  },
  button: {
    padding: "0.5rem 0.9rem",
    backgroundColor: "#007bff",
    color: "white",
    border: "none",
    borderRadius: "4px",
    cursor: "pointer",
    fontSize: "0.95rem",
  },
  list: {
    display: "flex",
    flexDirection: "column",
    gap: "0.75rem",
  },
  card: {
    backgroundColor: "white",
    border: "1px solid #e5e7eb",
    borderRadius: "8px",
    padding: "1rem",
    boxShadow: "0 1px 2px rgba(0,0,0,0.04)",
  },
  rowTop: {
    display: "flex",
    justifyContent: "space-between",
    gap: "1rem",
    alignItems: "center",
  },
  rowBottom: {
    display: "flex",
    flexWrap: "wrap",
    gap: "0.75rem",
    alignItems: "center",
    marginTop: "0.75rem",
  },
  name: {
    fontSize: "1.1rem",
    fontWeight: 700,
  },
  subtle: {
    color: "#6b7280",
    fontSize: "0.9rem",
    marginTop: "0.25rem",
  },
  actions: {
    display: "flex",
    gap: "0.5rem",
  },
  statusPill: (status: string) => ({
    textTransform: "uppercase",
    fontWeight: 700,
    fontSize: "0.8rem",
    padding: "0.3rem 0.75rem",
    borderRadius: "999px",
    backgroundColor: status === "done" ? "#d1e7dd" : status === "failed" ? "#f8d7da" : status === "processing" ? "#cfe2ff" : "#e2e3e5",
    color: status === "done" ? "#0f5132" : status === "failed" ? "#842029" : status === "processing" ? "#084298" : "#41464b",
  }),
  progressBar: {
    flex: "1 1 200px",
    height: "10px",
    backgroundColor: "#e9ecef",
    borderRadius: "999px",
    overflow: "hidden",
  },
  progressFill: {
    height: "100%",
    transition: "width 0.3s ease",
  },
  metaField: {
    color: "#444",
    fontSize: "0.9rem",
  },
  errorBox: {
    color: "#c53030",
    borderLeft: "4px solid #c53030",
  },
  modalBackdrop: {
    position: "fixed",
    inset: 0,
    backgroundColor: "rgba(0,0,0,0.45)",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    zIndex: 50,
  },
  modal: {
    backgroundColor: "white",
    borderRadius: "10px",
    padding: "1.5rem",
    width: "min(420px, 90vw)",
    boxShadow: "0 12px 40px rgba(0,0,0,0.18)",
  },
  input: {
    width: "100%",
    padding: "0.75rem 0.9rem",
    borderRadius: "8px",
    border: "1px solid #ced4da",
    fontSize: "1rem",
    marginBottom: "0.75rem",
  },
  modalActions: {
    display: "flex",
    justifyContent: "flex-end",
    gap: "0.5rem",
  },
};
