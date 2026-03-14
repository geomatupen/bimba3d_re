import { Link, useNavigate } from "react-router-dom";
import {
  Plus,
  FolderOpen,
  Activity,
  RefreshCw,
  CheckCircle2,
  Clock,
  AlertTriangle,
  X,
  MoreVertical,
} from "lucide-react";
import {
  useEffect,
  useMemo,
  useState,
  useCallback,
  type MouseEvent,
} from "react";
import { api } from "../api/client";
import UserMenu from "../components/UserMenu";

interface Project {
  project_id: string;
  name: string | null;
  status: string;
  progress: number;
  created_at: string | null;
  has_outputs: boolean;
}

export default function Dashboard() {
  const [projects, setProjects] = useState<Project[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [editingProject, setEditingProject] = useState<Project | null>(null);
  const [editName, setEditName] = useState("");
  const [editSaving, setEditSaving] = useState(false);
  const [editError, setEditError] = useState<string | null>(null);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [menuOpenId, setMenuOpenId] = useState<string | null>(null);
  const navigate = useNavigate();
  const [confirmProject, setConfirmProject] = useState<Project | null>(null);
  const [toast, setToast] = useState<{ message: string; type: "success" | "error" } | null>(null);

  const showToast = (message: string, type: "success" | "error" = "success") => {
    setToast({ message, type });
    window.setTimeout(() => setToast(null), 3000);
  };

  const loadProjects = useCallback(async () => {
    try {
      const res = await api.get("/projects");
      const payload = res.data;
      const list = Array.isArray(payload) ? payload : payload?.projects;
      setProjects(list || []);
    } catch (err) {
      console.error("Failed to load projects", err);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    loadProjects();
    const timer = setInterval(loadProjects, 5000);
    return () => clearInterval(timer);
  }, [loadProjects]);

  const stats = useMemo(() => {
    const total = projects.length;
    const processing = projects.filter((p) => p.status === "processing").length;
    const completed = projects.filter((p) => p.status === "completed").length;
    const failed = projects.filter((p) => p.status === "failed").length;
    return { total, processing, completed, failed };
  }, [projects]);

  const getStatusColor = (status: string) => {
    switch (status) {
      case "completed":
        return "bg-emerald-50 text-emerald-700 border-emerald-200";
      case "processing":
        return "bg-amber-50 text-amber-700 border-amber-200";
      case "failed":
        return "bg-rose-50 text-rose-700 border-rose-200";
      default:
        return "bg-slate-50 text-slate-700 border-slate-200";
    }
  };

  const refreshNow = async () => {
    setRefreshing(true);
    await loadProjects();
  };

  const openEdit = (project: Project) => {
    setEditingProject(project);
    setEditName(project.name || "");
    setEditError(null);
  };

  const closeEdit = () => {
    setEditingProject(null);
    setEditName("");
    setEditSaving(false);
    setEditError(null);
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
      setProjects((prev) => prev.map((p) => (p.project_id === editingProject.project_id ? { ...p, name: trimmed } : p)));
      closeEdit();
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Failed to update project";
      setEditError(msg);
    } finally {
      setEditSaving(false);
    }
  };

  const requestDelete = (project: Project, e?: MouseEvent) => {
    if (e) e.stopPropagation();
    setConfirmProject(project);
  };

  const performDelete = async () => {
    if (!confirmProject) return;
    setDeletingId(confirmProject.project_id);
    try {
      await api.delete(`/projects/${confirmProject.project_id}`);
      setProjects((prev) => prev.filter((p) => p.project_id !== confirmProject!.project_id));
      showToast("Deleted successfully", "success");
    } catch (err) {
      console.error("Failed to delete project", err);
      showToast("Failed to delete project", "error");
    } finally {
      setDeletingId(null);
      setConfirmProject(null);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-slate-50">
      {/* Hero Header */}
      <header className="bg-gradient-to-r from-blue-600 via-blue-700 to-indigo-700 shadow-xl">
        <div className="max-w-7xl mx-auto px-6 lg:px-8 py-8">
          <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-6">
            <div>
              <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-white/10 backdrop-blur-sm border border-white/20 mb-3">
                <Activity className="w-3 h-3 text-white" />
                <span className="text-xs font-medium text-white uppercase tracking-wider">Gaussian Splatting Platform</span>
              </div>
                <h1 className="text-3xl lg:text-4xl font-bold text-white mb-2 tracking-tight">Bimba3d</h1>
              <p className="text-base text-blue-100 max-w-2xl">
                Professional 3D reconstruction pipeline. Upload images, train Gaussian splats, and visualize results in real-time.
              </p>
            </div>
            <UserMenu />
            <div className="flex items-center gap-3">
              <button
                onClick={refreshNow}
                disabled={refreshing}
                className="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-white/10 hover:bg-white/20 backdrop-blur-sm border border-white/20 text-white text-sm font-medium transition-all duration-200 hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <RefreshCw className={`w-4 h-4 ${refreshing ? "animate-spin" : ""}`} />
                Refresh
              </button>
              <Link
                to="/create"
                className="inline-flex items-center gap-2 px-5 py-2 rounded-xl bg-white hover:bg-gray-50 text-blue-700 text-sm font-semibold shadow-lg hover:shadow-xl transition-all duration-200 hover:scale-105"
              >
                <Plus className="w-4 h-4" />
                New Project
              </Link>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-6 lg:px-8 py-6 space-y-6">
        {/* Stats Cards */}
        <div className="-mt-12 relative z-10 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="group bg-white rounded-2xl shadow-lg hover:shadow-xl transition-all duration-300 border border-slate-200/50 overflow-hidden">
            <div className="absolute inset-0 bg-gradient-to-br from-blue-500/5 to-blue-600/5 opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
            <div className="relative p-4">
              <div className="flex items-center justify-between mb-3">
                <div className="h-10 w-10 rounded-xl bg-gradient-to-br from-blue-500 to-blue-600 flex items-center justify-center shadow-md">
                  <FolderOpen className="w-5 h-5 text-white" />
                </div>
                <span className="text-2xl font-bold text-slate-900">{stats.total}</span>
              </div>
              <p className="text-xs font-semibold text-slate-600 uppercase tracking-wide">Total Projects</p>
            </div>
          </div>

          <div className="group bg-white rounded-2xl shadow-lg hover:shadow-xl transition-all duration-300 border border-slate-200/50 overflow-hidden">
            <div className="absolute inset-0 bg-gradient-to-br from-amber-500/5 to-amber-600/5 opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
            <div className="relative p-4">
              <div className="flex items-center justify-between mb-3">
                <div className="h-10 w-10 rounded-xl bg-gradient-to-br from-amber-500 to-amber-600 flex items-center justify-center shadow-md">
                  <Clock className="w-5 h-5 text-white" />
                </div>
                <span className="text-2xl font-bold text-slate-900">{stats.processing}</span>
              </div>
              <p className="text-xs font-semibold text-slate-600 uppercase tracking-wide">Processing</p>
            </div>
          </div>

          <div className="group bg-white rounded-2xl shadow-lg hover:shadow-xl transition-all duration-300 border border-slate-200/50 overflow-hidden">
            <div className="absolute inset-0 bg-gradient-to-br from-emerald-500/5 to-emerald-600/5 opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
            <div className="relative p-4">
              <div className="flex items-center justify-between mb-3">
                <div className="h-10 w-10 rounded-xl bg-gradient-to-br from-emerald-500 to-emerald-600 flex items-center justify-center shadow-md">
                  <CheckCircle2 className="w-5 h-5 text-white" />
                </div>
                <span className="text-2xl font-bold text-slate-900">{stats.completed}</span>
              </div>
              <p className="text-xs font-semibold text-slate-600 uppercase tracking-wide">Completed</p>
            </div>
          </div>

          <div className="group bg-white rounded-2xl shadow-lg hover:shadow-xl transition-all duration-300 border border-slate-200/50 overflow-hidden">
            <div className="absolute inset-0 bg-gradient-to-br from-rose-500/5 to-rose-600/5 opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
            <div className="relative p-4">
              <div className="flex items-center justify-between mb-3">
                <div className="h-10 w-10 rounded-xl bg-gradient-to-br from-rose-500 to-rose-600 flex items-center justify-center shadow-md">
                  <AlertTriangle className="w-5 h-5 text-white" />
                </div>
                <span className="text-2xl font-bold text-slate-900">{stats.failed}</span>
              </div>
              <p className="text-xs font-semibold text-slate-600 uppercase tracking-wide">Failed</p>
            </div>
          </div>
        </div>

        {loading ? (
          <div className="space-y-4">
            {[1, 2, 3].map((i) => (
              <div key={i} className="rounded-2xl border border-slate-200 bg-white p-6 animate-pulse flex items-center gap-6 shadow-sm">
                <div className="h-24 w-24 bg-slate-100 rounded-xl flex-shrink-0" />
                <div className="flex-1 space-y-3">
                  <div className="h-6 w-2/3 bg-slate-100 rounded" />
                  <div className="h-4 w-1/3 bg-slate-100 rounded" />
                  <div className="h-3 w-full bg-slate-100 rounded" />
                </div>
              </div>
            ))}
          </div>
        ) : projects.length === 0 ? (
          <div className="text-center py-16 bg-white border-2 border-dashed border-slate-300 rounded-2xl shadow-sm">
            <div className="h-16 w-16 rounded-xl bg-gradient-to-br from-blue-50 to-blue-100 flex items-center justify-center mx-auto mb-4">
              <FolderOpen className="w-8 h-8 text-blue-600" />
            </div>
            <h2 className="text-xl font-bold text-slate-900 mb-2">No projects yet</h2>
            <p className="text-sm text-slate-600 mb-6 max-w-md mx-auto">Create your first 3D reconstruction project to see it tracked here.</p>
            <Link
              to="/create"
              className="inline-flex items-center gap-2 bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800 text-white font-semibold px-6 py-3 rounded-xl transition-all duration-200 shadow-lg hover:shadow-xl hover:scale-105"
            >
              <Plus className="w-4 h-4" />
              Create Your First Project
            </Link>
          </div>
        ) : (
          <div>
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-bold text-slate-900">Your Projects</h2>
              <p className="text-xs text-slate-500">Auto-refreshing every 5s</p>
            </div>
            <div className="space-y-4">
              {projects.map((project) => (
                <div
                  key={project.project_id}
                  className="group relative block rounded-xl border border-slate-300 bg-white hover:shadow-lg transition-all duration-300 shadow-sm overflow-hidden hover:border-blue-400 cursor-pointer"
                  onClick={() => navigate(`/project/${project.project_id}`)}
                >
                  <div className="flex items-center gap-4 p-4">
                    {/* Thumbnail/Icon */}
                    <div className="flex-shrink-0 h-16 w-16 rounded-lg bg-gradient-to-br from-blue-500 to-blue-600 flex items-center justify-center shadow-md group-hover:scale-105 transition-transform duration-300">
                      <FolderOpen className="w-8 h-8 text-white" />
                    </div>

                    {/* Content */}
                    <div className="flex-1 min-w-0 space-y-2">
                      <div className="flex items-start justify-between gap-4">
                        <div className="flex-1 min-w-0">
                          <h3 className="text-base font-bold text-slate-900 group-hover:text-blue-600 transition-colors mb-0.5 truncate">
                            {project.name || `Project ${project.project_id.slice(0, 8)}`}
                          </h3>
                          <p className="text-xs text-slate-500 font-mono">ID: {project.project_id.slice(0, 16)}...</p>
                        </div>
                        <div className="flex items-center gap-1">
                          <span
                            className={`flex-shrink-0 px-3 py-1 rounded-full text-xs font-semibold border ${getStatusColor(
                              project.status
                            )}`}
                          >
                            {project.status}
                          </span>
                          <div className="relative">
                            <button
                              className="p-1.5 rounded-md hover:bg-slate-100 text-slate-500 hover:text-slate-700"
                              onClick={(e) => {
                                e.stopPropagation();
                                setMenuOpenId((prev) => (prev === project.project_id ? null : project.project_id));
                              }}
                              aria-label="Project actions"
                            >
                              <MoreVertical className="w-4 h-4" />
                            </button>
                            {menuOpenId === project.project_id && (
                              <div className="absolute right-0 mt-2 w-36 rounded-lg border border-slate-200 bg-white shadow-lg z-20">
                                <button
                                  className="w-full text-left px-3 py-2 text-sm hover:bg-slate-50"
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    setMenuOpenId(null);
                                    openEdit(project);
                                  }}
                                >
                                  Edit name
                                </button>
                                <button
                                  className="w-full text-left px-3 py-2 text-sm text-rose-600 hover:bg-rose-50 disabled:opacity-60"
                                  onClick={(e) => {
                                    setMenuOpenId(null);
                                    requestDelete(project, e);
                                  }}
                                  disabled={deletingId === project.project_id}
                                >
                                  {deletingId === project.project_id ? "Deleting..." : "Delete"}
                                </button>
                              </div>
                            )}
                          </div>
                        </div>
                      </div>

                      {/* Progress Bar */}
                      {project.status === "processing" && (
                        <div className="space-y-1">
                          <div className="flex justify-between text-xs font-medium text-slate-600">
                            <span>Processing Progress</span>
                            <span className="text-blue-600">{project.progress}%</span>
                          </div>
                          <div className="w-full h-2.5 rounded-full bg-slate-100 overflow-hidden shadow-inner">
                            <div
                              className="h-2.5 rounded-full bg-gradient-to-r from-blue-500 to-blue-600 transition-all duration-500 shadow-sm"
                              style={{ width: `${project.progress}%` }}
                            />
                          </div>
                        </div>
                      )}

                      {/* Metadata */}
                      <div className="flex flex-wrap items-center gap-4 text-xs text-slate-500">
                        {project.created_at && (
                          <span className="flex items-center gap-1.5">
                            <Clock className="w-3.5 h-3.5" />
                            Created {new Date(project.created_at).toLocaleDateString()}
                          </span>
                        )}
                        {project.has_outputs && (
                          <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-emerald-50 text-emerald-700 font-medium border border-emerald-200">
                            <CheckCircle2 className="w-3.5 h-3.5" /> Outputs ready
                          </span>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </main>

      {/* Toast */}
      {toast && (
        <div className={`fixed top-4 right-4 z-50 rounded-lg border px-3 py-2 text-sm shadow-lg ${
          toast.type === "success"
            ? "bg-emerald-50 text-emerald-700 border-emerald-200"
            : "bg-rose-50 text-rose-700 border-rose-200"
        }`}
        >
          {toast.message}
        </div>
      )}

      {/* Edit Modal */}
      {editingProject && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 px-4" onClick={closeEdit}>
          <div
            className="w-full max-w-md rounded-2xl bg-white shadow-2xl p-6 space-y-4"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-start justify-between">
              <div>
                <h3 className="text-lg font-semibold text-slate-900">Edit project name</h3>
                <p className="text-sm text-slate-500">Update the display name for this project.</p>
              </div>
              <button className="text-slate-400 hover:text-slate-600" onClick={closeEdit}>
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium text-slate-700">Project name</label>
              <input
                className="w-full rounded-lg border border-slate-200 px-3 py-2 text-sm focus:border-blue-500 focus:ring focus:ring-blue-100"
                value={editName}
                onChange={(e) => setEditName(e.target.value)}
                placeholder="Enter project name"
              />
              {editError && <p className="text-sm text-rose-600">{editError}</p>}
            </div>

            <div className="flex justify-end gap-3">
              <button className="px-4 py-2 text-sm font-medium text-slate-600 hover:text-slate-800" onClick={closeEdit}>
                Cancel
              </button>
              <button
                className="px-4 py-2 text-sm font-semibold rounded-lg bg-blue-600 text-white hover:bg-blue-700 disabled:opacity-60 disabled:cursor-not-allowed"
                onClick={saveEdit}
                disabled={editSaving || !editName.trim()}
              >
                {editSaving ? "Saving..." : "Save"}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Delete Confirm Modal */}
      {confirmProject && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 px-4" onClick={() => setConfirmProject(null)}>
          <div
            className="w-full max-w-md rounded-2xl bg-white shadow-2xl p-6 space-y-4"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-start justify-between">
              <div>
                <h3 className="text-lg font-semibold text-slate-900">Delete project?</h3>
                <p className="text-sm text-slate-500">Are you sure want to delete the model? This cannot be undone.</p>
              </div>
              <button className="text-slate-400 hover:text-slate-600" onClick={() => setConfirmProject(null)}>
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="flex justify-end gap-3">
              <button className="px-4 py-2 text-sm font-medium text-slate-600 hover:text-slate-800" onClick={() => setConfirmProject(null)}>
                Cancel
              </button>
              <button
                className="px-4 py-2 text-sm font-semibold rounded-lg bg-rose-600 text-white hover:bg-rose-700 disabled:opacity-60 disabled:cursor-not-allowed"
                onClick={performDelete}
                disabled={deletingId === confirmProject.project_id}
              >
                {deletingId === confirmProject.project_id ? "Deleting..." : "Delete"}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
