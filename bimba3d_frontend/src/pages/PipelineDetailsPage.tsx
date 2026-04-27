import { useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { ArrowLeft, Play, Pause, Square, RefreshCw, Clock, Check, RotateCcw } from "lucide-react";
import { api } from "../api/client";

interface Pipeline {
  id: string;
  name: string;
  status: string;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
  current_phase: number;
  current_pass: number;
  current_run: number;
  current_project_index: number;
  total_runs: number;
  completed_runs: number;
  failed_runs: number;
  mean_reward: number | null;
  success_rate: number | null;
  best_reward: number | null;
  last_error: string | null;
  cooldown_active: boolean;
  next_run_scheduled_at: string | null;
  config: any;
  runs: any[];
}

interface AILearningTableRow {
  project_name: string;
  run_id: string;
  run_name?: string | null;
  is_baseline_row?: boolean;
  ai_input_mode?: string | null;
  ai_selector_strategy?: string | null;
  baseline_run_id?: string | null;
  selected_preset?: string | null;
  phase?: string | null;
  is_warmup?: boolean;
  best_loss_step?: number | null;
  best_loss?: number | null;
  final_loss_step?: number | null;
  final_loss?: number | null;
  best_psnr_step?: number | null;
  best_psnr?: number | null;
  final_psnr_step?: number | null;
  final_psnr?: number | null;
  best_ssim_step?: number | null;
  best_ssim?: number | null;
  final_ssim_step?: number | null;
  final_ssim?: number | null;
  best_lpips_step?: number | null;
  best_lpips?: number | null;
  final_lpips_step?: number | null;
  final_lpips?: number | null;
  t_best?: number | null;
  t_eval_best?: number | null;
  t_end?: number | null;
  s_best?: number | null;
  s_end?: number | null;
  s_run?: number | null;
  s_base_best?: number | null;
  s_base_end?: number | null;
  s_base?: number | null;
  reward?: number | null;
  run_best_l?: number | null;
  run_best_q?: number | null;
  run_best_t?: number | null;
  run_best_s?: number | null;
  run_end_l?: number | null;
  run_end_q?: number | null;
  run_end_t?: number | null;
  run_end_s?: number | null;
  remarks?: string | null;
  learned_input_params?: Record<string, unknown> | null;
  learned_input_params_source?: string | null;
  learned_input_params_status?: string | null;
}

const BASELINE_PARAM_DEFAULTS: Record<string, number> = {
  feature_lr: 2.5e-3,
  position_lr_init: 1.6e-4,
  scaling_lr: 5.0e-3,
  opacity_lr: 5.0e-2,
  rotation_lr: 1.0e-3,
  densify_grad_threshold: 2.0e-4,
  opacity_threshold: 0.005,
  lambda_dssim: 0.2,
};

function buildLearningParamRows(
  params: Record<string, unknown> | null | undefined,
  isBaselineRow = false,
): Array<{ key: string; multiplier: number | null; actual: number | null }> {
  if ((!params || typeof params !== "object") && isBaselineRow) {
    return Object.entries(BASELINE_PARAM_DEFAULTS).map(([key, actual]) => ({
      key,
      multiplier: 1.0,
      actual,
    }));
  }

  if (!params || typeof params !== "object") return [];

  const rows: Array<{ key: string; multiplier: number | null; actual: number | null }> = [];

  for (const [rawKey, rawValue] of Object.entries(params)) {
    if (typeof rawValue !== "number" || !Number.isFinite(rawValue)) {
      rows.push({ key: rawKey, multiplier: null, actual: null });
      continue;
    }

    if (rawKey.endsWith("_mult")) {
      const baseKey = rawKey.slice(0, -5);
      const baseline = BASELINE_PARAM_DEFAULTS[baseKey];
      rows.push({
        key: baseKey,
        multiplier: rawValue,
        actual: typeof baseline === "number" ? baseline * rawValue : null,
      });
      continue;
    }

    const baseline = BASELINE_PARAM_DEFAULTS[rawKey];
    rows.push({
      key: rawKey,
      multiplier: typeof baseline === "number" && baseline !== 0 ? rawValue / baseline : null,
      actual: rawValue,
    });
  }

  return rows;
}

function formatParamNumber(value: number | null): string {
  if (value === null || !Number.isFinite(value)) return "-";
  const fixed = value.toFixed(6);
  const trimmed = fixed.replace(/\.0+$/, "").replace(/(\.\d*?)0+$/, "$1");
  return trimmed === "-0" ? "0" : trimmed;
}

export default function PipelineDetailsPage() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [pipeline, setPipeline] = useState<Pipeline | null>(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState<"overview" | "projects" | "models" | "configuration" | "logs">("overview");
  const [actioningId, setActioningId] = useState<string | null>(null);
  const [toast, setToast] = useState<{ message: string; type: "success" | "error" } | null>(null);
  const [learningRows, setLearningRows] = useState<AILearningTableRow[]>([]);
  const [learningLoading, setLearningLoading] = useState(false);
  const [logsView, setLogsView] = useState<"learning_table" | "worker_logs">("learning_table");
  const [workerLogs, setWorkerLogs] = useState<{ project: string; logs: string; lines: number }[]>([]);
  const [logsLoading, setLogsLoading] = useState(false);
  const [currentProjectStatus, setCurrentProjectStatus] = useState<any>(null);
  const [models, setModels] = useState<any[]>([]);
  const [modelsLoading, setModelsLoading] = useState(false);
  const [elevateModal, setElevateModal] = useState<{ open: boolean; modelName: string; detectedModes: string[]; selectedMode: string; elevating: boolean }>({
    open: false,
    modelName: "",
    detectedModes: [],
    selectedMode: "",
    elevating: false,
  });
  const [restartModal, setRestartModal] = useState<{ open: boolean; restarting: boolean }>({
    open: false,
    restarting: false,
  });

  const showToast = (message: string, type: "success" | "error" = "success") => {
    setToast({ message, type });
    window.setTimeout(() => setToast(null), 3000);
  };

  const loadPipeline = async () => {
    try {
      const res = await api.get(`/training-pipeline/${id}`);
      setPipeline(res.data);

      // If pipeline is running, load current project status
      if (res.data.status === "running" && res.data.config?.projects) {
        const currentProjectIdx = res.data.current_project_index;
        const currentProject = res.data.config.projects[currentProjectIdx];
        if (currentProject?.project_id) {
          try {
            const statusRes = await api.get(`/projects/${currentProject.project_id}/status`);
            setCurrentProjectStatus(statusRes.data);
          } catch (err) {
            console.error("Failed to load current project status", err);
            setCurrentProjectStatus(null);
          }
        }
      } else {
        setCurrentProjectStatus(null);
      }
    } catch (err) {
      console.error("Failed to load pipeline", err);
      showToast("Failed to load pipeline", "error");
    } finally {
      setLoading(false);
    }
  };

  const loadLearningTable = async () => {
    if (!id) return;
    setLearningLoading(true);
    try {
      const res = await api.get(`/training-pipeline/${id}/learning-table`);
      setLearningRows(res.data.rows || []);
    } catch (err) {
      console.error("Failed to load learning table", err);
      setLearningRows([]);
    } finally {
      setLearningLoading(false);
    }
  };

  const loadWorkerLogs = async () => {
    if (!pipeline) return;
    setLogsLoading(true);
    try {
      const res = await api.get(`/training-pipeline/${id}/worker-logs`);
      setWorkerLogs(res.data.logs || []);
    } catch (err) {
      console.error("Failed to load worker logs", err);
      setWorkerLogs([]);
    } finally {
      setLogsLoading(false);
    }
  };

  useEffect(() => {
    loadPipeline();
    const timer = setInterval(loadPipeline, 5000);
    return () => clearInterval(timer);
  }, [id]);

  useEffect(() => {
    if (activeTab === "logs") {
      loadLearningTable();
    } else if (activeTab === "models") {
      loadModels();
    }
  }, [id, activeTab]);

  const loadModels = async () => {
    if (!id) return;
    setModelsLoading(true);
    try {
      const res = await api.get(`/training-pipeline/${id}/models`);
      setModels(res.data.models || []);
    } catch (err) {
      console.error("Failed to load models", err);
      setModels([]);
    } finally {
      setModelsLoading(false);
    }
  };

  const handleAction = async (action: "start" | "pause" | "resume" | "stop") => {
    if (!pipeline || actioningId) return;
    setActioningId(pipeline.id);
    try {
      await api.post(`/training-pipeline/${pipeline.id}/${action}`);
      showToast(`Pipeline ${action}ed successfully`, "success");
      await loadPipeline();
    } catch (err: any) {
      showToast(err.response?.data?.detail || `Failed to ${action} pipeline`, "error");
    } finally {
      setActioningId(null);
    }
  };

  const handleRestart = async () => {
    if (!pipeline || !id) return;
    setRestartModal((prev) => ({ ...prev, restarting: true }));
    try {
      await api.post(`/training-pipeline/${id}/restart`);
      showToast("Pipeline restarted successfully. All non-baseline runs and learner weights have been cleared.", "success");
      setRestartModal({ open: false, restarting: false });
      await loadPipeline();
    } catch (err: any) {
      showToast(err.response?.data?.detail || "Failed to restart pipeline", "error");
      setRestartModal((prev) => ({ ...prev, restarting: false }));
    }
  };

  const handleElevateModel = async () => {
    if (!id || !elevateModal.modelName.trim()) return;
    setElevateModal((prev) => ({ ...prev, elevating: true }));
    try {
      const res = await api.post(`/training-pipeline/${id}/elevate-learner-model`, {
        model_name: elevateModal.modelName.trim(),
        mode: elevateModal.selectedMode,
      });
      showToast(
        res.data?.model_name
          ? `Model "${res.data.model_name}" elevated successfully! It will now appear in the AI Input Mode dropdown.`
          : "Model elevated to global registry successfully!",
        "success"
      );
      setElevateModal({ open: false, modelName: "", detectedModes: [], selectedMode: "", elevating: false });
      await loadModels();
    } catch (err: any) {
      showToast(err.response?.data?.detail || "Failed to elevate model", "error");
      setElevateModal((prev) => ({ ...prev, elevating: false }));
    }
  };


  const formatDuration = (start: string, end: string | null) => {
    const startTime = new Date(start).getTime();
    const endTime = end ? new Date(end).getTime() : Date.now();
    const diff = endTime - startTime;
    const hours = Math.floor(diff / 3600000);
    const minutes = Math.floor((diff % 3600000) / 60000);
    return `${hours}h ${minutes}m`;
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600"></div>
      </div>
    );
  }

  if (!pipeline) {
    return (
      <div className="max-w-7xl mx-auto px-6 py-8">
        <div className="text-center py-12 bg-red-50 rounded-lg border border-red-200">
          <p className="text-red-800 text-lg">Pipeline not found</p>
          <button
            onClick={() => navigate("/")}
            className="mt-4 inline-flex items-center gap-2 px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700"
          >
            <ArrowLeft className="w-4 h-4" />
            Back to Dashboard
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-slate-50">
      {/* Header */}
      <header className="bg-gradient-to-r from-blue-600 via-blue-700 to-indigo-700 shadow-xl">
        <div className="max-w-7xl mx-auto px-6 lg:px-8 py-7">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <button
                onClick={() => navigate("/")}
                className="inline-flex items-center gap-2 px-3 py-2 rounded-xl bg-white/10 hover:bg-white/20 backdrop-blur-sm border border-white/20 text-white text-sm font-medium transition-all duration-200 hover:scale-105"
              >
                <ArrowLeft className="w-4 h-4" />
                Back
              </button>
              <div>
                <div className="inline-flex items-center gap-2 px-2 py-0.5 rounded-full bg-white/10 backdrop-blur-sm border border-white/20 mb-1">
                  <span className="text-xs font-medium text-white uppercase tracking-wider">Training Pipeline</span>
                </div>
                <h1 className="text-2xl font-bold text-white mb-1">
                  {pipeline.name}
                </h1>
                <p className="text-xs text-blue-100">
                  Progress: {pipeline.completed_runs}/{pipeline.total_runs} runs ({Math.round((pipeline.completed_runs / pipeline.total_runs) * 100)}%)
                </p>
              </div>
            </div>
            <div className="flex items-center gap-3">
              {pipeline && (
                <span className={`px-4 py-2 rounded-xl text-xs font-semibold shadow-lg backdrop-blur-sm border-2 ${
                  pipeline.status === "completed"
                    ? "bg-emerald-50/90 text-emerald-700 border-emerald-200"
                    : pipeline.status === "running"
                    ? "bg-blue-50/90 text-blue-700 border-blue-200"
                    : pipeline.status === "failed"
                    ? "bg-rose-50/90 text-rose-700 border-rose-200"
                    : "bg-white/90 text-slate-700 border-slate-200"
                } inline-flex items-center gap-1.5`}>
                  {(pipeline.status === "running") && (
                    <Clock className="w-3.5 h-3.5 text-blue-600 animate-pulse" />
                  )}
                  {(pipeline.status === "completed") && (
                    <Check className="w-3.5 h-3.5 text-emerald-600" />
                  )}
                  {pipeline.status}
                </span>
              )}
            </div>
          </div>

        </div>
      </header>

      {/* Tabs Navigation */}
      <div className="bg-white border-b-2 border-slate-200 shadow-sm">
        <div className="max-w-7xl mx-auto px-6 lg:px-8">
          <div className="flex items-center justify-between">
            <nav className="flex space-x-1" aria-label="Tabs">
              {(["overview", "projects", "models", "configuration", "logs"] as const).map((tab) => (
                <button
                  key={tab}
                  onClick={() => setActiveTab(tab)}
                  className={`flex items-center gap-2 px-6 py-4 text-sm font-semibold border-b-3 transition-all duration-200 ${
                    activeTab === tab
                      ? "border-blue-600 text-blue-700 bg-blue-50/50"
                      : "border-transparent text-slate-600 hover:text-slate-900 hover:bg-slate-50 hover:border-slate-200"
                  }`}
                >
                  {tab.charAt(0).toUpperCase() + tab.slice(1)}
                </button>
              ))}
            </nav>
            <div className="flex items-center gap-1.5 py-2">
              <button
                onClick={loadPipeline}
                className="inline-flex items-center gap-1.5 px-2.5 py-1.5 text-xs border border-gray-300 rounded-lg hover:bg-gray-50"
              >
                <RefreshCw className="w-3.5 h-3.5" />
                Refresh
              </button>
              {pipeline.status === "pending" && (
                <button
                  onClick={() => handleAction("start")}
                  disabled={actioningId === pipeline.id}
                  className="inline-flex items-center gap-1.5 px-2.5 py-1.5 text-xs bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:opacity-50"
                >
                  <Play className="w-3.5 h-3.5" />
                  Start
                </button>
              )}
              {pipeline.status === "running" && (
                <>
                  <button
                    onClick={() => handleAction("pause")}
                    disabled={actioningId === pipeline.id}
                    className="inline-flex items-center gap-1.5 px-2.5 py-1.5 text-xs bg-yellow-600 text-white rounded-lg hover:bg-yellow-700 disabled:opacity-50"
                  >
                    <Pause className="w-3.5 h-3.5" />
                    Pause
                  </button>
                  <button
                    onClick={() => handleAction("stop")}
                    disabled={actioningId === pipeline.id}
                    className="inline-flex items-center gap-1.5 px-2.5 py-1.5 text-xs bg-red-600 text-white rounded-lg hover:bg-red-700 disabled:opacity-50"
                  >
                    <Square className="w-3.5 h-3.5" />
                    Stop
                  </button>
                </>
              )}
              {pipeline.status === "paused" && (
                <>
                  <button
                    onClick={() => handleAction("resume")}
                    disabled={actioningId === pipeline.id}
                    className="inline-flex items-center gap-1.5 px-2.5 py-1.5 text-xs bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:opacity-50"
                  >
                    <Play className="w-3.5 h-3.5" />
                    Resume
                  </button>
                  <button
                    onClick={() => handleAction("stop")}
                    disabled={actioningId === pipeline.id}
                    className="inline-flex items-center gap-1.5 px-2.5 py-1.5 text-xs bg-red-600 text-white rounded-lg hover:bg-red-700 disabled:opacity-50"
                  >
                    <Square className="w-3.5 h-3.5" />
                    Stop
                  </button>
                </>
              )}
              {pipeline.status === "stopped" && (
                <button
                  onClick={() => handleAction("resume")}
                  disabled={actioningId === pipeline.id}
                  className="inline-flex items-center gap-1.5 px-2.5 py-1.5 text-xs bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:opacity-50"
                >
                  <Play className="w-3.5 h-3.5" />
                  Resume
                </button>
              )}
              {/* Edit button — available for any non-running status */}
              {pipeline.status !== "running" && (
                <button
                  onClick={() => navigate(`/training-pipeline?edit=${pipeline.id}`)}
                  disabled={actioningId === pipeline.id}
                  className="inline-flex items-center gap-1.5 px-2.5 py-1.5 text-xs bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50"
                  title="Edit pipeline configuration (requires restart to apply)"
                >
                  Edit Config
                </button>
              )}
              {/* Restart button — available for any non-running status */}
              {pipeline.status !== "running" && (
                <button
                  onClick={() => setRestartModal({ open: true, restarting: false })}
                  disabled={actioningId === pipeline.id}
                  className="inline-flex items-center gap-1.5 px-2.5 py-1.5 text-xs bg-orange-600 text-white rounded-lg hover:bg-orange-700 disabled:opacity-50"
                  title="Restart pipeline from scratch (keeps baseline runs and images)"
                >
                  <RotateCcw className="w-3.5 h-3.5" />
                  Restart
                </button>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Toast */}
      {toast && (
        <div
          className={`fixed top-4 right-4 z-50 px-4 py-3 rounded-lg shadow-lg ${
            toast.type === "success" ? "bg-green-50 text-green-800" : "bg-red-50 text-red-800"
          }`}
        >
          {toast.message}
        </div>
      )}

      {/* Tab Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-2">
        {activeTab === "overview" && (
          <div className="space-y-3">
            {/* All Overview Cards in One Responsive Row */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3">
              {/* Pipeline Status */}
              <div className="bg-white rounded-lg border border-gray-200 p-3 shadow-sm">
                <h2 className="text-sm font-semibold text-gray-900 mb-2">Pipeline Status</h2>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div>
                    <p className="text-gray-500">Status</p>
                    <p className="text-sm font-medium text-gray-900">{pipeline.status}</p>
                  </div>
                  <div>
                    <p className="text-gray-500">Phase</p>
                    <p className="text-sm font-medium text-gray-900">
                      {pipeline.current_phase}/{pipeline.config?.phases?.length || 0}
                    </p>
                  </div>
                  <div>
                    <p className="text-gray-500">Pass</p>
                    <p className="text-sm font-medium text-gray-900">{pipeline.current_pass}</p>
                  </div>
                  <div>
                    <p className="text-gray-500">Current Project</p>
                    <p className="text-sm font-medium text-gray-900">
                      {pipeline.current_project_index + 1}/{pipeline.config?.projects?.length || 0}
                    </p>
                  </div>
                </div>
              </div>

              {/* Progress */}
              <div className="bg-white rounded-lg border border-gray-200 p-3 shadow-sm">
                <h2 className="text-sm font-semibold text-gray-900 mb-2">Progress</h2>
                <div className="space-y-2">
                  <div>
                    <div className="flex justify-between text-xs text-gray-600 mb-1">
                      <span>Total: {pipeline.completed_runs}/{pipeline.total_runs}</span>
                      <span>{Math.round((pipeline.completed_runs / pipeline.total_runs) * 100)}%</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div
                        className="bg-indigo-600 h-2 rounded-full transition-all"
                        style={{ width: `${(pipeline.completed_runs / pipeline.total_runs) * 100}%` }}
                      ></div>
                    </div>
                  </div>
                  <div className="grid grid-cols-3 gap-1 text-xs">
                    <div>
                      <p className="text-gray-500">Done</p>
                      <p className="text-sm font-semibold text-gray-900">{pipeline.completed_runs}</p>
                    </div>
                    <div>
                      <p className="text-gray-500">Failed</p>
                      <p className="text-sm font-semibold text-red-600">{pipeline.failed_runs}</p>
                    </div>
                    <div>
                      <p className="text-gray-500">Left</p>
                      <p className="text-sm font-semibold text-gray-900">{pipeline.total_runs - pipeline.completed_runs}</p>
                    </div>
                  </div>
                </div>
              </div>

              {/* Statistics */}
              <div className="bg-white rounded-lg border border-gray-200 p-3 shadow-sm">
                <h2 className="text-sm font-semibold text-gray-900 mb-2">Learning Stats</h2>
                <div className="space-y-1 text-xs">
                  <div className="flex justify-between">
                    <span className="text-gray-500">Mean:</span>
                    <span className="font-semibold text-gray-900">
                      {pipeline.mean_reward !== null && pipeline.mean_reward !== undefined ? pipeline.mean_reward.toFixed(4) : "N/A"}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-500">Best:</span>
                    <span className="font-semibold text-green-600">
                      {pipeline.best_reward !== null && pipeline.best_reward !== undefined ? pipeline.best_reward.toFixed(4) : "N/A"}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-500">Success:</span>
                    <span className="font-semibold text-gray-900">
                      {pipeline.success_rate !== null && pipeline.success_rate !== undefined ? pipeline.success_rate.toFixed(1) + "%" : "N/A"}
                    </span>
                  </div>
                </div>
              </div>

              {/* Time Information */}
              {pipeline.started_at && (
                <div className="bg-white rounded-lg border border-gray-200 p-3 shadow-sm">
                  <h2 className="text-sm font-semibold text-gray-900 mb-2">Time</h2>
                  <div className="space-y-1 text-xs">
                    <div className="flex justify-between">
                      <span className="text-gray-500">Started:</span>
                      <span className="font-medium text-gray-900 text-[10px]">{new Date(pipeline.started_at).toLocaleString()}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-500">Elapsed:</span>
                      <span className="font-medium text-gray-900">{formatDuration(pipeline.started_at, pipeline.completed_at)}</span>
                    </div>
                    {pipeline.completed_at && (
                      <div className="flex justify-between">
                        <span className="text-gray-500">Done:</span>
                        <span className="font-medium text-gray-900 text-[10px]">{new Date(pipeline.completed_at).toLocaleString()}</span>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>

            {/* Live Progress - Show when pipeline is running */}
            {pipeline.status === "running" && currentProjectStatus && (
              <div className="bg-gradient-to-br from-blue-50 to-indigo-50 border-2 border-blue-300 rounded-lg p-4 shadow-md">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 bg-blue-600 rounded-full animate-pulse"></div>
                    <h2 className="text-sm font-bold text-blue-900">Live Training Progress</h2>
                  </div>
                  {pipeline.config?.projects?.[pipeline.current_project_index]?.project_id && (
                    <button
                      onClick={() => navigate(`/project/${pipeline.config.projects[pipeline.current_project_index].project_id}`)}
                      className="px-3 py-1.5 text-xs font-medium bg-blue-600 text-white rounded hover:bg-blue-700 transition-colors"
                    >
                      Go to Project →
                    </button>
                  )}
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                  {/* Current Project & Run Info */}
                  <div className="bg-white/70 backdrop-blur-sm rounded-lg p-3 border border-blue-200">
                    <p className="text-xs font-semibold text-gray-700 mb-2">Current Execution</p>
                    <div className="space-y-1.5 text-xs">
                      <div className="flex justify-between">
                        <span className="text-gray-600">Project:</span>
                        <span className="font-semibold text-gray-900">
                          {pipeline.config?.projects[pipeline.current_project_index]?.name || "N/A"}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600">Run ID:</span>
                        <span className="font-mono text-[10px] text-gray-900">
                          {currentProjectStatus.current_run_id || "N/A"}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600">Status:</span>
                        <span className="font-medium text-blue-700">
                          {currentProjectStatus.status || "N/A"}
                        </span>
                      </div>
                    </div>
                  </div>

                  {/* Training Progress */}
                  <div className="bg-white/70 backdrop-blur-sm rounded-lg p-3 border border-blue-200">
                    <p className="text-xs font-semibold text-gray-700 mb-2">Training Progress</p>
                    <div className="space-y-2">
                      {/* Main Progress */}
                      {currentProjectStatus.progress !== undefined && (
                        <div>
                          <div className="flex justify-between text-xs text-gray-600 mb-1">
                            <span>Overall</span>
                            <span className="font-semibold">{currentProjectStatus.progress}%</span>
                          </div>
                          <div className="w-full bg-gray-200 rounded-full h-2">
                            <div
                              className="bg-gradient-to-r from-blue-500 to-indigo-600 h-2 rounded-full transition-all"
                              style={{ width: `${currentProjectStatus.progress}%` }}
                            ></div>
                          </div>
                        </div>
                      )}

                      {/* Stage Progress */}
                      {currentProjectStatus.stage && (
                        <div className="text-xs">
                          <div className="flex justify-between text-gray-600 mb-1">
                            <span className="font-medium">Stage: {currentProjectStatus.stage}</span>
                            {currentProjectStatus.stage_progress !== undefined && (
                              <span className="font-semibold">{currentProjectStatus.stage_progress}%</span>
                            )}
                          </div>
                          {currentProjectStatus.stage_progress !== undefined && (
                            <div className="w-full bg-gray-200 rounded-full h-1.5">
                              <div
                                className="bg-blue-400 h-1.5 rounded-full transition-all"
                                style={{ width: `${currentProjectStatus.stage_progress}%` }}
                              ></div>
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  </div>
                </div>

                {/* Detailed Stage Status Block */}
                {currentProjectStatus.stage && currentProjectStatus.message && (
                  <div className="mt-3 bg-blue-50 border border-blue-200 rounded-lg p-3">
                    <p className="text-xs font-semibold text-blue-700 mb-1">Stage Status</p>
                    <p className="text-xs text-blue-900 whitespace-pre-line mb-2">{currentProjectStatus.message}</p>

                    {/* Training Substep Progress */}
                    {currentProjectStatus.stage === 'training' &&
                     typeof currentProjectStatus.current_step === 'number' &&
                     typeof currentProjectStatus.max_steps === 'number' &&
                     currentProjectStatus.max_steps > 0 && (
                      <div className="mt-2">
                        <div className="flex items-center justify-between text-xs text-blue-700 mb-1">
                          <span>Training Step {currentProjectStatus.current_step.toLocaleString()} / {currentProjectStatus.max_steps.toLocaleString()}</span>
                          <span className="font-semibold">
                            {((currentProjectStatus.current_step / currentProjectStatus.max_steps) * 100).toFixed(1)}%
                          </span>
                        </div>
                        <div className="w-full bg-blue-100 rounded-full h-2 overflow-hidden">
                          <div
                            className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                            style={{ width: `${Math.min((currentProjectStatus.current_step / currentProjectStatus.max_steps) * 100, 100)}%` }}
                          />
                        </div>

                        {/* Training Metrics */}
                        {(currentProjectStatus.current_loss !== undefined || currentProjectStatus.psnr !== undefined) && (
                          <div className="mt-2 grid grid-cols-2 gap-2 text-xs">
                            {typeof currentProjectStatus.current_loss === 'number' && (
                              <div className="text-blue-700">
                                <span>Loss: </span>
                                <span className="font-semibold">{currentProjectStatus.current_loss.toFixed(6)}</span>
                              </div>
                            )}
                            {typeof currentProjectStatus.psnr === 'number' && (
                              <div className="text-blue-700">
                                <span>PSNR: </span>
                                <span className="font-semibold">{currentProjectStatus.psnr.toFixed(2)}</span>
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                    )}

                    {/* Non-training Stage Progress */}
                    {currentProjectStatus.stage !== 'training' &&
                     typeof currentProjectStatus.stage_progress === 'number' && (
                      <div className="mt-2">
                        <div className="flex items-center justify-between text-xs text-blue-700 mb-1">
                          <span>{currentProjectStatus.stage || 'Current Stage'} Progress</span>
                          <span className="font-semibold">{currentProjectStatus.stage_progress}%</span>
                        </div>
                        <div className="w-full bg-blue-100 rounded-full h-2 overflow-hidden">
                          <div
                            className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                            style={{ width: `${Math.max(0, Math.min(currentProjectStatus.stage_progress, 100))}%` }}
                          />
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}

            {/* Activity Logs */}
            <div className="bg-white border border-gray-200 rounded-lg p-4 shadow-sm">
              <h2 className="text-sm font-semibold text-gray-900 mb-3">Activity Logs</h2>
              <div className="space-y-2 text-sm max-h-96 overflow-y-auto">
                <div className="flex items-start gap-3 p-2 bg-gray-50 rounded border border-gray-200">
                  <span className="text-gray-500 font-mono text-xs whitespace-nowrap">{new Date(pipeline.created_at).toLocaleString()}</span>
                  <span className="text-gray-700 flex-1">Pipeline created: {pipeline.name}</span>
                </div>
                {pipeline.started_at && (
                  <div className="flex items-start gap-3 p-2 bg-gray-50 rounded border border-gray-200">
                    <span className="text-gray-500 font-mono text-xs whitespace-nowrap">{new Date(pipeline.started_at).toLocaleString()}</span>
                    <span className="text-gray-700 flex-1">Pipeline started</span>
                  </div>
                )}
                {pipeline.runs && pipeline.runs.length > 0 ? (
                  <>
                    {pipeline.runs.map((run: any, idx: number) => (
                      <div key={idx} className="flex items-start gap-3 p-2 bg-gray-50 rounded border border-gray-200">
                        <span className="text-gray-500 font-mono text-xs whitespace-nowrap">{new Date(run.timestamp).toLocaleString()}</span>
                        <span className="text-gray-700 flex-1">
                          <span className="font-medium">{run.project_name}</span> - {run.run_name || `Phase ${run.phase}, Run ${run.run}`}
                          {run.phase === 1 && (
                            <span className="ml-2 text-xs px-2 py-0.5 rounded bg-gray-100 text-gray-700">
                              Baseline
                            </span>
                          )}
                          <span className={`ml-2 text-xs px-2 py-0.5 rounded ${
                            run.status === "success" ? "bg-green-100 text-green-700" : "bg-red-100 text-red-700"
                          }`}>
                            {run.status}
                          </span>
                          {run.reward !== null && run.reward !== undefined && (
                            <span className="ml-2 text-xs text-gray-600">Reward: {run.reward.toFixed(4)}</span>
                          )}
                        </span>
                      </div>
                    ))}
                  </>
                ) : (
                  <div className="text-center py-4 text-gray-500 text-xs">
                    No runs yet. Pipeline runs will appear here.
                  </div>
                )}
                {pipeline.cooldown_active && pipeline.next_run_scheduled_at && (
                  <div className="flex items-start gap-3 p-2 bg-yellow-50 rounded border border-yellow-200">
                    <span className="text-yellow-600 font-mono text-xs whitespace-nowrap">{new Date().toLocaleString()}</span>
                    <span className="text-yellow-800 text-xs flex-1">
                      Cooldown active - Next run scheduled at {new Date(pipeline.next_run_scheduled_at).toLocaleTimeString()}
                    </span>
                  </div>
                )}
                {pipeline.status === "running" && !pipeline.cooldown_active && (
                  <div className="flex items-start gap-3 p-2 bg-blue-50 rounded border border-blue-200">
                    <span className="text-blue-600 font-mono text-xs whitespace-nowrap">{new Date().toLocaleString()}</span>
                    <span className="text-blue-800 flex-1">
                      Running: Phase {pipeline.current_phase}, Run {pipeline.current_run || pipeline.current_pass}, Project {pipeline.current_project_index + 1}/{pipeline.config?.projects?.length || 0}
                    </span>
                  </div>
                )}
                {pipeline.completed_at && (
                  <div className="flex items-start gap-3 p-2 bg-green-50 rounded border border-green-200">
                    <span className="text-green-600 font-mono text-xs whitespace-nowrap">{new Date(pipeline.completed_at).toLocaleString()}</span>
                    <span className="text-green-800 flex-1">Pipeline completed</span>
                  </div>
                )}
                {pipeline.last_error && (
                  <div className="flex items-start gap-3 p-2 bg-red-50 rounded border border-red-200">
                    <span className="text-red-600 font-mono text-xs whitespace-nowrap">{new Date().toLocaleString()}</span>
                    <span className="text-red-800 flex-1">Error: {pipeline.last_error}</span>
                  </div>
                )}
              </div>
            </div>

            {/* Errors and Messages */}
            {(pipeline.last_error || pipeline.status === "failed" || pipeline.failed_runs > 0) && (
              <div className="bg-red-50 border border-red-200 rounded-lg p-3 shadow-sm">
                <h2 className="text-sm font-semibold text-red-900 mb-2">Errors & Issues</h2>
                <div className="space-y-2">
                  {pipeline.last_error && (
                    <div className="bg-white border border-red-200 rounded p-2">
                      <p className="text-xs font-medium text-red-900 mb-1">Last Error:</p>
                      <p className="text-xs text-red-700 font-mono whitespace-pre-wrap">{pipeline.last_error}</p>
                    </div>
                  )}
                  {pipeline.failed_runs > 0 && (
                    <div className="bg-white border border-red-200 rounded p-2">
                      <p className="text-xs font-medium text-red-900 mb-1">Failed Runs:</p>
                      <p className="text-xs text-red-700">
                        {pipeline.failed_runs} run{pipeline.failed_runs !== 1 ? 's' : ''} failed.
                        Check <button onClick={() => setActiveTab("logs")} className="underline font-medium">Worker Logs</button> for details.
                      </p>
                    </div>
                  )}
                  {pipeline.status === "failed" && (
                    <div className="bg-white border border-red-200 rounded p-2">
                      <p className="text-xs text-red-700">
                        Pipeline execution failed. Review the error details above and check project logs for more information.
                      </p>
                    </div>
                  )}
                </div>
              </div>
            )}

          </div>
        )}

        {activeTab === "configuration" && (
          <div className="bg-white rounded-lg border border-gray-200 p-4 shadow-sm">
            <h2 className="text-base font-semibold text-gray-900 mb-3">Pipeline Configuration</h2>
            <div className="space-y-4">
              {/* Shared Config */}
              <div className="border border-gray-200 rounded-lg p-3">
                <h3 className="text-sm font-semibold text-gray-800 mb-2">Shared Configuration</h3>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  {pipeline.config?.shared_config && Object.entries(pipeline.config.shared_config).map(([key, value]) => (
                    <div key={key} className="flex justify-between border-b border-gray-100 py-1">
                      <span className="font-medium text-gray-600">{key}:</span>
                      <span className="text-gray-900">{JSON.stringify(value)}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Phases */}
              <div className="border border-gray-200 rounded-lg p-3">
                <h3 className="text-sm font-semibold text-gray-800 mb-2">Phases</h3>
                <div className="space-y-2">
                  {pipeline.config?.phases?.map((phase: any, idx: number) => (
                    <div key={idx} className="border border-gray-100 rounded p-2 bg-gray-50">
                      <p className="text-xs font-semibold text-gray-700 mb-1">
                        Phase {phase.phase_number}: {phase.name}
                      </p>
                      <div className="grid grid-cols-2 gap-1 text-xs">
                        <div className="flex justify-between">
                          <span className="text-gray-600">Runs per project:</span>
                          <span className="text-gray-900">{phase.runs_per_project}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-600">Passes:</span>
                          <span className="text-gray-900">{phase.passes}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-600">Strategy:</span>
                          <span className="text-gray-900">{phase.strategy_override || "default"}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-600">Update model:</span>
                          <span className="text-gray-900">{phase.update_model ? "Yes" : "No"}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-600">Context jitter:</span>
                          <span className="text-gray-900">{phase.context_jitter ? "Yes" : "No"}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-600">Shuffle order:</span>
                          <span className="text-gray-900">{phase.shuffle_order ? "Yes" : "No"}</span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Thermal Management */}
              {pipeline.config?.thermal_management && (
                <div className="border border-gray-200 rounded-lg p-3">
                  <h3 className="text-sm font-semibold text-gray-800 mb-2">Thermal Management</h3>
                  <div className="grid grid-cols-2 gap-2 text-xs">
                    {Object.entries(pipeline.config.thermal_management).map(([key, value]) => (
                      <div key={key} className="flex justify-between border-b border-gray-100 py-1">
                        <span className="font-medium text-gray-600">{key}:</span>
                        <span className="text-gray-900">{JSON.stringify(value)}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}


        {activeTab === "projects" && (
          <div className="bg-white rounded-lg border border-gray-200 shadow-sm">
            <div className="px-4 py-3 border-b border-gray-200">
              <h2 className="text-base font-semibold text-gray-900">Pipeline Projects</h2>
              <p className="text-xs text-gray-600 mt-0.5">
                All projects included in this pipeline
              </p>
            </div>
            <div className="p-4">
              {pipeline.config?.projects && pipeline.config.projects.length > 0 ? (
                <div className="space-y-3">
                  {pipeline.config.projects.map((project: any, idx: number) => {
                    // Find runs for this project
                    const projectRuns = pipeline.runs?.filter((r: any) => r.project_name === project.name) || [];
                    const completedRuns = projectRuns.filter((r: any) => r.status === "success").length;
                    const failedRuns = projectRuns.filter((r: any) => r.status === "failed").length;
                    const totalExpectedRuns = pipeline.config.phases?.reduce((sum: number, phase: any) =>
                      sum + (phase.runs_per_project * phase.passes), 0) || 0;
                    const isCurrentProject = pipeline.current_project_index === idx;
                    const hasStarted = projectRuns.length > 0;
                    const isCompleted = completedRuns >= totalExpectedRuns;

                    // Find project_id from pipeline folder structure
                    const projectId = project.project_id || project.id;

                    return (
                      <div
                        key={idx}
                        className={`border rounded-lg p-3 ${
                          isCurrentProject ? "border-indigo-500 bg-indigo-50" : "border-gray-200 bg-white"
                        }`}
                      >
                        <div className="flex items-start justify-between gap-2">
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center gap-2 flex-wrap">
                              <h3 className="text-sm font-semibold text-gray-900">{project.name}</h3>
                              {isCurrentProject && (
                                <span className="px-1.5 py-0.5 text-xs font-medium bg-indigo-600 text-white rounded">
                                  Current
                                </span>
                              )}
                              {isCompleted && (
                                <span className="px-1.5 py-0.5 text-xs font-medium bg-green-100 text-green-700 rounded">
                                  Completed
                                </span>
                              )}
                              {!hasStarted && !isCurrentProject && (
                                <span className="px-1.5 py-0.5 text-xs font-medium bg-gray-100 text-gray-600 rounded">
                                  Pending
                                </span>
                              )}
                            </div>
                            <p className="text-xs text-gray-600 mt-0.5 truncate" title={project.dataset_path}>
                              {project.image_count} images • {project.dataset_path}
                            </p>
                          </div>
                          <button
                            onClick={() => {
                              // Use project_id if available, otherwise show alert
                              if (projectId) {
                                navigate(`/project/${projectId}?returnToPipeline=${id}`);
                              } else {
                                alert("Project ID not available. The project may not be created yet.");
                              }
                            }}
                            className="shrink-0 px-2 py-1 text-xs font-medium border border-indigo-300 text-indigo-700 rounded hover:bg-indigo-50"
                          >
                            Go to Project
                          </button>
                        </div>

                        {/* Progress */}
                        <div className="mt-2">
                          <div className="flex items-center justify-between text-xs mb-1">
                            <span className="text-gray-600">
                              Progress: {completedRuns}/{totalExpectedRuns} runs
                            </span>
                            <span className="text-gray-500">
                              {Math.round((completedRuns / totalExpectedRuns) * 100)}%
                            </span>
                          </div>
                          <div className="w-full bg-gray-200 rounded-full h-1.5">
                            <div
                              className="bg-indigo-600 h-1.5 rounded-full transition-all"
                              style={{ width: `${(completedRuns / totalExpectedRuns) * 100}%` }}
                            ></div>
                          </div>
                        </div>

                        {/* Stats */}
                        {projectRuns.length > 0 && (
                          <div className="mt-2 grid grid-cols-3 gap-2 text-xs">
                  <div>
                    <p className="text-gray-500">Run</p>
                    <p className="text-sm font-medium text-gray-900">{pipeline.current_run || pipeline.current_pass}</p>
                  </div>
                            <div>
                              <span className="text-gray-500">Failed:</span>
                              <span className="ml-1 font-semibold text-red-600">{failedRuns}</span>
                            </div>
                            <div>
                              <span className="text-gray-500">Remaining:</span>
                              <span className="ml-1 font-semibold text-gray-600">{totalExpectedRuns - completedRuns}</span>
                            </div>
                          </div>
                        )}

                        {/* All Runs - Horizontal Table */}
                        {projectRuns.length > 0 && (
                          <div className="mt-2 pt-2 border-t border-gray-200">
                            <p className="text-xs font-semibold text-gray-700 mb-1">
                              All Runs ({projectRuns.length}):
                            </p>
                            <div className="overflow-x-auto">
                              <table className="w-full text-xs border-collapse">
                                <thead className="bg-gray-50">
                                  <tr>
                                    <th className="px-2 py-1 text-left font-medium text-gray-700 border border-gray-200">#</th>
                                    <th className="px-2 py-1 text-left font-medium text-gray-700 border border-gray-200">Run Name</th>
                                    <th className="px-2 py-1 text-left font-medium text-gray-700 border border-gray-200">Phase</th>
                                    <th className="px-2 py-1 text-left font-medium text-gray-700 border border-gray-200">Run</th>
                                    <th className="px-2 py-1 text-left font-medium text-gray-700 border border-gray-200">Status</th>
                                    <th className="px-2 py-1 text-left font-medium text-gray-700 border border-gray-200">Reward</th>
                                    <th className="px-2 py-1 text-left font-medium text-gray-700 border border-gray-200">Completed</th>
                                  </tr>
                                </thead>
                                <tbody>
                                  {projectRuns.map((run: any, runIdx: number) => (
                                    <tr key={runIdx} className="hover:bg-gray-50">
                                      <td className="px-2 py-1 text-gray-900 border border-gray-200">{runIdx + 1}</td>
                                      <td className="px-2 py-1 text-gray-900 font-medium border border-gray-200">
                                        {run.run_name || run.run_id || "-"}
                                      </td>
                                      <td className="px-2 py-1 text-gray-900 border border-gray-200">{run.phase}</td>
                                      <td className="px-2 py-1 text-gray-900 border border-gray-200">{run.run}</td>
                                      <td className="px-2 py-1 border border-gray-200">
                                        <span className={`inline-block px-1.5 py-0.5 text-xs font-medium rounded ${
                                          run.status === "success" ? "bg-green-100 text-green-700" : "bg-red-100 text-red-700"
                                        }`}>
                                          {run.status}
                                        </span>
                                      </td>
                                      <td className="px-2 py-1 text-gray-900 font-medium border border-gray-200">
                                        {run.reward !== null && run.reward !== undefined ? run.reward.toFixed(4) : "N/A"}
                                      </td>
                                      <td className="px-2 py-1 text-gray-600 border border-gray-200">
                                        {run.completed_at ? new Date(run.completed_at).toLocaleString() : "In progress"}
                                      </td>
                                    </tr>
                                  ))}
                                </tbody>
                              </table>
                            </div>
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              ) : (
                <div className="text-center py-8 text-gray-500 text-sm">
                  No projects configured for this pipeline
                </div>
              )}
            </div>
          </div>
        )}

        {activeTab === "models" && (
          <div className="bg-white rounded-lg border border-gray-200 p-4 shadow-sm">
            <div className="flex items-center justify-between mb-3">
              <div>
                <h2 className="text-base font-semibold text-gray-900">Pipeline Shared Models</h2>
                <p className="text-xs text-gray-600 mt-1">
                  Models trained across all projects in this pipeline
                </p>
              </div>
              <button
                onClick={loadModels}
                disabled={modelsLoading}
                className="inline-flex items-center gap-2 px-3 py-1.5 text-xs border border-gray-300 rounded-lg hover:bg-gray-50 disabled:opacity-50"
              >
                <RefreshCw className={`w-3.5 h-3.5 ${modelsLoading ? "animate-spin" : ""}`} />
                Refresh
              </button>
            </div>

            {modelsLoading ? (
              <div className="flex items-center justify-center py-12">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-600"></div>
              </div>
            ) : models.length === 0 ? (
              <div className="text-center py-12 bg-gray-50 rounded-lg border border-gray-200">
                <p className="text-gray-600 text-sm mb-2">No models found yet</p>
                <p className="text-gray-500 text-xs">
                  Models will be created after training runs complete. The shared_models directory will contain learned parameter selection models.
                </p>
              </div>
            ) : (
              <div className="space-y-3">
                {models.map((model, idx) => (
                  <div key={idx} className="border border-gray-200 rounded-lg p-4 bg-gray-50">
                    <div className="flex items-start justify-between mb-2">
                      <div>
                        <h3 className="text-sm font-semibold text-gray-900">{model.name}</h3>
                        <p className="text-xs text-gray-600 mt-0.5">
                          Type: {model.type} • Mode: {model.mode}
                        </p>
                      </div>
                      <span className="px-2 py-1 text-xs font-medium bg-green-100 text-green-700 rounded">
                        Available
                      </span>
                    </div>
                    <div className="grid grid-cols-2 gap-2 text-xs mb-3">
                      <div>
                        <span className="text-gray-500">Size:</span>
                        <span className="ml-1 font-medium text-gray-900">
                          {(model.size_bytes / 1024).toFixed(2)} KB
                        </span>
                      </div>
                      <div>
                        <span className="text-gray-500">Modified:</span>
                        <span className="ml-1 font-medium text-gray-900">
                          {new Date(model.modified_at).toLocaleString()}
                        </span>
                      </div>
                    </div>
                    <div className="text-xs text-gray-600 mb-2">
                      <span className="font-medium">Path:</span>
                      <code className="ml-1 bg-gray-100 px-1 py-0.5 rounded text-[10px]">{model.path}</code>
                    </div>
                    {model.data && (
                      <details className="mt-2">
                        <summary className="text-xs font-medium text-indigo-600 cursor-pointer hover:text-indigo-700">
                          View Model Data
                        </summary>
                        <pre className="mt-2 text-[10px] bg-white p-2 rounded border border-gray-200 overflow-auto max-h-60">
                          {JSON.stringify(model.data, null, 2)}
                        </pre>
                      </details>
                    )}
                  </div>
                ))}
                <div className="mt-4 p-4 bg-indigo-50 border border-indigo-200 rounded-lg">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm font-semibold text-indigo-900 mb-1">Elevate to Global Registry</p>
                      <p className="text-xs text-indigo-700">
                        Elevate the trained learner model to the global model registry so it can be selected and reused in any project's configuration.
                        {models.length > 0 && (
                          <span className="ml-1">
                            Detected mode{models.length > 1 ? "s" : ""}: <strong>{models.map((m: any) => m.mode).join(", ")}</strong>.
                          </span>
                        )}
                      </p>
                    </div>
                    <button
                      onClick={() => {
                        const detectedModes = models.map((m: any) => String(m.mode || "")).filter(Boolean);
                        const firstMode = detectedModes[0] || "exif_only";
                        setElevateModal({
                          open: true,
                          modelName: pipeline?.name ? `${pipeline.name} Model` : "Pipeline Model",
                          detectedModes,
                          selectedMode: firstMode,
                          elevating: false,
                        });
                      }}
                      className="ml-4 shrink-0 inline-flex items-center gap-2 px-4 py-2 text-sm font-semibold bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors shadow"
                    >
                      ↑ Elevate Model
                    </button>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {activeTab === "logs" && (
          <div className="bg-white rounded-lg border border-gray-200 shadow-sm">
            <div className="px-6 py-4 border-b border-gray-200">
              <div className="flex items-center justify-between mb-4">
                <div>
                  <h2 className="text-lg font-semibold text-gray-900">Pipeline Logs</h2>
                  <p className="text-xs text-gray-600 mt-1">
                    View pipeline execution activity or detailed learning metrics
                  </p>
                </div>
                <button
                  onClick={() => {
                    if (logsView === "learning_table") {
                      loadLearningTable();
                    }
                  }}
                  disabled={learningLoading}
                  className="inline-flex items-center gap-2 px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50 disabled:opacity-50"
                >
                  <RefreshCw className={`w-4 h-4 ${learningLoading ? "animate-spin" : ""}`} />
                  Refresh
                </button>
              </div>

              {/* View Toggle */}
              <div className="flex gap-2">
                <button
                  onClick={() => {
                    setLogsView("worker_logs");
                    loadWorkerLogs();
                  }}
                  className={`px-4 py-2 text-sm font-medium rounded-lg transition-colors ${
                    logsView === "worker_logs"
                      ? "bg-indigo-600 text-white"
                      : "bg-gray-100 text-gray-700 hover:bg-gray-200"
                  }`}
                >
                  Worker Logs
                </button>
                <button
                  onClick={() => setLogsView("learning_table")}
                  className={`px-4 py-2 text-sm font-medium rounded-lg transition-colors ${
                    logsView === "learning_table"
                      ? "bg-indigo-600 text-white"
                      : "bg-gray-100 text-gray-700 hover:bg-gray-200"
                  }`}
                >
                  AI Learning Table
                </button>
              </div>
            </div>
            <div className="p-4">
              {logsView === "worker_logs" ? (
                logsLoading ? (
                  <div className="flex items-center justify-center py-12">
                    <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-600"></div>
                  </div>
                ) : workerLogs.length === 0 ? (
                  <div className="text-center py-12 text-gray-500">
                    No worker logs available yet. Training runs will generate logs here.
                  </div>
                ) : (
                  <div className="space-y-4">
                    {workerLogs.map((log, idx) => (
                      <div key={idx} className="bg-gray-50 border border-gray-200 rounded-lg">
                        <div className="px-4 py-3 bg-gray-100 border-b border-gray-200 flex items-center justify-between">
                          <h4 className="text-sm font-semibold text-gray-900">{log.project}</h4>
                          <span className="text-xs text-gray-500">{log.lines} lines</span>
                        </div>
                        <div className="p-4">
                          <pre className="text-xs font-mono text-gray-700 whitespace-pre-wrap overflow-auto max-h-[400px] bg-white p-3 rounded border border-gray-200">
                            {log.logs}
                          </pre>
                        </div>
                      </div>
                    ))}
                  </div>
                )
              ) : learningLoading ? (
                <div className="flex items-center justify-center py-12">
                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-600"></div>
                </div>
              ) : learningRows.length === 0 ? (
                <div className="text-center py-12 text-gray-500">
                  No learning data available yet. Complete some training runs to see results.
                </div>
              ) : (
                <div className="overflow-auto max-h-[600px] bg-white border border-gray-200 rounded-lg">
                  <table className="min-w-[2400px] w-full text-xs">
                    <thead className="bg-slate-100 text-slate-700">
                      <tr>
                        <th className="sticky top-0 z-10 bg-slate-100 px-2 py-2 text-left font-semibold">Project</th>
                        <th className="sticky top-0 z-10 bg-slate-100 px-2 py-2 text-left font-semibold">Run</th>
                        <th className="sticky top-0 z-10 bg-slate-100 px-2 py-2 text-left font-semibold">Baseline</th>
                        <th className="sticky top-0 z-10 bg-slate-100 px-2 py-2 text-left font-semibold">Strategy</th>
                        <th className="sticky top-0 z-10 bg-slate-100 px-2 py-2 text-left font-semibold">Preset</th>
                        <th className="sticky top-0 z-10 bg-slate-100 px-2 py-2 text-left font-semibold">Actual Value (baseline x multiplier)</th>
                        <th className="sticky top-0 z-10 bg-slate-100 px-2 py-2 text-left font-semibold">Multiplier</th>
                        <th className="sticky top-0 z-10 bg-slate-100 px-2 py-2 text-left font-semibold">Best Loss</th>
                        <th className="sticky top-0 z-10 bg-slate-100 px-2 py-2 text-left font-semibold">Final Loss (- better)</th>
                        <th className="sticky top-0 z-10 bg-slate-100 px-2 py-2 text-left font-semibold">Best PSNR</th>
                        <th className="sticky top-0 z-10 bg-slate-100 px-2 py-2 text-left font-semibold">Final PSNR (+ better)</th>
                        <th className="sticky top-0 z-10 bg-slate-100 px-2 py-2 text-left font-semibold">Best SSIM</th>
                        <th className="sticky top-0 z-10 bg-slate-100 px-2 py-2 text-left font-semibold">Final SSIM (+ better)</th>
                        <th className="sticky top-0 z-10 bg-slate-100 px-2 py-2 text-left font-semibold">Best LPIPS</th>
                        <th className="sticky top-0 z-10 bg-slate-100 px-2 py-2 text-left font-semibold">Final LPIPS (- better)</th>
                        <th className="sticky top-0 z-10 bg-slate-100 px-2 py-2 text-left font-semibold">Run Best (l,q,t,s)</th>
                        <th className="sticky top-0 z-10 bg-slate-100 px-2 py-2 text-left font-semibold">Run End (l,q,t,s)</th>
                        <th className="sticky top-0 z-10 bg-slate-100 px-2 py-2 text-left font-semibold">S Best</th>
                        <th className="sticky top-0 z-10 bg-slate-100 px-2 py-2 text-left font-semibold">S End</th>
                        <th className="sticky top-0 z-10 bg-slate-100 px-2 py-2 text-left font-semibold">S Run</th>
                        <th className="sticky top-0 z-10 bg-slate-100 px-2 py-2 text-left font-semibold">S Base Best</th>
                        <th className="sticky top-0 z-10 bg-slate-100 px-2 py-2 text-left font-semibold">S Base End</th>
                        <th className="sticky top-0 z-10 bg-slate-100 px-2 py-2 text-left font-semibold">S Base</th>
                        <th className="sticky top-0 z-10 bg-slate-100 px-2 py-2 text-left font-semibold">Reward</th>
                        <th className="sticky top-0 z-10 bg-slate-100 px-2 py-2 text-left font-semibold">Remarks</th>
                      </tr>
                    </thead>
                    <tbody>
                      {learningRows.map((row) => (
                        <tr key={`${row.project_name}-${row.run_id}`} className={`border-t border-gray-100 align-top ${row.is_baseline_row ? "bg-amber-50" : ""}`}>
                          <td className="px-2 py-2 text-slate-800 font-medium">{row.project_name}</td>
                          <td className="px-2 py-2 text-slate-800">
                            <div className="font-semibold">{row.run_name || row.run_id}</div>
                            <div className="text-[10px] text-slate-500">{row.run_id}</div>
                          </td>
                          <td className="px-2 py-2 text-center">
                            {row.is_baseline_row ? (
                              <span className="inline-block px-2 py-0.5 text-xs font-medium bg-amber-200 text-amber-900 rounded">
                                Baseline
                              </span>
                            ) : (
                              <span className="text-slate-400">-</span>
                            )}
                          </td>
                          <td className="px-2 py-2 text-slate-700">{row.ai_selector_strategy || "-"}</td>
                          <td className="px-2 py-2 text-slate-700">{row.selected_preset || "-"}</td>
                          <td className="px-2 py-2 text-slate-700 font-mono text-[11px]">
                            {(() => {
                              const paramRows = buildLearningParamRows(row.learned_input_params, Boolean(row.is_baseline_row));
                              if (paramRows.length === 0) return "-";
                              return (
                                <div className="space-y-0.5">
                                  {paramRows.map((p) => (
                                    <div key={`${row.run_id}-actual-${p.key}`} className="text-slate-700">
                                      {p.key}: {formatParamNumber(p.actual)}
                                    </div>
                                  ))}
                                  {row.learned_input_params_source && (
                                    <div className="mt-1 text-[10px] font-sans text-slate-500 italic">
                                      ({row.learned_input_params_source})
                                    </div>
                                  )}
                                </div>
                              );
                            })()}
                          </td>
                          <td className="px-2 py-2 text-slate-700 font-mono text-[11px]">
                            {(() => {
                              const paramRows = buildLearningParamRows(row.learned_input_params, Boolean(row.is_baseline_row));
                              if (paramRows.length === 0) return "-";
                              return (
                                <div className="space-y-0.5">
                                  {paramRows.map((p) => (
                                    <div key={`${row.run_id}-mult-${p.key}`} className="text-slate-700">
                                      {p.key}: {formatParamNumber(p.multiplier)}
                                    </div>
                                  ))}
                                </div>
                              );
                            })()}
                          </td>
                          <td className="px-2 py-2 text-slate-700">{row.best_loss?.toFixed(6) || "-"} @ {row.best_loss_step || "-"}</td>
                          <td className="px-2 py-2 text-slate-700">{row.final_loss?.toFixed(6) || "-"} @ {row.final_loss_step || "-"}</td>
                          <td className="px-2 py-2 text-slate-700">{row.best_psnr?.toFixed(4) || "-"} @ {row.best_psnr_step || "-"}</td>
                          <td className="px-2 py-2 text-slate-700">{row.final_psnr?.toFixed(4) || "-"} @ {row.final_psnr_step || "-"}</td>
                          <td className="px-2 py-2 text-slate-700">{row.best_ssim?.toFixed(4) || "-"} @ {row.best_ssim_step || "-"}</td>
                          <td className="px-2 py-2 text-slate-700">{row.final_ssim?.toFixed(4) || "-"} @ {row.final_ssim_step || "-"}</td>
                          <td className="px-2 py-2 text-slate-700">{row.best_lpips?.toFixed(4) || "-"} @ {row.best_lpips_step || "-"}</td>
                          <td className="px-2 py-2 text-slate-700">{row.final_lpips?.toFixed(4) || "-"} @ {row.final_lpips_step || "-"}</td>
                          <td className="px-2 py-2 text-slate-700">
                            {row.run_best_l?.toFixed(4) || "-"}, {row.run_best_q?.toFixed(4) || "-"}, {row.run_best_t?.toFixed(4) || "-"}, {row.run_best_s?.toFixed(4) || "-"}
                          </td>
                          <td className="px-2 py-2 text-slate-700">
                            {row.run_end_l?.toFixed(4) || "-"}, {row.run_end_q?.toFixed(4) || "-"}, {row.run_end_t?.toFixed(4) || "-"}, {row.run_end_s?.toFixed(4) || "-"}
                          </td>
                          <td className="px-2 py-2 text-slate-700">{row.s_best?.toFixed(6) || "-"}</td>
                          <td className="px-2 py-2 text-slate-700">{row.s_end?.toFixed(6) || "-"}</td>
                          <td className="px-2 py-2 text-slate-700">{row.s_run?.toFixed(6) || "-"}</td>
                          <td className="px-2 py-2 text-slate-700">{row.s_base_best?.toFixed(6) || "-"}</td>
                          <td className="px-2 py-2 text-slate-700">{row.s_base_end?.toFixed(6) || "-"}</td>
                          <td className="px-2 py-2 text-slate-700">{row.s_base?.toFixed(6) || "-"}</td>
                          <td className="px-2 py-2 text-slate-700 font-semibold">{row.reward?.toFixed(6) || "-"}</td>
                          <td className="px-2 py-2 text-slate-600 text-[10px] max-w-[200px] break-words">{row.remarks || "-"}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          </div>
        )}
      </main>

      {/* Restart Confirmation Modal */}
      {restartModal.open && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-sm">
          <div className="bg-white rounded-xl shadow-2xl border border-orange-200 w-full max-w-md mx-4 p-6">
            <div className="flex items-center gap-3 mb-3">
              <div className="w-10 h-10 rounded-full bg-orange-100 flex items-center justify-center shrink-0">
                <RotateCcw className="w-5 h-5 text-orange-600" />
              </div>
              <h2 className="text-lg font-bold text-gray-900">Restart Pipeline?</h2>
            </div>
            <p className="text-sm text-gray-700 mb-3">
              This will reset the pipeline back to its initial state. The following will be <strong>permanently deleted</strong>:
            </p>
            <ul className="text-sm text-gray-700 mb-3 space-y-1 list-disc list-inside bg-orange-50 border border-orange-200 rounded-lg p-3">
              <li>All non-baseline training runs (per project)</li>
              <li>Trained splat models (<code className="text-xs bg-orange-100 px-1 rounded">outputs/engines/</code>)</li>
              <li>Local learner weights (<code className="text-xs bg-orange-100 px-1 rounded">models/</code>)</li>
              <li>Pipeline shared learner model (<code className="text-xs bg-orange-100 px-1 rounded">shared_models/</code>)</li>
              <li>Batch lineage and model state metadata</li>
            </ul>
            <p className="text-sm text-gray-700 mb-4">
              The following will be <strong>kept</strong>: original images, resized images, COLMAP sparse point clouds, and the baseline run for each project.
            </p>
            <div className="flex gap-3 justify-end">
              <button
                onClick={() => setRestartModal({ open: false, restarting: false })}
                disabled={restartModal.restarting}
                className="px-4 py-2 text-sm font-medium border border-gray-300 rounded-lg hover:bg-gray-50 disabled:opacity-50"
              >
                Cancel
              </button>
              <button
                onClick={handleRestart}
                disabled={restartModal.restarting}
                className="px-4 py-2 text-sm font-semibold bg-orange-600 text-white rounded-lg hover:bg-orange-700 disabled:opacity-50 inline-flex items-center gap-2"
              >
                {restartModal.restarting ? (
                  <>
                    <RefreshCw className="w-4 h-4 animate-spin" />
                    Restarting...
                  </>
                ) : (
                  <>
                    <RotateCcw className="w-4 h-4" />
                    Yes, Restart Pipeline
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Elevate Model Modal */}
      {elevateModal.open && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-sm">
          <div className="bg-white rounded-xl shadow-2xl border border-gray-200 w-full max-w-md mx-4 p-6">
            <h2 className="text-lg font-bold text-gray-900 mb-1">Elevate Model to Global Registry</h2>
            <p className="text-sm text-gray-600 mb-4">
              This will copy the pipeline's trained learner model (contextual continuous selector) to the global model registry.
              Once elevated, it will appear in the <strong>AI Input Mode</strong> model dropdown in any project's configuration,
              and can be selected for reuse in test runs or new projects.
            </p>
            <div className="mb-3">
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Model Name <span className="text-red-500">*</span>
              </label>
              <input
                type="text"
                value={elevateModal.modelName}
                onChange={(e) => setElevateModal((prev) => ({ ...prev, modelName: e.target.value }))}
                placeholder="e.g. Third Train Pipeline Model"
                className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
                disabled={elevateModal.elevating}
                onKeyDown={(e) => { if (e.key === "Enter") handleElevateModel(); }}
                autoFocus
              />
              <p className="text-xs text-gray-500 mt-1">
                This name will appear in the model selection dropdown in project configurations.
              </p>
            </div>
            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-700 mb-1">
                AI Input Mode (auto-detected)
              </label>
              {elevateModal.detectedModes.length === 1 ? (
                <div className="w-full px-3 py-2 border border-gray-200 rounded-lg text-sm bg-gray-50 text-gray-800 font-mono">
                  {elevateModal.selectedMode}
                  <span className="ml-2 text-xs text-green-600 font-sans">✓ auto-detected</span>
                </div>
              ) : elevateModal.detectedModes.length > 1 ? (
                <select
                  value={elevateModal.selectedMode}
                  onChange={(e) => setElevateModal((prev) => ({ ...prev, selectedMode: e.target.value }))}
                  disabled={elevateModal.elevating}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
                >
                  {elevateModal.detectedModes.map((m) => (
                    <option key={m} value={m}>{m}</option>
                  ))}
                </select>
              ) : (
                <div className="w-full px-3 py-2 border border-amber-200 rounded-lg text-sm bg-amber-50 text-amber-800">
                  No mode detected from models. Please check the Models tab.
                </div>
              )}
              <p className="text-xs text-gray-500 mt-1">
                The AI input mode is automatically detected from the trained model files in this pipeline.
              </p>
            </div>
            <div className="flex gap-3 justify-end">
              <button
                onClick={() => setElevateModal({ open: false, modelName: "", detectedModes: [], selectedMode: "", elevating: false })}
                disabled={elevateModal.elevating}
                className="px-4 py-2 text-sm font-medium border border-gray-300 rounded-lg hover:bg-gray-50 disabled:opacity-50"
              >
                Cancel
              </button>
              <button
                onClick={handleElevateModel}
                disabled={elevateModal.elevating || !elevateModal.modelName.trim()}
                className="px-4 py-2 text-sm font-semibold bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:opacity-50 inline-flex items-center gap-2"
              >
                {elevateModal.elevating ? (
                  <>
                    <RefreshCw className="w-4 h-4 animate-spin" />
                    Elevating...
                  </>
                ) : (
                  "↑ Elevate Model"
                )}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
