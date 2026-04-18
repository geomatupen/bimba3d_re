import { useEffect, useMemo, useState } from "react";
import { createPortal } from "react-dom";
import { Clock, Download, Eye, FolderTree, MoreHorizontal, Pencil, RefreshCw, Trash2, Upload } from "lucide-react";
import { api } from "../../api/client";

interface ModelRegistryTabProps {
  projectId: string;
}

interface ModelItem {
  model_id: string;
  model_name?: string | null;
  engine?: string | null;
  created_at?: string | null;
  source?: {
    project_id?: string | null;
    project_name?: string | null;
    run_id?: string | null;
  } | null;
  provenance_summary?: {
    contributor_count?: number;
    unique_project_count?: number;
    project_names?: string[];
  } | null;
  ai_profile?: {
    pipeline_kind?: "controller" | "input_mode" | null;
    ai_input_mode?: "exif_only" | "exif_plus_flight_plan" | "exif_plus_flight_plan_plus_external" | null;
    ai_selector_strategy?: "preset_bias" | "continuous_bandit_linear" | null;
  } | null;
}

interface ModelLineageDetail {
  model?: ModelItem;
  lineage?: {
    contributors?: Array<{
      contributor_id?: string;
      project_id?: string;
      project_name?: string | null;
      run_id?: string;
      captured_at?: string;
      files?: Record<string, { path?: string; size?: number; sha256?: string }>;
    }>;
  };
  provenance_summary?: {
    contributor_count?: number;
    unique_project_count?: number;
    project_names?: string[];
  };
  configs?: {
    projects?: Array<{
      project_id: string;
      runs?: Array<{
        run_id: string;
        files?: Array<{
          name: string;
          path: string;
          size?: number;
        }>;
      }>;
    }>;
  };
}

interface ProjectRunItem {
  run_id: string;
  run_name?: string | null;
  saved_at?: string | null;
  engine?: string | null;
  session_status?: "completed" | "pending" | string;
}

type ModelViewMode = "reusable" | "project-ready";

const toLocaleDate = (iso?: string | null) => {
  if (!iso) return "-";
  const parsed = new Date(iso);
  if (Number.isNaN(parsed.getTime())) return iso;
  return parsed.toLocaleString();
};

const formatSize = (bytes?: number) => {
  if (typeof bytes !== "number" || !Number.isFinite(bytes) || bytes < 0) return "-";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
};

const buildModelProfileBadges = (model?: ModelItem | null): string[] => {
  const profile = model?.ai_profile && typeof model.ai_profile === "object" ? model.ai_profile : null;
  if (!profile) return ["pipeline: unknown"];

  const pipeline = String(profile.pipeline_kind || "").trim().toLowerCase();
  const aiMode = String(profile.ai_input_mode || "").trim().toLowerCase();
  const selector = String(profile.ai_selector_strategy || "").trim().toLowerCase();

  const badges: string[] = [];
  if (pipeline === "input_mode") {
    badges.push("pipeline: input mode");
  } else if (pipeline === "controller") {
    badges.push("pipeline: controller");
  } else {
    badges.push("pipeline: unknown");
  }

  if (aiMode) {
    badges.push(`ai mode: ${aiMode}`);
  }
  if (selector) {
    badges.push(`selector: ${selector}`);
  }

  return badges;
};

export default function ModelRegistryTab({ projectId }: ModelRegistryTabProps) {
  const [models, setModels] = useState<ModelItem[]>([]);
  const [modelsLoading, setModelsLoading] = useState<boolean>(false);
  const [modelsError, setModelsError] = useState<string | null>(null);
  const [selectedModelId, setSelectedModelId] = useState<string>("");
  const [viewMode, setViewMode] = useState<ModelViewMode>("reusable");

  const [projectRuns, setProjectRuns] = useState<ProjectRunItem[]>([]);
  const [projectRunsLoading, setProjectRunsLoading] = useState<boolean>(false);
  const [projectRunsError, setProjectRunsError] = useState<string | null>(null);
  const [savingRunId, setSavingRunId] = useState<string>("");

  const [detail, setDetail] = useState<ModelLineageDetail | null>(null);
  const [detailLoading, setDetailLoading] = useState<boolean>(false);
  const [detailError, setDetailError] = useState<string | null>(null);

  const [showOnlyCurrentProject, setShowOnlyCurrentProject] = useState<boolean>(false);
  const [viewerOpen, setViewerOpen] = useState<boolean>(false);
  const [viewerTitle, setViewerTitle] = useState<string>("");
  const [viewerContent, setViewerContent] = useState<string>("");
  const [viewerLoading, setViewerLoading] = useState<boolean>(false);
  const [viewerError, setViewerError] = useState<string | null>(null);
  const [modelActionLoading, setModelActionLoading] = useState<boolean>(false);
  const [modelActionError, setModelActionError] = useState<string | null>(null);
  const [openMenuModelId, setOpenMenuModelId] = useState<string>("");
  const [openMenuPosition, setOpenMenuPosition] = useState<{ top: number; left: number } | null>(null);

  const openMenuModel = useMemo(
    () => models.find((item) => item.model_id === openMenuModelId) || null,
    [models, openMenuModelId],
  );

  const loadModels = async () => {
    setModelsLoading(true);
    setModelsError(null);
    try {
      const res = await api.get("/projects/models");
      const items = Array.isArray(res.data?.models) ? (res.data.models as ModelItem[]) : [];
      setModels(items);
      if (!selectedModelId && items.length > 0) {
        setSelectedModelId(items[0].model_id);
      } else if (selectedModelId && !items.some((m) => m.model_id === selectedModelId)) {
        setSelectedModelId(items[0]?.model_id || "");
      }
    } catch (err) {
      setModels([]);
      setModelsError(err instanceof Error ? err.message : "Failed to load models");
      setSelectedModelId("");
    } finally {
      setModelsLoading(false);
    }
  };

  const loadProjectRuns = async () => {
    setProjectRunsLoading(true);
    setProjectRunsError(null);
    try {
      const res = await api.get(`/projects/${projectId}/runs`);
      const items = Array.isArray(res.data?.runs) ? (res.data.runs as ProjectRunItem[]) : [];
      setProjectRuns(items);
    } catch (err) {
      setProjectRuns([]);
      setProjectRunsError(err instanceof Error ? err.message : "Failed to load project sessions");
    } finally {
      setProjectRunsLoading(false);
    }
  };

  useEffect(() => {
    void loadModels();
    void loadProjectRuns();
  }, [projectId]);

  useEffect(() => {
    const loadDetail = async () => {
      if (!selectedModelId) {
        setDetail(null);
        setDetailError(null);
        return;
      }
      setDetailLoading(true);
      setDetailError(null);
      try {
        const res = await api.get(`/projects/models/${selectedModelId}/lineage`);
        setDetail((res.data || null) as ModelLineageDetail | null);
      } catch (err) {
        setDetail(null);
        setDetailError(err instanceof Error ? err.message : "Failed to load model lineage");
      } finally {
        setDetailLoading(false);
      }
    };

    void loadDetail();
  }, [selectedModelId]);

  useEffect(() => {
    if (viewMode === "project-ready") {
      setOpenMenuModelId("");
      setOpenMenuPosition(null);
    }
  }, [viewMode]);

  useEffect(() => {
    if (!openMenuModelId) return;

    const closeMenu = () => {
      setOpenMenuModelId("");
      setOpenMenuPosition(null);
    };

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        closeMenu();
      }
    };

    window.addEventListener("resize", closeMenu);
    window.addEventListener("scroll", closeMenu, true);
    window.addEventListener("keydown", onKeyDown);

    return () => {
      window.removeEventListener("resize", closeMenu);
      window.removeEventListener("scroll", closeMenu, true);
      window.removeEventListener("keydown", onKeyDown);
    };
  }, [openMenuModelId]);

  const displayedContributors = useMemo(() => {
    const source = Array.isArray(detail?.lineage?.contributors) ? detail?.lineage?.contributors : [];
    if (!showOnlyCurrentProject) return source;
    return source.filter((item) => item?.project_id === projectId);
  }, [detail, showOnlyCurrentProject, projectId]);

  const contributorsByProject = useMemo(() => {
    const buckets = new Map<string, { projectLabel: string; runs: Array<{ contributorId: string; runId: string; capturedAt?: string }> }>();
    for (const item of displayedContributors) {
      const projectIdValue = String(item?.project_id || "").trim();
      const projectLabel = String(item?.project_name || projectIdValue || "Unknown project").trim();
      const bucketKey = projectIdValue || projectLabel;
      if (!buckets.has(bucketKey)) {
        buckets.set(bucketKey, { projectLabel, runs: [] });
      }
      const runId = String(item?.run_id || "unknown-run").trim();
      const contributorId = String(item?.contributor_id || `${bucketKey}:${runId}`).trim();
      const capturedAt = typeof item?.captured_at === "string" ? item.captured_at : undefined;
      buckets.get(bucketKey)?.runs.push({ contributorId, runId, capturedAt });
    }
    return Array.from(buckets.values());
  }, [displayedContributors]);

  const displayedConfigProjects = useMemo(() => {
    const source = Array.isArray(detail?.configs?.projects) ? detail?.configs?.projects : [];
    if (!showOnlyCurrentProject) return source;
    return source.filter((item) => item.project_id === projectId);
  }, [detail, showOnlyCurrentProject, projectId]);

  const elevatedSourceKeys = useMemo(() => {
    const keys = new Set<string>();
    for (const item of models) {
      const sourceProjectId = String(item?.source?.project_id || "").trim();
      const sourceRunId = String(item?.source?.run_id || "").trim();
      if (!sourceProjectId || !sourceRunId) continue;
      keys.add(`${sourceProjectId}::${sourceRunId}`);
    }
    return keys;
  }, [models]);

  const projectReadyRuns = useMemo(() => {
    return projectRuns.filter((run) => {
      const runId = String(run.run_id || "").trim();
      if (!runId) return false;
      if ((run.session_status || "").toLowerCase() !== "completed") return false;
      const engine = String(run.engine || "gsplat").toLowerCase();
      if (engine && engine !== "gsplat") return false;
      const key = `${projectId}::${runId}`;
      return !elevatedSourceKeys.has(key);
    });
  }, [projectRuns, projectId, elevatedSourceKeys]);

  const buildDefaultReusableName = (run: ProjectRunItem): string => {
    const projectToken = String(projectId || "project")
      .toLowerCase()
      .replace(/[^a-z0-9_-]+/g, "-")
      .replace(/-+/g, "-")
      .replace(/^[-_]+|[-_]+$/g, "") || "project";
    const runToken = String(run.run_name || run.run_id || "session")
      .toLowerCase()
      .replace(/[^a-z0-9_-]+/g, "-")
      .replace(/-+/g, "-")
      .replace(/^[-_]+|[-_]+$/g, "") || "session";
    return `model_${projectToken}_${runToken}`;
  };

  const openLineageForModel = (modelId: string) => {
    setSelectedModelId(modelId);
  };

  const downloadConfigSnapshot = (modelId: string, projectIdValue: string, runId: string, filename: string) => {
    const baseUrl = (api.defaults.baseURL || "").replace(/\/$/, "");
    const path = `/projects/models/${encodeURIComponent(modelId)}/configs/${encodeURIComponent(projectIdValue)}/${encodeURIComponent(runId)}/${encodeURIComponent(filename)}/download`;
    const url = baseUrl ? `${baseUrl}${path}` : path;
    window.open(url, "_blank", "noopener,noreferrer");
  };

  const viewConfigSnapshot = async (modelId: string, projectIdValue: string, runId: string, filename: string) => {
    setViewerOpen(true);
    setViewerTitle(`${projectIdValue}/${runId}/${filename}`);
    setViewerContent("");
    setViewerError(null);
    setViewerLoading(true);
    try {
      const res = await api.get(
        `/projects/models/${encodeURIComponent(modelId)}/configs/${encodeURIComponent(projectIdValue)}/${encodeURIComponent(runId)}/${encodeURIComponent(filename)}`,
      );
      const content = res.data?.content;
      setViewerContent(JSON.stringify(content, null, 2));
    } catch (err) {
      setViewerError(err instanceof Error ? err.message : "Failed to load config snapshot");
    } finally {
      setViewerLoading(false);
    }
  };

  const renameModel = async (modelId: string) => {
    if (!modelId || modelActionLoading) return;
    const selected = models.find((item) => item.model_id === modelId);
    const currentName = selected?.model_name || selected?.model_id || modelId;
    const nextName = window.prompt("Enter a new model name:", currentName);
    if (nextName === null) return;

    const trimmedName = nextName.trim();
    if (!trimmedName) {
      setModelActionError("Model name cannot be empty.");
      return;
    }

    setModelActionLoading(true);
    setModelActionError(null);
    try {
      const res = await api.patch(`/projects/models/${encodeURIComponent(modelId)}`, {
        model_name: trimmedName,
      });
      const updated = res.data?.model as ModelItem | undefined;
      if (updated?.model_id) {
        setModels((prev) => prev.map((item) => (item.model_id === updated.model_id ? { ...item, ...updated } : item)));
        setDetail((prev) => {
          if (!prev) return prev;
          if (prev.model?.model_id !== updated.model_id) return prev;
          return { ...prev, model: { ...prev.model, ...updated } };
        });
      } else {
        await loadModels();
      }
    } catch (err) {
      setModelActionError(err instanceof Error ? err.message : "Failed to rename model");
    } finally {
      setModelActionLoading(false);
      setOpenMenuModelId("");
      setOpenMenuPosition(null);
    }
  };

  const deleteModel = async (modelId: string) => {
    if (!modelId || modelActionLoading) return;
    const selected = models.find((item) => item.model_id === modelId);
    const displayName = selected?.model_name || selected?.model_id || modelId;
    const confirmed = window.confirm(`Delete reusable model '${displayName}'? This cannot be undone.`);
    if (!confirmed) return;

    setModelActionLoading(true);
    setModelActionError(null);
    try {
      await api.delete(`/projects/models/${encodeURIComponent(modelId)}`);
      const deletedModelId = modelId;
      setModels((prev) => prev.filter((item) => item.model_id !== deletedModelId));
      setDetail((prev) => (prev?.model?.model_id === deletedModelId ? null : prev));
      const remaining = models.filter((item) => item.model_id !== deletedModelId);
      setSelectedModelId(remaining[0]?.model_id || "");
      await loadModels();
    } catch (err) {
      setModelActionError(err instanceof Error ? err.message : "Failed to delete model");
    } finally {
      setModelActionLoading(false);
      setOpenMenuModelId("");
      setOpenMenuPosition(null);
    }
  };

  const saveRunAsReusableModel = async (run: ProjectRunItem) => {
    const runId = String(run.run_id || "").trim();
    if (!runId || savingRunId) return;

    const suggested = buildDefaultReusableName(run);
    const promptValue = window.prompt("Name for reusable model:", suggested);
    if (promptValue === null) return;
    const modelName = promptValue.trim();

    setSavingRunId(runId);
    setModelActionError(null);
    try {
      const res = await api.post(`/projects/${projectId}/runs/${encodeURIComponent(runId)}/elevate-model`, {
        model_name: modelName || undefined,
      });
      const created = res.data?.model as ModelItem | undefined;
      await loadModels();
      await loadProjectRuns();
      setViewMode("reusable");
      if (created?.model_id) {
        setSelectedModelId(created.model_id);
      }
    } catch (err) {
      setModelActionError(err instanceof Error ? err.message : "Failed to save reusable model");
    } finally {
      setSavingRunId("");
    }
  };

  return (
    <div className="space-y-3">
      <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-3">
        <div className="flex flex-wrap items-center justify-between gap-2 mb-2">
          <div>
            <p className="text-xs uppercase tracking-wide font-semibold text-slate-500">Model Registry</p>
            <h3 className="text-base font-bold text-slate-900">Model Library</h3>
            <p className="text-[11px] text-slate-600 mt-0.5">
              Save finished sessions as reusable models, then manage and inspect lineage.
            </p>
          </div>
          <div className="inline-flex items-center gap-2">
            <div className="inline-flex rounded-md border border-slate-300 overflow-hidden text-xs">
              <button
                type="button"
                onClick={() => setViewMode("project-ready")}
                className={`px-2.5 py-1.5 font-semibold ${viewMode === "project-ready" ? "bg-blue-600 text-white" : "bg-white text-slate-700 hover:bg-slate-50"}`}
              >
                Session Models (This Project)
              </button>
              <button
                type="button"
                onClick={() => setViewMode("reusable")}
                className={`px-2.5 py-1.5 font-semibold ${viewMode === "reusable" ? "bg-blue-600 text-white" : "bg-white text-slate-700 hover:bg-slate-50"}`}
              >
                Reusable Models (All Projects)
              </button>
            </div>
            <button
              type="button"
              onClick={() => {
                void loadModels();
                void loadProjectRuns();
              }}
              disabled={modelsLoading || modelActionLoading}
              className="inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-md border border-slate-300 text-xs font-semibold text-slate-700 hover:bg-slate-50 disabled:opacity-50"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${modelsLoading ? "animate-spin" : ""}`} />
              Refresh
            </button>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-3">
          <div className="lg:col-span-1 rounded-lg border border-slate-200 bg-slate-50 p-2.5 space-y-2">
            {viewMode === "reusable" ? (
              <>
                <div>
                  <label className="block text-xs font-semibold text-slate-600 mb-1">Model</label>
                  <select
                    value={selectedModelId}
                    onChange={(e) => setSelectedModelId(e.target.value)}
                    className="w-full px-2.5 py-1.5 text-xs border border-slate-300 rounded-md"
                    disabled={modelsLoading || models.length === 0}
                  >
                    <option value="">
                      {modelsLoading ? "Loading models..." : models.length > 0 ? "Select model" : "No models available"}
                    </option>
                    {models.map((model) => (
                      <option key={model.model_id} value={model.model_id}>
                        {model.model_name || model.model_id}
                      </option>
                    ))}
                  </select>
                </div>

                <div>
                  <p className="text-xs font-semibold text-slate-600 mb-1">Quick Open</p>
                  <div className="h-64 overflow-auto rounded-md border border-slate-200 divide-y divide-slate-100 bg-white">
                    {models.length === 0 ? (
                      <div className="px-2 py-2 text-xs text-slate-500">No reusable models yet.</div>
                    ) : (
                      models.map((model) => (
                        <div key={`quick_${model.model_id}`} className="px-2 py-1 flex items-center justify-between gap-2">
                          <div className="min-w-0">
                            <p className="text-xs text-slate-700 truncate" title={model.model_name || model.model_id}>
                              {model.model_name || model.model_id}
                            </p>
                            <p className="mt-0.5 text-[10px] text-slate-500 truncate" title={`${model.source?.project_name || model.source?.project_id || "Unknown project"} / ${model.source?.run_id || "unknown-run"}`}>
                              {`${model.source?.project_name || model.source?.project_id || "Unknown project"} / ${model.source?.run_id || "unknown-run"}`}
                            </p>
                            <div className="mt-0.5 flex flex-wrap gap-1">
                              {buildModelProfileBadges(model).map((badge) => (
                                <span
                                  key={`${model.model_id}_${badge}`}
                                  className="inline-flex items-center rounded-full border border-slate-200 bg-slate-100 px-1.5 py-0.5 text-[10px] text-slate-700"
                                >
                                  {badge}
                                </span>
                              ))}
                            </div>
                          </div>
                          <div className="relative shrink-0">
                            <button
                              type="button"
                              onClick={(event) => {
                                if (openMenuModelId === model.model_id) {
                                  setOpenMenuModelId("");
                                  setOpenMenuPosition(null);
                                  return;
                                }

                                const rect = (event.currentTarget as HTMLButtonElement).getBoundingClientRect();
                                setOpenMenuModelId(model.model_id);
                                setOpenMenuPosition({
                                  top: Math.max(12, rect.top - 6),
                                  left: Math.max(12, rect.right - 160),
                                });
                              }}
                              className="inline-flex items-center justify-center p-1 rounded border border-slate-300 text-slate-700 hover:bg-slate-100"
                              title="Model actions"
                            >
                              <MoreHorizontal className="w-4 h-4" />
                            </button>
                          </div>
                        </div>
                      ))
                    )}
                  </div>
                  {openMenuModelId && openMenuModel && openMenuPosition && typeof document !== "undefined"
                    ? createPortal(
                        <div
                          className="fixed inset-0 z-[1400]"
                          onClick={() => {
                            setOpenMenuModelId("");
                            setOpenMenuPosition(null);
                          }}
                        >
                          <div
                            className="fixed w-40 rounded-md border border-slate-200 bg-white shadow-lg py-1"
                            style={{ top: openMenuPosition.top, left: openMenuPosition.left, transform: "translateY(-100%)" }}
                            onClick={(event) => event.stopPropagation()}
                          >
                            <button
                              type="button"
                              onClick={() => {
                                openLineageForModel(openMenuModel.model_id);
                                setOpenMenuModelId("");
                                setOpenMenuPosition(null);
                              }}
                              className="w-full px-3 py-1.5 text-left text-xs text-slate-700 hover:bg-slate-100"
                            >
                              Open lineage
                            </button>
                            <button
                              type="button"
                              onClick={() => void renameModel(openMenuModel.model_id)}
                              disabled={modelActionLoading}
                              className="w-full px-3 py-1.5 text-left text-xs text-slate-700 hover:bg-slate-100 inline-flex items-center gap-1 disabled:opacity-50"
                            >
                              <Pencil className="w-3 h-3" />
                              Rename
                            </button>
                            <button
                              type="button"
                              onClick={() => void deleteModel(openMenuModel.model_id)}
                              disabled={modelActionLoading}
                              className="w-full px-3 py-1.5 text-left text-xs text-rose-700 hover:bg-rose-50 inline-flex items-center gap-1 disabled:opacity-50"
                            >
                              <Trash2 className="w-3 h-3" />
                              Delete
                            </button>
                          </div>
                        </div>,
                        document.body,
                      )
                    : null}
                </div>

                <label className="inline-flex items-center gap-2 text-xs font-medium text-slate-700">
                  <input
                    type="checkbox"
                    className="w-4 h-4"
                    checked={showOnlyCurrentProject}
                    onChange={(e) => setShowOnlyCurrentProject(e.target.checked)}
                  />
                  Show only contributors from this project
                </label>
              </>
            ) : (
              <>
                <div className="rounded-md border border-blue-200 bg-blue-50 px-2.5 py-2 text-xs text-blue-900">
                  <p className="font-semibold">Project sessions ready to save</p>
                  <p className="mt-0.5 text-blue-800">
                    Choose a completed gsplat session and click <span className="font-semibold">Save as reusable</span>.
                  </p>
                </div>
                <div className="h-64 overflow-auto rounded-md border border-slate-200 divide-y divide-slate-100 bg-white">
                  {projectRunsLoading ? (
                    <div className="px-2 py-2 text-xs text-slate-500">Loading sessions...</div>
                  ) : projectReadyRuns.length === 0 ? (
                    <div className="px-2 py-2 text-xs text-slate-500">No completed gsplat sessions pending save.</div>
                  ) : (
                    projectReadyRuns.map((run) => (
                      <div key={`run_${run.run_id}`} className="px-2 py-1.5 flex items-center justify-between gap-2">
                        <div className="min-w-0">
                          <p className="text-xs font-semibold text-slate-800 truncate">{run.run_name || run.run_id}</p>
                          <p className="text-[11px] text-slate-500 truncate">{run.run_id} • {toLocaleDate(run.saved_at)}</p>
                        </div>
                        <button
                          type="button"
                          onClick={() => void saveRunAsReusableModel(run)}
                          disabled={Boolean(savingRunId) || modelActionLoading}
                          className="shrink-0 inline-flex items-center gap-1 px-2 py-1 rounded border border-blue-300 text-[11px] font-semibold text-blue-700 hover:bg-blue-50 disabled:opacity-50"
                        >
                          <Upload className="w-3 h-3" />
                          {savingRunId === run.run_id ? "Saving..." : "Save as reusable"}
                        </button>
                      </div>
                    ))
                  )}
                </div>
              </>
            )}

            {modelsError && (
              <div className="text-xs text-rose-700 bg-rose-50 border border-rose-200 rounded-md px-2 py-1.5">
                {modelsError}
              </div>
            )}
            {projectRunsError && viewMode === "project-ready" && (
              <div className="text-xs text-rose-700 bg-rose-50 border border-rose-200 rounded-md px-2 py-1.5">
                {projectRunsError}
              </div>
            )}
            {modelActionError && (
              <div className="text-xs text-rose-700 bg-rose-50 border border-rose-200 rounded-md px-2 py-1.5">
                {modelActionError}
              </div>
            )}
          </div>

          <div id="model-lineage-detail" className="lg:col-span-2 rounded-lg border border-slate-200 bg-slate-50 p-2.5">
            {viewMode === "project-ready" && (
              <div className="rounded-md border border-slate-200 bg-white p-3 text-sm text-slate-700 space-y-2">
                <p className="font-semibold text-slate-900">How this works</p>
                <p className="text-xs text-slate-600">
                  1. Train a session in this project until it is completed.
                </p>
                <p className="text-xs text-slate-600">
                  2. In <span className="font-semibold">Project Sessions</span>, click <span className="font-semibold">Save as reusable</span>.
                </p>
                <p className="text-xs text-slate-600">
                  3. The saved item appears under <span className="font-semibold">Reusable Models</span> and can be reused in other projects.
                </p>
                <div className="pt-1 text-xs text-slate-700">
                  <span className="font-semibold">Pending sessions:</span> {projectReadyRuns.length}
                </div>
              </div>
            )}
            {viewMode === "reusable" && !selectedModelId && <p className="text-sm text-slate-500">Select a model to inspect lineage and copied configs.</p>}
            {viewMode === "reusable" && detailLoading && (
              <p className="text-sm text-blue-700 inline-flex items-center gap-2">
                <Clock className="w-4 h-4 animate-pulse" />
                Loading lineage...
              </p>
            )}
            {viewMode === "reusable" && detailError && (
              <div className="text-xs text-rose-700 bg-rose-50 border border-rose-200 rounded-md px-2 py-1.5">
                {detailError}
              </div>
            )}
            {viewMode === "reusable" && !detailLoading && !detailError && detail && (
              <div className="space-y-3">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-2 text-sm">
                  <div className="rounded-md border border-slate-200 bg-slate-50 p-2.5">
                    <p className="text-xs font-semibold text-slate-500">Model</p>
                    <p className="font-semibold text-slate-900">{detail.model?.model_name || detail.model?.model_id || "-"}</p>
                    <p className="text-xs text-slate-600 mt-1">Created: {toLocaleDate(detail.model?.created_at)}</p>
                    <div className="mt-1.5 flex flex-wrap gap-1">
                      {buildModelProfileBadges(detail.model).map((badge) => (
                        <span
                          key={`detail_${detail.model?.model_id || "unknown"}_${badge}`}
                          className="inline-flex items-center rounded-full border border-slate-200 bg-slate-100 px-1.5 py-0.5 text-[10px] text-slate-700"
                        >
                          {badge}
                        </span>
                      ))}
                    </div>
                  </div>
                  <div className="rounded-md border border-slate-200 bg-slate-50 p-2.5">
                    <p className="text-xs font-semibold text-slate-500">Provenance Summary</p>
                    <p className="text-slate-900">Contributor Runs: {detail.provenance_summary?.contributor_count ?? 0}</p>
                    <p className="text-slate-900">Projects: {detail.provenance_summary?.unique_project_count ?? 0}</p>
                  </div>
                </div>

                <div>
                  <h4 className="text-sm font-semibold text-slate-900 mb-2">Contributors (Grouped by Project)</h4>
                  <div className="max-h-48 overflow-auto rounded-md border border-slate-200 divide-y divide-slate-100 bg-white">
                    {contributorsByProject.length === 0 ? (
                      <div className="px-3 py-2 text-xs text-slate-500">No contributors match this filter.</div>
                    ) : (
                      contributorsByProject.map((projectBucket) => (
                        <div key={`project_bucket_${projectBucket.projectLabel}`} className="px-3 py-2 text-xs">
                          <p className="font-semibold text-slate-800">{projectBucket.projectLabel}</p>
                          <div className="mt-1 ml-2 space-y-1 border-l border-slate-200 pl-2">
                            {projectBucket.runs.map((run) => (
                              <div key={run.contributorId}>
                                <p className="text-slate-700">{run.runId}</p>
                                <p className="text-slate-500">Captured: {toLocaleDate(run.capturedAt)}</p>
                              </div>
                            ))}
                          </div>
                        </div>
                      ))
                    )}
                  </div>
                </div>

                <div>
                  <h4 className="text-sm font-semibold text-slate-900 mb-2 inline-flex items-center gap-2">
                    <FolderTree className="w-4 h-4" />
                    Copied Config Tree
                  </h4>
                  <div className="max-h-60 overflow-auto rounded-md border border-slate-200 divide-y divide-slate-100 bg-white">
                    {displayedConfigProjects.length === 0 ? (
                      <div className="px-3 py-2 text-xs text-slate-500">No copied config snapshots match this filter.</div>
                    ) : (
                      displayedConfigProjects.map((projectNode) => (
                        <div key={projectNode.project_id} className="px-3 py-2">
                          <p className="text-xs font-semibold text-slate-800">Project: {projectNode.project_id}</p>
                          <div className="mt-1 space-y-1">
                            {(projectNode.runs || []).map((runNode) => (
                              <div key={runNode.run_id} className="rounded border border-slate-200 bg-slate-50 px-2 py-1.5">
                                <p className="text-xs font-medium text-slate-700">Run: {runNode.run_id}</p>
                                <ul className="mt-1 space-y-0.5">
                                  {(runNode.files || []).map((file) => (
                                    <li key={`${runNode.run_id}_${file.path}`} className="text-[11px] text-slate-600 flex items-center justify-between gap-2">
                                      <span className="truncate">{file.name} ({formatSize(file.size)})</span>
                                      <span className="inline-flex items-center gap-1 shrink-0">
                                        <button
                                          type="button"
                                          onClick={() => void viewConfigSnapshot(selectedModelId, projectNode.project_id, runNode.run_id, file.name)}
                                          className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded border border-slate-300 text-slate-700 hover:bg-slate-100"
                                        >
                                          <Eye className="w-3 h-3" />
                                          View
                                        </button>
                                        <button
                                          type="button"
                                          onClick={() => downloadConfigSnapshot(selectedModelId, projectNode.project_id, runNode.run_id, file.name)}
                                          className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded border border-slate-300 text-slate-700 hover:bg-slate-100"
                                        >
                                          <Download className="w-3 h-3" />
                                          Download
                                        </button>
                                      </span>
                                    </li>
                                  ))}
                                </ul>
                              </div>
                            ))}
                          </div>
                        </div>
                      ))
                    )}
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {viewerOpen && (
        <div className="fixed inset-0 z-50">
          <div className="absolute inset-0 bg-black/50" onClick={() => setViewerOpen(false)} />
          <div className="absolute inset-0 flex items-center justify-center p-4">
            <div className="w-[980px] max-w-full h-[82vh] bg-white rounded-xl shadow-2xl border border-slate-200 overflow-hidden flex flex-col">
              <div className="flex items-center justify-between px-4 py-3 border-b border-slate-200">
                <div>
                  <p className="text-xs uppercase font-semibold text-slate-500">Config Snapshot</p>
                  <h3 className="text-base font-bold text-slate-900">{viewerTitle || "JSON Viewer"}</h3>
                </div>
                <button
                  type="button"
                  onClick={() => setViewerOpen(false)}
                  className="px-3 py-1.5 rounded-lg border border-slate-300 text-sm text-slate-700 hover:bg-slate-50"
                >
                  Close
                </button>
              </div>
              <div className="flex-1 overflow-auto bg-slate-950 text-slate-100">
                {viewerLoading ? (
                  <div className="p-4 text-sm text-blue-200">Loading...</div>
                ) : viewerError ? (
                  <div className="p-4 text-sm text-rose-300">{viewerError}</div>
                ) : (
                  <pre className="p-4 text-xs leading-relaxed whitespace-pre-wrap break-words">{viewerContent}</pre>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
