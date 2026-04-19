import { useEffect, useMemo, useRef, useState } from "react";
import { Download } from "lucide-react";
import { jsPDF } from "jspdf";
import autoTable from "jspdf-autotable";
import { api } from "../../api/client";

type JsPdfWithAutoTable = jsPDF & { lastAutoTable?: { finalY?: number } };

const getAutoTableFinalY = (pdf: jsPDF): number | undefined => {
  return (pdf as JsPdfWithAutoTable).lastAutoTable?.finalY;
};

interface ComparisonTabProps {
  currentProjectId: string;
}

interface ProjectListItem {
  project_id: string;
  name?: string | null;
  status: string;
}

interface ProjectRunInfo {
  run_id: string;
  run_name?: string | null;
  saved_at?: string | null;
  stage?: string | null;
  session_status?: "completed" | "pending" | string;
  is_base?: boolean;
}

interface SummaryPayload {
  project_id: string;
  run_id?: string | null;
  run_name?: string | null;
  name?: string | null;
  status?: string;
  mode?: string;
  engine?: string | null;
  metrics?: Record<string, number | null | undefined>;
  major_params?: Record<string, number | boolean | null | undefined>;
  ai_insights?: {
    ai_input_mode?: string | null;
    baseline_session_id?: string | null;
    selected_preset?: string | null;
    heuristic_preset?: string | null;
    cache_used?: boolean | null;
    reward?: number | null;
    reward_positive?: boolean | null;
    reward_label?: string | null;
    initial_params?: Record<string, unknown>;
    missing_flags?: Record<string, unknown>;
  };
  tuning?: {
    initial?: Record<string, number | null | undefined>;
    final?: Record<string, number | null | undefined>;
    end_params?: Record<string, unknown>;
    end_step?: number | null;
    tune_interval?: number | null;
    log_interval?: number | null;
    runs?: number | null;
    history_count?: number;
    runtime_series?: Array<{
      step?: number;
      params?: Record<string, unknown>;
    }>;
    history?: Array<{
      step?: number;
      adjustments?: string[];
      convergence?: Record<string, unknown>;
      instability?: Record<string, unknown>;
      params?: Record<string, unknown>;
    }>;
  };
  loss_milestones?: Record<string, number | null | undefined>;
  log_loss_series?: Array<{ step?: number; loss?: number }>;
  log_time_series?: Array<{ step?: number; elapsed_seconds?: number }>;
  eval_series?: Array<{ step?: number; loss?: number }>;
  eval_time_series?: Array<{ step?: number; elapsed_seconds?: number }>;
  eval_psnr_series?: Array<{ step?: number; value?: number }>;
  eval_ssim_series?: Array<{ step?: number; value?: number }>;
  eval_lpips_series?: Array<{ step?: number; value?: number }>;
  preview_url?: string | null;
  eval_points?: number;
}

interface FilesPayload {
  files?: {
    engines?: Record<
      string,
      {
        previews?: {
          items?: Array<{
            name?: string;
            url?: string;
          }>;
        };
      }
    >;
  };
}

const metricRows: Array<{ key: string; label: string; lowerIsBetter?: boolean }> = [
  { key: "total_time_seconds", label: "Total Time", lowerIsBetter: true },
  { key: "convergence_speed", label: "Convergence Speed" },
  { key: "final_loss", label: "Final Loss", lowerIsBetter: true },
  { key: "best_splat_step", label: "Best Splat Step", lowerIsBetter: true },
  { key: "best_splat_loss", label: "Best Splat Loss", lowerIsBetter: true },
  { key: "lpips_mean", label: "LPIPS", lowerIsBetter: true },
  { key: "sharpness_mean", label: "Sharpness" },
  { key: "num_gaussians", label: "Gaussian Count" },
];

const graphMetricRows: Array<{ key: string; label: string; type: "loss_log" | "time_log" | "loss_eval" | "time_eval" | "psnr_eval" | "ssim_eval" | "lpips_eval" | "tuning" | "major"; path?: string[] }> = [
  { key: "loss_log", label: "Step vs Training Loss (log interval)", type: "loss_log" },
  { key: "time_log", label: "Step vs Training Time (log interval)", type: "time_log" },
  { key: "loss_eval", label: "Step vs Eval Loss (eval interval)", type: "loss_eval" },
  { key: "time_eval", label: "Step vs Eval Time (eval interval)", type: "time_eval" },
  { key: "psnr_eval", label: "Step vs PSNR (eval interval)", type: "psnr_eval" },
  { key: "ssim_eval", label: "Step vs SSIM (eval interval)", type: "ssim_eval" },
  { key: "lpips_eval", label: "Step vs LPIPS (eval interval)", type: "lpips_eval" },
  { key: "means_lr", label: "Step vs Means LR (tuning)", type: "tuning", path: ["learning_rates", "means"] },
  { key: "grow_grad2d", label: "Step vs Grow Grad2D (tuning)", type: "tuning", path: ["strategy", "grow_grad2d"] },
];

type GraphPoint = { x: number; y: number };

function getNestedNumber(source: unknown, path?: string[]): number | undefined {
  if (!path || !path.length) {
    return typeof source === "number" ? source : undefined;
  }
  let cur: unknown = source;
  for (const seg of path) {
    if (!cur || typeof cur !== "object") return undefined;
    cur = (cur as Record<string, unknown>)[seg];
  }
  return typeof cur === "number" ? cur : undefined;
}

function fmt(v: unknown): string {
  if (typeof v !== "number" || Number.isNaN(v)) return "-";
  if (Math.abs(v) >= 1000) return v.toLocaleString();
  if (Math.abs(v) >= 1) return v.toFixed(4);
  return v.toPrecision(4);
}

function fmtTickLabel(v: number): string {
  if (!Number.isFinite(v)) return "0";
  if (Math.abs(v) >= 1000) return Math.round(v).toLocaleString();
  return v.toFixed(3).replace(/\.0+$/, "").replace(/(\.\d*?)0+$/, "$1");
}

function formatDuration(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return "-";
  const rounded = Math.round(seconds);
  const h = Math.floor(rounded / 3600);
  const m = Math.floor((rounded % 3600) / 60);
  const s = rounded % 60;
  if (h > 0) return `${h}h ${m}m ${s}s`;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
}

function fmtMetricValue(key: string, v: unknown): string {
  if (key === "total_time_seconds" && typeof v === "number") return formatDuration(v);
  if (key === "best_splat_step" && typeof v === "number" && Number.isFinite(v)) return Math.round(v).toLocaleString();
  return fmt(v);
}

function fmtMajorParamValue(v: unknown): string {
  if (typeof v === "boolean") return v ? "Enabled" : "Disabled";
  return fmt(v);
}

function fmtLooseValue(v: unknown): string {
  if (typeof v === "boolean") return v ? "true" : "false";
  if (typeof v === "string") return v.trim() ? v : "-";
  if (typeof v === "number") return fmt(v);
  return "-";
}

function deltaText(left?: number | null, right?: number | null, lowerIsBetter?: boolean): string {
  if (typeof left !== "number" || typeof right !== "number") return "-";
  const delta = right - left;
  const pct = left !== 0 ? (delta / left) * 100 : null;
  const directionGood = lowerIsBetter ? delta < 0 : delta > 0;
  const arrow = delta === 0 ? "=" : directionGood ? "better" : "worse";
  if (pct === null || !Number.isFinite(pct)) return `${fmt(delta)} (${arrow})`;
  return `${fmt(delta)} (${pct.toFixed(2)}%, ${arrow})`;
}

function extractStepFromPreviewName(name: string): number | null {
  const primary = name.match(/^preview_(\d+)\.png$/i);
  const match = primary ?? name.match(/(\d+)(?!.*\d)/);
  if (!match) return null;
  const step = Number.parseInt(match[1], 10);
  return Number.isFinite(step) ? step : null;
}

function getLossSeriesPoints(summary: SummaryPayload | null | undefined, source: "log" | "eval"): GraphPoint[] {
  const sourceSeries =
    source === "log"
      ? ((summary?.log_loss_series && summary.log_loss_series.length > 0)
          ? summary.log_loss_series
          : (summary?.eval_series ?? []))
      : (summary?.eval_series ?? []);
  return sourceSeries
    .map((item) => {
      if (!item || typeof item.step !== "number" || typeof item.loss !== "number") return null;
      if (!Number.isFinite(item.step) || !Number.isFinite(item.loss)) return null;
      return { x: item.step, y: item.loss };
    })
    .filter((item): item is GraphPoint => item !== null)
    .sort((a, b) => a.x - b.x);
}

function getTimeSeriesPoints(summary: SummaryPayload | null | undefined, source: "log" | "eval"): GraphPoint[] {
  const sourceSeries =
    source === "log"
      ? ((summary?.log_time_series && summary.log_time_series.length > 0)
          ? summary.log_time_series
          : (summary?.eval_time_series ?? []))
      : (summary?.eval_time_series ?? []);
  return sourceSeries
    .map((item) => {
      if (!item || typeof item.step !== "number" || typeof item.elapsed_seconds !== "number") return null;
      if (!Number.isFinite(item.step) || !Number.isFinite(item.elapsed_seconds)) return null;
      return { x: item.step, y: item.elapsed_seconds };
    })
    .filter((item): item is GraphPoint => item !== null)
    .sort((a, b) => a.x - b.x);
}

function getEvalMetricSeriesPoints(
  summary: SummaryPayload | null | undefined,
  metric: "psnr" | "ssim" | "lpips",
): GraphPoint[] {
  const sourceSeries =
    metric === "psnr"
      ? (summary?.eval_psnr_series ?? [])
      : metric === "ssim"
        ? (summary?.eval_ssim_series ?? [])
        : (summary?.eval_lpips_series ?? []);
  return sourceSeries
    .map((item) => {
      if (!item || typeof item.step !== "number" || typeof item.value !== "number") return null;
      if (!Number.isFinite(item.step) || !Number.isFinite(item.value)) return null;
      return { x: item.step, y: item.value };
    })
    .filter((item): item is GraphPoint => item !== null)
    .sort((a, b) => a.x - b.x);
}

function getTuningSeriesPoints(summary: SummaryPayload | null | undefined, path?: string[]): GraphPoint[] {
  const runtimeSeries = summary?.tuning?.runtime_series ?? [];
  const historySeries = summary?.tuning?.history?.map((h) => ({ step: h.step, params: h.params })) ?? [];
  const sourceSeries = runtimeSeries.length ? runtimeSeries : historySeries;
  if (!sourceSeries.length || !path) return [];
  return sourceSeries
    .map((item) => {
      if (typeof item.step !== "number") return null;
      const y = getNestedNumber(item.params, path);
      if (typeof y !== "number" || !Number.isFinite(y)) return null;
      return { x: item.step, y };
    })
    .filter((item): item is GraphPoint => item !== null)
    .sort((a, b) => a.x - b.x);
}

function getTuningChangeMarkers(summary: SummaryPayload | null | undefined, path?: string[]): GraphPoint[] {
  if (!summary?.tuning?.history?.length || !path) return [];
  const sorted = [...summary.tuning.history]
    .filter((item) => typeof item.step === "number")
    .sort((a, b) => (a.step as number) - (b.step as number));

  const markers: GraphPoint[] = [];
  let prevValue: number | undefined;
  for (const item of sorted) {
    const cur = getNestedNumber(item.params, path);
    if (typeof cur !== "number" || !Number.isFinite(cur)) continue;
    if (typeof prevValue !== "number" || Math.abs(cur - prevValue) > 1e-12) {
      markers.push({ x: item.step as number, y: cur });
    }
    prevValue = cur;
  }
  return markers;
}

function getTuningChangeSteps(summary: SummaryPayload | null | undefined): number[] {
  if (!summary?.tuning?.history?.length) return [];
  const steps = summary.tuning.history
    .map((item) => {
      if (typeof item.step !== "number" || !Number.isFinite(item.step)) return null;
      const hasAdjustments = Array.isArray(item.adjustments) && item.adjustments.length > 0;
      const hasParams = !!item.params && typeof item.params === "object";
      return hasAdjustments || hasParams ? item.step : null;
    })
    .filter((step): step is number => step !== null);

  return Array.from(new Set(steps)).sort((a, b) => a - b);
}

function getMajorParamSeriesPoints(summary: SummaryPayload | null | undefined, key: string, xMax: number): GraphPoint[] {
  const y = summary?.major_params?.[key];
  if (typeof y !== "number" || !Number.isFinite(y)) return [];
  return [
    { x: 0, y },
    { x: xMax > 0 ? xMax : 1, y },
  ];
}

function toPath(points: GraphPoint[], xMin: number, xMax: number, yMin: number, yMax: number, width: number, height: number, pad: number): string {
  if (!points.length) return "";
  const innerW = width - pad * 2;
  const innerH = height - pad * 2;
  const xSpan = xMax - xMin || 1;
  const ySpan = yMax - yMin || 1;

  const coords = points.map((p) => {
    const x = pad + ((p.x - xMin) / xSpan) * innerW;
    const y = height - pad - ((p.y - yMin) / ySpan) * innerH;
    return `${x},${y}`;
  });
  return `M ${coords.join(" L ")}`;
}

function stepToSvgX(step: number, xMin: number, xMax: number, width: number, pad: number): number {
  const innerW = width - pad * 2;
  const xSpan = xMax - xMin || 1;
  return pad + ((step - xMin) / xSpan) * innerW;
}

function valueToSvgY(value: number, yMin: number, yMax: number, height: number, pad: number): number {
  const innerH = height - pad * 2;
  const ySpan = yMax - yMin || 1;
  return height - pad - ((value - yMin) / ySpan) * innerH;
}

function nearestPointValue(points: GraphPoint[], step: number): number | null {
  if (!points.length) return null;
  const exact = points.find((p) => p.x === step);
  if (exact) return exact.y;
  if (points.length === 1) return points[0].y;

  const sorted = [...points].sort((a, b) => a.x - b.x);
  for (let i = 0; i < sorted.length - 1; i += 1) {
    const a = sorted[i];
    const b = sorted[i + 1];
    if (step >= a.x && step <= b.x) {
      const t = (step - a.x) / (b.x - a.x || 1);
      return a.y + (b.y - a.y) * t;
    }
  }

  return step < sorted[0].x ? sorted[0].y : sorted[sorted.length - 1].y;
}

export default function ComparisonTab({ currentProjectId }: ComparisonTabProps) {
  const [projects, setProjects] = useState<ProjectListItem[]>([]);
  const [leftId, setLeftId] = useState<string>(currentProjectId);
  const [rightId, setRightId] = useState<string>("");
  const [leftRuns, setLeftRuns] = useState<ProjectRunInfo[]>([]);
  const [rightRuns, setRightRuns] = useState<ProjectRunInfo[]>([]);
  const [leftRunId, setLeftRunId] = useState<string>("");
  const [rightRunId, setRightRunId] = useState<string>("");
  const [leftSummary, setLeftSummary] = useState<SummaryPayload | null>(null);
  const [rightSummary, setRightSummary] = useState<SummaryPayload | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedGraphMetric, setSelectedGraphMetric] = useState<string>("loss_log");
  const [leftPreviewByStep, setLeftPreviewByStep] = useState<Record<number, string>>({});
  const [rightPreviewByStep, setRightPreviewByStep] = useState<Record<number, string>>({});
  const [selectedEvalStep, setSelectedEvalStep] = useState<number | null>(null);
  const [swipePercent, setSwipePercent] = useState<number>(50);
  const [showGroundTruthCompare, setShowGroundTruthCompare] = useState<boolean>(false);
  const [bottomSwipePercent, setBottomSwipePercent] = useState<number>(100);
  const [isSwipeDragging, setIsSwipeDragging] = useState<boolean>(false);
  const [isSecondSwipeDragging, setIsSecondSwipeDragging] = useState<boolean>(false);
  const [showLeftSeries, setShowLeftSeries] = useState<boolean>(true);
  const [showRightSeries, setShowRightSeries] = useState<boolean>(true);
  const [showTunerChangedMarkers, setShowTunerChangedMarkers] = useState<boolean>(true);
  const [showTuneEndMarkers, setShowTuneEndMarkers] = useState<boolean>(true);
  const [hoverStep, setHoverStep] = useState<number | null>(null);
  const [isViewSwitching, setIsViewSwitching] = useState(false);
  const [contentHoldHeight, setContentHoldHeight] = useState<number>(0);
  const [comparisonPdfBusy, setComparisonPdfBusy] = useState<boolean>(false);
  const comparisonContentRef = useRef<HTMLDivElement | null>(null);
  const swipeAreaRef = useRef<HTMLDivElement | null>(null);

  const friendlyErrorMessage = (err: unknown, fallback: string): string => {
    const maybeObj = err as { response?: { data?: { detail?: unknown } }; message?: string };
    const detail = maybeObj?.response?.data?.detail;
    if (typeof detail === "string" && detail.trim()) return detail;
    if (maybeObj?.message) return maybeObj.message;
    return fallback;
  };

  const beginRefreshWithStableLayout = () => {
    const measured = comparisonContentRef.current?.getBoundingClientRect().height ?? 0;
    if (measured > 0) {
      setContentHoldHeight(Math.ceil(measured));
    }
    setIsViewSwitching(true);
    setLeftSummary(null);
    setRightSummary(null);
    setLeftPreviewByStep({});
    setRightPreviewByStep({});
    setSelectedEvalStep(null);
    setSwipePercent(50);
    setBottomSwipePercent(100);
    setIsSecondSwipeDragging(false);
    setHoverStep(null);
  };

  useEffect(() => {
    let mounted = true;
    const loadProjects = async () => {
      try {
        const res = await api.get("/projects/");
        if (!mounted) return;
        const items = (res.data || []) as ProjectListItem[];
        setProjects(items);
        if (!rightId) {
          const candidate = items.find((p) => p.project_id !== currentProjectId);
          if (candidate) setRightId(candidate.project_id);
        }
      } catch (err) {
        if (!mounted) return;
        setError(err instanceof Error ? err.message : "Failed to load projects");
      }
    };
    loadProjects();
    return () => {
      mounted = false;
    };
  }, [currentProjectId, rightId]);

  useEffect(() => {
    let mounted = true;
    const loadLeftRuns = async () => {
      if (!leftId) {
        setLeftRuns([]);
        setLeftRunId("");
        return;
      }
      try {
        const res = await api.get(`/projects/${leftId}/runs`);
        if (!mounted) return;
        const runsRaw = Array.isArray(res.data?.runs) ? (res.data.runs as ProjectRunInfo[]) : [];
        const runs = runsRaw.filter((run) => run.session_status === "completed");
        setLeftRuns(runs);
        if (!leftRunId || !runs.some((r) => r.run_id === leftRunId)) {
          setLeftRunId(runs[0]?.run_id || "");
        }
      } catch {
        if (!mounted) return;
        setLeftRuns([]);
        setLeftRunId("");
      }
    };
    loadLeftRuns();
    return () => {
      mounted = false;
    };
  }, [leftId, leftRunId]);

  useEffect(() => {
    let mounted = true;
    const loadRightRuns = async () => {
      if (!rightId) {
        setRightRuns([]);
        setRightRunId("");
        return;
      }
      try {
        const res = await api.get(`/projects/${rightId}/runs`);
        if (!mounted) return;
        const runsRaw = Array.isArray(res.data?.runs) ? (res.data.runs as ProjectRunInfo[]) : [];
        const runs = runsRaw.filter((run) => run.session_status === "completed");
        setRightRuns(runs);
        if (!rightRunId || !runs.some((r) => r.run_id === rightRunId)) {
          setRightRunId(runs[0]?.run_id || "");
        }
      } catch {
        if (!mounted) return;
        setRightRuns([]);
        setRightRunId("");
      }
    };
    loadRightRuns();
    return () => {
      mounted = false;
    };
  }, [rightId, rightRunId]);

  useEffect(() => {
    if (!leftId || !rightId) return;
    setLeftSummary(null);
    setRightSummary(null);
    setLeftPreviewByStep({});
    setRightPreviewByStep({});
    setSelectedEvalStep(null);
    setSwipePercent(50);
    setBottomSwipePercent(100);
    setIsSecondSwipeDragging(false);
    setHoverStep(null);
    let mounted = true;
    const loadSummaries = async () => {
      try {
        setLoading(true);
        setError(null);

        const loadPreviewSteps = async (projectId: string, runId: string, engine?: string | null): Promise<Record<number, string>> => {
          try {
            const filesRes = await api.get(`/projects/${projectId}/files`, {
              params: { run_id: runId || undefined },
            });
            const payload = (filesRes.data || {}) as FilesPayload;
            const engines = payload.files?.engines || {};
            const chosenEngine = engine && engines[engine] ? engine : Object.keys(engines)[0];
            if (!chosenEngine) return {};
            const items = engines[chosenEngine]?.previews?.items || [];
            const mapped: Record<number, string> = {};
            for (const item of items) {
              if (!item?.name || !item?.url) continue;
              const step = extractStepFromPreviewName(item.name);
              if (step === null) continue;
              mapped[step] = `${api.defaults.baseURL}${item.url}`;
            }
            return mapped;
          } catch {
            return {};
          }
        };

        const [leftRes, rightRes] = await Promise.all([
          api.get(`/projects/${leftId}/experiment-summary`, {
            params: { run_id: leftRunId || undefined },
          }),
          api.get(`/projects/${rightId}/experiment-summary`, {
            params: { run_id: rightRunId || undefined },
          }),
        ]);
        const leftData = leftRes.data as SummaryPayload;
        const rightData = rightRes.data as SummaryPayload;
        const [leftSteps, rightSteps] = await Promise.all([
          loadPreviewSteps(leftId, leftRunId, leftData.engine),
          loadPreviewSteps(rightId, rightRunId, rightData.engine),
        ]);
        if (!mounted) return;
        setLeftSummary(leftData);
        setRightSummary(rightData);
        setLeftPreviewByStep(leftSteps);
        setRightPreviewByStep(rightSteps);
        setContentHoldHeight(0);
      } catch (err) {
        if (!mounted) return;
        setLeftSummary(null);
        setRightSummary(null);
        setError(friendlyErrorMessage(err, "Failed to load comparison summary"));
      } finally {
        if (mounted) setLoading(false);
      }
    };
    loadSummaries();
    return () => {
      mounted = false;
    };
  }, [leftId, rightId, leftRunId, rightRunId, projects]);

  useEffect(() => {
    setHoverStep(null);
  }, [selectedGraphMetric]);

  useEffect(() => {
    if (!isViewSwitching) return;
    const timer = window.setTimeout(() => setIsViewSwitching(false), 0);
    return () => window.clearTimeout(timer);
  }, [isViewSwitching, selectedGraphMetric, leftId, rightId]);

  const commonEvalSteps = useMemo(() => {
    const leftSteps = new Set(Object.keys(leftPreviewByStep).map((k) => Number.parseInt(k, 10)));
    const rightSteps = new Set(Object.keys(rightPreviewByStep).map((k) => Number.parseInt(k, 10)));
    return Array.from(leftSteps)
      .filter((step) => rightSteps.has(step))
      .sort((a, b) => a - b);
  }, [leftPreviewByStep, rightPreviewByStep]);

  useEffect(() => {
    if (!commonEvalSteps.length) {
      setSelectedEvalStep(null);
      return;
    }
    setSelectedEvalStep((prev) => {
      if (typeof prev === "number" && commonEvalSteps.includes(prev)) return prev;
      return commonEvalSteps[commonEvalSteps.length - 1];
    });
  }, [commonEvalSteps]);

  const leftSelectedPreview = selectedEvalStep === null ? null : leftPreviewByStep[selectedEvalStep] || null;
  const rightSelectedPreview = selectedEvalStep === null ? null : rightPreviewByStep[selectedEvalStep] || null;
  const fixedGroundTruthPreview = useMemo(() => {
    const firstStep = Object.keys(leftPreviewByStep)
      .map((k) => Number.parseInt(k, 10))
      .filter((n) => Number.isFinite(n))
      .sort((a, b) => a - b)[0];
    if (typeof firstStep !== "number") return null;
    return leftPreviewByStep[firstStep] || null;
  }, [leftPreviewByStep]);

  const updateSwipeFromClientX = (clientX: number) => {
    const rect = swipeAreaRef.current?.getBoundingClientRect();
    if (!rect || rect.width <= 0) return;
    const raw = ((clientX - rect.left) / rect.width) * 100;
    const clamped = Math.max(0, Math.min(100, raw));
    setSwipePercent(clamped);
  };

  const updateSecondSwipeFromClientX = (clientX: number) => {
    const rect = swipeAreaRef.current?.getBoundingClientRect();
    if (!rect || rect.width <= 0) return;
    const raw = ((clientX - rect.left) / rect.width) * 100;
    setBottomSwipePercent(Math.max(0, Math.min(100, raw)));
  };

  const options = projects.map((project) => ({
    value: project.project_id,
    label: `${project.name || project.project_id.slice(0, 8)} (${project.status})`,
  }));

  const milestoneKeys = useMemo(() => {
    const keySet = new Set<string>();
    [leftSummary?.loss_milestones, rightSummary?.loss_milestones].forEach((milestones) => {
      if (!milestones) return;
      Object.keys(milestones).forEach((k) => keySet.add(k));
    });
    return Array.from(keySet).sort((a, b) => {
      const ai = parseInt(a.replace("loss_at_", ""), 10);
      const bi = parseInt(b.replace("loss_at_", ""), 10);
      return ai - bi;
    });
  }, [leftSummary, rightSummary]);

  const leftHistory = useMemo(() => leftSummary?.tuning?.history ?? [], [leftSummary?.tuning?.history]);
  const rightHistory = useMemo(() => rightSummary?.tuning?.history ?? [], [rightSummary?.tuning?.history]);
  const leftSelectedRun = useMemo(() => leftRuns.find((run) => run.run_id === leftRunId) || null, [leftRuns, leftRunId]);
  const rightSelectedRun = useMemo(() => rightRuns.find((run) => run.run_id === rightRunId) || null, [rightRuns, rightRunId]);
  const leftIsBase = Boolean(leftSelectedRun?.is_base);
  const rightIsBase = Boolean(rightSelectedRun?.is_base);
  const aiInitialParamKeys = useMemo(() => {
    const keys = new Set<string>();
    Object.keys(leftSummary?.ai_insights?.initial_params || {}).forEach((k) => keys.add(k));
    Object.keys(rightSummary?.ai_insights?.initial_params || {}).forEach((k) => keys.add(k));
    const preferredOrder = [
      "tune_start_step",
      "tune_end_step",
      "tune_interval",
      "tune_min_improvement",
      "trend_scope",
      "feature_lr",
      "opacity_lr",
      "scaling_lr",
      "rotation_lr",
      "position_lr_init",
      "position_lr_final",
      "densification_interval",
      "densify_grad_threshold",
      "opacity_threshold",
      "lambda_dssim",
    ];
    const ordered = preferredOrder.filter((k) => keys.has(k));
    const extras = Array.from(keys).filter((k) => !preferredOrder.includes(k)).sort();
    return [...ordered, ...extras];
  }, [leftSummary?.ai_insights?.initial_params, rightSummary?.ai_insights?.initial_params]);
  const leftHasSummaryData = (leftSummary?.eval_points ?? 0) > 0;
  const rightHasSummaryData = (rightSummary?.eval_points ?? 0) > 0;
  const selectedGraphRow = graphMetricRows.find((row) => row.key === selectedGraphMetric) ?? graphMetricRows[0];
  const graphXMax = useMemo(() => {
    const points = [
      ...getLossSeriesPoints(leftSummary, "log"),
      ...getLossSeriesPoints(rightSummary, "log"),
      ...getLossSeriesPoints(leftSummary, "eval"),
      ...getLossSeriesPoints(rightSummary, "eval"),
      ...getTimeSeriesPoints(leftSummary, "log"),
      ...getTimeSeriesPoints(rightSummary, "log"),
      ...getTimeSeriesPoints(leftSummary, "eval"),
      ...getTimeSeriesPoints(rightSummary, "eval"),
      ...getEvalMetricSeriesPoints(leftSummary, "psnr"),
      ...getEvalMetricSeriesPoints(rightSummary, "psnr"),
      ...getEvalMetricSeriesPoints(leftSummary, "ssim"),
      ...getEvalMetricSeriesPoints(rightSummary, "ssim"),
      ...getEvalMetricSeriesPoints(leftSummary, "lpips"),
      ...getEvalMetricSeriesPoints(rightSummary, "lpips"),
      ...leftHistory
        .map((h) => (typeof h.step === "number" ? h.step : null))
        .filter((v): v is number => v !== null),
      ...rightHistory
        .map((h) => (typeof h.step === "number" ? h.step : null))
        .filter((v): v is number => v !== null),
      leftSummary?.major_params?.max_steps,
      rightSummary?.major_params?.max_steps,
      leftSummary?.tuning?.end_step,
      rightSummary?.tuning?.end_step,
    ].filter((v): v is number => typeof v === "number" && Number.isFinite(v));
    return points.length ? Math.max(...points) : 1;
  }, [leftSummary, rightSummary, leftHistory, rightHistory]);

  const leftGraphPoints = useMemo(() => {
    if (selectedGraphRow.type === "loss_log") return getLossSeriesPoints(leftSummary, "log");
    if (selectedGraphRow.type === "time_log") return getTimeSeriesPoints(leftSummary, "log");
    if (selectedGraphRow.type === "loss_eval") return getLossSeriesPoints(leftSummary, "eval");
    if (selectedGraphRow.type === "time_eval") return getTimeSeriesPoints(leftSummary, "eval");
    if (selectedGraphRow.type === "psnr_eval") return getEvalMetricSeriesPoints(leftSummary, "psnr");
    if (selectedGraphRow.type === "ssim_eval") return getEvalMetricSeriesPoints(leftSummary, "ssim");
    if (selectedGraphRow.type === "lpips_eval") return getEvalMetricSeriesPoints(leftSummary, "lpips");
    if (selectedGraphRow.type === "tuning") return getTuningSeriesPoints(leftSummary, selectedGraphRow.path);
    return getMajorParamSeriesPoints(leftSummary, selectedGraphRow.key, graphXMax);
  }, [leftSummary, selectedGraphRow, graphXMax]);

  const rightGraphPoints = useMemo(() => {
    if (selectedGraphRow.type === "loss_log") return getLossSeriesPoints(rightSummary, "log");
    if (selectedGraphRow.type === "time_log") return getTimeSeriesPoints(rightSummary, "log");
    if (selectedGraphRow.type === "loss_eval") return getLossSeriesPoints(rightSummary, "eval");
    if (selectedGraphRow.type === "time_eval") return getTimeSeriesPoints(rightSummary, "eval");
    if (selectedGraphRow.type === "psnr_eval") return getEvalMetricSeriesPoints(rightSummary, "psnr");
    if (selectedGraphRow.type === "ssim_eval") return getEvalMetricSeriesPoints(rightSummary, "ssim");
    if (selectedGraphRow.type === "lpips_eval") return getEvalMetricSeriesPoints(rightSummary, "lpips");
    if (selectedGraphRow.type === "tuning") return getTuningSeriesPoints(rightSummary, selectedGraphRow.path);
    return getMajorParamSeriesPoints(rightSummary, selectedGraphRow.key, graphXMax);
  }, [rightSummary, selectedGraphRow, graphXMax]);

  const allGraphPoints = useMemo(() => [...leftGraphPoints, ...rightGraphPoints], [leftGraphPoints, rightGraphPoints]);
  const graphHasData = allGraphPoints.length > 0;
  const forceZeroXStart =
    selectedGraphRow.type === "loss_log" ||
    selectedGraphRow.type === "time_log" ||
    selectedGraphRow.type === "loss_eval" ||
    selectedGraphRow.type === "time_eval" ||
    selectedGraphRow.type === "psnr_eval" ||
    selectedGraphRow.type === "ssim_eval" ||
    selectedGraphRow.type === "lpips_eval" ||
    selectedGraphRow.type === "major";
  const graphXMin = graphHasData
    ? (forceZeroXStart ? 0 : Math.min(...allGraphPoints.map((p) => p.x)))
    : 0;
  const graphXMaxUsed = graphHasData ? Math.max(...allGraphPoints.map((p) => p.x)) : 1;
  const graphYMinRaw = graphHasData ? Math.min(...allGraphPoints.map((p) => p.y)) : 0;
  const graphYMaxRaw = graphHasData ? Math.max(...allGraphPoints.map((p) => p.y)) : 1;
  const graphYPad = graphYMaxRaw === graphYMinRaw ? Math.max(1, Math.abs(graphYMaxRaw) * 0.1 || 1) : (graphYMaxRaw - graphYMinRaw) * 0.08;
  const graphYMin = graphYMinRaw - graphYPad;
  const graphYMax = graphYMaxRaw + graphYPad;
  const graphWidth = 1200;
  const graphHeight = 340;
  const graphPad = 44;
  const leftPath = toPath(leftGraphPoints, graphXMin, graphXMaxUsed, graphYMin, graphYMax, graphWidth, graphHeight, graphPad);
  const rightPath = toPath(rightGraphPoints, graphXMin, graphXMaxUsed, graphYMin, graphYMax, graphWidth, graphHeight, graphPad);
  const leftMainChangeMarkers = useMemo(() => {
    if (selectedGraphRow.type === "tuning") return getTuningChangeMarkers(leftSummary, selectedGraphRow.path);
    const steps = getTuningChangeSteps(leftSummary);
    return steps
      .map((step) => {
        const y = nearestPointValue(leftGraphPoints, step);
        if (y === null) return null;
        return { x: step, y };
      })
      .filter((p): p is GraphPoint => p !== null);
  }, [leftSummary, selectedGraphRow, leftGraphPoints]);
  const rightMainChangeMarkers = useMemo(() => {
    if (selectedGraphRow.type === "tuning") return getTuningChangeMarkers(rightSummary, selectedGraphRow.path);
    const steps = getTuningChangeSteps(rightSummary);
    return steps
      .map((step) => {
        const y = nearestPointValue(rightGraphPoints, step);
        if (y === null) return null;
        return { x: step, y };
      })
      .filter((p): p is GraphPoint => p !== null);
  }, [rightSummary, selectedGraphRow, rightGraphPoints]);
  const graphSeriesIdentical = leftGraphPoints.length === rightGraphPoints.length
    && leftGraphPoints.length > 0
    && leftGraphPoints.every((p, idx) => {
      const r = rightGraphPoints[idx];
      return !!r && p.x === r.x && Math.abs(p.y - r.y) < 1e-12;
    });

  const graphDirectionHint = useMemo(() => {
    if (selectedGraphRow.type === "psnr_eval" || selectedGraphRow.type === "ssim_eval") {
      return "Direction: higher (+) is better.";
    }
    if (selectedGraphRow.type === "loss_log" || selectedGraphRow.type === "loss_eval" || selectedGraphRow.type === "lpips_eval") {
      return "Direction: lower (-) is better.";
    }
    return null;
  }, [selectedGraphRow.type]);
  const graphXTicks = useMemo(() => {
    const tickCount = 6;
    return Array.from({ length: tickCount + 1 }, (_, i) => {
      const ratio = i / tickCount;
      return graphXMin + (graphXMaxUsed - graphXMin) * ratio;
    });
  }, [graphXMin, graphXMaxUsed]);
  const graphXValues = useMemo(() => {
    const vals = new Set<number>();
    allGraphPoints.forEach((p) => vals.add(p.x));
    return Array.from(vals).sort((a, b) => a - b);
  }, [allGraphPoints]);
  const hoverLeftValue = hoverStep === null || !showLeftSeries ? null : nearestPointValue(leftGraphPoints, hoverStep);
  const hoverRightValue = hoverStep === null || !showRightSeries ? null : nearestPointValue(rightGraphPoints, hoverStep);
  const hoverX = hoverStep === null ? null : stepToSvgX(hoverStep, graphXMin, graphXMaxUsed, graphWidth, graphPad);
  const leftTuneEndStep = typeof leftSummary?.tuning?.end_step === "number" ? leftSummary.tuning.end_step : null;
  const rightTuneEndStep = typeof rightSummary?.tuning?.end_step === "number" ? rightSummary.tuning.end_step : null;
  const leftTuneEndValue = leftTuneEndStep === null ? null : nearestPointValue(leftGraphPoints, leftTuneEndStep);
  const rightTuneEndValue = rightTuneEndStep === null ? null : nearestPointValue(rightGraphPoints, rightTuneEndStep);
  const showLeftTuneEndMarker = showTuneEndMarkers && !leftIsBase && leftTuneEndStep !== null && leftTuneEndValue !== null;
  const showRightTuneEndMarker = showTuneEndMarkers && !rightIsBase && rightTuneEndStep !== null && rightTuneEndValue !== null;
  const secondSwipeXPercent = bottomSwipePercent;
  const leftSessionLabel = leftSummary?.run_name || leftSummary?.run_id || "Left Session";
  const rightSessionLabel = rightSummary?.run_name || rightSummary?.run_id || "Right Session";
  const topLayerLabel = leftSessionLabel;
  const middleLayerLabel = rightSessionLabel;
  const groundTruthLayerLabel = "Ground truth";

  const downloadComparisonPdf = async () => {
    if (!leftSummary || !rightSummary) return;

    try {
      setComparisonPdfBusy(true);
      const pdf = new jsPDF({ format: "a4", compress: true });
      pdf.setFont("helvetica", "normal");

      const pageWidth = pdf.internal.pageSize.getWidth();
      const pageHeight = pdf.internal.pageSize.getHeight();
      const margin = 12;
      const contentWidth = pageWidth - margin * 2;
      let yPos = 15;

      const addHeading = (text: string, size = 14) => {
        if (yPos > pageHeight - 20) {
          pdf.addPage();
          yPos = 15;
        }
        pdf.setFont("helvetica", "bold");
        pdf.setFontSize(size);
        pdf.text(text, margin, yPos);
        pdf.setFont("helvetica", "normal");
        yPos += size / 2.5 + 2;
      };

      const addLine = (text: string, size = 9) => {
        pdf.setFontSize(size);
        const lines = pdf.splitTextToSize(text, contentWidth);
        pdf.text(lines, margin, yPos);
        yPos += lines.length * 4 + 1;
      };

      const toDataUrl = async (url: string): Promise<string> => {
        const res = await fetch(url);
        if (!res.ok) {
          throw new Error(`Failed to load image: ${res.status}`);
        }
        const blob = await res.blob();
        const reader = new FileReader();
        return await new Promise<string>((resolve, reject) => {
          reader.onload = () => resolve(String(reader.result || ""));
          reader.onerror = () => reject(new Error("Failed to read image blob"));
          reader.readAsDataURL(blob);
        });
      };

      const drawImageFit = (dataUrl: string, x: number, y: number, maxW: number, maxH: number) => {
        const props = pdf.getImageProperties(dataUrl);
        const srcW = Math.max(1, Number(props.width || 1));
        const srcH = Math.max(1, Number(props.height || 1));
        const scale = Math.min(maxW / srcW, maxH / srcH);
        const drawW = srcW * scale;
        const drawH = srcH * scale;
        const dx = x + (maxW - drawW) / 2;
        const dy = y + (maxH - drawH) / 2;
        pdf.setDrawColor(226, 232, 240);
        pdf.rect(x, y, maxW, maxH);
        pdf.addImage(dataUrl, "PNG", dx, dy, drawW, drawH);
      };

      addHeading("Comparison Report", 17);
      addLine(`Exported: ${new Date().toISOString()}`);
      addLine(`Left: ${leftSummary.name || leftSummary.project_id} | Run: ${leftSummary.run_name || leftSummary.run_id || "latest"}`);
      addLine(`Right: ${rightSummary.name || rightSummary.project_id} | Run: ${rightSummary.run_name || rightSummary.run_id || "latest"}`);
      addLine(`Graph parameter: ${selectedGraphRow.label}`);
      yPos += 2;

      const majorParamRows = [
        ["max_steps", "Configured max steps"],
        ["total_steps_completed", "Total steps completed"],
        ["densify_from_iter", "Start densification"],
        ["densify_until_iter", "End densification"],
        ["densification_interval", "Densification interval"],
        ["eval_interval", "Eval interval"],
        ["best_splat_interval", "Best splat interval"],
        ["auto_early_stop", "Auto early stop"],
        ["batch_size", "Batch size"],
      ] as const;

      addHeading("Major Params", 12);
      autoTable(pdf, {
        startY: yPos,
        margin: { left: margin, right: margin, top: margin, bottom: margin },
        head: [["Major Param", "Left", "Right"]],
        body: majorParamRows.map(([key, label]) => [
          label,
          fmtMajorParamValue(leftSummary.major_params?.[key]),
          fmtMajorParamValue(rightSummary.major_params?.[key]),
        ]),
        headStyles: { fontSize: 8, textColor: [255, 255, 255], fillColor: [25, 55, 120], font: "helvetica", fontStyle: "bold" },
        bodyStyles: { fontSize: 7, font: "helvetica" },
      });
      yPos = (getAutoTableFinalY(pdf) ?? yPos) + 6;

      addHeading("Metrics", 12);
      autoTable(pdf, {
        startY: yPos,
        margin: { left: margin, right: margin, top: margin, bottom: margin },
        head: [["Metric", "Left", "Right", "Delta (Right - Left)"]],
        body: metricRows.map((row) => {
          const leftVal = leftSummary.metrics?.[row.key] as number | undefined;
          const rightVal = rightSummary.metrics?.[row.key] as number | undefined;
          return [
            row.label,
            fmtMetricValue(row.key, leftVal),
            fmtMetricValue(row.key, rightVal),
            deltaText(leftVal, rightVal, row.lowerIsBetter),
          ];
        }),
        headStyles: { fontSize: 8, textColor: [255, 255, 255], fillColor: [25, 55, 120], font: "helvetica", fontStyle: "bold" },
        bodyStyles: { fontSize: 7, font: "helvetica" },
      });
      yPos = (getAutoTableFinalY(pdf) ?? yPos) + 6;

      if (milestoneKeys.length > 0) {
        addHeading("Loss Milestones", 12);
        autoTable(pdf, {
          startY: yPos,
          margin: { left: margin, right: margin, top: margin, bottom: margin },
          head: [["Milestone", "Left", "Right", "Delta"]],
          body: milestoneKeys.map((key) => {
            const leftVal = leftSummary.loss_milestones?.[key] as number | undefined;
            const rightVal = rightSummary.loss_milestones?.[key] as number | undefined;
            return [
              `Loss @ ${key.replace("loss_at_", "")}`,
              fmt(leftVal),
              fmt(rightVal),
              deltaText(leftVal, rightVal, true),
            ];
          }),
          headStyles: { fontSize: 8, textColor: [255, 255, 255], fillColor: [25, 55, 120], font: "helvetica", fontStyle: "bold" },
          bodyStyles: { fontSize: 7, font: "helvetica" },
        });
        yPos = (getAutoTableFinalY(pdf) ?? yPos) + 6;
      }

      const buildPdfTicks = (min: number, max: number, size: number, vertical = false) => {
        const span = Math.max(max - min, 1e-12);
        const minSpacing = vertical ? 16 : 42;
        const targetCount = Math.max(4, Math.min(vertical ? 12 : 14, Math.floor(size / minSpacing) + 1));
        const step = niceNumber(span / Math.max(targetCount - 1, 1));
        const start = vertical ? Math.floor(min / step) * step : Math.ceil(min / step) * step;
        const end = vertical ? Math.ceil(max / step) * step : Math.floor(max / step) * step;
        const values: number[] = [];
        for (let v = start; vertical ? v <= end + step * 0.5 : v <= end; v += step) {
          values.push(v);
          if (values.length > 18) break;
        }
        if (values.length < 2) {
          return [min, max];
        }
        return values;
      };

      const niceNumber = (value: number): number => {
        if (!Number.isFinite(value) || value <= 0) return 1;
        const exponent = Math.floor(Math.log10(value));
        const fraction = value / Math.pow(10, exponent);
        let niceFraction = 1;
        if (fraction <= 1) niceFraction = 1;
        else if (fraction <= 2) niceFraction = 2;
        else if (fraction <= 5) niceFraction = 5;
        else niceFraction = 10;
        return niceFraction * Math.pow(10, exponent);
      };
      const getPdfGraphPoints = (summary: SummaryPayload | null, row: (typeof graphMetricRows)[number]) => {
        if (row.type === "loss_log") return getLossSeriesPoints(summary, "log");
        if (row.type === "time_log") return getTimeSeriesPoints(summary, "log");
        if (row.type === "loss_eval") return getLossSeriesPoints(summary, "eval");
        if (row.type === "time_eval") return getTimeSeriesPoints(summary, "eval");
        if (row.type === "psnr_eval") return getEvalMetricSeriesPoints(summary, "psnr");
        if (row.type === "ssim_eval") return getEvalMetricSeriesPoints(summary, "ssim");
        if (row.type === "lpips_eval") return getEvalMetricSeriesPoints(summary, "lpips");
        if (row.type === "tuning") return getTuningSeriesPoints(summary, row.path);
        return getMajorParamSeriesPoints(summary, row.key, graphXMax);
      };

      const renderPdfGraphPage = (row: (typeof graphMetricRows)[number]) => {
        const leftPoints = getPdfGraphPoints(leftSummary, row);
        const rightPoints = getPdfGraphPoints(rightSummary, row);
        const allPoints = [...leftPoints, ...rightPoints];
        if (!allPoints.length) return;

        const forceZeroXStartRow =
          row.type === "loss_log" ||
          row.type === "time_log" ||
          row.type === "loss_eval" ||
          row.type === "time_eval" ||
          row.type === "psnr_eval" ||
          row.type === "ssim_eval" ||
          row.type === "lpips_eval" ||
          row.type === "major";
        const graphXMinRow = forceZeroXStartRow ? 0 : Math.min(...allPoints.map((p) => p.x));
        const graphXMaxRow = Math.max(...allPoints.map((p) => p.x));
        const graphYMinRowRaw = Math.min(...allPoints.map((p) => p.y));
        const graphYMaxRowRaw = Math.max(...allPoints.map((p) => p.y));
        const graphYPadRow = graphYMaxRowRaw === graphYMinRowRaw ? Math.max(1, Math.abs(graphYMaxRowRaw) * 0.1 || 1) : (graphYMaxRowRaw - graphYMinRowRaw) * 0.08;
        const graphYMinRow = graphYMinRowRaw - graphYPadRow;
        const graphYMaxRow = graphYMaxRowRaw + graphYPadRow;
        const leftChangeMarkers = row.type === "tuning"
          ? getTuningChangeMarkers(leftSummary, row.path)
          : getTuningChangeSteps(leftSummary)
              .map((step) => {
                const y = nearestPointValue(leftPoints, step);
                if (y === null) return null;
                return { x: step, y };
              })
              .filter((p): p is GraphPoint => p !== null);
        const rightChangeMarkers = row.type === "tuning"
          ? getTuningChangeMarkers(rightSummary, row.path)
          : getTuningChangeSteps(rightSummary)
              .map((step) => {
                const y = nearestPointValue(rightPoints, step);
                if (y === null) return null;
                return { x: step, y };
              })
              .filter((p): p is GraphPoint => p !== null);
        const leftEndValueRow = leftTuneEndStep === null ? null : nearestPointValue(leftPoints, leftTuneEndStep);
        const rightEndValueRow = rightTuneEndStep === null ? null : nearestPointValue(rightPoints, rightTuneEndStep);
        const showLeftEndRow = showTuneEndMarkers && !leftIsBase && leftTuneEndStep !== null && leftEndValueRow !== null;
        const showRightEndRow = showTuneEndMarkers && !rightIsBase && rightTuneEndStep !== null && rightEndValueRow !== null;

        pdf.addPage();
        yPos = 15;
        addHeading("Comparison Graph", 14);
        addLine(`Parameter: ${row.label}`);

        const chartTopRow = yPos + 4;
        const chartLeftRow = margin;
        const chartWidthRow = contentWidth;
        const chartHeightRow = 124;
        const chartPadLeftRow = 12;
        const chartPadRightRow = 8;
        const chartPadTopRow = 10;
        const chartPadBottomRow = 22;
        const plotLeftRow = chartLeftRow + chartPadLeftRow;
        const plotTopRow = chartTopRow + chartPadTopRow;
        const plotWidthRow = chartWidthRow - chartPadLeftRow - chartPadRightRow;
        const plotHeightRow = chartHeightRow - chartPadTopRow - chartPadBottomRow;

        pdf.setDrawColor(226, 232, 240);
        pdf.rect(chartLeftRow, chartTopRow, chartWidthRow, chartHeightRow);

        const yTicksRow = buildPdfTicks(graphYMinRow, graphYMaxRow, plotHeightRow, true);
        for (const value of yTicksRow) {
          const ratio = (graphYMaxRow - value) / (graphYMaxRow - graphYMinRow || 1);
          const y = plotTopRow + ratio * plotHeightRow;
          pdf.setDrawColor(203, 213, 225);
          pdf.line(plotLeftRow, y, plotLeftRow + plotWidthRow, y);
          pdf.setDrawColor(148, 163, 184);
          pdf.line(plotLeftRow - 4, y, plotLeftRow, y);
          pdf.setFontSize(7);
          pdf.setTextColor(100, 116, 139);
          pdf.text(fmtTickLabel(value), chartLeftRow - 1, y + 1);
        }

        const xTicksRow = buildPdfTicks(graphXMinRow, graphXMaxRow, plotWidthRow, false);
        for (let i = 0; i < xTicksRow.length; i += 1) {
          const tick = xTicksRow[i];
          const x = plotLeftRow + ((tick - graphXMinRow) / (graphXMaxRow - graphXMinRow || 1)) * plotWidthRow;
          pdf.setDrawColor(241, 245, 249);
          pdf.line(x, plotTopRow, x, plotTopRow + plotHeightRow);
          pdf.setDrawColor(148, 163, 184);
          pdf.line(x, plotTopRow + plotHeightRow, x, plotTopRow + plotHeightRow + 3);
          pdf.setFontSize(7);
          pdf.setTextColor(100, 116, 139);
          pdf.text(fmtTickLabel(tick), x - 5, plotTopRow + plotHeightRow + 7);
        }

        pdf.setDrawColor(148, 163, 184);
        pdf.line(plotLeftRow, plotTopRow + plotHeightRow, plotLeftRow + plotWidthRow, plotTopRow + plotHeightRow);
        pdf.line(plotLeftRow, plotTopRow, plotLeftRow, plotTopRow + plotHeightRow);

        const mapXRow = (x: number) => plotLeftRow + ((x - graphXMinRow) / (graphXMaxRow - graphXMinRow || 1)) * plotWidthRow;
        const mapYRow = (y: number) => plotTopRow + plotHeightRow - ((y - graphYMinRow) / (graphYMaxRow - graphYMinRow || 1)) * plotHeightRow;

        if (graphYMinRow < 0 && graphYMaxRow > 0) {
          const zeroY = mapYRow(0);
          pdf.setDrawColor(203, 213, 225);
          pdf.line(plotLeftRow, zeroY, plotLeftRow + plotWidthRow, zeroY);
        }

        if (leftPoints.length > 1) {
          pdf.setDrawColor(14, 165, 233);
          for (let i = 1; i < leftPoints.length; i += 1) {
            pdf.line(mapXRow(leftPoints[i - 1].x), mapYRow(leftPoints[i - 1].y), mapXRow(leftPoints[i].x), mapYRow(leftPoints[i].y));
          }
        }
        if (rightPoints.length > 1) {
          pdf.setDrawColor(244, 63, 94);
          for (let i = 1; i < rightPoints.length; i += 1) {
            pdf.line(mapXRow(rightPoints[i - 1].x), mapYRow(rightPoints[i - 1].y), mapXRow(rightPoints[i].x), mapYRow(rightPoints[i].y));
          }
        }

        for (const p of leftPoints) {
          pdf.setFillColor(14, 165, 233);
          pdf.circle(mapXRow(p.x), mapYRow(p.y), 0.45, "F");
        }
        for (const p of rightPoints) {
          pdf.setFillColor(244, 63, 94);
          pdf.circle(mapXRow(p.x), mapYRow(p.y), 0.45, "F");
        }

        if (showTunerChangedMarkers) {
          for (const p of leftChangeMarkers) {
            pdf.setDrawColor(14, 165, 233);
            pdf.circle(mapXRow(p.x), mapYRow(p.y), 1.0, "S");
          }
          for (const p of rightChangeMarkers) {
            pdf.setDrawColor(244, 63, 94);
            pdf.circle(mapXRow(p.x), mapYRow(p.y), 1.0, "S");
          }
        }

        if (showTuneEndMarkers) {
          if (showLeftSeries && showLeftEndRow) {
            const x = mapXRow(leftTuneEndStep) - 1;
            const y = mapYRow(leftEndValueRow as number);
            pdf.setDrawColor(139, 92, 246);
            pdf.line(x, y - 6, x, y + 6);
          }
          if (showRightSeries && showRightEndRow) {
            const x = mapXRow(rightTuneEndStep) + 1;
            const y = mapYRow(rightEndValueRow as number);
            pdf.setDrawColor(139, 92, 246);
            pdf.line(x, y - 6, x, y + 6);
          }
        }

        pdf.setTextColor(15, 23, 42);
        pdf.setFontSize(9);
        pdf.text(`Step`, plotLeftRow + plotWidthRow / 2, plotTopRow + plotHeightRow + 16, { align: "center" });
        pdf.text(row.label, chartLeftRow + 1, plotTopRow + plotHeightRow / 2, { angle: 90, align: "center" });
        pdf.text(`Left: ${leftSessionLabel}`, margin, chartTopRow + chartHeightRow + 7);
        pdf.text(`Right: ${rightSessionLabel}`, margin + 65, chartTopRow + chartHeightRow + 7);
        pdf.setFontSize(8);
        pdf.setTextColor(71, 85, 105);
        pdf.text(`Blue = left series`, margin, chartTopRow + chartHeightRow + 13);
        pdf.text(`Red = right series`, margin + 58, chartTopRow + chartHeightRow + 13);
        pdf.text(`Hollow dots = tuner changed`, margin + 112, chartTopRow + chartHeightRow + 13);
        if (showLeftEndRow || showRightEndRow) {
          pdf.text(`Purple bars = tune end`, margin + 187, chartTopRow + chartHeightRow + 13);
        }
      };

      renderPdfGraphPage(selectedGraphRow);
      for (const row of graphMetricRows) {
        if (row.key === selectedGraphRow.key) continue;
        renderPdfGraphPage(row);
      }

      // Eval Image Comparison snapshot page
      pdf.addPage();
      yPos = 15;
      addHeading("Eval Image Comparison Snapshot", 14);
      addLine(`Eval step: ${selectedEvalStep === null ? "-" : fmt(selectedEvalStep)}`);
      addLine(`Top layer: ${topLayerLabel} | Middle layer: ${middleLayerLabel}`);
      if (showGroundTruthCompare) {
        addLine(`Ground truth layer: ${groundTruthLayerLabel}`);
      }

      if (leftSelectedPreview && rightSelectedPreview) {
        const [leftImg, rightImg, gtImg] = await Promise.all([
          toDataUrl(leftSelectedPreview),
          toDataUrl(rightSelectedPreview),
          showGroundTruthCompare && fixedGroundTruthPreview ? toDataUrl(fixedGroundTruthPreview) : Promise.resolve<string>(""),
        ]);

        if (showGroundTruthCompare && gtImg) {
          pdf.setFont("helvetica", "bold");
          pdf.setFontSize(9);
          pdf.text(`1. ${groundTruthLayerLabel}`, margin, yPos);
          yPos += 3;
          drawImageFit(gtImg, margin, yPos, contentWidth, 68);
          yPos += 73;
          pdf.setFont("helvetica", "normal");
        }

        pdf.setFont("helvetica", "bold");
        pdf.setFontSize(9);
        pdf.text(`2. ${topLayerLabel}`, margin, yPos);
        yPos += 3;
        pdf.setFont("helvetica", "normal");

        drawImageFit(leftImg, margin, yPos, contentWidth, 72);
        yPos += 77;

        pdf.setFont("helvetica", "bold");
        pdf.setFontSize(9);
        pdf.text(`3. ${middleLayerLabel}`, margin, yPos);
        yPos += 3;
        pdf.setFont("helvetica", "normal");

        drawImageFit(rightImg, margin, yPos, contentWidth, 72);
      } else {
        addLine("No matching eval step snapshots are available for both projects.");
      }

      const leftToken = (leftSummary.name || leftSummary.project_id || "left").replace(/[^a-zA-Z0-9_-]+/g, "-");
      const rightToken = (rightSummary.name || rightSummary.project_id || "right").replace(/[^a-zA-Z0-9_-]+/g, "-");
      const stamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
      pdf.save(`comparison_${leftToken}_vs_${rightToken}_${stamp}.pdf`);
    } catch (err) {
      setError(friendlyErrorMessage(err, "Failed to generate comparison PDF"));
    } finally {
      setComparisonPdfBusy(false);
    }
  };

  return (
    <div className="space-y-3">
      <div className="bg-white rounded-lg border border-slate-200 p-3">
        <div className="flex items-center justify-between gap-3 mb-2">
          <h3 className="text-base font-semibold text-slate-900">Run Comparison</h3>
          <button
            type="button"
            onClick={downloadComparisonPdf}
            disabled={comparisonPdfBusy || loading || !leftSummary || !rightSummary}
            className="inline-flex items-center gap-2 px-3 py-1.5 rounded-lg bg-indigo-600 hover:bg-indigo-700 text-white text-sm disabled:opacity-60 disabled:cursor-not-allowed"
          >
            <Download className="w-4 h-4" />
            {comparisonPdfBusy ? "Preparing PDF..." : "Download Comparison PDF"}
          </button>
        </div>
        <p className="text-sm text-slate-600 mb-3">Pick two projects and compare metrics, tuning values, and preview snapshots side-by-side.</p>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          <div>
            <label className="block text-xs font-semibold text-slate-600 mb-1">Left project</label>
            <select
              value={leftId}
              onChange={(e) => {
                beginRefreshWithStableLayout();
                setLeftId(e.target.value);
              }}
              className="w-full px-3 py-1.5 border border-slate-300 rounded-lg"
            >
              {options.map((option) => (
                <option key={`left-${option.value}`} value={option.value}>{option.label}</option>
              ))}
            </select>
            <label className="block text-xs font-semibold text-slate-600 mt-3 mb-1">Left run</label>
            <select
              value={leftRunId}
              onChange={(e) => {
                beginRefreshWithStableLayout();
                setLeftRunId(e.target.value);
              }}
              className="w-full px-3 py-1.5 border border-slate-300 rounded-lg"
            >
              {leftRuns.length === 0 ? (
                <option value="">No completed sessions</option>
              ) : (
                leftRuns.map((run) => (
                  <option key={`left-run-${run.run_id}`} value={run.run_id}>
                    {(run.run_name || run.run_id) + (run.is_base ? " [BASE]" : "")}
                  </option>
                ))
              )}
            </select>
          </div>
          <div>
            <label className="block text-xs font-semibold text-slate-600 mb-1">Right project</label>
            <select
              value={rightId}
              onChange={(e) => {
                beginRefreshWithStableLayout();
                setRightId(e.target.value);
              }}
              className="w-full px-3 py-1.5 border border-slate-300 rounded-lg"
            >
              <option value="">Select project</option>
              {options.map((option) => (
                <option key={`right-${option.value}`} value={option.value}>{option.label}</option>
              ))}
            </select>
            <label className="block text-xs font-semibold text-slate-600 mt-3 mb-1">Right run</label>
            <select
              value={rightRunId}
              onChange={(e) => {
                beginRefreshWithStableLayout();
                setRightRunId(e.target.value);
              }}
              className="w-full px-3 py-1.5 border border-slate-300 rounded-lg"
            >
              {rightRuns.length === 0 ? (
                <option value="">No completed sessions</option>
              ) : (
                rightRuns.map((run) => (
                  <option key={`right-run-${run.run_id}`} value={run.run_id}>
                    {(run.run_name || run.run_id) + (run.is_base ? " [BASE]" : "")}
                  </option>
                ))
              )}
            </select>
          </div>
        </div>
      </div>

      {error && <div className="bg-rose-50 border border-rose-200 text-rose-700 px-4 py-3 rounded-lg text-sm">{error}</div>}

      {loading && <div className="text-sm text-slate-500">Loading comparison data...</div>}
      {isViewSwitching && !loading && <div className="text-sm text-slate-500">Updating comparison view...</div>}

      {!leftSummary && !rightSummary && (loading || isViewSwitching) && contentHoldHeight > 0 && (
        <div className="bg-white rounded-lg border border-slate-200" style={{ minHeight: `${contentHoldHeight}px` }} />
      )}

      {!loading && !isViewSwitching && leftSummary && rightSummary && (
        <div ref={comparisonContentRef}>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
            <div className="bg-white rounded-lg border border-slate-200 p-3">
              <p className="text-sm font-semibold text-slate-800">{leftSummary.name || leftSummary.project_id}</p>
              <p className="text-xs text-slate-500">run: {leftSummary.run_name || leftSummary.run_id || "latest"}</p>
              <p className="text-xs text-slate-500">mode: {leftSummary.mode || "-"} | engine: {leftSummary.engine || "-"}</p>
              <p className="text-xs text-slate-500">eval points: {leftSummary.eval_points ?? 0} (number of evaluation records)</p>
              <p className="text-xs text-slate-500">tune end step: {leftSummary.tuning?.end_step ?? "-"}</p>
              {!leftHasSummaryData && (
                <p className="text-xs text-amber-700 mt-1">No completed summary yet for this project.</p>
              )}
            </div>
            <div className="bg-white rounded-lg border border-slate-200 p-3">
              <p className="text-sm font-semibold text-slate-800">{rightSummary.name || rightSummary.project_id}</p>
              <p className="text-xs text-slate-500">run: {rightSummary.run_name || rightSummary.run_id || "latest"}</p>
              <p className="text-xs text-slate-500">mode: {rightSummary.mode || "-"} | engine: {rightSummary.engine || "-"}</p>
              <p className="text-xs text-slate-500">eval points: {rightSummary.eval_points ?? 0} (number of evaluation records)</p>
              <p className="text-xs text-slate-500">tune end step: {rightSummary.tuning?.end_step ?? "-"}</p>
              {!rightHasSummaryData && (
                <p className="text-xs text-amber-700 mt-1">No completed summary yet for this project.</p>
              )}
            </div>
          </div>

          <div className="bg-white rounded-lg border border-slate-200 overflow-hidden">
            <table className="w-full text-sm">
              <thead className="bg-slate-50 text-slate-700">
                <tr>
                  <th className="text-left px-3 py-2">AI Setting</th>
                  <th className="text-left px-3 py-2">Left</th>
                  <th className="text-left px-3 py-2">Right</th>
                </tr>
              </thead>
              <tbody>
                {[
                  ["ai_input_mode", "AI input mode"],
                  ["baseline_session_id", "Baseline session"],
                  ["selected_preset", "Selected preset"],
                  ["heuristic_preset", "Heuristic preset"],
                  ["cache_used", "Feature cache used"],
                  ["reward", "End-of-run reward"],
                  ["reward_label", "Reward outcome"],
                ].map(([key, label]) => {
                  const leftVal = leftSummary.ai_insights?.[key as keyof NonNullable<SummaryPayload["ai_insights"]>];
                  const rightVal = rightSummary.ai_insights?.[key as keyof NonNullable<SummaryPayload["ai_insights"]>];
                  return (
                    <tr key={key} className="border-t border-slate-100">
                      <td className="px-3 py-1.5 text-slate-700">{label}</td>
                      <td className="px-3 py-1.5 text-slate-900">{fmtLooseValue(leftVal)}</td>
                      <td className="px-3 py-1.5 text-slate-900">{fmtLooseValue(rightVal)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          {aiInitialParamKeys.length > 0 && (
            <div className="bg-white rounded-lg border border-slate-200 overflow-hidden">
              <table className="w-full text-sm">
                <thead className="bg-slate-50 text-slate-700">
                  <tr>
                    <th className="text-left px-3 py-2">AI Initial Parameter</th>
                    <th className="text-left px-3 py-2">Left</th>
                    <th className="text-left px-3 py-2">Right</th>
                  </tr>
                </thead>
                <tbody>
                  {aiInitialParamKeys.map((key) => {
                    const leftVal = leftSummary.ai_insights?.initial_params?.[key];
                    const rightVal = rightSummary.ai_insights?.initial_params?.[key];
                    return (
                      <tr key={key} className="border-t border-slate-100">
                        <td className="px-3 py-1.5 text-slate-700">{key}</td>
                        <td className="px-3 py-1.5 text-slate-900">{fmtLooseValue(leftVal)}</td>
                        <td className="px-3 py-1.5 text-slate-900">{fmtLooseValue(rightVal)}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}

          <div className="bg-white rounded-lg border border-slate-200 overflow-hidden">
            <table className="w-full text-sm">
              <thead className="bg-slate-50 text-slate-700">
                <tr>
                  <th className="text-left px-3 py-2">Major Param</th>
                  <th className="text-left px-3 py-2">Left</th>
                  <th className="text-left px-3 py-2">Right</th>
                </tr>
              </thead>
              <tbody>
                {[
                  ["max_steps", "Configured max steps"],
                  ["total_steps_completed", "Total steps completed"],
                  ["densify_from_iter", "Start densification"],
                  ["densify_until_iter", "End densification"],
                  ["densification_interval", "Densification interval"],
                  ["eval_interval", "Eval interval"],
                  ["best_splat_interval", "Best splat interval"],
                  ["auto_early_stop", "Auto early stop"],
                  ["batch_size", "Batch size"],
                ].map(([key, label]) => (
                  <tr key={key} className="border-t border-slate-100">
                    <td className="px-3 py-1.5 text-slate-700">{label}</td>
                    <td className="px-3 py-1.5 text-slate-900">{fmtMajorParamValue(leftSummary.major_params?.[key])}</td>
                    <td className="px-3 py-1.5 text-slate-900">{fmtMajorParamValue(rightSummary.major_params?.[key])}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="bg-white rounded-lg border border-slate-200 overflow-hidden">
            <table className="w-full text-sm">
              <thead className="bg-slate-50 text-slate-700">
                <tr>
                  <th className="text-left px-3 py-2">Metric</th>
                  <th className="text-left px-3 py-2">Left</th>
                  <th className="text-left px-3 py-2">Right</th>
                  <th className="text-left px-3 py-2">Delta (Right - Left)</th>
                </tr>
              </thead>
              <tbody>
                {metricRows.map((row) => {
                  const leftVal = leftSummary.metrics?.[row.key] as number | undefined;
                  const rightVal = rightSummary.metrics?.[row.key] as number | undefined;
                  return (
                    <tr key={row.key} className="border-t border-slate-100">
                      <td className="px-3 py-1.5 text-slate-700">{row.label}</td>
                      <td className="px-3 py-1.5 text-slate-900">{fmtMetricValue(row.key, leftVal)}</td>
                      <td className="px-3 py-1.5 text-slate-900">{fmtMetricValue(row.key, rightVal)}</td>
                      <td className="px-3 py-1.5 text-slate-600">{deltaText(leftVal, rightVal, row.lowerIsBetter)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          {milestoneKeys.length > 0 && (
            <div className="bg-white rounded-lg border border-slate-200 overflow-hidden">
              <table className="w-full text-sm">
                <thead className="bg-slate-50 text-slate-700">
                  <tr>
                    <th className="text-left px-3 py-2">Loss Milestone (500-step)</th>
                    <th className="text-left px-3 py-2">Left</th>
                    <th className="text-left px-3 py-2">Right</th>
                    <th className="text-left px-3 py-2">Delta (Right - Left)</th>
                  </tr>
                </thead>
                <tbody>
                  {milestoneKeys.map((key) => {
                    const leftVal = leftSummary.loss_milestones?.[key] as number | undefined;
                    const rightVal = rightSummary.loss_milestones?.[key] as number | undefined;
                    const stepLabel = key.replace("loss_at_", "");
                    return (
                      <tr key={key} className="border-t border-slate-100">
                        <td className="px-3 py-1.5 text-slate-700">Loss @ {stepLabel}</td>
                        <td className="px-3 py-1.5 text-slate-900">{fmt(leftVal)}</td>
                        <td className="px-3 py-1.5 text-slate-900">{fmt(rightVal)}</td>
                        <td className="px-3 py-1.5 text-slate-600">{deltaText(leftVal, rightVal, true)}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}

          <div className="bg-white rounded-lg border border-slate-200 p-3">
            <div className="flex flex-wrap items-end justify-between gap-2 mb-2">
              <div>
                <p className="text-sm font-semibold text-slate-800">Comparison Line Graph</p>
                <p className="text-xs text-slate-500">Select loss, tuning, or major params from dropdown. (X: Step, Y: Value). For tuning params, outlined points mark exact tuner-change steps.</p>
              </div>
              <div className="flex flex-wrap items-end gap-2">
                <div className="min-w-[220px]">
                  <label className="block text-xs font-semibold text-slate-600 mb-1">Graph parameter</label>
                  <select
                    value={selectedGraphMetric}
                    onChange={(e) => {
                      setHoverStep(null);
                      setSelectedGraphMetric(e.target.value);
                    }}
                    className="w-full px-3 py-2 border border-slate-300 rounded-lg"
                  >
                    {graphMetricRows.map((row) => (
                      <option key={row.key} value={row.key}>{row.label}</option>
                    ))}
                  </select>
                </div>
                <div className="flex items-center gap-2 text-[10px] leading-tight pb-1">
                  <button
                    type="button"
                    onClick={() => setShowLeftSeries((prev) => !prev)}
                    aria-pressed={showLeftSeries}
                    className={`flex items-center gap-1 ${showLeftSeries ? "text-slate-700" : "text-slate-400"}`}
                    title="Toggle left series"
                  >
                    <span className={`inline-block w-2 h-2 rounded-full ${showLeftSeries ? "bg-sky-500" : "bg-slate-300"}`} />
                    <span>{leftSessionLabel}</span>
                  </button>
                  <button
                    type="button"
                    onClick={() => setShowRightSeries((prev) => !prev)}
                    aria-pressed={showRightSeries}
                    className={`flex items-center gap-1 ${showRightSeries ? "text-slate-700" : "text-slate-400"}`}
                    title="Toggle right series"
                  >
                    <span className={`inline-block w-2 h-2 rounded-full ${showRightSeries ? "bg-rose-500" : "bg-slate-300"}`} />
                    <span>{rightSessionLabel}</span>
                  </button>
                  {(leftMainChangeMarkers.length > 0 || rightMainChangeMarkers.length > 0) && (
                    <button
                      type="button"
                      onClick={() => setShowTunerChangedMarkers((prev) => !prev)}
                      aria-pressed={showTunerChangedMarkers}
                      className={`flex items-center gap-1 ${showTunerChangedMarkers ? "text-slate-700" : "text-slate-400"}`}
                      title="Toggle tuner-changed markers"
                    >
                      <span className={`inline-block w-2 h-2 rounded-full border ${showTunerChangedMarkers ? "border-slate-500 bg-white" : "border-slate-300 bg-slate-100"}`} />
                      <span>tuner changed value at this step</span>
                    </button>
                  )}
                  <button
                    type="button"
                    onClick={() => setShowTuneEndMarkers((prev) => !prev)}
                    aria-pressed={showTuneEndMarkers}
                    className={`flex items-center gap-1 ${showTuneEndMarkers ? "text-slate-700" : "text-slate-400"}`}
                    title="Toggle tune-end markers"
                  >
                    <span className={`inline-block w-[1.5px] h-2 rounded-sm ${showTuneEndMarkers ? "bg-violet-500" : "bg-slate-300"}`} />
                    <span>tune-end step</span>
                  </button>
                  {graphSeriesIdentical && (
                    <span className="text-amber-700">Both series overlap exactly for this parameter.</span>
                  )}
                </div>
              </div>
            </div>
            {graphHasData ? (
              <div className="w-full overflow-x-auto">
                <svg
                  className="w-full min-w-[900px] h-[340px]"
                  viewBox={`0 0 ${graphWidth} ${graphHeight}`}
                  role="img"
                  aria-label="Unified comparison graph"
                  onMouseMove={(e) => {
                    if (!graphXValues.length) return;
                    const rect = e.currentTarget.getBoundingClientRect();
                    const px = ((e.clientX - rect.left) / rect.width) * graphWidth;
                    const clamped = Math.max(graphPad, Math.min(graphWidth - graphPad, px));
                    const ratio = (clamped - graphPad) / (graphWidth - graphPad * 2 || 1);
                    const stepGuess = graphXMin + ratio * (graphXMaxUsed - graphXMin);
                    let nearest = graphXValues[0];
                    let bestDist = Math.abs(stepGuess - nearest);
                    for (let i = 1; i < graphXValues.length; i += 1) {
                      const d = Math.abs(stepGuess - graphXValues[i]);
                      if (d < bestDist) {
                        bestDist = d;
                        nearest = graphXValues[i];
                      }
                    }
                    setHoverStep(nearest);
                  }}
                  onMouseLeave={() => setHoverStep(null)}
                >
                  {[0, 1, 2, 3, 4].map((i) => {
                    const ratio = i / 4;
                    const y = graphPad + (graphHeight - graphPad * 2) * ratio;
                    const value = graphYMax - (graphYMax - graphYMin) * ratio;
                    return (
                      <g key={`grid-${i}`}>
                        <line x1={graphPad} y1={y} x2={graphWidth - graphPad} y2={y} stroke="#cbd5e1" strokeWidth="1.1" />
                        <line x1={graphPad - 4} y1={y} x2={graphPad} y2={y} stroke="#94a3b8" strokeWidth="1.1" />
                        <text x={graphPad - 8} y={y + 4} fill="#64748b" fontSize="11" textAnchor="end">{fmt(value)}</text>
                      </g>
                    );
                  })}
                  {graphXTicks.map((tick, idx) => {
                    const x = stepToSvgX(tick, graphXMin, graphXMaxUsed, graphWidth, graphPad);
                  {graphDirectionHint && (
                    <p className="mt-2 text-xs text-slate-600">{graphDirectionHint}</p>
                  )}
                    return (
                      <line
                        key={`xgrid-${idx}`}
                        x1={x}
                        y1={graphPad}
                        x2={x}
                        y2={graphHeight - graphPad}
                        stroke="#d1d5db"
                        strokeWidth="1.1"
                      />
                    );
                  })}
                  <line x1={graphPad} y1={graphHeight - graphPad} x2={graphWidth - graphPad} y2={graphHeight - graphPad} stroke="#94a3b8" strokeWidth="1.2" />
                  <line x1={graphPad} y1={graphPad} x2={graphPad} y2={graphHeight - graphPad} stroke="#94a3b8" strokeWidth="1.2" />

                  {graphXTicks.map((tick, idx) => {
                    const x = stepToSvgX(tick, graphXMin, graphXMaxUsed, graphWidth, graphPad);
                    return (
                      <g key={`xtick-${idx}`}>
                        <line x1={x} y1={graphHeight - graphPad} x2={x} y2={graphHeight - graphPad + 6} stroke="#94a3b8" strokeWidth="1" />
                        <text x={x} y={graphHeight - graphPad + 18} fill="#64748b" fontSize="11" textAnchor="middle">{fmtTickLabel(tick)}</text>
                      </g>
                    );
                  })}

                  {showLeftSeries && leftPath && <path d={leftPath} fill="none" stroke="#0ea5e9" strokeWidth="1.8" strokeLinejoin="round" strokeLinecap="round" />}
                  {showRightSeries && rightPath && (
                    <path
                      d={rightPath}
                      fill="none"
                      stroke="#f43f5e"
                      strokeWidth="1.7"
                      strokeDasharray={graphSeriesIdentical ? "6 5" : undefined}
                      strokeLinejoin="round"
                      strokeLinecap="round"
                    />
                  )}

                  {showTunerChangedMarkers && showLeftSeries && leftMainChangeMarkers.map((p, idx) => (
                    <circle
                      key={`lmc-${idx}`}
                      cx={stepToSvgX(p.x, graphXMin, graphXMaxUsed, graphWidth, graphPad)}
                      cy={valueToSvgY(p.y, graphYMin, graphYMax, graphHeight, graphPad)}
                      r="4.8"
                      fill="#ffffff"
                      stroke="#0ea5e9"
                      strokeWidth="2"
                    />
                  ))}
                  {showTunerChangedMarkers && showRightSeries && rightMainChangeMarkers.map((p, idx) => (
                    <circle
                      key={`rmc-${idx}`}
                      cx={stepToSvgX(p.x, graphXMin, graphXMaxUsed, graphWidth, graphPad)}
                      cy={valueToSvgY(p.y, graphYMin, graphYMax, graphHeight, graphPad)}
                      r="4.8"
                      fill="#ffffff"
                      stroke="#f43f5e"
                      strokeWidth="2"
                    />
                  ))}

                  {showLeftTuneEndMarker && showLeftSeries && (
                    <line
                      x1={stepToSvgX(leftTuneEndStep, graphXMin, graphXMaxUsed, graphWidth, graphPad) - 1}
                      y1={valueToSvgY(leftTuneEndValue, graphYMin, graphYMax, graphHeight, graphPad) - 7}
                      x2={stepToSvgX(leftTuneEndStep, graphXMin, graphXMaxUsed, graphWidth, graphPad) - 1}
                      y2={valueToSvgY(leftTuneEndValue, graphYMin, graphYMax, graphHeight, graphPad) + 7}
                      stroke="#8b5cf6"
                      strokeWidth="1.6"
                      strokeLinecap="round"
                    />
                  )}
                  {showRightTuneEndMarker && showRightSeries && (
                    <line
                      x1={stepToSvgX(rightTuneEndStep, graphXMin, graphXMaxUsed, graphWidth, graphPad) + 1}
                      y1={valueToSvgY(rightTuneEndValue, graphYMin, graphYMax, graphHeight, graphPad) - 7}
                      x2={stepToSvgX(rightTuneEndStep, graphXMin, graphXMaxUsed, graphWidth, graphPad) + 1}
                      y2={valueToSvgY(rightTuneEndValue, graphYMin, graphYMax, graphHeight, graphPad) + 7}
                      stroke="#8b5cf6"
                      strokeWidth="1.6"
                      strokeLinecap="round"
                    />
                  )}

                  {showLeftSeries && leftGraphPoints.map((p, idx) => (
                    <circle key={`lp-${idx}`} cx={stepToSvgX(p.x, graphXMin, graphXMaxUsed, graphWidth, graphPad)} cy={valueToSvgY(p.y, graphYMin, graphYMax, graphHeight, graphPad)} r="2.6" fill="#0ea5e9" />
                  ))}
                  {showRightSeries && rightGraphPoints.map((p, idx) => (
                    <circle key={`rp-${idx}`} cx={stepToSvgX(p.x, graphXMin, graphXMaxUsed, graphWidth, graphPad)} cy={valueToSvgY(p.y, graphYMin, graphYMax, graphHeight, graphPad)} r="2.6" fill="#f43f5e" />
                  ))}

                  {hoverX !== null && (
                    <line x1={hoverX} y1={graphPad} x2={hoverX} y2={graphHeight - graphPad} stroke="#64748b" strokeDasharray="4 4" strokeWidth="1" />
                  )}

                  {hoverStep !== null && hoverX !== null && (
                    <g>
                      <rect
                        x={Math.max(graphPad + 8, Math.min(graphWidth - graphPad - 240, hoverX + 10))}
                        y={graphPad + 8}
                        width="232"
                        height="56"
                        rx="6"
                        fill="#ffffff"
                        stroke="#cbd5e1"
                      />
                      <text
                        x={Math.max(graphPad + 18, Math.min(graphWidth - graphPad - 230, hoverX + 20))}
                        y={graphPad + 27}
                        fill="#0f172a"
                        fontSize="12"
                        fontWeight="600"
                      >
                        Step {fmt(hoverStep)}
                      </text>
                      <text
                        x={Math.max(graphPad + 18, Math.min(graphWidth - graphPad - 230, hoverX + 20))}
                        y={graphPad + 43}
                        fill="#0369a1"
                        fontSize="11"
                      >
                        Left: {fmt(hoverLeftValue)}
                      </text>
                      <text
                        x={Math.max(graphPad + 120, Math.min(graphWidth - graphPad - 120, hoverX + 122))}
                        y={graphPad + 43}
                        fill="#be123c"
                        fontSize="11"
                      >
                        Right: {fmt(hoverRightValue)}
                      </text>
                    </g>
                  )}

                  <text x={graphPad} y={graphHeight - 8} fill="#64748b" fontSize="11">Step {fmt(graphXMin)}</text>
                  <text x={graphWidth - graphPad - 78} y={graphHeight - 8} fill="#64748b" fontSize="11">Step {fmt(graphXMaxUsed)}</text>
                </svg>
              </div>
            ) : (
              <p className="text-xs text-slate-500">No graph data available for this parameter in the selected projects.</p>
            )}
          </div>

          <div className="bg-white rounded-lg border border-slate-200 p-4 space-y-4">
            <div className="flex flex-wrap items-end justify-between gap-3">
              <div>
                <p className="text-sm font-semibold text-slate-800">Eval Step Image Comparison</p>
                <p className="text-xs text-slate-500">Choose one eval step; both sessions show that same step. Left side is Ground truth, right side is the compared output.</p>
              </div>
              <div className="min-w-[220px]">
                <label className="block text-xs font-semibold text-slate-600 mb-1">Eval step</label>
                <select
                  value={selectedEvalStep === null ? "" : String(selectedEvalStep)}
                  onChange={(e) => {
                    const raw = e.target.value;
                    if (!raw) {
                      setSelectedEvalStep(null);
                      return;
                    }
                    setSelectedEvalStep(Number.parseInt(raw, 10));
                  }}
                  className="w-full px-3 py-2 border border-slate-300 rounded-lg"
                  disabled={!commonEvalSteps.length}
                >
                  {!commonEvalSteps.length && <option value="">No common eval steps</option>}
                  {commonEvalSteps.map((step) => (
                    <option key={`eval-step-${step}`} value={step}>Step {fmt(step)}</option>
                  ))}
                </select>
                <label className="mt-2 inline-flex items-center gap-2 text-xs text-slate-700">
                  <input
                    type="checkbox"
                    checked={showGroundTruthCompare}
                    onChange={(e) => setShowGroundTruthCompare(e.target.checked)}
                    className="rounded border-slate-300"
                  />
                  Compare with ground truth (extra bottom swipe)
                </label>
              </div>
            </div>

            {leftSelectedPreview && rightSelectedPreview ? (
              <div className="space-y-3">
                <div
                  ref={swipeAreaRef}
                  className="relative w-full overflow-hidden rounded-lg border border-slate-200 bg-slate-50 touch-none select-none"
                >
                  <img
                    src={rightSelectedPreview}
                    alt={`Right session at step ${selectedEvalStep ?? ""}`}
                    className={`block w-full h-auto ${showGroundTruthCompare ? "opacity-0 pointer-events-none" : ""}`}
                    draggable={false}
                  />
                  {!showGroundTruthCompare && (
                    <div className="absolute bottom-2 right-2 px-2 py-1 rounded bg-slate-900/75 text-white text-[11px] font-semibold whitespace-nowrap z-20">
                      {middleLayerLabel}
                    </div>
                  )}
                  {showGroundTruthCompare && (
                    <>
                      <div className="absolute inset-0 overflow-hidden z-10">
                        <img
                          src={fixedGroundTruthPreview || leftSelectedPreview}
                          alt="Ground truth base layer"
                          className="block w-full h-full object-cover"
                          style={{ transform: "translateX(50%)" }}
                          draggable={false}
                        />
                        <div className="absolute bottom-2 right-2 px-2 py-1 rounded bg-slate-900/75 text-white text-[11px] font-semibold whitespace-nowrap">
                          {groundTruthLayerLabel}
                        </div>
                      </div>
                      <div
                        className="absolute inset-0 overflow-hidden z-20"
                        style={{ clipPath: `inset(0 ${100 - secondSwipeXPercent}% 0 0)` }}
                      >
                        <img
                          src={rightSelectedPreview}
                          alt={`Middle layer at step ${selectedEvalStep ?? ""}`}
                          className="block w-full h-full object-cover"
                          draggable={false}
                        />
                        <div className="absolute bottom-2 right-2 px-2 py-1 rounded bg-slate-900/75 text-white text-[11px] font-semibold whitespace-nowrap">
                          {middleLayerLabel}
                        </div>
                      </div>
                    </>
                  )}
                  <div
                    className={`absolute inset-0 overflow-hidden z-40 ${isSwipeDragging ? "cursor-ew-resize" : "cursor-col-resize"}`}
                    style={{ clipPath: `inset(0 ${100 - swipePercent}% 0 0)` }}
                    onPointerDown={(e) => {
                      setIsSwipeDragging(true);
                      updateSwipeFromClientX(e.clientX);
                      e.currentTarget.setPointerCapture(e.pointerId);
                    }}
                    onPointerMove={(e) => {
                      if (!isSwipeDragging) return;
                      updateSwipeFromClientX(e.clientX);
                    }}
                    onPointerUp={(e) => {
                      setIsSwipeDragging(false);
                      if (e.currentTarget.hasPointerCapture(e.pointerId)) {
                        e.currentTarget.releasePointerCapture(e.pointerId);
                      }
                    }}
                    onPointerCancel={(e) => {
                      setIsSwipeDragging(false);
                      if (e.currentTarget.hasPointerCapture(e.pointerId)) {
                        e.currentTarget.releasePointerCapture(e.pointerId);
                      }
                    }}
                  >
                    <img src={leftSelectedPreview} alt={`Left session at step ${selectedEvalStep ?? ""}`} className="block w-full h-auto" draggable={false} />
                    <div className="absolute bottom-2 left-2 px-2 py-1 rounded bg-sky-950/75 text-white text-[11px] font-semibold whitespace-nowrap">
                      {topLayerLabel}
                    </div>
                  </div>

                  <div
                    className="absolute top-0 bottom-0 w-[2px] bg-white/90 shadow pointer-events-none z-50"
                    style={{ left: `${swipePercent}%`, transform: "translateX(-1px)" }}
                  />
                  <div
                    className="absolute top-1/2 w-4 h-4 rounded-full border border-white bg-sky-500 shadow pointer-events-none z-50"
                    style={{ left: `${swipePercent}%`, transform: "translate(-50%, -50%)" }}
                  />

                  {showGroundTruthCompare && (
                    <>
                      <div
                        className={`absolute top-0 bottom-0 w-10 -ml-5 touch-none z-[55] ${isSecondSwipeDragging ? "cursor-ew-resize" : "cursor-col-resize"}`}
                        style={{ left: `${secondSwipeXPercent}%` }}
                        onPointerDown={(e) => {
                          e.stopPropagation();
                          setIsSecondSwipeDragging(true);
                          updateSecondSwipeFromClientX(e.clientX);
                          e.currentTarget.setPointerCapture(e.pointerId);
                        }}
                        onPointerMove={(e) => {
                          e.stopPropagation();
                          if (!isSecondSwipeDragging) return;
                          updateSecondSwipeFromClientX(e.clientX);
                        }}
                        onPointerUp={(e) => {
                          e.stopPropagation();
                          setIsSecondSwipeDragging(false);
                          if (e.currentTarget.hasPointerCapture(e.pointerId)) {
                            e.currentTarget.releasePointerCapture(e.pointerId);
                          }
                        }}
                        onPointerCancel={(e) => {
                          e.stopPropagation();
                          setIsSecondSwipeDragging(false);
                          if (e.currentTarget.hasPointerCapture(e.pointerId)) {
                            e.currentTarget.releasePointerCapture(e.pointerId);
                          }
                        }}
                      />
                      <div
                        className="absolute top-0 bottom-0 w-[2px] bg-amber-300/95 shadow pointer-events-none z-35"
                        style={{ left: `${secondSwipeXPercent}%`, transform: "translateX(-1px)" }}
                      />
                      <div
                        className="absolute top-1/2 w-3.5 h-3.5 rounded-full border border-white bg-amber-500 shadow pointer-events-none z-35"
                        style={{ left: `${secondSwipeXPercent}%`, transform: "translate(-50%, -50%)" }}
                      />
                    </>
                  )}

                </div>

                <div className="space-y-1">
                  <div className="flex items-center justify-between text-xs text-slate-600">
                    <span>Swipe: {topLayerLabel}</span>
                    <span>{Math.round(swipePercent)}%</span>
                  </div>
                  <input
                    type="range"
                    min={0}
                    max={100}
                    value={swipePercent}
                    onChange={(e) => setSwipePercent(Number.parseFloat(e.target.value))}
                    onMouseDown={() => setIsSwipeDragging(true)}
                    onMouseUp={() => setIsSwipeDragging(false)}
                    onTouchStart={() => setIsSwipeDragging(true)}
                    onTouchEnd={() => setIsSwipeDragging(false)}
                    className="w-full"
                  />
                </div>

                {showGroundTruthCompare && (
                  <div className="space-y-1 pt-2 border-t border-slate-200">
                    <div className="flex items-center justify-between text-xs text-slate-600">
                      <span>Swipe: {middleLayerLabel}</span>
                      <span>{Math.round(bottomSwipePercent)}%</span>
                    </div>
                    <input
                      type="range"
                      min={0}
                      max={100}
                      value={bottomSwipePercent}
                      onChange={(e) => setBottomSwipePercent(Number.parseFloat(e.target.value))}
                      className="w-full"
                    />
                  </div>
                )}
              </div>
            ) : (
              <p className="text-xs text-slate-500">Matching preview images at the same eval step are not available for both projects yet.</p>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
