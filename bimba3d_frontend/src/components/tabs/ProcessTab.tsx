import { useMemo, useState, useEffect, useRef, useCallback } from "react";
import { Play, Settings2, Layers, Map as MapIcon, Boxes, Check, X, Clock, Square, Download, Info as LucideInfo } from "lucide-react";
import Map, { NavigationControl } from "react-map-gl/maplibre";
import maplibregl from "maplibre-gl";
import { jsPDF } from "jspdf";
import autoTable from "jspdf-autotable";
import { api } from "../../api/client";
import ViewerTab from "./ViewerTab";
import SparseViewer from "../SparseViewer.tsx";
import ConfirmModal from "../ConfirmModal";
import CoreAiSessionTestControls, { type SessionExecutionMode } from "./CoreAiSessionTestControls";

// Small Info wrapper: render a smaller info icon throughout the modal
const Info = (props: React.ComponentProps<typeof LucideInfo>) => (
  <LucideInfo className={props.className ? props.className + " w-3 h-3" : "w-3 h-3"} {...props} />
);

interface ProcessTabProps {
  projectId: string;
}

interface SnapshotEntry {
  name: string;
  url: string;
  step: number | null;
  size?: number;
  format?: string;
}

interface PreviewFile {
  name: string;
  url: string;
}

interface EngineOutputBundle {
  name: string;
  label: string;
  hasModel: boolean;
  finalModelUrl: string | null;
  bestModelUrl: string | null;
  previews: PreviewFile[];
  snapshots: SnapshotEntry[];
}

interface TelemetryTrainingRow {
  timestamp?: string | null;
  step?: number | null;
  max_steps?: number | null;
  loss?: number | null;
  elapsed_seconds?: number | null;
  eta?: string | null;
  speed?: string | null;
  source?: string | null;
}

interface TelemetryEvalRow {
  step?: number | null;
  psnr?: number | null;
  lpips?: number | null;
  ssim?: number | null;
  num_gaussians?: number | null;
}

interface TelemetryEventRow {
  timestamp?: string | null;
  type?: string | null;
  step?: number | null;
  summary?: string | null;
}

interface TelemetryPayload {
  project_id: string;
  project_name?: string | null;
  run_id?: string | null;
  generated_at?: string;
  training_rows?: TelemetryTrainingRow[];
  event_rows?: TelemetryEventRow[];
  eval_rows?: TelemetryEvalRow[];
  latest_eval?: TelemetryEvalRow | null;
  training_summary?: {
    first_step?: number | null;
    last_step?: number | null;
    start_timestamp?: string | null;
    end_timestamp?: string | null;
    total_elapsed_seconds?: number | null;
    row_count?: number | null;
    best_loss?: number | null;
    best_loss_step?: number | null;
  };
  run_config?: {
    requested_params?: Record<string, unknown>;
    resolved_params?: Record<string, unknown>;
    shared_config_version?: number | null;
    active_sparse_shared_version?: number | null;
    run_shared_config_version?: number | null;
    shared_outdated?: boolean | null;
    base_session_id?: string | null;
    effective_shared_config?: Record<string, unknown> | null;
  } | null;
  ai_insights?: {
    ai_input_mode?: string | null;
    baseline_session_id?: string | null;
    selected_preset?: string | null;
    heuristic_preset?: string | null;
    cache_used?: boolean | null;
    reward?: number | null;
    reward_positive?: boolean | null;
    reward_label?: string | null;
    reward_mode?: string | null;
    reward_preset?: string | null;
    feature_source?: string | null;
    initial_params?: Record<string, unknown>;
    feature_details?: Record<string, unknown>;
    feature_sources?: Record<string, unknown>;
    missing_flags?: Record<string, unknown>;
    learn_snapshot?: Record<string, unknown>;
  } | null;
  status?: {
    stage?: string | null;
    message?: string | null;
    currentStep?: number | null;
    maxSteps?: number | null;
    current_loss?: number | null;
  };
}

interface MergeReportSourceDetail {
  relative_path?: string;
  used?: boolean;
  reason?: string;
  points?: number;
  aligned?: boolean;
  overlap_images?: number;
  scale?: number;
}

interface SparseMergeReport {
  anchor_relative_path?: string;
  selected_relative_paths?: string[];
  merged_points?: number;
  created_at?: number;
  alignment?: string;
  source_details?: MergeReportSourceDetail[];
}

interface ProjectRunInfo {
  run_id: string;
  run_name?: string | null;
  saved_at?: string | null;
  mode?: string | null;
  stage?: string | null;
  engine?: string | null;
  session_status?: "completed" | "pending" | string;
  max_steps?: number | null;
  tune_scope?: string | null;
  trend_scope?: string | null;
  adaptive_event_count?: number;
  has_run_config?: boolean;
  has_run_log?: boolean;
  is_base?: boolean;
  shared_config_version?: number | null;
  active_sparse_shared_version?: number | null;
  shared_outdated?: boolean;
  batch_plan_id?: string | null;
  batch_index?: number | null;
  batch_total?: number | null;
}

type NewSessionConfigSource = "current" | "defaults";

type TrainingEngine = "gsplat" | "litegs";
type TuneScope = "core_individual" | "core_only" | "core_ai_optimization" | "core_individual_plus_strategy";
type TrendScope = "run" | "phase";
type AiInputMode = "" | "exif_only" | "exif_plus_flight_plan" | "exif_plus_flight_plan_plus_external";
type AiSelectorStrategy = "preset_bias" | "continuous_bandit_linear";
type RunJitterMode = "fixed" | "random";
type TuneScopeDropdownValue =
  | TuneScope
  | "core_ai_optimization__exif_only"
  | "core_ai_optimization__exif_plus_flight_plan"
  | "core_ai_optimization__exif_plus_flight_plan_plus_external";
type StartModelMode = "scratch" | "reuse";

interface ReusableModelEntry {
  model_id: string;
  model_name?: string | null;
  source_project_id?: string | null;
  source_run_id?: string | null;
  created_at?: string | null;
  ai_profile?: {
    pipeline_kind?: "controller" | "input_mode" | null;
    ai_input_mode?: AiInputMode | null;
    ai_selector_strategy?: AiSelectorStrategy | null;
  } | null;
}

const extractSnapshotStep = (name?: string): number | null => {
  if (!name) return null;
  const match = name.match(/(\d+)(?!.*\d)/);
  return match ? parseInt(match[1], 10) : null;
};

const formatEngineLabel = (name: string) =>
  name
    .replace(/_/g, " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());

const sanitizeRunToken = (value: string): string =>
  value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9_-]+/g, "-")
    .replace(/-+/g, "-")
    .replace(/^[-_]+|[-_]+$/g, "")
    .slice(0, 80);

const sanitizeModelToken = (value: string): string =>
  value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9_-]+/g, "-")
    .replace(/-+/g, "-")
    .replace(/^[-_]+|[-_]+$/g, "")
    .slice(0, 80);

const sanitizeFilenameToken = (value: string): string =>
  value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9_-]+/g, "-")
    .replace(/-+/g, "-")
    .replace(/^[-_]+|[-_]+$/g, "")
    .slice(0, 80);

const formatDurationCompact = (seconds?: number | null): string => {
  if (typeof seconds !== "number" || !Number.isFinite(seconds) || seconds < 0) return "-";
  const total = Math.floor(seconds);
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  if (h > 0) return `${h}h ${m}m ${s}s`;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
};

const formatTelemetryScalar = (value: unknown): string => {
  if (value === null || value === undefined || value === "") return "-";
  if (typeof value === "boolean") return value ? "Yes" : "No";
  if (typeof value === "number") {
    if (Number.isInteger(value)) return value.toLocaleString();
    return value.toFixed(6).replace(/\.?0+$/, "");
  }
  if (typeof value === "object") {
    try {
      return JSON.stringify(value);
    } catch {
      return String(value);
    }
  }
  return String(value);
};

const formatTelemetryFieldLabel = (key: string): string =>
  key
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());

const LEARNABLE_AI_PARAM_KEYS = new Set([
  "feature_lr",
  "opacity_lr",
  "scaling_lr",
  "rotation_lr",
  "position_lr_init",
  "densify_grad_threshold",
  "opacity_threshold",
  "lambda_dssim",
]);

const isMissingFlagField = (key: string): boolean => /_missing$/i.test(key);

const parseMissingFlag = (value: unknown): boolean | null => {
  if (value === null || value === undefined) return null;
  if (typeof value === "boolean") return value;
  if (typeof value === "number") return value !== 0;
  if (typeof value === "string") {
    const token = value.trim().toLowerCase();
    if (!token) return null;
    if (["1", "true", "yes", "y", "missing"].includes(token)) return true;
    if (["0", "false", "no", "n", "available", "present"].includes(token)) return false;
  }
  return null;
};

const featureMissingFlagKey = (key: string): string | null => {
  const directMap: Record<string, string> = {
    focal_length_mm: "focal_missing",
    aperture_f: "aperture_missing",
    iso: "iso_missing",
    shutter_s: "shutter_missing",
    gps_lat_mean: "gps_missing",
    gps_lon_mean: "gps_missing",
    gps_alt_mean: "gps_missing",
    timestamp_mode: "timestamp_missing",
    camera_make: "camera_meta_missing",
    camera_model: "camera_meta_missing",
    lens_model: "lens_missing",
    camera_angle_bucket: "angle_missing",
    img_orientation: "orientation_missing",
    img_width_median: "img_size_missing",
    img_height_median: "img_size_missing",
    flight_type: "flight_type_missing",
    camera_angle_profile: "angle_profile_missing",
    average_altitude: "altitude_missing",
    heading_consistency: "heading_missing",
    coverage_spread: "coverage_missing",
    overlap_proxy: "overlap_missing",
    vegetation_cover_percentage: "green_area_missing",
    vegetation_complexity_score: "veg_complexity_missing",
    terrain_roughness_proxy: "roughness_missing",
    texture_density: "texture_missing",
    blur_motion_risk: "blur_missing",
  };
  return directMap[key] || null;
};

type FeatureStatus = "present" | "defaulted" | "unknown";

interface TelemetryFeatureRow {
  key: string;
  value: unknown;
  status: FeatureStatus;
  source: string;
}

const formatFeatureSourceLabel = (source: unknown): string => {
  const token = String(source || "").trim().toLowerCase();
  if (token === "original_metadata") return "Original metadata";
  if (token === "processed_dimensions") return "Processed dimensions";
  if (token === "processed_pixels") return "Processed pixels";
  return "Unknown";
};

const buildTelemetryFeatureRows = (
  features: Record<string, unknown> | null | undefined,
  missingFlags?: Record<string, unknown> | null,
  featureSources?: Record<string, unknown> | null,
): TelemetryFeatureRow[] => {
  if (!features || typeof features !== "object") {
    return [];
  }

  return Object.entries(features)
    .filter(([key]) => !isMissingFlagField(key))
    .map(([key, value]) => {
      const missingKey = featureMissingFlagKey(key);
      const missingRaw = missingKey ? (features[missingKey] ?? missingFlags?.[missingKey]) : null;
      const missingValue = parseMissingFlag(missingRaw);
      const status: FeatureStatus = missingValue === true ? "defaulted" : missingValue === false ? "present" : "unknown";
      const source = formatFeatureSourceLabel(featureSources?.[key]);
      return { key, value, status, source };
    });
};

const buildDefaultModelName = (
  projectLabel: string | null | undefined,
  projectId: string,
  runLabel: string | null | undefined,
): string => {
  const projectToken = sanitizeModelToken(projectLabel || projectId || "project") || "project";
  const runToken = sanitizeModelToken(runLabel || "session") || "session";
  return `model_${projectToken}_${runToken}`;
};

const buildDefaultRunName = (
  projectLabel: string | null | undefined,
  projectId: string,
  runs: ProjectRunInfo[] = []
): string => {
  const base = sanitizeRunToken(projectLabel || projectId || "project") || "project";
  const matcher = new RegExp(`^${base}_session(\\d+)$`);
  let nextIdx = 1;

  runs.forEach((run) => {
    const candidate = String(run.run_id || run.run_name || "").trim().toLowerCase();
    const match = candidate.match(matcher);
    if (!match) return;
    nextIdx = Math.max(nextIdx, Number.parseInt(match[1], 10) + 1);
  });

  return `${base}_session${nextIdx}`;
};

const COLMAP_CAMERA_MODELS = [
  "SIMPLE_PINHOLE",
  "PINHOLE",
  "SIMPLE_RADIAL",
  "RADIAL",
  "OPENCV",
  "OPENCV_FISHEYE",
  "FULL_OPENCV",
  "FOV",
  "SIMPLE_RADIAL_FISHEYE",
  "RADIAL_FISHEYE",
  "THIN_PRISM_FISHEYE",
] as const;

const getDefaultProcessConfig = () => ({
  mode: "baseline" as "baseline" | "modified",
  tune_start_step: 100,
  tune_min_improvement: 0.005,
  tune_end_step: 15000,
  tune_interval: 100,
  tune_scope: "core_individual_plus_strategy" as TuneScope,
  trend_scope: "run" as TrendScope,
  ai_input_mode: "" as AiInputMode,
  ai_selector_strategy: "preset_bias" as AiSelectorStrategy,
  baseline_session_id: "",
  warmup_at_start: false,
  run_count: 1,
  run_jitter_mode: "fixed" as RunJitterMode,
  run_jitter_factor: 1,
  run_jitter_min: 0.5,
  run_jitter_max: 1.5,
  continue_on_failure: true,
  start_model_mode: "scratch" as StartModelMode,
  project_model_name: "",
  source_model_id: "",
  engine: "gsplat" as TrainingEngine,
  maxSteps: 15000,
  logInterval: 100,
  splatInterval: 31000,
  bestSplatInterval: 100,
  bestSplatStartStep: 2000,
  saveBestSplat: false,
  auto_early_stop: false,
  earlyStopMonitorInterval: 200,
  earlyStopDecisionPoints: 10,
  earlyStopMinEvalPoints: 6,
  earlyStopMinStepRatio: 0.25,
  earlyStopMonitorMinRelativeImprovement: 0.0015,
  earlyStopEvalMinRelativeImprovement: 0.003,
  earlyStopMaxVolatilityRatio: 0.01,
  earlyStopEmaAlpha: 0.1,
  pngInterval: 50,
  evalInterval: 1000,
  saveInterval: 31000,
  sparse_preference: "best",
  sparse_merge_selection: [] as string[],
  colmap: {
    max_image_size: 1600,
    peak_threshold: 0.02,
    guided_matching: true,
    camera_model: "OPENCV",
    single_camera: true,
    camera_params: "",
    matching_type: "sequential",
    mapper_num_threads: 4,
    mapper_min_num_matches: 12,
    mapper_abs_pose_min_num_inliers: 15,
    mapper_init_min_num_inliers: 60,
    sift_matching_min_num_inliers: 12,
    run_image_registrator: true,
  },
  images_max_size: 1600,
  images_resize_enabled: true,
  densifyFromIter: 500,
  densifyUntilIter: 10000,
  densificationInterval: 100,
  densifyGradThreshold: 0.0002,
  opacityThreshold: 0.005,
  lambdaDssim: 0.2,
  litegs_target_primitives: 50000,
  litegs_alpha_shrink: 0.95,
});

export default function ProcessTab({ projectId }: ProcessTabProps) {
  const getSharedConfigStorageKey = useCallback(() => `processSharedConfig_${projectId}`, [projectId]);
  const getSelectedRunStorageKey = useCallback(() => `processSelectedRun_${projectId}`, [projectId]);
  const getTrainingConfigStorageKey = useCallback(
    (runId?: string) => `processTrainingConfig_${projectId}_${(runId || "__default").trim() || "__default"}`,
    [projectId],
  );
  const getApiErrorMessage = useCallback((err: unknown, fallback: string): string => {
    const candidate = err as {
      response?: { data?: { detail?: unknown } };
      message?: unknown;
    };
    const detail = candidate?.response?.data?.detail;
    if (typeof detail === "string" && detail.trim()) {
      return detail.trim();
    }
    if (Array.isArray(detail) && detail.length > 0) {
      const joined = detail
        .map((item) => {
          if (typeof item === "string") return item;
          if (item && typeof item?.msg === "string") return item.msg;
          return "";
        })
        .filter(Boolean)
        .join("; ");
      if (joined) return joined;
    }
    const message = candidate?.message;
    if (typeof message === "string" && message.trim()) {
      return message.trim();
    }
    return fallback;
  }, []);

  // Load config from localStorage or use defaults
  const loadConfig = () => {
    const defaults = getDefaultProcessConfig();
    try {
      const sharedSaved = localStorage.getItem(getSharedConfigStorageKey());
      const trainingSaved = localStorage.getItem(getTrainingConfigStorageKey());

      const sharedParsed = sharedSaved ? JSON.parse(sharedSaved) : {};
      const trainingParsed = trainingSaved ? JSON.parse(trainingSaved) : {};

      const merged = {
        ...defaults,
        ...sharedParsed,
        ...trainingParsed,
        colmap: {
          ...defaults.colmap,
          ...(sharedParsed?.colmap || {}),
        },
      } as ReturnType<typeof getDefaultProcessConfig>;

      return merged;
    } catch (e) {
      console.error('Failed to load config:', e);
    }
    // Do not load user's persisted advanced toggle; always start hidden
    return { ...defaults };
  };
  
  const [selectedInfoKey, setSelectedInfoKey] = useState<string | null>(null);

  // Load persisted config and expose individual controls
  const cfg = loadConfig();
  const cfgLegacy = cfg as Record<string, unknown>;
  const cfgMaxSteps = typeof cfgLegacy["max_steps"] === "number" ? cfgLegacy["max_steps"] : undefined;
  const cfgLogInterval = typeof cfgLegacy["log_interval"] === "number" ? cfgLegacy["log_interval"] : undefined;
  const cfgEvalInterval = typeof cfgLegacy["eval_interval"] === "number" ? cfgLegacy["eval_interval"] : undefined;
  const cfgSplatExportInterval = typeof cfgLegacy["splat_export_interval"] === "number" ? cfgLegacy["splat_export_interval"] : undefined;
  const cfgBestSplatInterval = typeof cfgLegacy["best_splat_interval"] === "number" ? cfgLegacy["best_splat_interval"] : undefined;
  const cfgSaveBestSplat = typeof cfgLegacy["save_best_splat"] === "boolean" ? cfgLegacy["save_best_splat"] : undefined;
  const cfgBestSplatStartStep = typeof cfgLegacy["best_splat_start_step"] === "number" ? cfgLegacy["best_splat_start_step"] : undefined;
  const cfgEarlyStopMonitorInterval = typeof cfgLegacy["early_stop_monitor_interval"] === "number" ? cfgLegacy["early_stop_monitor_interval"] : undefined;
  const cfgEarlyStopDecisionPoints = typeof cfgLegacy["early_stop_decision_points"] === "number" ? cfgLegacy["early_stop_decision_points"] : undefined;
  const cfgEarlyStopMinEvalPoints = typeof cfgLegacy["early_stop_min_eval_points"] === "number" ? cfgLegacy["early_stop_min_eval_points"] : undefined;
  const cfgEarlyStopMinStepRatio = typeof cfgLegacy["early_stop_min_step_ratio"] === "number" ? cfgLegacy["early_stop_min_step_ratio"] : undefined;
  const cfgEarlyStopMonitorMinRelativeImprovement =
    typeof cfgLegacy["early_stop_monitor_min_relative_improvement"] === "number"
      ? cfgLegacy["early_stop_monitor_min_relative_improvement"]
      : undefined;
  const cfgEarlyStopEvalMinRelativeImprovement =
    typeof cfgLegacy["early_stop_eval_min_relative_improvement"] === "number"
      ? cfgLegacy["early_stop_eval_min_relative_improvement"]
      : undefined;
  const cfgEarlyStopMaxVolatilityRatio = typeof cfgLegacy["early_stop_max_volatility_ratio"] === "number" ? cfgLegacy["early_stop_max_volatility_ratio"] : undefined;
  const cfgEarlyStopEmaAlpha = typeof cfgLegacy["early_stop_ema_alpha"] === "number" ? cfgLegacy["early_stop_ema_alpha"] : undefined;
  const cfgSaveInterval = typeof cfgLegacy["save_interval"] === "number" ? cfgLegacy["save_interval"] : undefined;
  const [mode, setMode] = useState<"baseline" | "modified">(cfg.mode ?? "baseline");
  const [tuneStartStep, setTuneStartStep] = useState<number>(cfg.tune_start_step ?? 100);
  const [tuneMinImprovement, setTuneMinImprovement] = useState<number>(cfg.tune_min_improvement ?? 0.005);
  const [tuneEndStep, setTuneEndStep] = useState<number>(cfg.tune_end_step ?? 15000);
  const [tuneInterval, setTuneInterval] = useState<number>(cfg.tune_interval ?? 100);
  const [tuneScope, setTuneScope] = useState<TuneScope>(cfg.tune_scope ?? "core_individual_plus_strategy");
  const [trendScope, setTrendScope] = useState<TrendScope>(cfg.trend_scope === "phase" ? "phase" : "run");
  const [aiInputMode, setAiInputMode] = useState<AiInputMode>(
    cfg.ai_input_mode === "exif_only" ||
      cfg.ai_input_mode === "exif_plus_flight_plan" ||
      cfg.ai_input_mode === "exif_plus_flight_plan_plus_external"
      ? cfg.ai_input_mode
      : ""
  );
  const [aiSelectorStrategy, setAiSelectorStrategy] = useState<AiSelectorStrategy>(
    cfg.ai_selector_strategy === "continuous_bandit_linear" ? "continuous_bandit_linear" : "preset_bias"
  );
  const [baselineSessionIdForAi, setBaselineSessionIdForAi] = useState<string>(cfg.baseline_session_id ?? "");
  const [warmupAtStart, setWarmupAtStart] = useState<boolean>(cfg.warmup_at_start ?? false);
  const [runCount, setRunCount] = useState<number>(cfg.run_count ?? 1);
  const [runJitterMode, setRunJitterMode] = useState<RunJitterMode>(cfg.run_jitter_mode === "random" ? "random" : "fixed");
  const [runJitterFactor, setRunJitterFactor] = useState<number>(cfg.run_jitter_factor ?? 1);
  const [runJitterMin, setRunJitterMin] = useState<number>(cfg.run_jitter_min ?? 0.5);
  const [runJitterMax, setRunJitterMax] = useState<number>(cfg.run_jitter_max ?? 1.5);
  const [continueOnFailure, setContinueOnFailure] = useState<boolean>(cfg.continue_on_failure ?? true);
  const [startModelMode, setStartModelMode] = useState<StartModelMode>(cfg.start_model_mode === "reuse" ? "reuse" : "scratch");
  const [sessionExecutionMode, setSessionExecutionMode] = useState<SessionExecutionMode>(
    (cfg as Record<string, unknown>).session_execution_mode === "test" ? "test" : "train"
  );
  const [projectModelName, setProjectModelName] = useState<string>(cfg.project_model_name ?? "");
  const [sourceModelId, setSourceModelId] = useState<string>(cfg.source_model_id ?? "");
  const [reusableModels, setReusableModels] = useState<ReusableModelEntry[]>([]);
  const [reusableModelsLoading, setReusableModelsLoading] = useState<boolean>(false);
  const [reusableModelsError, setReusableModelsError] = useState<string | null>(null);
  const [engine, setEngine] = useState<TrainingEngine>(cfg.engine ?? "gsplat");
  const [maxSteps, setMaxSteps] = useState<number>(cfgMaxSteps ?? cfg.maxSteps ?? 15000);
  const [logInterval, setLogInterval] = useState<number>(cfgLogInterval ?? cfg.logInterval ?? 100);
  const [splatInterval, setSplatInterval] = useState<number>(cfgSplatExportInterval ?? cfg.splatInterval ?? 31000);
  const [bestSplatInterval, setBestSplatInterval] = useState<number>(cfgBestSplatInterval ?? cfg.bestSplatInterval ?? 100);
  const [saveBestSplat, setSaveBestSplat] = useState<boolean>(cfgSaveBestSplat ?? cfg.saveBestSplat ?? false);
  const [bestSplatStartStep, setBestSplatStartStep] = useState<number>(cfgBestSplatStartStep ?? cfg.bestSplatStartStep ?? 2000);
  const [autoEarlyStop, setAutoEarlyStop] = useState<boolean>(cfg.auto_early_stop ?? false);
  const [earlyStopMonitorInterval, setEarlyStopMonitorInterval] = useState<number>(cfg.earlyStopMonitorInterval ?? cfgEarlyStopMonitorInterval ?? 200);
  const [earlyStopDecisionPoints, setEarlyStopDecisionPoints] = useState<number>(cfg.earlyStopDecisionPoints ?? cfgEarlyStopDecisionPoints ?? 10);
  const [earlyStopMinEvalPoints, setEarlyStopMinEvalPoints] = useState<number>(cfg.earlyStopMinEvalPoints ?? cfgEarlyStopMinEvalPoints ?? 6);
  const [earlyStopMinStepRatio, setEarlyStopMinStepRatio] = useState<number>(cfg.earlyStopMinStepRatio ?? cfgEarlyStopMinStepRatio ?? 0.25);
  const [earlyStopMonitorMinRelativeImprovement, setEarlyStopMonitorMinRelativeImprovement] = useState<number>(cfg.earlyStopMonitorMinRelativeImprovement ?? cfgEarlyStopMonitorMinRelativeImprovement ?? 0.0015);
  const [earlyStopEvalMinRelativeImprovement, setEarlyStopEvalMinRelativeImprovement] = useState<number>(cfg.earlyStopEvalMinRelativeImprovement ?? cfgEarlyStopEvalMinRelativeImprovement ?? 0.003);
  const [earlyStopMaxVolatilityRatio, setEarlyStopMaxVolatilityRatio] = useState<number>(cfg.earlyStopMaxVolatilityRatio ?? cfgEarlyStopMaxVolatilityRatio ?? 0.01);
  const [earlyStopEmaAlpha, setEarlyStopEmaAlpha] = useState<number>(cfg.earlyStopEmaAlpha ?? cfgEarlyStopEmaAlpha ?? 0.1);
  const [pngInterval, setPngInterval] = useState<number>(cfg.pngInterval ?? 50);
  const [evalInterval, setEvalInterval] = useState<number>(cfgEvalInterval ?? cfg.evalInterval ?? 1000);
  const [saveInterval, setSaveInterval] = useState<number>(cfgSaveInterval ?? cfg.saveInterval ?? 31000);
  const [imagesMaxSize, setImagesMaxSize] = useState<number | undefined>(cfg.images_max_size ?? 1600);
  const [imagesResizeEnabled, setImagesResizeEnabled] = useState<boolean>(cfg.images_resize_enabled ?? true);
  const [, setShowAdvancedTraining] = useState<boolean>(false);

  const [litegsTargetPrimitives, setLitegsTargetPrimitives] = useState<number>(cfg.litegs_target_primitives ?? 50000);
  const [litegsAlphaShrink, setLitegsAlphaShrink] = useState<number>(cfg.litegs_alpha_shrink ?? 0.95);
  const [sparsePreference, setSparsePreference] = useState<string>(cfg.sparse_preference ?? "best");
  const [sparseMergeSelection, setSparseMergeSelection] = useState<string[]>(Array.isArray(cfg.sparse_merge_selection) ? cfg.sparse_merge_selection : []);
  const [sparseOptions, setSparseOptions] = useState<Array<{ value: string; label: string }>>([
    { value: "best", label: "Auto (best available)" },
  ]);
  const [sparseOptionsLoading, setSparseOptionsLoading] = useState<boolean>(false);
  const [sparseMergeReport, setSparseMergeReport] = useState<SparseMergeReport | null>(null);
  const [sparseMergeReportCandidate, setSparseMergeReportCandidate] = useState<string | null>(null);
  const [sparseMergeReportLoading, setSparseMergeReportLoading] = useState<boolean>(false);
  const [sparseMergeReportError, setSparseMergeReportError] = useState<string | null>(null);
  const [sparseMergeBuildLoading, setSparseMergeBuildLoading] = useState<boolean>(false);
  const [sparseMergeBuildMessage, setSparseMergeBuildMessage] = useState<string | null>(null);
  const [densifyFromIter, setDensifyFromIter] = useState<number>(cfg.densifyFromIter ?? 500);
  const [densifyUntilIter, setDensifyUntilIter] = useState<number>(cfg.densifyUntilIter ?? 10000);
  const [densificationInterval, setDensificationInterval] = useState<number>(cfg.densificationInterval ?? 100);
  const [densifyGradThreshold, setDensifyGradThreshold] = useState<number>(cfg.densifyGradThreshold ?? 0.0002);
  const [opacityThreshold, setOpacityThreshold] = useState<number>(cfg.opacityThreshold ?? 0.005);
  const [lambdaDssim, setLambdaDssim] = useState<number>(cfg.lambdaDssim ?? 0.2);

  const showCoreAiSessionControls =
    engine === "gsplat" && mode === "modified" && tuneScope === "core_ai_optimization";
  const hasAiInputModeFlow = showCoreAiSessionControls && Boolean(aiInputMode);
  const hasAiInputModeTrainFlow = hasAiInputModeFlow && sessionExecutionMode === "train";
  const hasAiInputModeCompareFlow = hasAiInputModeFlow;
  const hasLegacyControllerFlow = showCoreAiSessionControls && !aiInputMode;
  const isSessionTestMode = showCoreAiSessionControls && sessionExecutionMode === "test";
  const effectiveStartModelMode: StartModelMode = isSessionTestMode ? "reuse" : startModelMode;
  const effectiveWarmupAtStart = showCoreAiSessionControls && !isSessionTestMode ? warmupAtStart : false;
  const effectiveRunCount = showCoreAiSessionControls && !isSessionTestMode ? runCount : 1;
  const showManualModifiedTuneControls = engine === "gsplat" && mode === "modified" && !showCoreAiSessionControls;
  const showManualDensificationControls = engine === "gsplat" && !showCoreAiSessionControls;
  const showBatchActions = showCoreAiSessionControls && !effectiveWarmupAtStart && effectiveRunCount > 1;
  const isReusableWarmStartSelected = effectiveStartModelMode === "reuse" && Boolean(sourceModelId);
  const modeCompatibleReusableModels = useMemo(() => {
    if (!showCoreAiSessionControls) {
      return reusableModels;
    }

    return reusableModels.filter((item) => {
      const profile = item.ai_profile && typeof item.ai_profile === "object" ? item.ai_profile : {};
      const pipelineKind = String(profile.pipeline_kind || "").trim().toLowerCase();
      const modelAiMode = String(profile.ai_input_mode || "").trim().toLowerCase();
      const modelSelector = String(profile.ai_selector_strategy || "").trim().toLowerCase();

      if (!hasAiInputModeFlow) {
        return pipelineKind === "controller";
      }

      if (!aiInputMode || modelAiMode !== aiInputMode) {
        return false;
      }

      // In test mode, allow both selector families for the chosen EXIF mode.
      // This lets users compare preset-bias and continuous-bandit models side by side.
      if (isSessionTestMode) {
        return true;
      }

      if (!aiSelectorStrategy) {
        return true;
      }

      return modelSelector === aiSelectorStrategy;
    });
  }, [
    showCoreAiSessionControls,
    reusableModels,
    hasAiInputModeFlow,
    isSessionTestMode,
    aiInputMode,
    aiSelectorStrategy,
  ]);
  const modeModelEmptyLabel = hasAiInputModeFlow
    ? "No reusable models match selected EXIF mode + selector strategy"
    : "No reusable models match controller pipeline";

  const tuneScopeDropdownValue: TuneScopeDropdownValue =
    tuneScope === "core_ai_optimization"
      ? (aiInputMode
          ? (`core_ai_optimization__${aiInputMode}` as TuneScopeDropdownValue)
          : "core_ai_optimization")
      : tuneScope;

  const densifyScheduleBlocked =
    !showCoreAiSessionControls && (densificationInterval <= 0 || densifyFromIter >= densifyUntilIter);
  const densifyBlockedReason = useMemo(() => {
    if (showCoreAiSessionControls) {
      return null;
    }
    if (densificationInterval <= 0) {
      return "Set a positive densification interval so gsplat can schedule refinements.";
    }
    if (densifyFromIter >= densifyUntilIter) {
      return `Start step (${densifyFromIter.toLocaleString()}) must be lower than the stop step (${densifyUntilIter.toLocaleString()}).`;
    }
    return null;
  }, [showCoreAiSessionControls, densificationInterval, densifyFromIter, densifyUntilIter]);

  const [colmapMaxImageSize, setColmapMaxImageSize] = useState<number | undefined>(cfg.colmap?.max_image_size ?? 1600);
  const [colmapPeakThreshold, setColmapPeakThreshold] = useState<number | undefined>(cfg.colmap?.peak_threshold ?? undefined);
  const [colmapGuidedMatching, setColmapGuidedMatching] = useState<boolean>(cfg.colmap?.guided_matching ?? true);
  const [colmapCameraModel, setColmapCameraModel] = useState<string>(cfg.colmap?.camera_model ?? "OPENCV");
  const [colmapSingleCamera, setColmapSingleCamera] = useState<boolean>(cfg.colmap?.single_camera ?? true);
  const [colmapCameraParams, setColmapCameraParams] = useState<string>(cfg.colmap?.camera_params ?? "");
  const [colmapMatchingType, setColmapMatchingType] = useState<string>(cfg.colmap?.matching_type ?? "sequential");
  const [colmapMapperThreads, setColmapMapperThreads] = useState<number | undefined>(cfg.colmap?.mapper_num_threads ?? undefined);
  const [colmapMapperMinNumMatches, setColmapMapperMinNumMatches] = useState<number | undefined>(cfg.colmap?.mapper_min_num_matches ?? 12);
  const [colmapMapperAbsPoseMinNumInliers, setColmapMapperAbsPoseMinNumInliers] = useState<number | undefined>(cfg.colmap?.mapper_abs_pose_min_num_inliers ?? 15);
  const [colmapMapperInitMinNumInliers, setColmapMapperInitMinNumInliers] = useState<number | undefined>(cfg.colmap?.mapper_init_min_num_inliers ?? 60);
  const [colmapSiftMatchingMinNumInliers, setColmapSiftMatchingMinNumInliers] = useState<number | undefined>(cfg.colmap?.sift_matching_min_num_inliers ?? 12);
  const [colmapRunImageRegistrator, setColmapRunImageRegistrator] = useState<boolean>(cfg.colmap?.run_image_registrator ?? true);

  const [runColmap, setRunColmap] = useState<boolean>(false);
  const [runTraining, setRunTraining] = useState<boolean>(true);
  const [runExport, setRunExport] = useState<boolean>(true);

  const [stoppedStage, setStoppedStage] = useState<string | null>(null);
  const [wasStopped, setWasStopped] = useState<boolean>(false);

  const [configTab, setConfigTab] = useState<"images"|"colmap"|"training">("training");

  const trainingInfo: Record<string, string> = {
    mode: 'Training profile. Baseline keeps default behavior; Modified applies rule-based or adaptive tuning during training depending on selected scope.',
    tune_start_step: 'For Modified mode, this is the first step where tuning checks are allowed. Before this step, no rule-based LR updates are applied.',
    tune_min_improvement: 'For Core individual and Core AI optimization scopes, this is the baseline minimum-improvement anchor (example: 0.005 = 0.5%).',
    tune_end_step: 'For Modified mode, this is the last step where rule-based tuning updates are allowed. The worker keeps applying rule checks until this step, then continues normal training.',
    tune_interval: 'For Modified mode, worker evaluates and applies rule-based updates every N steps during the tuning window.',
    tune_scope: 'Rule tuning scope: Core individual updates only LR groups. Core only updates LR groups + core strategy threshold. Core AI optimization uses AI input-mode preset selection and run-end best/end-anchor learning updates. Core individual + strategy updates LR groups and full strategy controls.',
    trend_scope: 'Core AI optimization trend scope setting retained for compatibility with existing payloads.',
    ai_input_mode: 'Initial preset mode for Core AI optimization. Leave empty to use the legacy controller-only flow. EXIF only uses image metadata, EXIF + flight plan adds sequence-derived flight features, and + external adds cheap image-derived scene features (no manual external inputs).',
    ai_selector_strategy: 'Core AI optimization selector strategy. Preset bias uses discrete preset learning. Continuous bandit linear predicts bounded continuous multipliers per run.',
    baseline_session_id: 'Completed baseline gsplat session used as reference for baseline-relative scoring in Core AI optimization modes. Required in train mode, optional in test mode.',
    warmup_at_start: 'Runs an automatic 3-phase warmup from this project base-session config (keeps base max_steps and densify_until_iter). Phase A forces rotating presets (balanced, conservative, geometry_fast, appearance_fast) with wider random jitter; Phases B and C switch back to adaptive preset selection with tighter jitter. Manual batch jitter controls are ignored while enabled.',
    run_count: 'Total sessions in this batch, including the selected session as run 1. Default 1 keeps manual behavior.',
    run_jitter_mode: 'Jitter behavior for batch runs starting from run 2. Fixed uses deterministic multiplier growth; Random samples a new multiplier per run within min/max bounds. Bounds: fixed factor >= 0.1 (frontend), random min/max >= 0.000001.',
    run_jitter_factor: 'Fixed mode only: per-run deterministic multiplier for LR-related params (applied from run 2 onward). Bounds: minimum 0.1, no frontend hard max; 1 means no fixed jitter.',
    run_jitter_min: 'Random mode only: lower bound for per-run random jitter multiplier (applied from run 2 onward). Bounds: minimum 0.000001, no frontend hard max. If min > max, backend auto-swaps.',
    run_jitter_max: 'Random mode only: upper bound for per-run random jitter multiplier (applied from run 2 onward). Bounds: minimum 0.000001, no frontend hard max. If max < min, backend auto-swaps.',
    continue_on_failure: 'If enabled, remaining runs continue even when one run fails/stops.',
    session_execution_mode: 'Session intent for Core AI optimization. Train keeps warmup/batch controls. Test hides those controls and requires selecting one reusable model.',
    start_model_mode: 'Choose training initialization mode. Scratch starts a new project-scoped model series. Reuse continues the active project-scoped series unless a global elevated model is selected below.',
    project_model_name: 'Optional display name for the project-scoped model series. If empty, run name is used. New series keys are created only when start mode is Scratch.',
    source_model_id: 'Optional global elevated model for warm-start. Leave empty to reuse the latest checkpoint from the active project-scoped model series.',
    // --- ORIGINAL KERBL PARAMETERS ---
    maxSteps: 'Total training iterations. This value is sent from frontend in both baseline and modified modes. [original]',
    logInterval: 'How often (in steps) to print consolidated training snapshots in worker logs. Lower values are more verbose. [custom]',
    splatInterval: 'How often (in steps) to export intermediate .splat/.ply files during training. [original]',
    bestSplatInterval: 'How often (in steps) to evaluate and update best.splat using measured training loss. Final export cadence remains controlled by Splat export interval. [custom]',
    bestSplatStartStep: 'First step where best.splat tracking becomes active. Use this to skip expensive best-splat checks early in training. [custom]',
    auto_early_stop: 'Enable early stop: monitor trend every monitor interval and confirm plateau only at eval steps. [custom]',
    earlyStopMonitorInterval: 'Cadence for fast EMA trend checks between eval passes. These checks only mark candidate status; they do not stop training directly. [custom]',
    earlyStopDecisionPoints: 'Window size (points) used for both monitor trend and eval confirmation. Recommended 10 for stable decisions. [custom]',
    earlyStopMinEvalPoints: 'Minimum number of eval points required before early-stop confirmation is allowed. [custom]',
    earlyStopMinStepRatio: 'Minimum fraction of max steps that must be completed before early-stop confirmation is allowed. [custom]',
    earlyStopMonitorMinRelativeImprovement: 'Minimum relative improvement required over monitor window to avoid candidate plateau state. [custom]',
    earlyStopEvalMinRelativeImprovement: 'Minimum relative improvement required over eval window to avoid confirmed plateau stop. [custom]',
    earlyStopMaxVolatilityRatio: 'Maximum normalized volatility allowed in eval window for plateau confirmation. [custom]',
    earlyStopEmaAlpha: 'EMA smoothing factor for monitor loss trend (0-1). Higher values react faster, lower values are smoother. [custom]',
    pngInterval: 'Deprecated for gsplat: previews are generated on eval steps. Use eval interval to control preview cadence.',
    evalInterval: 'How often to run eval passes + metrics collection. Preview images are generated on each eval step. This value is configurable from frontend in both modes. [original]',
    saveInterval: 'Checkpoint frequency for gsplat. This value is configurable from frontend. [original]',
    densify_from_iter: 'Iteration to start densifying Gaussians. This value is configurable from frontend in both modes. [original]',
    densify_until_iter: 'Iteration after which densification stops. This value is configurable from frontend in both modes. [original]',
    densification_interval: 'Spacing between densification passes. This value is configurable from frontend in both modes. [original]',
    densify_grad_threshold: 'Gradient threshold used to trigger densification. Lower values densify more aggressively; higher values densify more conservatively. [original]',
    opacity_threshold: 'Minimum opacity for densification pruning. This value is configurable from frontend in both modes. [original]',
    lambda_dssim: 'Weight for DSSIM vs L2 loss. This value is configurable from frontend in both modes. [original]',
    // --- CUSTOM PARAMETERS ---
    engine: 'Choose between gsplat and LiteGS backends. [custom]',
    sparse_preference: 'Choose which COLMAP reconstruction to seed training from. Auto sticks with the best-scoring run. [custom]',
    sparse_merge_selection: 'When preference is set to merge, pick multiple sparse folders to combine into one merged initialization. [custom]',
    litegs_target_primitives: 'LiteGS keeps growing Gaussians until it nears this count. Lower it for faster runs or tight GPU budgets. [custom]',
    litegs_alpha_shrink: 'LiteGS-specific alpha shrink factor. Values below 1 tighten each Gaussian lobe every densify pass to improve sharp edges. [custom]'
  };

  const colmapInfo: Record<string, string> = {
    max_image_size: 'Optional cap on COLMAP feature-extraction resolution. Helps keep SIFT affordable on ultra-high-res uploads without touching the originals.',
    peak_threshold: 'SIFT detection threshold. Higher values reduce number of keypoints, speeding up processing.',
    guided_matching: 'Enable guided matching to improve accuracy using estimated geometry (may be slower).',
    camera_model: 'COLMAP camera model used during feature extraction. For most DJI RGB captures, OPENCV is a good default.',
    single_camera: 'When enabled, all images share one camera intrinsics set. Disable if intrinsics vary across frames/sensors.',
    camera_params: 'Optional explicit camera params string for the selected model (comma-separated in COLMAP order). Leave blank to estimate from EXIF/data.',
    matching_type: 'Matching strategy: exhaustive compares all pairs; sequential is faster for ordered captures.',
    mapper_num_threads: 'Number of CPU threads the COLMAP mapper can use (increase to speed up bundle adjustment).',
    mapper_min_num_matches: 'Lower values let mapper try weaker image pairs. This can increase registered images on hard captures, but may also add outliers.',
    mapper_abs_pose_min_num_inliers: 'Minimum inliers for registering a new image pose. Lowering helps difficult images register.',
    mapper_init_min_num_inliers: 'Inlier threshold used during mapper initialization. Lower values can reduce early dropouts.',
    sift_matching_min_num_inliers: 'Minimum inliers after geometric verification in matching. Lower values keep more tentative pairs.',
    run_image_registrator: 'Run COLMAP image_registrator + triangulation pass after mapper to recover additional images that failed first pass.'
  };
  const imagesInfo: Record<string, string> = {
    resize_mode: 'Control whether the pipeline clones your uploads into a resized working set. When enabled, both COLMAP and gsplat share the same downsized images, keeping intrinsics consistent and reducing VRAM use.',
    images_max_size: 'Largest dimension (width or height) allowed in the resized set. Originals stay untouched; the copies live beside your uploads and are reused for both reconstruction and training.'
  };
  const [processing, setProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [gpuAvailable, setGpuAvailable] = useState(true);
  const [showConfig, setShowConfig] = useState(false);
  const [topView, setTopView] = useState<"map" | "viewer" | "png">("map");
  const [mapDim, setMapDim] = useState<"2d" | "3d">("2d");
  const [headerCompact, setHeaderCompact] = useState(false);
  const [processingStatus, setProcessingStatus] = useState<React.ReactNode>(null);
  const [pngFiles, setPngFiles] = useState<PreviewFile[]>([]);
  const [engineOutputMap, setEngineOutputMap] = useState<Record<string, EngineOutputBundle>>({});
  const [selectedEngineName, setSelectedEngineName] = useState<string | null>(null);
  const [selectedPng, setSelectedPng] = useState<string | null>(null);
  const [modelSnapshots, setModelSnapshots] = useState<SnapshotEntry[]>([]);
  const [selectedModelSnapshot, setSelectedModelSnapshot] = useState<string | null>(null);
  const [selectedModelLayer, setSelectedModelLayer] = useState<"final" | "best">("final");
  const [basemap, setBasemap] = useState<"satellite" | "osm">("satellite");
  const [showImagesLayer, setShowImagesLayer] = useState(true);
  const [, setShowSparseLayer] = useState(true);
  const [show3DModel, setShow3DModel] = useState(false);
  const [locations, setLocations] = useState<Array<{ name: string; lat: number; lon: number }>>([]);
  const [locLoading, setLocLoading] = useState(true);
  const [hasSparseCloud, setHasSparseCloud] = useState(false);
  const [has3DModel, setHas3DModel] = useState(false);
  const [viewerOutput, setViewerOutput] = useState<'model' | 'pointcloud'>('model');
  const [focusTarget, setFocusTarget] = useState<[number, number, number] | null>(null);
  const [stageStatus, setStageStatus] = useState<{ colmap: string; training: string; export: string }>({
    colmap: "pending",
    training: "pending",
    export: "pending"
  });
  const [canResume, setCanResume] = useState(false);
  const [isStopping, setIsStopping] = useState(false);
  const [stoppingMessage, setStoppingMessage] = useState<string | null>(null);
  const [trainingCurrentStep, setTrainingCurrentStep] = useState<number | undefined>(undefined);
  const [trainingMaxSteps, setTrainingMaxSteps] = useState<number | undefined>(undefined);
  const [overallProgress, setOverallProgress] = useState<number>(0);
  const [currentStage, setCurrentStage] = useState<string>("");
  const [currentStageKey, setCurrentStageKey] = useState<"docker"|"colmap"|"training"|"export"|"">("");
  const [stageProgress, setStageProgress] = useState<number | undefined>(undefined);
  const [batchTotal, setBatchTotal] = useState<number>(0);
  const [batchCompleted, setBatchCompleted] = useState<number>(0);
  const [batchCurrentIndex, setBatchCurrentIndex] = useState<number>(0);
  const [showTelemetryModal, setShowTelemetryModal] = useState<boolean>(false);
  const [telemetryLoading, setTelemetryLoading] = useState<boolean>(false);
  const [telemetryError, setTelemetryError] = useState<string | null>(null);
  const [telemetryData, setTelemetryData] = useState<TelemetryPayload | null>(null);
  const [telemetryDownloadBusy, setTelemetryDownloadBusy] = useState<boolean>(false);
  const [pipelineDone, setPipelineDone] = useState(false);
  const [projectRuns, setProjectRuns] = useState<ProjectRunInfo[]>([]);
  const [processingRunId, setProcessingRunId] = useState<string>("");
  const [startRequestAtMs, setStartRequestAtMs] = useState<number>(0);
  const [selectedRunId, setSelectedRunId] = useState<string>("");
  const selectedRunIdRef = useRef<string>("");
  const [baseSessionId, setBaseSessionId] = useState<string>("");
  const [projectDisplayName, setProjectDisplayName] = useState<string>(projectId);
  const [newRunName, setNewRunName] = useState<string>(buildDefaultRunName(projectId, projectId, projectRuns));
  const [isRenamingRun, setIsRenamingRun] = useState<boolean>(false);
  const [showRestartConfirmModal, setShowRestartConfirmModal] = useState<boolean>(false);
  const [showRenameSessionModal, setShowRenameSessionModal] = useState<boolean>(false);
  const [renameSessionDraft, setRenameSessionDraft] = useState<string>("");
  const [showElevateModelModal, setShowElevateModelModal] = useState<boolean>(false);
  const [elevateModelNameDraft, setElevateModelNameDraft] = useState<string>("");
  const [isElevatingModel, setIsElevatingModel] = useState<boolean>(false);
  const [showNewSessionModal, setShowNewSessionModal] = useState<boolean>(false);
  const [newSessionNameDraft, setNewSessionNameDraft] = useState<string>("");
  const [newSessionConfigSource, setNewSessionConfigSource] = useState<NewSessionConfigSource>("current");
  const [isCreatingSessionDraft, setIsCreatingSessionDraft] = useState<boolean>(false);
  const [isSavingConfig, setIsSavingConfig] = useState<boolean>(false);
  const [configSavedToast, setConfigSavedToast] = useState<string>("");
  const [processInfoToast, setProcessInfoToast] = useState<string>("");
  const [canCreateSessionDraft, setCanCreateSessionDraft] = useState<boolean>(false);
  const [createSessionDisabledReason, setCreateSessionDisabledReason] = useState<string>("Complete COLMAP on the base session before creating new sessions.");
  const [baseColmapProfile, setBaseColmapProfile] = useState<{
    runId: string;
    runName: string;
    imageSize: number | null;
    resizeEnabled: boolean;
  } | null>(null);

  const selectedRunMeta = useMemo(
    () => projectRuns.find((r) => r.run_id === selectedRunId) || null,
    [projectRuns, selectedRunId],
  );
  const baselineCandidateRuns = useMemo(() => {
    return projectRuns.filter((r) => {
      const statusVal = String(r.session_status || "").trim().toLowerCase();
      const engineVal = String(r.engine || "").trim().toLowerCase();
      const modeVal = String(r.mode || "").trim().toLowerCase();
      const completed = statusVal === "completed" || statusVal === "succeeded";
      return completed && modeVal === "baseline" && (!engineVal || engineVal === "gsplat");
    });
  }, [projectRuns]);
  const selectedRunIsProcessing = Boolean(processingRunId) && selectedRunId === processingRunId;
  const canManageColmapImages = !selectedRunId || selectedRunMeta?.is_base === true;
  const selectedRunSharedOutdated = Boolean(!canManageColmapImages && selectedRunMeta?.shared_outdated);

  useEffect(() => {
    selectedRunIdRef.current = (selectedRunId || "").trim();
  }, [selectedRunId]);

  useEffect(() => {
    if (!hasAiInputModeCompareFlow) return;
    const valid = baselineCandidateRuns.some((r) => r.run_id === baselineSessionIdForAi);
    if (!valid) {
      setBaselineSessionIdForAi(baselineCandidateRuns[0]?.run_id || "");
    }
  }, [hasAiInputModeCompareFlow, baselineCandidateRuns, baselineSessionIdForAi]);

  const applyTrainingDefaults = (defaults: ReturnType<typeof getDefaultProcessConfig>) => {
      setSaveBestSplat(typeof defaults.saveBestSplat === "boolean" ? defaults.saveBestSplat : false);
    setMode(defaults.mode ?? "baseline");
    setTuneStartStep(defaults.tune_start_step ?? 100);
    setTuneMinImprovement(defaults.tune_min_improvement ?? 0.005);
    setTuneEndStep(defaults.tune_end_step ?? 15000);
    setTuneInterval(defaults.tune_interval ?? 100);
    setTuneScope(defaults.tune_scope ?? "core_individual_plus_strategy");
    setTrendScope(defaults.trend_scope === "phase" ? "phase" : "run");
    setAiInputMode(defaults.ai_input_mode ?? "");
    setAiSelectorStrategy(defaults.ai_selector_strategy === "continuous_bandit_linear" ? "continuous_bandit_linear" : "preset_bias");
    setBaselineSessionIdForAi(defaults.baseline_session_id ?? "");
    setWarmupAtStart(defaults.warmup_at_start ?? false);
    setRunCount(defaults.run_count ?? 1);
    setRunJitterMode(defaults.run_jitter_mode === "random" ? "random" : "fixed");
    setRunJitterFactor(defaults.run_jitter_factor ?? 1);
    setRunJitterMin(defaults.run_jitter_min ?? 0.5);
    setRunJitterMax(defaults.run_jitter_max ?? 1.5);
    setContinueOnFailure(defaults.continue_on_failure ?? true);
    setSessionExecutionMode("train");
    setStartModelMode(defaults.start_model_mode ?? "scratch");
    setProjectModelName(defaults.project_model_name ?? "");
    setSourceModelId(defaults.source_model_id ?? "");
    setEngine(defaults.engine ?? "gsplat");
    setMaxSteps(defaults.maxSteps);
    setLogInterval(defaults.logInterval ?? 100);
    setSplatInterval(defaults.splatInterval);
    setBestSplatInterval(defaults.bestSplatInterval ?? 100);
    setBestSplatStartStep(defaults.bestSplatStartStep ?? 2000);
    setAutoEarlyStop(defaults.auto_early_stop ?? true);
    setEarlyStopMonitorInterval(defaults.earlyStopMonitorInterval ?? 200);
    setEarlyStopDecisionPoints(defaults.earlyStopDecisionPoints ?? 10);
    setEarlyStopMinEvalPoints(defaults.earlyStopMinEvalPoints ?? 6);
    setEarlyStopMinStepRatio(defaults.earlyStopMinStepRatio ?? 0.25);
    setEarlyStopMonitorMinRelativeImprovement(defaults.earlyStopMonitorMinRelativeImprovement ?? 0.0015);
    setEarlyStopEvalMinRelativeImprovement(defaults.earlyStopEvalMinRelativeImprovement ?? 0.003);
    setEarlyStopMaxVolatilityRatio(defaults.earlyStopMaxVolatilityRatio ?? 0.01);
    setEarlyStopEmaAlpha(defaults.earlyStopEmaAlpha ?? 0.1);
    setPngInterval(defaults.pngInterval);
    setEvalInterval(defaults.evalInterval);
    setSaveInterval(defaults.saveInterval);
    setLitegsTargetPrimitives(defaults.litegs_target_primitives);
    setLitegsAlphaShrink(defaults.litegs_alpha_shrink);
    setSparsePreference(defaults.sparse_preference ?? "best");
    setSparseMergeSelection(Array.isArray(defaults.sparse_merge_selection) ? defaults.sparse_merge_selection : []);
    setDensifyFromIter(defaults.densifyFromIter);
    setDensifyUntilIter(defaults.densifyUntilIter);
    setDensificationInterval(defaults.densificationInterval);
    setOpacityThreshold(defaults.opacityThreshold);
    setLambdaDssim(defaults.lambdaDssim);
    setShowAdvancedTraining(false);
  };

  const applyColmapDefaults = (defaults: ReturnType<typeof getDefaultProcessConfig>) => {
    setColmapMaxImageSize(defaults.colmap.max_image_size);
    setColmapPeakThreshold(defaults.colmap.peak_threshold);
    setColmapGuidedMatching(defaults.colmap.guided_matching);
    setColmapCameraModel(defaults.colmap.camera_model);
    setColmapSingleCamera(defaults.colmap.single_camera);
    setColmapCameraParams(defaults.colmap.camera_params);
    setColmapMatchingType(defaults.colmap.matching_type);
    setColmapMapperThreads(defaults.colmap.mapper_num_threads);
    setColmapMapperMinNumMatches(defaults.colmap.mapper_min_num_matches);
    setColmapMapperAbsPoseMinNumInliers(defaults.colmap.mapper_abs_pose_min_num_inliers);
    setColmapMapperInitMinNumInliers(defaults.colmap.mapper_init_min_num_inliers);
    setColmapSiftMatchingMinNumInliers(defaults.colmap.sift_matching_min_num_inliers);
    setColmapRunImageRegistrator(defaults.colmap.run_image_registrator);
  };

  const applyImageDefaults = (defaults: ReturnType<typeof getDefaultProcessConfig>) => {
    setImagesResizeEnabled(defaults.images_resize_enabled);
    setImagesMaxSize(defaults.images_max_size);
  };

  const applyResolvedParamsToForm = (
    resolved: Record<string, unknown>,
    options: { includeTraining?: boolean; includeShared?: boolean } = {},
  ) => {
    const includeTraining = options.includeTraining !== false;
    const includeShared = options.includeShared !== false;

    if (includeTraining) {
      if (resolved.mode === "baseline" || resolved.mode === "modified") setMode(resolved.mode);
      if (typeof resolved.tune_start_step === "number") setTuneStartStep(resolved.tune_start_step);
      if (typeof resolved.tune_min_improvement === "number") setTuneMinImprovement(resolved.tune_min_improvement);
      if (typeof resolved.tune_end_step === "number") setTuneEndStep(resolved.tune_end_step);
      if (typeof resolved.tune_interval === "number") setTuneInterval(resolved.tune_interval);
      if (resolved.tune_scope) setTuneScope(resolved.tune_scope as TuneScope);
      if (resolved.trend_scope === "run" || resolved.trend_scope === "phase") setTrendScope(resolved.trend_scope as TrendScope);
      if (
        resolved.ai_input_mode === "exif_only" ||
        resolved.ai_input_mode === "exif_plus_flight_plan" ||
        resolved.ai_input_mode === "exif_plus_flight_plan_plus_external"
      ) {
        setAiInputMode(resolved.ai_input_mode as AiInputMode);
      }
      if (resolved.ai_selector_strategy === "preset_bias" || resolved.ai_selector_strategy === "continuous_bandit_linear") {
        setAiSelectorStrategy(resolved.ai_selector_strategy as AiSelectorStrategy);
      }
      if (typeof resolved.baseline_session_id === "string") setBaselineSessionIdForAi(resolved.baseline_session_id);
      if (resolved.session_execution_mode === "train" || resolved.session_execution_mode === "test") {
        setSessionExecutionMode(resolved.session_execution_mode as SessionExecutionMode);
      }
      if (typeof resolved.warmup_at_start === "boolean") setWarmupAtStart(resolved.warmup_at_start);
      if (typeof resolved.run_count === "number") setRunCount(Math.max(1, Math.floor(resolved.run_count)));
      if (resolved.run_jitter_mode === "fixed" || resolved.run_jitter_mode === "random") {
        setRunJitterMode(resolved.run_jitter_mode as RunJitterMode);
      }
      if (typeof resolved.run_jitter_factor === "number") setRunJitterFactor(Math.max(0.1, resolved.run_jitter_factor));
      if (typeof resolved.run_jitter_min === "number") setRunJitterMin(Math.max(1e-6, resolved.run_jitter_min));
      if (typeof resolved.run_jitter_max === "number") setRunJitterMax(Math.max(1e-6, resolved.run_jitter_max));
      if (typeof resolved.continue_on_failure === "boolean") setContinueOnFailure(resolved.continue_on_failure);
      if (resolved.start_model_mode === "scratch" || resolved.start_model_mode === "reuse") setStartModelMode(resolved.start_model_mode);
      if (typeof resolved.project_model_name === "string") setProjectModelName(resolved.project_model_name);
      if (typeof resolved.source_model_id === "string") setSourceModelId(resolved.source_model_id);
      if (resolved.engine === "gsplat" || resolved.engine === "litegs") setEngine(resolved.engine);
      if (typeof resolved.max_steps === "number") setMaxSteps(resolved.max_steps);
      if (typeof resolved.log_interval === "number") setLogInterval(resolved.log_interval);
      if (typeof resolved.splat_export_interval === "number") setSplatInterval(resolved.splat_export_interval);
      if (typeof resolved.best_splat_interval === "number") setBestSplatInterval(resolved.best_splat_interval);
      if (typeof resolved.best_splat_start_step === "number") setBestSplatStartStep(resolved.best_splat_start_step);
      if (typeof resolved.save_best_splat === "boolean") setSaveBestSplat(resolved.save_best_splat);
      else if (typeof resolved.saveBestSplat === "boolean") setSaveBestSplat(resolved.saveBestSplat);
      if (typeof resolved.auto_early_stop === "boolean") setAutoEarlyStop(resolved.auto_early_stop);
      if (typeof resolved.early_stop_monitor_interval === "number") setEarlyStopMonitorInterval(resolved.early_stop_monitor_interval);
      if (typeof resolved.early_stop_decision_points === "number") setEarlyStopDecisionPoints(resolved.early_stop_decision_points);
      if (typeof resolved.early_stop_min_eval_points === "number") setEarlyStopMinEvalPoints(resolved.early_stop_min_eval_points);
      if (typeof resolved.early_stop_min_step_ratio === "number") setEarlyStopMinStepRatio(resolved.early_stop_min_step_ratio);
      if (typeof resolved.early_stop_monitor_min_relative_improvement === "number") setEarlyStopMonitorMinRelativeImprovement(resolved.early_stop_monitor_min_relative_improvement);
      if (typeof resolved.early_stop_eval_min_relative_improvement === "number") setEarlyStopEvalMinRelativeImprovement(resolved.early_stop_eval_min_relative_improvement);
      if (typeof resolved.early_stop_max_volatility_ratio === "number") setEarlyStopMaxVolatilityRatio(resolved.early_stop_max_volatility_ratio);
      if (typeof resolved.early_stop_ema_alpha === "number") setEarlyStopEmaAlpha(resolved.early_stop_ema_alpha);
      if (typeof resolved.eval_interval === "number") setEvalInterval(resolved.eval_interval);
      if (typeof resolved.save_interval === "number") setSaveInterval(resolved.save_interval);
      if (typeof resolved.densify_from_iter === "number") setDensifyFromIter(resolved.densify_from_iter);
      if (typeof resolved.densify_until_iter === "number") setDensifyUntilIter(resolved.densify_until_iter);
      if (typeof resolved.densification_interval === "number") setDensificationInterval(resolved.densification_interval);
      if (typeof resolved.densify_grad_threshold === "number") setDensifyGradThreshold(resolved.densify_grad_threshold);
      if (typeof resolved.opacity_threshold === "number") setOpacityThreshold(resolved.opacity_threshold);
      if (typeof resolved.lambda_dssim === "number") setLambdaDssim(resolved.lambda_dssim);
      if (typeof resolved.litegs_target_primitives === "number") setLitegsTargetPrimitives(resolved.litegs_target_primitives);
      if (typeof resolved.litegs_alpha_shrink === "number") setLitegsAlphaShrink(resolved.litegs_alpha_shrink);
      if (typeof resolved.sparse_preference === "string") setSparsePreference(resolved.sparse_preference);
      if (Array.isArray(resolved.sparse_merge_selection)) setSparseMergeSelection(resolved.sparse_merge_selection);
    }

    if (includeShared) {
      if (typeof resolved.images_resize_enabled === "boolean") {
        setImagesResizeEnabled(resolved.images_resize_enabled);
      }
      if (typeof resolved.images_max_size === "number") {
        if (typeof resolved.images_resize_enabled !== "boolean") {
          setImagesResizeEnabled(true);
        }
        setImagesMaxSize(resolved.images_max_size);
      }
    }

    const colmap = (resolved.colmap && typeof resolved.colmap === "object") ? (resolved.colmap as Record<string, unknown>) : null;
    if (includeShared && colmap) {
      if (typeof colmap["max_image_size"] === "number") setColmapMaxImageSize(colmap["max_image_size"]);
      if (typeof colmap["peak_threshold"] === "number") setColmapPeakThreshold(colmap["peak_threshold"]);
      if (typeof colmap["guided_matching"] === "boolean") setColmapGuidedMatching(colmap["guided_matching"]);
      if (typeof colmap["camera_model"] === "string") setColmapCameraModel(colmap["camera_model"]);
      if (typeof colmap["single_camera"] === "boolean") setColmapSingleCamera(colmap["single_camera"]);
      if (typeof colmap["camera_params"] === "string") setColmapCameraParams(colmap["camera_params"]);
      if (typeof colmap["matching_type"] === "string") setColmapMatchingType(colmap["matching_type"]);
      if (typeof colmap["mapper_num_threads"] === "number") setColmapMapperThreads(colmap["mapper_num_threads"]);
      if (typeof colmap["mapper_min_num_matches"] === "number") setColmapMapperMinNumMatches(colmap["mapper_min_num_matches"]);
      if (typeof colmap["mapper_abs_pose_min_num_inliers"] === "number") setColmapMapperAbsPoseMinNumInliers(colmap["mapper_abs_pose_min_num_inliers"]);
      if (typeof colmap["mapper_init_min_num_inliers"] === "number") setColmapMapperInitMinNumInliers(colmap["mapper_init_min_num_inliers"]);
      if (typeof colmap["sift_matching_min_num_inliers"] === "number") setColmapSiftMatchingMinNumInliers(colmap["sift_matching_min_num_inliers"]);
      if (typeof colmap["run_image_registrator"] === "boolean") setColmapRunImageRegistrator(colmap["run_image_registrator"]);
    }
  };

  const normalizeTrainingConfigForForm = (raw: Record<string, unknown>): Record<string, unknown> => {
    const normalized = { ...raw };
    if (typeof normalized.save_best_splat !== "boolean" && typeof raw.saveBestSplat === "boolean") {
      normalized.save_best_splat = raw.saveBestSplat;
    }
    if (typeof normalized.trend_scope !== "string" && typeof raw.trendScope === "string") normalized.trend_scope = raw.trendScope;
    if (typeof normalized.max_steps !== "number" && typeof raw.maxSteps === "number") normalized.max_steps = raw.maxSteps;
    if (typeof normalized.log_interval !== "number" && typeof raw.logInterval === "number") normalized.log_interval = raw.logInterval;
    if (typeof normalized.splat_export_interval !== "number" && typeof raw.splatInterval === "number") normalized.splat_export_interval = raw.splatInterval;
    if (typeof normalized.best_splat_interval !== "number" && typeof raw.bestSplatInterval === "number") normalized.best_splat_interval = raw.bestSplatInterval;
    if (typeof normalized.best_splat_start_step !== "number" && typeof raw.bestSplatStartStep === "number") normalized.best_splat_start_step = raw.bestSplatStartStep;
    if (typeof normalized.auto_early_stop !== "boolean" && typeof raw.auto_early_stop === "boolean") normalized.auto_early_stop = raw.auto_early_stop;
    if (typeof normalized.early_stop_monitor_interval !== "number" && typeof raw.earlyStopMonitorInterval === "number") normalized.early_stop_monitor_interval = raw.earlyStopMonitorInterval;
    if (typeof normalized.early_stop_decision_points !== "number" && typeof raw.earlyStopDecisionPoints === "number") normalized.early_stop_decision_points = raw.earlyStopDecisionPoints;
    if (typeof normalized.early_stop_min_eval_points !== "number" && typeof raw.earlyStopMinEvalPoints === "number") normalized.early_stop_min_eval_points = raw.earlyStopMinEvalPoints;
    if (typeof normalized.early_stop_min_step_ratio !== "number" && typeof raw.earlyStopMinStepRatio === "number") normalized.early_stop_min_step_ratio = raw.earlyStopMinStepRatio;
    if (typeof normalized.early_stop_monitor_min_relative_improvement !== "number" && typeof raw.earlyStopMonitorMinRelativeImprovement === "number") normalized.early_stop_monitor_min_relative_improvement = raw.earlyStopMonitorMinRelativeImprovement;
    if (typeof normalized.early_stop_eval_min_relative_improvement !== "number" && typeof raw.earlyStopEvalMinRelativeImprovement === "number") normalized.early_stop_eval_min_relative_improvement = raw.earlyStopEvalMinRelativeImprovement;
    if (typeof normalized.early_stop_max_volatility_ratio !== "number" && typeof raw.earlyStopMaxVolatilityRatio === "number") normalized.early_stop_max_volatility_ratio = raw.earlyStopMaxVolatilityRatio;
    if (typeof normalized.early_stop_ema_alpha !== "number" && typeof raw.earlyStopEmaAlpha === "number") normalized.early_stop_ema_alpha = raw.earlyStopEmaAlpha;
    if (typeof normalized.eval_interval !== "number" && typeof raw.evalInterval === "number") normalized.eval_interval = raw.evalInterval;
    if (typeof normalized.save_interval !== "number" && typeof raw.saveInterval === "number") normalized.save_interval = raw.saveInterval;
    if (typeof normalized.densify_from_iter !== "number" && typeof raw.densifyFromIter === "number") normalized.densify_from_iter = raw.densifyFromIter;
    if (typeof normalized.densify_until_iter !== "number" && typeof raw.densifyUntilIter === "number") normalized.densify_until_iter = raw.densifyUntilIter;
    if (typeof normalized.densification_interval !== "number" && typeof raw.densificationInterval === "number") normalized.densification_interval = raw.densificationInterval;
    if (typeof normalized.densify_grad_threshold !== "number" && typeof raw.densifyGradThreshold === "number") normalized.densify_grad_threshold = raw.densifyGradThreshold;
    if (typeof normalized.opacity_threshold !== "number" && typeof raw.opacityThreshold === "number") normalized.opacity_threshold = raw.opacityThreshold;
    if (typeof normalized.lambda_dssim !== "number" && typeof raw.lambdaDssim === "number") normalized.lambda_dssim = raw.lambdaDssim;
    return normalized;
  };

  const resetConfigToDefaults = () => {
    const defaults = getDefaultProcessConfig();
    const tab: "images" | "colmap" | "training" = configTab;
    if (tab === "training") {
      applyTrainingDefaults(defaults);
      localStorage.removeItem(getTrainingConfigStorageKey(selectedRunId));
      return;
    }
    if (tab === "colmap") {
      applyColmapDefaults(defaults);
      return;
    }
    applyImageDefaults(defaults);
  };

  // Auto-switch to viewer when 3D model layer is enabled
  

  // Persist training config per session.
  useEffect(() => {
    if (!selectedRunId || hydratedTrainingRunIdRef.current !== selectedRunId) {
      return;
    }

    const config = {
      mode,
      tune_start_step: tuneStartStep,
      tune_min_improvement: tuneMinImprovement,
      tune_end_step: tuneEndStep,
      tune_interval: tuneInterval,
      tune_scope: tuneScope,
      trend_scope: trendScope,
      ai_input_mode: aiInputMode,
      ai_selector_strategy: aiSelectorStrategy,
      baseline_session_id: baselineSessionIdForAi,
      warmup_at_start: effectiveWarmupAtStart,
      run_count: effectiveRunCount,
      run_jitter_mode: sessionExecutionMode === "train" ? runJitterMode : undefined,
      run_jitter_factor: sessionExecutionMode === "train" ? runJitterFactor : undefined,
      run_jitter_min: sessionExecutionMode === "train" ? runJitterMin : undefined,
      run_jitter_max: sessionExecutionMode === "train" ? runJitterMax : undefined,
      continue_on_failure: sessionExecutionMode === "train" ? continueOnFailure : undefined,
      session_execution_mode: sessionExecutionMode,
      start_model_mode: effectiveStartModelMode,
      project_model_name: sessionExecutionMode === "train" ? projectModelName : "",
      source_model_id: effectiveStartModelMode === "reuse" ? sourceModelId : "",
      engine,
      max_steps: maxSteps,
      log_interval: logInterval,
      splat_export_interval: splatInterval,
      best_splat_interval: bestSplatInterval,
      best_splat_start_step: bestSplatStartStep,
      save_best_splat: saveBestSplat,
      auto_early_stop: autoEarlyStop,
      early_stop_monitor_interval: earlyStopMonitorInterval,
      early_stop_decision_points: earlyStopDecisionPoints,
      early_stop_min_eval_points: earlyStopMinEvalPoints,
      early_stop_min_step_ratio: earlyStopMinStepRatio,
      early_stop_monitor_min_relative_improvement: earlyStopMonitorMinRelativeImprovement,
      early_stop_eval_min_relative_improvement: earlyStopEvalMinRelativeImprovement,
      early_stop_max_volatility_ratio: earlyStopMaxVolatilityRatio,
      early_stop_ema_alpha: earlyStopEmaAlpha,
      png_export_interval: pngInterval,
      eval_interval: evalInterval,
      save_interval: saveInterval,
      sparse_preference: sparsePreference,
      sparse_merge_selection: sparseMergeSelection,
      densify_from_iter: densifyFromIter,
      densify_until_iter: densifyUntilIter,
      densification_interval: densificationInterval,
      densify_grad_threshold: densifyGradThreshold,
      opacity_threshold: opacityThreshold,
      lambda_dssim: lambdaDssim,
      litegs_target_primitives: litegsTargetPrimitives,
      litegs_alpha_shrink: litegsAlphaShrink,
    };
    localStorage.setItem(getTrainingConfigStorageKey(selectedRunId), JSON.stringify(config));
  }, [mode, tuneStartStep, tuneMinImprovement, tuneEndStep, tuneInterval, tuneScope, trendScope, aiInputMode, baselineSessionIdForAi, warmupAtStart, runCount, runJitterMode, runJitterFactor, runJitterMin, runJitterMax, continueOnFailure, sessionExecutionMode, startModelMode, projectModelName, sourceModelId, engine, maxSteps, logInterval, splatInterval, bestSplatInterval, bestSplatStartStep, saveBestSplat, autoEarlyStop, earlyStopMonitorInterval, earlyStopDecisionPoints, earlyStopMinEvalPoints, earlyStopMinStepRatio, earlyStopMonitorMinRelativeImprovement, earlyStopEvalMinRelativeImprovement, earlyStopMaxVolatilityRatio, earlyStopEmaAlpha, pngInterval, evalInterval, saveInterval, sparsePreference, sparseMergeSelection, densifyFromIter, densifyUntilIter, densificationInterval, densifyGradThreshold, opacityThreshold, lambdaDssim, litegsTargetPrimitives, litegsAlphaShrink, selectedRunId, getTrainingConfigStorageKey]);

  useEffect(() => {
    let cancelled = false;
    const loadReusableModels = async () => {
      setReusableModelsLoading(true);
      setReusableModelsError(null);
      try {
        const res = await api.get("/projects/models");
        const items = Array.isArray(res.data?.models) ? (res.data.models as ReusableModelEntry[]) : [];
        if (!cancelled) {
          setReusableModels(items);
        }
      } catch (err) {
        if (!cancelled) {
          setReusableModels([]);
          setReusableModelsError(err instanceof Error ? err.message : "Failed to load reusable models");
          setSourceModelId("");
        }
      } finally {
        if (!cancelled) {
          setReusableModelsLoading(false);
        }
      }
    };

    loadReusableModels();
    return () => {
      cancelled = true;
    };
  }, [projectId]);

  useEffect(() => {
    if (!showCoreAiSessionControls) {
      return;
    }
    if (sourceModelId && !modeCompatibleReusableModels.some((item) => item.model_id === sourceModelId)) {
      setSourceModelId("");
    }
  }, [
    showCoreAiSessionControls,
    sourceModelId,
    modeCompatibleReusableModels,
  ]);

  // Persist shared image/COLMAP config once per project.
  useEffect(() => {
    if (!canManageColmapImages) {
      // Non-base sessions should not mutate project-level shared config state.
      return;
    }
    if (hydratedSharedProjectRef.current !== projectId) {
      return;
    }
    const sharedConfig = {
      images_resize_enabled: imagesResizeEnabled,
      images_max_size: imagesMaxSize,
      colmap: {
        max_image_size: colmapMaxImageSize,
        peak_threshold: colmapPeakThreshold,
        guided_matching: colmapGuidedMatching,
        camera_model: colmapCameraModel,
        single_camera: colmapSingleCamera,
        camera_params: colmapCameraParams,
        matching_type: colmapMatchingType,
        mapper_num_threads: colmapMapperThreads,
        mapper_min_num_matches: colmapMapperMinNumMatches,
        mapper_abs_pose_min_num_inliers: colmapMapperAbsPoseMinNumInliers,
        mapper_init_min_num_inliers: colmapMapperInitMinNumInliers,
        sift_matching_min_num_inliers: colmapSiftMatchingMinNumInliers,
        run_image_registrator: colmapRunImageRegistrator,
      },
    };
    localStorage.setItem(getSharedConfigStorageKey(), JSON.stringify(sharedConfig));
  }, [canManageColmapImages, imagesResizeEnabled, imagesMaxSize, colmapMaxImageSize, colmapPeakThreshold, colmapGuidedMatching, colmapCameraModel, colmapSingleCamera, colmapCameraParams, colmapMatchingType, colmapMapperThreads, colmapMapperMinNumMatches, colmapMapperAbsPoseMinNumInliers, colmapMapperInitMinNumInliers, colmapSiftMatchingMinNumInliers, colmapRunImageRegistrator, getSharedConfigStorageKey]);

  useEffect(() => {
    let cancelled = false;
    const loadSelectedRunTraining = async () => {
      if (!selectedRunId) return;
      hydratedTrainingRunIdRef.current = "";

      try {
        const res = await api.get(`/projects/${projectId}/runs/${selectedRunId}/config`);
        const runConfig = res.data?.run_config;
        const resolved = runConfig?.resolved_params;
        const requested = runConfig?.requested_params;
        if (!cancelled && runConfig && typeof runConfig === "object") {
          const hydrated = normalizeTrainingConfigForForm({
            ...(requested && typeof requested === "object" ? (requested as Record<string, any>) : {}),
            ...(resolved && typeof resolved === "object" ? (resolved as Record<string, any>) : {}),
          });
          applyResolvedParamsToForm(hydrated, { includeTraining: true, includeShared: false });
          hydratedTrainingRunIdRef.current = selectedRunId;
          return;
        }
      } catch {
        // Run might be a new draft without persisted config yet.
      }

      try {
        const saved = localStorage.getItem(getTrainingConfigStorageKey(selectedRunId));
        if (saved) {
          const parsed = JSON.parse(saved);
          if (!cancelled && parsed && typeof parsed === "object") {
            applyResolvedParamsToForm(normalizeTrainingConfigForForm(parsed as Record<string, any>), { includeTraining: true, includeShared: false });
            hydratedTrainingRunIdRef.current = selectedRunId;
          }
          return;
        }

        // Fallback: if a run-specific cache is missing, use the default training cache.
        const defaultSaved = localStorage.getItem(getTrainingConfigStorageKey());
        if (defaultSaved) {
          const parsedDefault = JSON.parse(defaultSaved);
          if (!cancelled && parsedDefault && typeof parsedDefault === "object") {
            applyResolvedParamsToForm(normalizeTrainingConfigForForm(parsedDefault as Record<string, any>), { includeTraining: true, includeShared: false });
            hydratedTrainingRunIdRef.current = selectedRunId;
            return;
          }
        }
      } catch (err) {
        console.warn("Failed to load saved training config for run", selectedRunId, err);
      }

      if (!cancelled) {
        applyTrainingDefaults(getDefaultProcessConfig());
        hydratedTrainingRunIdRef.current = selectedRunId;
      }
    };

    loadSelectedRunTraining();
    return () => {
      cancelled = true;
    };
  }, [projectId, selectedRunId, getTrainingConfigStorageKey]);

  useEffect(() => {
    let cancelled = false;
    const loadSharedConfigAndBaseProfile = async () => {
      try {
        const res = await api.get(`/projects/${projectId}/shared-config`);
        const shared = res.data?.shared;
        if (shared && typeof shared === "object" && !cancelled) {
          const sharedPayload = {
            images_resize_enabled: typeof shared.images_resize_enabled === "boolean" ? shared.images_resize_enabled : undefined,
            images_max_size: typeof shared.images_max_size === "number" ? shared.images_max_size : undefined,
            colmap: typeof shared.colmap === "object" && shared.colmap ? shared.colmap : undefined,
          };
          applyResolvedParamsToForm(sharedPayload as Record<string, any>, { includeTraining: false, includeShared: true });
          hydratedSharedProjectRef.current = projectId;
        }

        const activeSparseVersion = typeof res.data?.active_sparse_version === "number" ? res.data.active_sparse_version : null;
        const resolvedBaseRunId = typeof res.data?.base_session_id === "string" && res.data.base_session_id
          ? res.data.base_session_id
          : baseSessionId;
        if (!resolvedBaseRunId || !activeSparseVersion) {
          if (!cancelled) setBaseColmapProfile(null);
          return;
        }
        const size = typeof shared?.images_max_size === "number"
          ? shared.images_max_size
          : typeof shared?.colmap?.max_image_size === "number"
            ? shared.colmap.max_image_size
            : null;
        const runName = projectRuns.find((r) => r.run_id === resolvedBaseRunId)?.run_name || resolvedBaseRunId;
        if (!cancelled) {
          setBaseColmapProfile({
            runId: resolvedBaseRunId,
            runName,
            imageSize: size,
            resizeEnabled: typeof size === "number",
          });
        }
      } catch {
        if (canManageColmapImages) {
          try {
            const sharedSaved = localStorage.getItem(getSharedConfigStorageKey());
            if (sharedSaved) {
              const localShared = JSON.parse(sharedSaved);
              if (localShared && typeof localShared === "object" && !cancelled) {
                applyResolvedParamsToForm(localShared as Record<string, any>, { includeTraining: false, includeShared: true });
                hydratedSharedProjectRef.current = projectId;
              }
            }
          } catch {
            // Keep defaults when both backend and local fallback fail.
          }
        }
        if (!cancelled) setBaseColmapProfile(null);
      }
    };

    loadSharedConfigAndBaseProfile();
    return () => {
      cancelled = true;
    };
  }, [projectId, baseSessionId, projectRuns, canManageColmapImages, getSharedConfigStorageKey]);

  const sharedImageSizeMismatch = useMemo(() => {
    if (!baseColmapProfile || !canCreateSessionDraft || !selectedRunSharedOutdated) return false;
    const currentSize = imagesResizeEnabled ? imagesMaxSize ?? null : null;
    const baseSize = baseColmapProfile.resizeEnabled ? baseColmapProfile.imageSize : null;
    return currentSize !== baseSize;
  }, [baseColmapProfile, canCreateSessionDraft, imagesResizeEnabled, imagesMaxSize, selectedRunSharedOutdated]);

  useEffect(() => {
    const checkGpu = async () => {
      try {
        const res = await api.get("/health/gpu");
        // Support both gpu_health response shapes
        setGpuAvailable(Boolean(res.data.gpu_available ?? res.data.available));
      } catch {
        setGpuAvailable(false);
      }
    };
    checkGpu();
  }, []);

  useEffect(() => {
    const fetchLocations = async () => {
      try {
        setLocLoading(true);
        const res = await api.get(`/projects/${projectId}/images/locations`);
        setLocations(res.data.locations || []);
      } catch (err) {
        console.error("Failed to load image locations", err);
        setLocations([]);
      } finally {
        setLocLoading(false);
      }
    };
    fetchLocations();
  }, [projectId]);

  useEffect(() => {
    let isMounted = true;
    const fetchRuns = async () => {
      try {
        const res = await api.get(`/projects/${projectId}/runs`);
        const runs = Array.isArray(res.data?.runs) ? (res.data.runs as ProjectRunInfo[]) : [];
        const baseId = typeof res.data?.base_session_id === "string" ? res.data.base_session_id : "";
        if (!isMounted) return;
        setProjectRuns(runs);
        setBaseSessionId(baseId);
        setCanCreateSessionDraft(Boolean(res.data?.can_create_session));
        setCreateSessionDisabledReason(
          typeof res.data?.can_create_session_reason === "string" && res.data.can_create_session_reason.trim()
            ? res.data.can_create_session_reason
            : "Complete COLMAP on the base session before creating new sessions.",
        );
        const persistedRunId = (localStorage.getItem(getSelectedRunStorageKey()) || "").trim();
        if (!selectedRunId && runs.length > 0) {
          const preferred = persistedRunId && runs.some((r) => r.run_id === persistedRunId)
            ? persistedRunId
            : runs[0].run_id;
          setSelectedRunId(preferred);
        } else if (selectedRunId && !runs.some((r) => r.run_id === selectedRunId)) {
          setSelectedRunId(runs.length > 0 ? runs[0].run_id : "");
        }
      } catch {
        if (!isMounted) return;
        setProjectRuns([]);
        setCanCreateSessionDraft(false);
        setCreateSessionDisabledReason("Complete COLMAP on the base session before creating new sessions.");
      }
    };
    fetchRuns();
    const id = setInterval(fetchRuns, 10000);
    return () => {
      isMounted = false;
      clearInterval(id);
    };
  }, [projectId, selectedRunId, getSelectedRunStorageKey]);

  useEffect(() => {
    if (!selectedRunId) return;
    localStorage.setItem(getSelectedRunStorageKey(), selectedRunId);
  }, [selectedRunId, getSelectedRunStorageKey]);

  useEffect(() => {
    const batchActive = batchTotal > 1 || batchCurrentIndex > 0;
    if (!batchActive) return;
    if (!processingRunId) return;
    if (processingRunId === selectedRunId) return;
    setSelectedRunId(processingRunId);
  }, [batchTotal, batchCurrentIndex, processingRunId, selectedRunId]);

  useEffect(() => {
    if (!canManageColmapImages) {
      if (runColmap) setRunColmap(false);
      if (configTab !== "training") setConfigTab("training");
    }
  }, [canManageColmapImages, runColmap, configTab]);

  useEffect(() => {
    setSelectedModelSnapshot(null);
    setModelSnapshots([]);
    setSelectedModelLayer("final");
  }, [projectId]);

  useEffect(() => {
    const checkOutputs = async () => {
      try {
        const runIdParam = selectedRunId || undefined;
        const [filesRes, statusRes] = await Promise.all([
          api.get(`/projects/${projectId}/files`, { params: { run_id: runIdParam } }),
          api.get(`/projects/${projectId}/status`).catch(() => ({ data: { stage: "idle", message: null, status: "pending" } }))
        ]);

        if (typeof statusRes?.data?.name === "string" && statusRes.data.name.trim()) {
          const resolvedProjectName = statusRes.data.name.trim();
          setProjectDisplayName(resolvedProjectName);
          // If the current run-name still looks like an auto-generated project-id session name,
          // swap only its prefix to the human project name while preserving the session number.
          setNewRunName((prev) => {
            const match = prev.match(/^(.*?)(_session\d+)$/);
            if (!match) return prev;
            const currentPrefix = match[1];
            const suffix = match[2];
            const oldPrefix = sanitizeRunToken(projectId || "project") || "project";
            const newPrefix = sanitizeRunToken(resolvedProjectName || projectId) || oldPrefix;
            if (currentPrefix !== oldPrefix) return prev;
            return `${newPrefix}${suffix}`;
          });
        }
        
        const files = filesRes.data.files;
        // Try to infer outputs presence across possible shapes
        // Treat COLMAP sparse as present only if a reconstruction directory exists and is marked complete
        let sparse = false;
        if (files.sparse) {
          if (Array.isArray(files.sparse) && files.sparse.length > 0) {
            sparse = files.sparse.some((r: any) => {
              if (typeof r === 'object' && r !== null) {
                if (r.complete === true) return true;
                if (Array.isArray(r.files)) {
                  return r.files.some((f: any) => f && (f.name === 'points3D.bin' || f.name === 'points3D.txt'));
                }
              }
              return false;
            });
          } else if (typeof files.sparse === 'object') {
            // Backwards compatibility: if sparse is an object, assume it's present
            sparse = true;
          }
        }
        // Accept splats (object or array), ply, or any other model output
        const model = Boolean(
          files.splats ||
          (Array.isArray(files.ply) && files.ply.length > 0) ||
          (files.ply && !Array.isArray(files.ply))
        );

        const baseUrl = (api.defaults.baseURL || "").replace(/\/$/, "");
        const normalizeUrl = (url?: string | null) => {
          if (!url) return null;
          if (/^https?:\/\//i.test(url)) return url;
          const prefixed = url.startsWith("/") ? url : `/${url}`;
          return baseUrl ? `${baseUrl}${prefixed}` : prefixed;
        };

        const parsePreviewItems = (source: any): PreviewFile[] => {
          if (!source) return [];
          const entries = Array.isArray(source?.items)
            ? source.items
            : Array.isArray(source)
              ? source
              : [];
          return entries
            .map((item: any) => {
              const rawName = item?.name || (typeof item === 'string' ? item.split('/').pop() : undefined) || 'preview.png';
              const relUrl = item?.url || (rawName ? `/projects/${projectId}/previews/${encodeURIComponent(rawName)}` : null);
              const absoluteUrl = normalizeUrl(relUrl);
              if (!absoluteUrl) return null;
              return { name: rawName, url: absoluteUrl };
            })
            .filter(Boolean) as PreviewFile[];
        };

        const buildSnapshotEntries = (source: any): SnapshotEntry[] => {
          if (!Array.isArray(source)) return [];
          return source
            .map((item: any) => {
              const name = item?.name || item?.filename || (typeof item === 'string' ? item.split('/').pop() : undefined);
              const relUrl = item?.url || (name ? `/projects/${projectId}/download/snapshots/${encodeURIComponent(name)}` : null);
              const absoluteUrl = normalizeUrl(relUrl);
              if (!absoluteUrl) return null;
              return {
                name: name ?? 'snapshot',
                url: absoluteUrl,
                step: typeof item?.step === 'number' ? item.step : extractSnapshotStep(name),
                size: item?.size,
                format: item?.format || (name?.toLowerCase().endsWith('.ply') ? 'ply' : 'splat'),
              } as SnapshotEntry;
            })
            .filter(Boolean) as SnapshotEntry[];
        };

        const legacyPreviewList = parsePreviewItems(files.previews);
        const legacySnapshotSource = Array.isArray(files.model_snapshots)
          ? files.model_snapshots
          : Array.isArray(files.ply)
            ? files.ply
            : [];
        const legacySnapshots = buildSnapshotEntries(legacySnapshotSource);

        const enginesData: Record<string, any> = files.engines || {};
        const nextEngineMap: Record<string, EngineOutputBundle> = {};
        Object.entries(enginesData).forEach(([engineName, bundle]: [string, any]) => {
          const previews = parsePreviewItems(bundle?.previews);
          const snapshots = buildSnapshotEntries(Array.isArray(bundle?.model_snapshots) ? bundle.model_snapshots : []);
          const finalModelUrl = normalizeUrl(bundle?.splats?.url);
          const bestModelUrl = normalizeUrl(bundle?.best_splat?.url);
          nextEngineMap[engineName] = {
            name: engineName,
            label: formatEngineLabel(engineName),
            hasModel: Boolean(finalModelUrl || bestModelUrl),
            finalModelUrl,
            bestModelUrl,
            previews,
            snapshots,
          };
        });
        setEngineOutputMap(nextEngineMap);

        const enginesWithModels = Object.values(nextEngineMap).filter((bundle) => bundle.hasModel);
        const availableEngines = Object.keys(nextEngineMap);
        const previousEngineSelection = selectedEngineRef.current;
        let resolvedEngineSelection = previousEngineSelection && nextEngineMap[previousEngineSelection]
          ? previousEngineSelection
          : null;
        if (!resolvedEngineSelection && engine && nextEngineMap[engine]) {
          resolvedEngineSelection = engine;
        }
        if (!resolvedEngineSelection && enginesWithModels.length > 0) {
          const preferred = enginesWithModels.find((bundle) => bundle.name === engine) ?? enginesWithModels[0];
          resolvedEngineSelection = preferred.name;
        }
        if (!resolvedEngineSelection && availableEngines.length > 0) {
          const preferredWithPreviews = availableEngines.find((name) => nextEngineMap[name]?.previews?.length > 0);
          resolvedEngineSelection = preferredWithPreviews ?? availableEngines[0];
        }
        if (resolvedEngineSelection !== selectedEngineRef.current) {
          setSelectedEngineName(resolvedEngineSelection);
        } else if (!resolvedEngineSelection && selectedEngineRef.current !== null && enginesWithModels.length === 0) {
          setSelectedEngineName(null);
        }

        const activeEngineBundle = resolvedEngineSelection ? nextEngineMap[resolvedEngineSelection] : null;
        const previewList = activeEngineBundle ? activeEngineBundle.previews : legacyPreviewList;
        setPngFiles(previewList);
        setSelectedPng((prev) => (prev && previewList.some((item) => item.url === prev) ? prev : null));

        const snapshotEntries = activeEngineBundle ? activeEngineBundle.snapshots : legacySnapshots;
        setModelSnapshots(snapshotEntries);
        setSelectedModelSnapshot((prev) => (prev && snapshotEntries.some((snap) => snap.url === prev) ? prev : null));

        setHasSparseCloud(sparse);
        setHas3DModel(model || enginesWithModels.length > 0);
        
        // Auto-enable layers if outputs exist
        if (sparse) setShowSparseLayer(true);
        // If a 3D model becomes available, mark it present but do not automatically
        // open the viewer to avoid unintentional WebGL canvases capturing input.
        
        // Update processing status
        const status = statusRes.data;
        setBatchTotal(typeof status.batch_total === "number" ? Math.max(0, status.batch_total) : 0);
        setBatchCompleted(typeof status.batch_completed === "number" ? Math.max(0, status.batch_completed) : 0);
        setBatchCurrentIndex(typeof status.batch_current_index === "number" ? Math.max(0, status.batch_current_index) : 0);
        const activeRunIdFromStatus =
          (status.status === "processing" || status.status === "stopping") && typeof status.current_run_id === "string"
            ? status.current_run_id
            : "";
        const statusBusy = status?.status === "processing" || status?.status === "stopping";
        setProcessingRunId(activeRunIdFromStatus);
        const startupWindowActive =
          restartPendingRef.current &&
          startRequestAtMs > 0 &&
          Date.now() - startRequestAtMs < 30000;

        const resolvedCurrentStep =
          typeof status.currentStep === "number"
            ? status.currentStep
            : (typeof status?.last_tuning?.step === "number" ? status.last_tuning.step : undefined);
        const resolvedMaxSteps =
          typeof status.maxSteps === "number"
            ? status.maxSteps
            : (typeof maxSteps === "number" ? maxSteps : undefined);

        const startupStaleTerminalTraining =
          startupWindowActive &&
          status?.status === "processing" &&
          status?.stage === "training" &&
          typeof resolvedCurrentStep === "number" &&
          typeof resolvedMaxSteps === "number" &&
          resolvedMaxSteps > 0 &&
          resolvedCurrentStep >= resolvedMaxSteps;

        const hasFreshStartSignal =
          status?.status === "processing" &&
          (
            status?.stage === "docker" ||
            status?.stage === "queued" ||
            status?.stage === "colmap" ||
            status?.stage === "colmap_only" ||
            status?.stage === "export" ||
            (typeof resolvedCurrentStep === "number" && typeof resolvedMaxSteps === "number" && resolvedCurrentStep < resolvedMaxSteps) ||
            (typeof status?.stage_progress === "number" && status.stage_progress < 100)
          );

        if (
          startupWindowActive &&
          hasFreshStartSignal &&
          activeRunIdFromStatus &&
          selectedRunId &&
          activeRunIdFromStatus === selectedRunId
        ) {
          restartPendingRef.current = false;
          setStartRequestAtMs(0);
        }
        const suppressStoppedAtStart =
          startRequestAtMs > 0 &&
          Date.now() - startRequestAtMs < 15000 &&
          status?.current_run_id === selectedRunId;

        // Immediately after requesting start/restart, ignore stale completed/idle snapshots
        // until worker reports processing/stopping for the new run.
        const waitingForFreshStart =
          startRequestAtMs > 0 &&
          Date.now() - startRequestAtMs < 15000 &&
          (
            (status?.status !== "processing" && status?.status !== "stopping") ||
            startupStaleTerminalTraining
          );
        if (waitingForFreshStart) {
          setProcessing(true);
          setProcessingStatus("Starting...");
          setOverallProgress(0);
          setTrainingCurrentStep(undefined);
          setTrainingMaxSteps(undefined);
          setStageProgress(undefined);
          setCurrentStage("");
          setCurrentStageKey("");
          setStageStatus({ colmap: "pending", training: "pending", export: "pending" });
          setPipelineDone(false);
          setWasStopped(false);
          setStoppedStage(null);
          setCanResume(false);
          setIsStopping(false);
          setStoppingMessage(null);
          return;
        }

        if (restartPendingRef.current && !startupWindowActive) {
          restartPendingRef.current = false;
        }
        const selectedRunIsActive = Boolean(
          selectedRunId &&
          status?.current_run_id === selectedRunId &&
          (["processing", "stopping"].includes(String(status?.status || "")) ||
            (!suppressStoppedAtStart && ["stopped", "failed"].includes(String(status?.status || "")))),
        );
        const batchRunIsActive = Boolean(
          ((typeof status?.batch_total === "number" && status.batch_total > 1) ||
            (typeof status?.batch_current_index === "number" && status.batch_current_index > 0)) &&
          (status?.status === "processing" || status?.status === "stopping"),
        );
        const anyRunIsActiveFromStatus = Boolean(
          statusBusy &&
          (
            (typeof status?.current_run_id === "string" && status.current_run_id.trim()) ||
            (typeof status?.stage === "string" && status.stage !== "idle" && status.stage !== "pending") ||
            typeof resolvedCurrentStep === "number"
          ),
        );
        const statusContextActive = selectedRunIsActive || batchRunIsActive || anyRunIsActiveFromStatus;
        
        // Store training telemetry only for the actively running session.
        setTrainingCurrentStep(statusContextActive ? resolvedCurrentStep : undefined);
        setTrainingMaxSteps(statusContextActive ? resolvedMaxSteps : undefined);
        setStageProgress(statusContextActive ? status.stage_progress : undefined);

        // Use stopped_percentage when stopped, else 'percentage' or 'progress'
        let percent = undefined as number | undefined;
        if (statusContextActive) {
          if (status.status === 'stopped' && typeof status.stopped_percentage === 'number') {
            percent = status.stopped_percentage;
          }
          if (percent === undefined) {
            percent = typeof status.percentage === 'number' ? status.percentage : undefined;
          }
          if (percent === undefined) {
            const p = typeof status.progress === 'number' ? status.progress : parseInt(status.progress || '0');
            percent = Number.isFinite(p) ? p : undefined;
          }
        }
        const pctNum = typeof percent === 'number' ? percent : 0;
        setOverallProgress(Math.max(0, Math.min(100, isNaN(pctNum) ? 0 : pctNum)));

        // Current stage label
        let stageName = "" as string;
        let stageKey: "docker"|"colmap"|"training"|"export"|"" = "";
        const rawStopped = statusContextActive && status.status === 'stopped' && !suppressStoppedAtStart;
        if (rawStopped) {
          stoppedPollCountRef.current += 1;
        } else {
          stoppedPollCountRef.current = 0;
        }
        // Require two consecutive stopped polls before surfacing stopped state.
        // This avoids transient stopped -> processing flicker from polling races.
        const confirmedStopped = rawStopped && stoppedPollCountRef.current >= 2;

        // If stopped, prefer worker-provided stopped_stage for accurate location.
        const effectiveStage = statusContextActive
          ? ((confirmedStopped && status.stopped_stage) ? status.stopped_stage : status.stage)
          : null;
        // workerStoppedStage: the stage where the worker actually stopped (null if not stopped).
        const workerStoppedStage = confirmedStopped ? (status.stopped_stage || status.stage) : null;
        if (effectiveStage === "docker" || effectiveStage === "queued") { stageName = "Docker Worker (Starting)"; stageKey = "docker"; }
        else if (effectiveStage === "colmap" || effectiveStage === "colmap_only") { stageName = "COLMAP (Structure from Motion)"; stageKey = "colmap"; }
        else if (effectiveStage === "training") { stageName = "Training (Gaussian Splatting)"; stageKey = "training"; }
        else if (effectiveStage === "export") { stageName = "Exporting Results"; stageKey = "export"; }
        // If all steps are completed, clear currentStage
        if (sparse && model && stageStatus && stageStatus.colmap === "success" && stageStatus.training === "success" && stageStatus.export === "success") {
          stageName = "";
          stageKey = "";
        }
        setCurrentStageKey(stageKey);
        setCurrentStage(stageName);
        
        // Check if resumable (has sparse or checkpoints)
        const resumable = statusContextActive ? (status.can_resume || sparse) : sparse;
        setCanResume(resumable && status.status !== "processing" && status.status !== "stopping");
        // Track which stage the worker stopped at (prefer stopped_stage)
        setStoppedStage(workerStoppedStage);
        
        // Handle stopping state
        if (statusContextActive && status.status === "stopping") {
          setIsStopping(true);
          setStoppingMessage(status.message || "Will stop after current step completes...");
        } else {
          setIsStopping(false);
          setStoppingMessage(null);
        }
        
        // Update processing flag based on status
        if (statusContextActive && (status.status === "processing" || status.status === "stopping")) {
          setProcessing(true);
        } else {
          setProcessing(false);
        }
        
        const sessionCompleted =
          !statusBusy &&
          (Boolean(model) || selectedRunMeta?.session_status === "completed");

        // Build detailed status message
        let statusMsg: React.ReactNode = null;
        if (!statusContextActive) {
          if (sessionCompleted) {
            statusMsg = (
              <span className="inline-flex items-center gap-1 text-green-700 font-semibold">
                <Check className="inline w-4 h-4 text-green-600" />
                Completed
              </span>
            );
          } else {
            statusMsg = "Session pending";
          }
        } else if (status.status === "failed") {
          statusMsg = (
            <span className="inline-flex items-center gap-1 text-red-700 font-semibold">
              <X className="inline w-4 h-4 text-red-600" />
              Failed: {status.error || status.message || 'Unknown error'}
            </span>
          );
        } else if (status.status === "processing" || status.status === "stopping" || 
            (status.stage && status.stage !== "idle" && status.stage !== "pending")) {
          if (status.message) {
            // Remove emojis from backend messages
            const cleanMsg = status.message.replace(/[\u{1F300}-\u{1F6FF}\u{2700}-\u{27BF}\u{1F900}-\u{1F9FF}\u{1FA70}-\u{1FAFF}\u{2600}-\u{26FF}\u{1F1E6}-\u{1F1FF}\u{1F700}-\u{1F77F}\u{1F780}-\u{1F7FF}\u{1F800}-\u{1F8FF}\u{1F680}-\u{1F6FF}\u{1F600}-\u{1F64F}\u{1F300}-\u{1F5FF}\u{1F900}-\u{1F9FF}\u{1F680}-\u{1F6FF}\u{2600}-\u{26FF}\u{2700}-\u{27BF}]/gu, "");
            statusMsg = cleanMsg;
          } else if (status.stage && status.stage !== "idle" && status.stage !== null) {
            statusMsg = status.stage.charAt(0).toUpperCase() + status.stage.slice(1).replace(/_/g, ' ');
          } else {
            statusMsg = "Processing...";
          }
          // Add step progress if available and only for training
          if (
            status.stage === "training" &&
            typeof status.currentStep === "number" &&
            typeof status.maxSteps === "number" &&
            !isNaN(status.currentStep) &&
            !isNaN(status.maxSteps)
          ) {
            const progress = ((status.currentStep / status.maxSteps) * 100).toFixed(1);
            statusMsg = <>{statusMsg}<br /><br />Progress: Step {status.currentStep.toLocaleString()} / {status.maxSteps.toLocaleString()} ({progress}%)</>;
          }
        } else if (status.stage === "completed" || status.status === "completed") {
          statusMsg = (
            <span className="inline-flex items-center gap-1 text-green-700 font-semibold">
              <Check className="inline w-4 h-4 text-green-600" />
              Completed
            </span>
          );
        }
        console.log('Status data:', { status: status.status, stage: status.stage, message: status.message, currentStep: status.currentStep, maxSteps: status.maxSteps });
        console.log('Generated statusMsg:', statusMsg);
        setProcessingStatus(statusMsg);
        
        // Detect if pipeline is stably stopped.
        const stopped = confirmedStopped;
        setWasStopped(stopped);

        // Determine stage status based on actual pipeline state.
        // Mark previously completed stages as 'success' even when stopped,
        // but keep `pipelineDone` false when the pipeline was stopped.
        const newStatus = { colmap: "pending", training: "pending", export: "pending" };

        // COLMAP: consider it complete if sparse outputs exist, or if COLMAP reached 100% stage_progress,
        // or if the worker stopped after COLMAP (stoppedStage later than colmap), or overall status indicates completed.
        // Consider COLMAP complete only when there's an explicit sparse reconstruction present
        // OR the pipeline reports full completion for that stage (stage_progress >= 100 and overall status is completed).
        const colmapReadyForSession = Boolean(sparse) || Boolean(canCreateSessionDraft);
        let colmapComplete = colmapReadyForSession || ((status.stage === 'colmap' || status.stage === 'colmap_only') && typeof status.stage_progress === 'number' && status.stage_progress >= 100 && status.status === 'completed') || workerStoppedStage === 'training' || workerStoppedStage === 'export' || (statusContextActive && status.status === 'completed' && (status.stage === 'colmap' || status.stage === 'training' || status.stage === 'export'));
        // If the worker stopped at COLMAP but the COLMAP substep actually did NOT finish, do not treat as complete.
        if (workerStoppedStage === 'colmap' && !(Boolean(sparse) || ((status.stage === 'colmap' || status.stage === 'colmap_only') && typeof status.stage_progress === 'number' && status.stage_progress >= 100 && status.status === 'completed'))) {
          colmapComplete = false;
        }
        if (colmapComplete) {
          newStatus.colmap = 'success';
        } else if (statusContextActive && (status.stage === 'colmap' || status.stage === 'colmap_only')) {
          // Show running when actively in COLMAP
          newStatus.colmap = (status.status === 'processing' || status.status === 'stopping') ? 'running' : 'pending';
        }

        // TRAINING: consider it complete if model outputs exist, or training reached 100% stage_progress,
        // or the worker stopped after training (stoppedStage === export) or overall completed.
        let trainingComplete = sessionCompleted || (statusContextActive && status.stage === 'training' && typeof status.stage_progress === 'number' && status.stage_progress >= 100) || workerStoppedStage === 'export' || (statusContextActive && status.status === 'completed' && (status.stage === 'training' || status.stage === 'export'));
        // If the worker stopped at training but training hadn't finished, do not mark as complete
        if (workerStoppedStage === 'training' && !(Boolean(model) || (status.stage === 'training' && typeof status.stage_progress === 'number' && status.stage_progress >= 100))) {
          trainingComplete = false;
        }
        if (trainingComplete) {
          newStatus.training = 'success';
        } else if (statusContextActive && status.stage === 'training') {
          newStatus.training = (status.status === 'processing' || status.status === 'stopping') ? 'running' : 'pending';
        }

        // EXPORT: consider it complete if model outputs exist or overall status is completed
        let exportComplete = sessionCompleted || (statusContextActive && status.status === 'completed' && status.stage === 'export');
        // If worker stopped during export and export did not finish, do not mark as complete
        if (workerStoppedStage === 'export' && !(Boolean(model) || (status.status === 'completed' && status.stage === 'export'))) {
          exportComplete = false;
        }
        if (exportComplete) {
          newStatus.export = 'success';
        } else if (statusContextActive && status.stage === 'export' && status.status === 'processing') {
          newStatus.export = 'running';
        }

        setStageStatus(newStatus);

        // --- Show Completed and hide Stage Status if all are success and not stopped ---
        const allStagesSuccess =
          !statusBusy &&
          ((newStatus.colmap === "success" && newStatus.training === "success" && newStatus.export === "success" && !stopped) ||
            (!statusContextActive && sessionCompleted));
        setPipelineDone(allStagesSuccess);

        if (allStagesSuccess) {
          setOverallProgress(100);
        }

        // --- Fix overall status label ---
        if (allStagesSuccess) {
          setCurrentStage("");
          setCurrentStageKey("");
        }
      } catch (err) {
        console.error("Failed to check outputs", err);
      }
    };
    checkOutputs();
    
    // Poll every 3 seconds to update status
    const interval = setInterval(checkOutputs, 3000);
    return () => clearInterval(interval);
  }, [projectId, show3DModel, selectedRunId, selectedRunMeta?.session_status, startRequestAtMs, showBatchActions]);

  // Compute map center and bounds for auto-fit
  const mapCenter = useMemo(() => {
    if (!locations.length) return [0, 0] as [number, number];
    const avgLat = locations.reduce((sum, p) => sum + p.lat, 0) / locations.length;
    const avgLon = locations.reduce((sum, p) => sum + p.lon, 0) / locations.length;
    return [avgLat, avgLon] as [number, number];
  }, [locations]);

  // Compute bounds for fitBounds
  const mapBounds = useMemo(() => {
    if (!locations.length) return null;
    let minLat = Infinity, maxLat = -Infinity, minLon = Infinity, maxLon = -Infinity;
    locations.forEach(loc => {
      if (loc.lat < minLat) minLat = loc.lat;
      if (loc.lat > maxLat) maxLat = loc.lat;
      if (loc.lon < minLon) minLon = loc.lon;
      if (loc.lon > maxLon) maxLon = loc.lon;
    });
    return [[minLon, minLat], [maxLon, maxLat]];
  }, [locations]);

  const formatBytes = (bytes?: number) => {
    if (!bytes || bytes <= 0) return "â€”";
    const units = ["B", "KB", "MB", "GB"];
    const order = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
    const value = bytes / Math.pow(1024, order);
    return `${order === 0 ? value.toFixed(0) : value.toFixed(1)} ${units[order]}`;
  };

  const [, setMapViewState] = useState<Record<string, number> | null>(null);
  const [hasAutoFitted, setHasAutoFitted] = useState(false);
  const mapRef = useRef<any | null>(null);
  const selectedEngineRef = useRef<string | null>(null);
  const isMountedRef = useRef(false);
  const prevProcessingRef = useRef<boolean>(false);
  const prevSparsePresenceRef = useRef<boolean>(false);
  const stoppedPollCountRef = useRef<number>(0);
  const saveToastTimeoutRef = useRef<number | null>(null);
  const processInfoToastTimeoutRef = useRef<number | null>(null);
  const telemetryGenerationRef = useRef<number>(0);
  const telemetryModalOpenRef = useRef<boolean>(false);
  const restartPendingRef = useRef<boolean>(false);
  const hydratedTrainingRunIdRef = useRef<string>("");
  const hydratedSharedProjectRef = useRef<string>("");
  const etaSampleRef = useRef<{ step: number; ts: number } | null>(null);
  const etaSecondsPerStepRef = useRef<number | null>(null);

  useEffect(() => {
    const trainingActive =
      processing &&
      currentStageKey === "training" &&
      typeof trainingCurrentStep === "number" &&
      typeof trainingMaxSteps === "number" &&
      Number.isFinite(trainingCurrentStep) &&
      Number.isFinite(trainingMaxSteps) &&
      trainingMaxSteps > 0;

    if (!trainingActive) {
      etaSampleRef.current = null;
      etaSecondsPerStepRef.current = null;
      return;
    }

    const step = Math.max(0, Math.floor(trainingCurrentStep));
    const now = Date.now();
    const prev = etaSampleRef.current;

    if (prev && step > prev.step) {
      const deltaTimeSec = (now - prev.ts) / 1000;
      const deltaSteps = step - prev.step;
      const secondsPerStep = deltaTimeSec / deltaSteps;
      if (Number.isFinite(secondsPerStep) && secondsPerStep > 0 && secondsPerStep < 600) {
        const prevEstimate = etaSecondsPerStepRef.current;
        etaSecondsPerStepRef.current =
          prevEstimate === null ? secondsPerStep : (prevEstimate * 0.7 + secondsPerStep * 0.3);
      }
    }

    etaSampleRef.current = { step, ts: now };
  }, [processing, currentStageKey, trainingCurrentStep, trainingMaxSteps, processingRunId, selectedRunId]);

  const resetProgressDisplayForNewRun = () => {
    stoppedPollCountRef.current = 0;
    restartPendingRef.current = true;
    setOverallProgress(0);
    setTrainingCurrentStep(undefined);
    setTrainingMaxSteps(undefined);
    setStageProgress(undefined);
    setCurrentStage("");
    setCurrentStageKey("");
    setStageStatus({ colmap: "pending", training: "pending", export: "pending" });
    setPipelineDone(false);
    setProcessingStatus("Starting...");
    setBatchCompleted(0);
    setBatchCurrentIndex(0);
    telemetryGenerationRef.current += 1;
    setTelemetryData(null);
    setTelemetryError(null);
    setShowTelemetryModal(false);
    setPngFiles([]);
    setSelectedPng(null);
    setModelSnapshots([]);
    setSelectedModelSnapshot(null);
    setHasSparseCloud(false);
    setHas3DModel(false);
  };

  useEffect(() => {
    stoppedPollCountRef.current = 0;
  }, [projectId, selectedRunId]);

  useEffect(() => {
    isMountedRef.current = true;
    return () => {
      isMountedRef.current = false;
      if (saveToastTimeoutRef.current !== null) {
        window.clearTimeout(saveToastTimeoutRef.current);
      }
      if (processInfoToastTimeoutRef.current !== null) {
        window.clearTimeout(processInfoToastTimeoutRef.current);
      }
    };
  }, []);

  useEffect(() => {
    selectedEngineRef.current = selectedEngineName;
  }, [selectedEngineName]);

  useEffect(() => {
    telemetryModalOpenRef.current = showTelemetryModal;
    if (!showTelemetryModal) {
      telemetryGenerationRef.current += 1;
      setTelemetryLoading(false);
      setTelemetryData(null);
      setTelemetryError(null);
    }
  }, [showTelemetryModal]);

  const refreshSparseOptions = useCallback(async () => {
    const fallback = [
      { value: "best", label: "Auto (best available)" },
      { value: "merge_selected", label: "Merge selected folders (manual)" },
    ];
    const formatStats = (entry: any) => {
      if (!entry) return null;
      const parts: string[] = [];
      if (typeof entry.images === "number" && entry.images >= 0) {
        parts.push(`${entry.images.toLocaleString()} img`);
      }
      if (typeof entry.points === "number" && entry.points > 0) {
        parts.push(`${entry.points.toLocaleString()} pt`);
      }
      return parts.length ? parts.join(" ") : null;
    };
    const describeEntry = (rel: string, entry: any) => {
      const base = entry?.label || (rel === "." ? "root" : rel);
      const stats = formatStats(entry);
      return stats ? `${base} (${stats})` : base;
    };
    const applyOptions = (options: Array<{ value: string; label: string }>) => {
      if (!isMountedRef.current) return;
      const list = options.length ? options : fallback;
      setSparseOptions(list);
      setSparsePreference((prev) => (list.some((opt) => opt.value === prev) ? prev : "best"));
      const validValues = new Set(list.map((opt) => opt.value));
      setSparseMergeSelection((prev) => prev.filter((value) => validValues.has(value) && value !== "best" && value !== "merge_selected"));
    };
    setSparseOptionsLoading(true);
    try {
      const res = await api.get(`/projects/${projectId}/sparse/candidates`);
      const data = res.data || {};
      const bestRel = data?.best_relative_path ?? null;
      const rawList = Array.isArray(data?.candidates) ? data.candidates : [];
      const bestEntry = bestRel ? rawList.find((entry: any) => (entry?.relative_path ?? ".") === bestRel) : null;
      const bestTarget = bestRel ? describeEntry(bestRel, bestEntry) : null;
      const bestLabel = bestTarget ? `Auto (best â†’ ${bestTarget})` : "Auto (best available)";

      const formatted: Array<{ value: string; label: string }> = [
        { value: "best", label: bestLabel },
        { value: "merge_selected", label: "Merge selected folders (manual)" },
        ...rawList.map((entry: any) => {
          const rel = entry?.relative_path ?? ".";
          return {
            value: rel,
            label: describeEntry(rel, entry),
          };
        })
      ];
      applyOptions(formatted);
    } catch (err) {
      console.warn("Failed to load sparse candidates", err);
      applyOptions(fallback);
    } finally {
      if (isMountedRef.current) {
        setSparseOptionsLoading(false);
      }
    }
  }, [projectId]);

  useEffect(() => {
    if (newRunName.trim()) return;
    setNewRunName(buildDefaultRunName(projectDisplayName, projectId, projectRuns));
  }, [projectDisplayName, projectId, projectRuns, newRunName]);

  useEffect(() => {
    refreshSparseOptions();
  }, [refreshSparseOptions]);

  useEffect(() => {
    if (prevProcessingRef.current && !processing) {
      refreshSparseOptions();
    }
    prevProcessingRef.current = processing;
  }, [processing, refreshSparseOptions]);

  useEffect(() => {
    if (!prevSparsePresenceRef.current && hasSparseCloud) {
      refreshSparseOptions();
    }
    prevSparsePresenceRef.current = hasSparseCloud;
  }, [hasSparseCloud, refreshSparseOptions]);

  const sparseMergeCandidates = useMemo(
    () => sparseOptions.filter((opt) => opt.value !== "best" && opt.value !== "merge_selected"),
    [sparseOptions],
  );

  const toggleSparseMergeSelection = (value: string) => {
    setSparseMergeSelection((prev) => {
      if (prev.includes(value)) {
        return prev.filter((item) => item !== value);
      }
      return [...prev, value];
    });
  };

  const showMergeReportPanel = sparsePreference === "merge_selected" || sparsePreference.startsWith("_merged/");

  const formatMergeDate = (epoch?: number) => {
    if (!epoch || !Number.isFinite(epoch)) return "Unknown";
    try {
      return new Date(epoch * 1000).toLocaleString();
    } catch {
      return "Unknown";
    }
  };

  const buildSparseMergeNow = async () => {
    if (sparseMergeSelection.length < 2) {
      setSparseMergeBuildMessage("Select at least two folders before building a merge.");
      return;
    }
    setSparseMergeBuildLoading(true);
    setSparseMergeBuildMessage(null);
    setSparseMergeReportError(null);
    try {
      const res = await api.post(`/projects/${projectId}/sparse/merge`, {
        selections: sparseMergeSelection,
      });
      const data = res.data || {};
      const candidate = data.candidate as string | undefined;
      const report = data.report as SparseMergeReport | null | undefined;

      if (candidate) {
        setSparsePreference(candidate);
      }
      if (report) {
        setSparseMergeReport(report);
      }
      setSparseMergeReportCandidate(candidate ?? null);
      await refreshSparseOptions();
      setSparseMergeBuildMessage(candidate ? `Merged model ready: ${candidate}` : "Merged model built.");
    } catch (err: any) {
      const msg = err?.response?.data?.detail || err?.message || "Failed to build merged model";
      setSparseMergeBuildMessage(msg);
    } finally {
      setSparseMergeBuildLoading(false);
    }
  };

  useEffect(() => {
    if (!showMergeReportPanel) {
      setSparseMergeReport(null);
      setSparseMergeReportCandidate(null);
      setSparseMergeReportError(null);
      setSparseMergeReportLoading(false);
      return;
    }

    const candidate = sparsePreference.startsWith("_merged/") ? sparsePreference : undefined;
    let cancelled = false;

    const fetchMergeReport = async () => {
      setSparseMergeReportLoading(true);
      setSparseMergeReportError(null);
      try {
        const res = await api.get(`/projects/${projectId}/sparse/merge-report`, {
          params: candidate ? { candidate } : {},
        });
        if (cancelled) return;
        const data = res.data || {};
        if (!data.available || !data.report) {
          setSparseMergeReport(null);
          setSparseMergeReportCandidate(data.candidate ?? null);
          return;
        }
        setSparseMergeReport(data.report as SparseMergeReport);
        setSparseMergeReportCandidate(data.candidate ?? null);
      } catch (err: any) {
        if (cancelled) return;
        const msg = err?.response?.data?.detail || err?.message || "Failed to load merge report";
        setSparseMergeReportError(msg);
        setSparseMergeReport(null);
      } finally {
        if (!cancelled) {
          setSparseMergeReportLoading(false);
        }
      }
    };

    fetchMergeReport();
    return () => {
      cancelled = true;
    };
  }, [projectId, sparsePreference, showMergeReportPanel]);

  // Auto-fit map to locations only once after locations load
  useEffect(() => {
    if (locations.length && mapBounds && !hasAutoFitted) {
      const padding = 60;
      // If it's a single point, center and zoom in. Otherwise use the map instance to fit bounds.
      if (mapBounds[0][0] === mapBounds[1][0] && mapBounds[0][1] === mapBounds[1][1]) {
        const lat = mapBounds[0][1];
        const lon = mapBounds[0][0];
        // Set direct view state briefly so initialViewState is respected
        setMapViewState({
          latitude: lat,
          longitude: lon,
          zoom: 16,
          transitionDuration: 800
        });
        // After centering, clear controlled viewState so the map becomes interactive
        setTimeout(() => setMapViewState(null), 900);
      } else {
        // Use the map ref to perform a fitBounds call which correctly computes center/zoom
        try {
          if (mapRef.current && typeof mapRef.current.fitBounds === 'function') {
            // maplibre fitBounds expects [[west, south],[east, north]] as provided
            mapRef.current.fitBounds(mapBounds, { padding, duration: 800 });
          } else if (mapRef.current && mapRef.current.getMap && typeof mapRef.current.getMap().fitBounds === 'function') {
            mapRef.current.getMap().fitBounds(mapBounds, { padding, duration: 800 });
          }
        } catch (err) {
          console.warn('Map fitBounds failed', err);
        }
        // Clear any controlled viewState so user interactions remain responsive
        setMapViewState(null);
      }
      setHasAutoFitted(true);
    }
    if (!locations.length) setHasAutoFitted(false);
  }, [locations, mapBounds, hasAutoFitted]);

  

  // Header compacting on scroll
  useEffect(() => {
    let raf = 0;
    const onScroll = () => {
      if (raf) cancelAnimationFrame(raf);
      raf = requestAnimationFrame(() => {
        setHeaderCompact(window.scrollY > 80);
      });
    };
    window.addEventListener('scroll', onScroll, { passive: true });
    // init
    setHeaderCompact(window.scrollY > 80);
    return () => { window.removeEventListener('scroll', onScroll); if (raf) cancelAnimationFrame(raf); };
  }, []);

  const satelliteStyle = useMemo(() => ({
    version: 8,
    sources: {
      esri: {
        type: "raster",
        tiles: [
          "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        ],
        tileSize: 256,
        attribution: "Â© Esri World Imagery",
      },
    },
    layers: [
      { id: "esri", type: "raster", source: "esri" },
    ],
  }), []);

  const mapStyle = useMemo(() => {
    if (basemap === "satellite") return satelliteStyle;
    // OSM raster tile style
    if (basemap === "osm") {
      return {
        version: 8,
        sources: {
          osm: {
            type: 'raster',
            tiles: [
              'https://a.tile.openstreetmap.org/{z}/{x}/{y}.png',
              'https://b.tile.openstreetmap.org/{z}/{x}/{y}.png',
              'https://c.tile.openstreetmap.org/{z}/{x}/{y}.png'
            ],
            tileSize: 256,
            attribution: 'Â© OpenStreetMap contributors'
          }
        },
        layers: [{ id: 'osm', type: 'raster', source: 'osm' }]
      } as any;
    }
    return "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json";
  }, [basemap, satelliteStyle]);

  // When switching between 2D and 3D, animate the map pitch/bearing so users see the change.
  useEffect(() => {
    try {
      const map = mapRef.current && (mapRef.current.getMap ? mapRef.current.getMap() : mapRef.current);
      if (!map) return;
      const target = mapDim === '3d' ? { pitch: 45, bearing: -20, duration: 600 } : { pitch: 0, bearing: 0, duration: 600 };
      if (typeof map.easeTo === 'function') {
        map.easeTo(target);
      }
    } catch (err) {
      console.warn('Failed to animate map for 3D/2D switch', err);
    }
  }, [mapDim]);

  // Optionally enable terrain when in 3D mode if a DEM URL is provided via window.__MAP_DEM_URL
  useEffect(() => {
    try {
      const demUrl = (window as any).__MAP_DEM_URL;
      const map = mapRef.current && (mapRef.current.getMap ? mapRef.current.getMap() : mapRef.current);
      if (!map) return;
      const demSrc = 'bimba3d-dem-src';
      if (mapDim === '3d' && demUrl) {
        if (!map.getSource(demSrc)) {
          try {
            map.addSource(demSrc, { type: 'raster-dem', url: demUrl });
            map.setTerrain({ source: demSrc, exaggeration: 1 });
            // optional sky layer for better 3D feel
            if (!map.getLayer('sky')) {
              map.addLayer({
                id: 'sky',
                type: 'sky',
                paint: {
                  'sky-type': 'atmosphere',
                  'sky-atmosphere-sun': [0.0, 0.0],
                  'sky-atmosphere-sun-intensity': 15
                }
              });
            }
          } catch (e) {
            console.warn('Failed to add DEM source for terrain', e);
          }
        }
      } else {
        // disable terrain if present
        try {
          if (map.getTerrain && map.getTerrain()) {
            map.setTerrain(null);
          }
          if (map.getLayer && map.getLayer('sky')) {
            map.removeLayer('sky');
          }
          if (map.getSource && map.getSource(demSrc)) {
            map.removeSource(demSrc);
          }
        } catch {
          // ignore
        }
      }
    } catch (err) {
      console.warn('Terrain toggle failed', err);
    }
  }, [mapDim]);

  // Add a GeoJSON layer for image locations (reliable rendering and hit-testing)
  useEffect(() => {
    const map = mapRef.current && (mapRef.current.getMap ? mapRef.current.getMap() : mapRef.current);
    if (!map) return;

    const srcId = 'image-locations-src';
    const layerId = 'image-locations-layer';

    const buildGeo = () => ({
      type: 'FeatureCollection',
      features: locations.map(loc => ({ type: 'Feature', properties: { name: loc.name }, geometry: { type: 'Point', coordinates: [loc.lon, loc.lat] } }))
    });

    const onClick = async (e: any) => {
      try {
        const features = e.features && e.features[0];
        if (!features) return;
        setShow3DModel(true);
        setTopView('viewer');
        const base = (window as any).__API_BASE__ || (api && api.defaults && api.defaults.baseURL) || `${window.location.protocol}//${window.location.hostname}:8005`;
        const res = await fetch(`${base}/projects/${projectId}/download/sparse.json`);
        if (!res.ok) return;
        const data = await res.json();
        if (!data.points || !Array.isArray(data.points) || data.points.length === 0) return;
        const sampleCount = Math.min(2000, data.points.length);
        let sx = 0, sy = 0, sz = 0;
        const step = Math.max(1, Math.floor(data.points.length / sampleCount));
        let n = 0;
        for (let i = 0; i < data.points.length; i += step) {
          const p = data.points[i];
          if (!isFinite(p.x) || !isFinite(p.y) || !isFinite(p.z)) continue;
          sx += p.x; sy += p.y; sz += p.z; n += 1;
        }
        if (n > 0) setFocusTarget([sx / n, sy / n, sz / n]);
      } catch (err) {
        console.error('Failed to handle map location click', err);
      }
    };

    const addLayerFn = () => {
      try {
        console.debug('image-locations: addLayerFn running', { showImagesLayer, locationsLength: locations.length });
        if (!showImagesLayer) {
          // ensure removed
          if (map.getLayer && map.getLayer(layerId)) {
            map.off('click', layerId, onClick);
            map.removeLayer(layerId);
          }
          if (map.getSource && map.getSource(srcId)) {
            map.removeSource(srcId);
          }
          return;
        }

        if (!map.getSource(srcId)) {
          map.addSource(srcId, { type: 'geojson', data: buildGeo() });
        } else {
          (map.getSource(srcId) as any).setData(buildGeo());
        }
        if (!map.getLayer(layerId)) {
          map.addLayer({
            id: layerId,
            type: 'circle',
            source: srcId,
            paint: {
              'circle-radius': 6,
              'circle-color': '#10b981',
              'circle-stroke-color': '#ffffff',
              'circle-stroke-width': 2,
              'circle-opacity': 0.95
            }
          });
          map.on('click', layerId, onClick);
          console.debug('image-locations: layer added', layerId);
          // ensure visibility is applied even if style reloads; try a few times if necessary
          const ensureVisibility = (retries = 3) => {
            try {
              map.setLayoutProperty(layerId, 'visibility', showImagesLayer ? 'visible' : 'none');
              console.debug('image-locations: setLayoutProperty visibility', { layerId, visibility: showImagesLayer ? 'visible' : 'none' });
            } catch {
              if (retries > 0) {
                setTimeout(() => ensureVisibility(retries - 1), 200);
              }
            }
          };
          ensureVisibility(5);
        } else {
          // ensure visibility matches `showImagesLayer`
          try {
            map.setLayoutProperty(layerId, 'visibility', showImagesLayer ? 'visible' : 'none');
            console.debug('image-locations: updated visibility', { layerId, visibility: showImagesLayer ? 'visible' : 'none' });
          } catch {
            // ignore if setLayoutProperty fails on some styles
          }
        }
      } catch (err) {
        console.warn('Failed to add image locations layer (inside addLayerFn)', err);
      }
    };

    try {
      if (typeof map.isStyleLoaded === 'function' && !map.isStyleLoaded()) {
        const onLoad = () => {
          addLayerFn();
          map.off('styledata', onLoad);
        };
        map.on('styledata', onLoad);
      } else {
        addLayerFn();
      }
      // also re-run addLayerFn whenever the style updates (covers remounts/style reloads)
      map.on('styledata', addLayerFn);
    } catch (err) {
      console.warn('Failed to schedule image locations layer add', err);
    }

    return () => {
      try {
        map.off('styledata', addLayerFn);
        if (map.getLayer && map.getLayer(layerId)) {
          map.off('click', layerId, onClick);
          map.removeLayer(layerId);
        }
        if (map.getSource && map.getSource(srcId)) {
          map.removeSource(srcId);
        }
      } catch {
        // ignore cleanup errors
      }
    };
  }, [mapRef, locations, projectId, basemap, showImagesLayer, topView]);

  // Ensure immediate layer creation/update right after locations load so markers appear on first load
  useEffect(() => {
    const map = mapRef.current && (mapRef.current.getMap ? mapRef.current.getMap() : mapRef.current);
    if (!map) return;
    const srcId = 'image-locations-src';
    const layerId = 'image-locations-layer';

    if (!showImagesLayer) {
      try { if (map.getLayer && map.getLayer(layerId)) map.setLayoutProperty(layerId, 'visibility', 'none'); } catch { /* ignore visibility update errors */ }
      return;
    }

    if (!locations || locations.length === 0) {
      // nothing to show yet
      return;
    }

    const buildGeo = () => ({
      type: 'FeatureCollection',
      features: locations.map(loc => ({ type: 'Feature', properties: { name: loc.name }, geometry: { type: 'Point', coordinates: [loc.lon, loc.lat] } }))
    });

    try {
      if (!map.getSource(srcId)) {
        map.addSource(srcId, { type: 'geojson', data: buildGeo() });
      } else {
        const s = map.getSource(srcId) as any;
        if (s && typeof s.setData === 'function') s.setData(buildGeo());
      }

      if (!map.getLayer(layerId)) {
        map.addLayer({
          id: layerId,
          type: 'circle',
          source: srcId,
          paint: {
            'circle-radius': 6,
            'circle-color': '#10b981',
            'circle-stroke-color': '#ffffff',
            'circle-stroke-width': 2,
            'circle-opacity': 0.95
          }
        });
      }

      // ensure visibility
      const ensureVisibility = (retries = 5) => {
        try {
          map.setLayoutProperty(layerId, 'visibility', 'visible');
        } catch {
          if (retries > 0) setTimeout(() => ensureVisibility(retries - 1), 150);
        }
      };
      ensureVisibility();
    } catch {
      // ignore
    }
  }, [mapRef, locations, showImagesLayer, topView]);

  // Force-refresh the image layer when returning to the Map tab so markers appear without toggling
  useEffect(() => {
    if (topView !== 'map') return;
    const map = mapRef.current && (mapRef.current.getMap ? mapRef.current.getMap() : mapRef.current);
    if (!map) return;
    console.debug('image-locations: topView effect running, refreshing layer', { topView });
    const srcId = 'image-locations-src';
    const layerId = 'image-locations-layer';

    const buildGeo = () => ({
      type: 'FeatureCollection',
      features: locations.map(loc => ({ type: 'Feature', properties: { name: loc.name }, geometry: { type: 'Point', coordinates: [loc.lon, loc.lat] } }))
    });

    try {
      if (map.getSource && map.getSource(srcId)) {
        const s = map.getSource(srcId) as any;
        if (s && typeof s.setData === 'function') s.setData(buildGeo());
      }

      if (map.getLayer && map.getLayer(layerId)) {
        try {
          map.setLayoutProperty(layerId, 'visibility', showImagesLayer ? 'visible' : 'none');
        } catch {
          // ignore
        }
      } else if (showImagesLayer) {
        // Add layer if missing
        try {
          if (!map.getSource(srcId)) map.addSource(srcId, { type: 'geojson', data: buildGeo() });
          map.addLayer({ id: layerId, type: 'circle', source: srcId, paint: { 'circle-radius': 6, 'circle-color': '#10b981', 'circle-stroke-color': '#ffffff', 'circle-stroke-width': 2, 'circle-opacity': 0.95 } });
        } catch {
          // ignore
        }
      }
    } catch {
      // ignore
    }
  }, [topView, mapRef, locations, showImagesLayer]);

  const handleEngineSelection = (engineName: string) => {
    if (!engineName || !engineOutputMap[engineName]) {
      return;
    }
    const bundle = engineOutputMap[engineName];
    setSelectedEngineName(engineName);
    setPngFiles(bundle.previews);
    setSelectedPng(null);
    setModelSnapshots(bundle.snapshots);
    setSelectedModelSnapshot(null);
    setSelectedModelLayer("final");
    setViewerOutput('model');
    setTopView('viewer');
  };

  const handleProcess = async (skipRestartConfirm = false) => {
    const selectedRunIdAtStart = selectedRunIdRef.current || (selectedRunId || "").trim();
    const selectedRunExists = projectRuns.some((r) => r.run_id === selectedRunIdAtStart);
    const effectiveModeForIntent = engine === "gsplat" ? mode : "baseline";
    const includeSessionControlsForIntent =
      engine === "gsplat" && effectiveModeForIntent === "modified" && tuneScope === "core_ai_optimization";
    const isWarmupStart = includeSessionControlsForIntent && effectiveWarmupAtStart;
    const wantsBatchStart = includeSessionControlsForIntent && !effectiveWarmupAtStart && effectiveRunCount > 1;
    const isRestart = !isWarmupStart && !wantsBatchStart && Boolean(selectedRunIdAtStart);
    const shouldReuseSelectedSession = Boolean(selectedRunIdAtStart) && !wantsBatchStart && !isWarmupStart;
    const batchSeedRunId = wantsBatchStart && selectedRunExists ? selectedRunIdAtStart : "";
    if (isRestart && !skipRestartConfirm) {
      setShowRestartConfirmModal(true);
      return;
    }

    if (isRestart && !selectedRunIdAtStart) {
      setError("Select a session to restart.");
      return;
    }

    if (wantsBatchStart && !batchSeedRunId) {
      setError("Select the target session from Active Session Output before starting batch.");
      return;
    }

    const runNameForRequest = shouldReuseSelectedSession
      ? selectedRunIdAtStart
      : ((batchSeedRunId || newRunName.trim()) || buildDefaultRunName(projectDisplayName, projectId, projectRuns));

    setProcessing(true);
    setError(null);
    setWasStopped(false);
    setStoppedStage(null);
    setStartRequestAtMs(Date.now());
    stoppedPollCountRef.current = 0;

    if (sparsePreference === "merge_selected" && sparseMergeSelection.length < 2) {
      setError("Select at least two sparse folders when using merge mode.");
      setProcessing(false);
      return;
    }

    if (showCoreAiSessionControls && effectiveStartModelMode === "reuse" && !sourceModelId) {
      setError("Select a reusable model or switch start mode to scratch.");
      setProcessing(false);
      return;
    }
    if (hasAiInputModeTrainFlow && !baselineSessionIdForAi) {
      setError("Select a completed baseline session for comparison.");
      setProcessing(false);
      return;
    }

    resetProgressDisplayForNewRun();

    // Determine stage based on checkboxes
    let stage: "full" | "colmap_only" | "train_only";
    if (runColmap && runTraining && runExport) {
      stage = "full";
    } else if (runColmap && !runTraining && !runExport) {
      stage = "colmap_only";
    } else if (!runColmap && runTraining) {
      stage = "train_only";
    } else {
      stage = "full"; // default
    }

    try {
      const effectiveMode = engine === "gsplat" ? mode : "baseline";
      const includeSessionControls =
        engine === "gsplat" && effectiveMode === "modified" && tuneScope === "core_ai_optimization";
      const includeBatchControls = includeSessionControls && !effectiveWarmupAtStart && effectiveRunCount > 1;
      const includeWarmupControls = includeSessionControls && effectiveWarmupAtStart;
      const res = await api.post(`/projects/${projectId}/process`, {
        run_name: runNameForRequest,
        restart_fresh: isRestart,
        mode: effectiveMode,
        tune_start_step: effectiveMode === "modified" ? tuneStartStep : undefined,
        tune_min_improvement: effectiveMode === "modified" ? tuneMinImprovement : undefined,
        tune_end_step: effectiveMode === "modified" ? tuneEndStep : undefined,
        tune_interval: effectiveMode === "modified" ? tuneInterval : undefined,
        tune_scope: effectiveMode === "modified" ? tuneScope : undefined,
        trend_scope:
          effectiveMode === "modified" && tuneScope === "core_ai_optimization" && !aiInputMode && sessionExecutionMode === "train"
            ? trendScope
            : undefined,
        ai_input_mode:
          effectiveMode === "modified" && tuneScope === "core_ai_optimization" && aiInputMode && sessionExecutionMode === "train"
            ? aiInputMode
            : undefined,
        ai_selector_strategy:
          effectiveMode === "modified" && tuneScope === "core_ai_optimization" && sessionExecutionMode === "train"
            ? aiSelectorStrategy
            : undefined,
        baseline_session_id:
          effectiveMode === "modified" && tuneScope === "core_ai_optimization" && aiInputMode
            ? (baselineSessionIdForAi || undefined)
            : undefined,
        session_execution_mode: includeSessionControls ? sessionExecutionMode : undefined,
        warmup_at_start: includeSessionControls ? effectiveWarmupAtStart : undefined,
        run_count: includeBatchControls || includeWarmupControls ? effectiveRunCount : 1,
        run_jitter_mode: includeBatchControls ? runJitterMode : undefined,
        run_jitter_factor: includeBatchControls ? runJitterFactor : undefined,
        run_jitter_min: includeBatchControls ? runJitterMin : undefined,
        run_jitter_max: includeBatchControls ? runJitterMax : undefined,
        continue_on_failure: includeBatchControls ? continueOnFailure : undefined,
        start_model_mode: includeSessionControls ? effectiveStartModelMode : undefined,
        project_model_name:
          includeSessionControls && sessionExecutionMode === "train" && !isReusableWarmStartSelected
            ? (projectModelName.trim() || runNameForRequest || selectedRunId || undefined)
            : undefined,
        source_model_id: includeSessionControls && effectiveStartModelMode === "reuse" ? sourceModelId || undefined : undefined,
        stage,
        engine,
        max_steps: maxSteps,
        log_interval: logInterval,
        splat_export_interval: splatInterval,
        best_splat_interval: bestSplatInterval,
        best_splat_start_step: bestSplatStartStep,
        save_best_splat: saveBestSplat,
        auto_early_stop: autoEarlyStop,
        early_stop_monitor_interval: earlyStopMonitorInterval,
        early_stop_decision_points: earlyStopDecisionPoints,
        early_stop_min_eval_points: earlyStopMinEvalPoints,
        early_stop_min_step_ratio: earlyStopMinStepRatio,
        early_stop_monitor_min_relative_improvement: earlyStopMonitorMinRelativeImprovement,
        early_stop_eval_min_relative_improvement: earlyStopEvalMinRelativeImprovement,
        early_stop_max_volatility_ratio: earlyStopMaxVolatilityRatio,
        early_stop_ema_alpha: earlyStopEmaAlpha,
        png_export_interval: evalInterval,
        eval_interval: evalInterval,
        save_interval: saveInterval,
        densify_from_iter: densifyFromIter,
        densify_until_iter: densifyUntilIter,
        densification_interval: densificationInterval,
        densify_grad_threshold: densifyGradThreshold,
        opacity_threshold: opacityThreshold,
        lambda_dssim: lambdaDssim,
        images_max_size: imagesResizeEnabled ? imagesMaxSize : undefined,
        sparse_preference: sparsePreference,
        sparse_merge_selection: sparsePreference === "merge_selected" ? sparseMergeSelection : undefined,
        litegs_target_primitives: litegsTargetPrimitives,
        litegs_alpha_shrink: litegsAlphaShrink,
        resume: false,
        colmap: {
          ...(imagesResizeEnabled && imagesMaxSize ? { max_image_size: imagesMaxSize } : {}),
          peak_threshold: colmapPeakThreshold,
          guided_matching: colmapGuidedMatching,
          camera_model: colmapCameraModel,
          single_camera: colmapSingleCamera,
          camera_params: colmapCameraParams?.trim() ? colmapCameraParams.trim() : undefined,
          matching_type: colmapMatchingType,
          mapper_num_threads: colmapMapperThreads,
          mapper_min_num_matches: colmapMapperMinNumMatches,
          mapper_abs_pose_min_num_inliers: colmapMapperAbsPoseMinNumInliers,
          mapper_init_min_num_inliers: colmapMapperInitMinNumInliers,
          sift_matching_min_num_inliers: colmapSiftMatchingMinNumInliers,
          run_image_registrator: colmapRunImageRegistrator,
        }
      });

      const seedActionNote = typeof res?.data?.seed_action_note === "string"
        ? res.data.seed_action_note.trim()
        : "";
      if (seedActionNote) {
        setProcessInfoToast(seedActionNote);
        if (processInfoToastTimeoutRef.current !== null) {
          window.clearTimeout(processInfoToastTimeoutRef.current);
        }
        processInfoToastTimeoutRef.current = window.setTimeout(() => {
          setProcessInfoToast("");
          processInfoToastTimeoutRef.current = null;
        }, 9000);
      }
    } catch (err) {
      setError(getApiErrorMessage(err, "Failed to start processing"));
      setProcessing(false);
    }
  };

  const handleContinueBatchFromSelected = async () => {
    if (!selectedRunId) {
      setError("Select a session first.");
      return;
    }
    setProcessing(true);
    setError(null);
    setWasStopped(false);
    setStoppedStage(null);
    setStartRequestAtMs(Date.now());
    stoppedPollCountRef.current = 0;
    resetProgressDisplayForNewRun();
    try {
      await api.post(`/projects/${projectId}/runs/${selectedRunId}/continue-batch`, {
        restart_current: true,
      });
    } catch (err) {
      setError(getApiErrorMessage(err, "Failed to continue batch chain"));
      setProcessing(false);
    }
  };

  const handleResumeProcess = async () => {
    setProcessing(true);
    setError(null);
    setWasStopped(false);
    setStoppedStage(null);
    setStartRequestAtMs(Date.now());
    stoppedPollCountRef.current = 0;
    resetProgressDisplayForNewRun();
    if (sparsePreference === "merge_selected" && sparseMergeSelection.length < 2) {
      setError("Select at least two sparse folders when using merge mode.");
      setProcessing(false);
      return;
    }
    if (showCoreAiSessionControls && effectiveStartModelMode === "reuse" && !sourceModelId) {
      setError("Select a reusable model or switch start mode to scratch.");
      setProcessing(false);
      return;
    }
    if (hasAiInputModeTrainFlow && !baselineSessionIdForAi) {
      setError("Select a completed baseline session for comparison.");
      setProcessing(false);
      return;
    }
    try {
      // Decide which stage to request for resume based on where the worker stopped
      let resumeStage: "full" | "colmap_only" | "train_only" = "train_only";
      if (stoppedStage === 'colmap') resumeStage = 'colmap_only';
      else if (stoppedStage === 'training') resumeStage = 'train_only';
      else resumeStage = 'full';

      // Ensure necessary checkboxes are enabled for resume
      if (stoppedStage === 'colmap') setRunColmap(true);
      if (stoppedStage === 'training') setRunTraining(true);
      if (stoppedStage === 'export') setRunExport(true);

      const effectiveMode = engine === "gsplat" ? mode : "baseline";
      const includeSessionControls =
        engine === "gsplat" && effectiveMode === "modified" && tuneScope === "core_ai_optimization";
      await api.post(`/projects/${projectId}/process`, {
        run_name: selectedRunId || newRunName.trim() || buildDefaultRunName(projectDisplayName, projectId, projectRuns),
        mode: effectiveMode,
        tune_start_step: effectiveMode === "modified" ? tuneStartStep : undefined,
        tune_min_improvement: effectiveMode === "modified" ? tuneMinImprovement : undefined,
        tune_end_step: effectiveMode === "modified" ? tuneEndStep : undefined,
        tune_interval: effectiveMode === "modified" ? tuneInterval : undefined,
        tune_scope: effectiveMode === "modified" ? tuneScope : undefined,
        trend_scope:
          effectiveMode === "modified" && tuneScope === "core_ai_optimization" && !aiInputMode && sessionExecutionMode === "train"
            ? trendScope
            : undefined,
        ai_input_mode:
          effectiveMode === "modified" && tuneScope === "core_ai_optimization" && aiInputMode && sessionExecutionMode === "train"
            ? aiInputMode
            : undefined,
        ai_selector_strategy:
          effectiveMode === "modified" && tuneScope === "core_ai_optimization" && sessionExecutionMode === "train"
            ? aiSelectorStrategy
            : undefined,
        baseline_session_id:
          effectiveMode === "modified" && tuneScope === "core_ai_optimization" && aiInputMode
            ? (baselineSessionIdForAi || undefined)
            : undefined,
        session_execution_mode: includeSessionControls ? sessionExecutionMode : undefined,
        warmup_at_start: includeSessionControls ? effectiveWarmupAtStart : undefined,
        run_count: includeSessionControls ? effectiveRunCount : undefined,
        run_jitter_mode: includeSessionControls && sessionExecutionMode === "train" ? runJitterMode : undefined,
        run_jitter_factor: includeSessionControls && sessionExecutionMode === "train" ? runJitterFactor : undefined,
        run_jitter_min: includeSessionControls && sessionExecutionMode === "train" ? runJitterMin : undefined,
        run_jitter_max: includeSessionControls && sessionExecutionMode === "train" ? runJitterMax : undefined,
        continue_on_failure: includeSessionControls && sessionExecutionMode === "train" ? continueOnFailure : undefined,
        start_model_mode: includeSessionControls ? effectiveStartModelMode : undefined,
        project_model_name:
          includeSessionControls && sessionExecutionMode === "train" && !isReusableWarmStartSelected
            ? (projectModelName.trim() || selectedRunId || newRunName.trim() || undefined)
            : undefined,
        source_model_id: includeSessionControls && effectiveStartModelMode === "reuse" ? sourceModelId || undefined : undefined,
        stage: resumeStage,
        engine,
        max_steps: maxSteps,
        log_interval: logInterval,
        splat_export_interval: splatInterval,
        best_splat_interval: bestSplatInterval,
        best_splat_start_step: bestSplatStartStep,
        save_best_splat: saveBestSplat,
        auto_early_stop: autoEarlyStop,
        early_stop_monitor_interval: earlyStopMonitorInterval,
        early_stop_decision_points: earlyStopDecisionPoints,
        early_stop_min_eval_points: earlyStopMinEvalPoints,
        early_stop_min_step_ratio: earlyStopMinStepRatio,
        early_stop_monitor_min_relative_improvement: earlyStopMonitorMinRelativeImprovement,
        early_stop_eval_min_relative_improvement: earlyStopEvalMinRelativeImprovement,
        early_stop_max_volatility_ratio: earlyStopMaxVolatilityRatio,
        early_stop_ema_alpha: earlyStopEmaAlpha,
        png_export_interval: evalInterval,
        eval_interval: evalInterval,
        save_interval: saveInterval,
        densify_from_iter: densifyFromIter,
        densify_until_iter: densifyUntilIter,
        densification_interval: densificationInterval,
        densify_grad_threshold: densifyGradThreshold,
        opacity_threshold: opacityThreshold,
        lambda_dssim: lambdaDssim,
        images_max_size: imagesResizeEnabled ? imagesMaxSize : undefined,
        sparse_preference: sparsePreference,
        sparse_merge_selection: sparsePreference === "merge_selected" ? sparseMergeSelection : undefined,
        litegs_target_primitives: litegsTargetPrimitives,
        litegs_alpha_shrink: litegsAlphaShrink,
        resume: true,
        colmap: {
          ...(imagesResizeEnabled && imagesMaxSize ? { max_image_size: imagesMaxSize } : {}),
          peak_threshold: colmapPeakThreshold,
          guided_matching: colmapGuidedMatching,
          camera_model: colmapCameraModel,
          single_camera: colmapSingleCamera,
          camera_params: colmapCameraParams?.trim() ? colmapCameraParams.trim() : undefined,
          matching_type: colmapMatchingType,
          mapper_num_threads: colmapMapperThreads,
          mapper_min_num_matches: colmapMapperMinNumMatches,
          mapper_abs_pose_min_num_inliers: colmapMapperAbsPoseMinNumInliers,
          mapper_init_min_num_inliers: colmapMapperInitMinNumInliers,
          sift_matching_min_num_inliers: colmapSiftMatchingMinNumInliers,
          run_image_registrator: colmapRunImageRegistrator,
        }
      });
    } catch (err) {
      setError(getApiErrorMessage(err, "Failed to resume processing"));
      setProcessing(false);
    }
  };

  const handleRenameSelectedRun = async (proposedNameInput?: string): Promise<boolean> => {
    if (!selectedRunId) {
      setError("Select a run to rename.");
      return false;
    }
    const proposedName = (proposedNameInput ?? newRunName).trim();
    if (!proposedName) {
      setError("Enter a new run name before renaming.");
      return false;
    }
    setError(null);
    setIsRenamingRun(true);
    try {
      const res = await api.patch(`/projects/${projectId}/runs/${selectedRunId}`, {
        run_name: proposedName,
      });
      const renamedId = String(res?.data?.run_id || proposedName);
      setSelectedRunId(renamedId);
      setNewRunName(renamedId);
      const runsRes = await api.get(`/projects/${projectId}/runs`);
      const runs = Array.isArray(runsRes.data?.runs) ? (runsRes.data.runs as ProjectRunInfo[]) : [];
      setProjectRuns(runs);
      return true;
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to rename run");
      return false;
    } finally {
      setIsRenamingRun(false);
    }
  };

  const promptRenameCurrentSession = () => {
    if (!selectedRunId) {
      setError("Select a session first.");
      return;
    }
    const current = projectRuns.find((r) => r.run_id === selectedRunId);
    const suggested = (current?.run_name || current?.run_id || selectedRunId).trim();
    setRenameSessionDraft(suggested);
    setShowRenameSessionModal(true);
  };

  const openElevateModelModal = () => {
    if (!selectedRunId) {
      setError("Select a session first.");
      return;
    }
    const current = projectRuns.find((r) => r.run_id === selectedRunId);
    const runLabel = (current?.run_name || current?.run_id || selectedRunId).trim();
    const suggested = buildDefaultModelName(projectDisplayName, projectId, runLabel);
    setElevateModelNameDraft(suggested);
    setShowElevateModelModal(true);
  };

  const confirmElevateModel = async () => {
    if (!selectedRunId) {
      setError("Select a session first.");
      return;
    }
    setIsElevatingModel(true);
    setError(null);
    try {
      const payloadName = elevateModelNameDraft.trim();
      const res = await api.post(`/projects/${projectId}/runs/${selectedRunId}/elevate-model`, {
        model_name: payloadName || undefined,
      });
      const model = res.data?.model as ReusableModelEntry | undefined;
      if (model?.model_id) {
        setReusableModels((prev) => [model, ...prev.filter((item) => item.model_id !== model.model_id)]);
        setStartModelMode("reuse");
        setSourceModelId(model.model_id);
      } else {
        const modelsRes = await api.get("/projects/models");
        const items = Array.isArray(modelsRes.data?.models) ? (modelsRes.data.models as ReusableModelEntry[]) : [];
        setReusableModels(items);
      }
      setShowElevateModelModal(false);
      setElevateModelNameDraft("");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to elevate model");
    } finally {
      setIsElevatingModel(false);
    }
  };

  const confirmRenameCurrentSession = async () => {
    const ok = await handleRenameSelectedRun(renameSessionDraft);
    if (ok) {
      setShowRenameSessionModal(false);
      setRenameSessionDraft("");
    }
  };

  const openNewSessionModal = () => {
    if (!canCreateSessionDraft) {
      setError(createSessionDisabledReason || "Create Session is disabled until base COLMAP is ready.");
      return;
    }
    setError(null);
    setNewSessionConfigSource("current");
    setNewSessionNameDraft(buildDefaultRunName(projectDisplayName, projectId, projectRuns));
    setShowNewSessionModal(true);
  };

  const buildCurrentSessionTrainingParams = (): Record<string, any> => {
    const effectiveMode = engine === "gsplat" ? mode : "baseline";
    return {
      mode: effectiveMode,
      tune_start_step: tuneStartStep,
      tune_min_improvement: tuneMinImprovement,
      tune_end_step: tuneEndStep,
      tune_interval: tuneInterval,
      tune_scope: tuneScope,
      trend_scope: tuneScope === "core_ai_optimization" && !aiInputMode ? trendScope : undefined,
      ai_input_mode: tuneScope === "core_ai_optimization" && aiInputMode ? aiInputMode : undefined,
      ai_selector_strategy:
        tuneScope === "core_ai_optimization" && sessionExecutionMode === "train" ? aiSelectorStrategy : undefined,
      baseline_session_id:
        tuneScope === "core_ai_optimization" && aiInputMode ? (baselineSessionIdForAi || undefined) : undefined,
      warmup_at_start: showCoreAiSessionControls ? effectiveWarmupAtStart : undefined,
      run_count: effectiveRunCount,
      run_jitter_mode: sessionExecutionMode === "train" ? runJitterMode : undefined,
      run_jitter_factor: sessionExecutionMode === "train" ? runJitterFactor : undefined,
      run_jitter_min: sessionExecutionMode === "train" ? runJitterMin : undefined,
      run_jitter_max: sessionExecutionMode === "train" ? runJitterMax : undefined,
      continue_on_failure: sessionExecutionMode === "train" ? continueOnFailure : undefined,
      session_execution_mode: showCoreAiSessionControls ? sessionExecutionMode : undefined,
      start_model_mode: effectiveStartModelMode,
      project_model_name: sessionExecutionMode === "train" && !isReusableWarmStartSelected ? (projectModelName || undefined) : undefined,
      source_model_id: effectiveStartModelMode === "reuse" ? (sourceModelId || undefined) : undefined,
      stage: "train_only",
      engine,
      max_steps: maxSteps,
      log_interval: logInterval,
      splat_export_interval: splatInterval,
      best_splat_interval: bestSplatInterval,
      best_splat_start_step: bestSplatStartStep,
      save_best_splat: saveBestSplat,
      auto_early_stop: autoEarlyStop,
      early_stop_monitor_interval: earlyStopMonitorInterval,
      early_stop_decision_points: earlyStopDecisionPoints,
      early_stop_min_eval_points: earlyStopMinEvalPoints,
      early_stop_min_step_ratio: earlyStopMinStepRatio,
      early_stop_monitor_min_relative_improvement: earlyStopMonitorMinRelativeImprovement,
      early_stop_eval_min_relative_improvement: earlyStopEvalMinRelativeImprovement,
      early_stop_max_volatility_ratio: earlyStopMaxVolatilityRatio,
      early_stop_ema_alpha: earlyStopEmaAlpha,
      png_export_interval: evalInterval,
      eval_interval: evalInterval,
      save_interval: saveInterval,
      densify_from_iter: densifyFromIter,
      densify_until_iter: densifyUntilIter,
      densification_interval: densificationInterval,
      densify_grad_threshold: densifyGradThreshold,
      opacity_threshold: opacityThreshold,
      lambda_dssim: lambdaDssim,
      sparse_preference: sparsePreference,
      sparse_merge_selection: sparseMergeSelection,
      litegs_target_primitives: litegsTargetPrimitives,
      litegs_alpha_shrink: litegsAlphaShrink,
    };
  };

  const persistCurrentConfigLocally = () => {
    const trainingConfig = {
      mode,
      tune_start_step: tuneStartStep,
      tune_min_improvement: tuneMinImprovement,
      tune_end_step: tuneEndStep,
      tune_interval: tuneInterval,
      tune_scope: tuneScope,
      trend_scope: tuneScope === "core_ai_optimization" && !aiInputMode ? trendScope : undefined,
      ai_input_mode: tuneScope === "core_ai_optimization" && aiInputMode ? aiInputMode : undefined,
      ai_selector_strategy:
        tuneScope === "core_ai_optimization" && sessionExecutionMode === "train" ? aiSelectorStrategy : undefined,
      baseline_session_id:
        tuneScope === "core_ai_optimization" && aiInputMode ? (baselineSessionIdForAi || undefined) : undefined,
      warmup_at_start: showCoreAiSessionControls ? effectiveWarmupAtStart : undefined,
      run_count: effectiveRunCount,
      run_jitter_mode: sessionExecutionMode === "train" ? runJitterMode : undefined,
      run_jitter_factor: sessionExecutionMode === "train" ? runJitterFactor : undefined,
      run_jitter_min: sessionExecutionMode === "train" ? runJitterMin : undefined,
      run_jitter_max: sessionExecutionMode === "train" ? runJitterMax : undefined,
      continue_on_failure: sessionExecutionMode === "train" ? continueOnFailure : undefined,
      session_execution_mode: showCoreAiSessionControls ? sessionExecutionMode : undefined,
      start_model_mode: effectiveStartModelMode,
      project_model_name: sessionExecutionMode === "train" && !isReusableWarmStartSelected ? projectModelName : "",
      source_model_id: effectiveStartModelMode === "reuse" ? sourceModelId : "",
      engine,
      max_steps: maxSteps,
      log_interval: logInterval,
      splat_export_interval: splatInterval,
      best_splat_interval: bestSplatInterval,
      best_splat_start_step: bestSplatStartStep,
      save_best_splat: saveBestSplat,
      auto_early_stop: autoEarlyStop,
      early_stop_monitor_interval: earlyStopMonitorInterval,
      early_stop_decision_points: earlyStopDecisionPoints,
      early_stop_min_eval_points: earlyStopMinEvalPoints,
      early_stop_min_step_ratio: earlyStopMinStepRatio,
      early_stop_monitor_min_relative_improvement: earlyStopMonitorMinRelativeImprovement,
      early_stop_eval_min_relative_improvement: earlyStopEvalMinRelativeImprovement,
      early_stop_max_volatility_ratio: earlyStopMaxVolatilityRatio,
      early_stop_ema_alpha: earlyStopEmaAlpha,
      png_export_interval: pngInterval,
      eval_interval: evalInterval,
      save_interval: saveInterval,
      sparse_preference: sparsePreference,
      sparse_merge_selection: sparseMergeSelection,
      densify_from_iter: densifyFromIter,
      densify_until_iter: densifyUntilIter,
      densification_interval: densificationInterval,
      densify_grad_threshold: densifyGradThreshold,
      opacity_threshold: opacityThreshold,
      lambda_dssim: lambdaDssim,
      litegs_target_primitives: litegsTargetPrimitives,
      litegs_alpha_shrink: litegsAlphaShrink,
    };

    localStorage.setItem(getTrainingConfigStorageKey(selectedRunId), JSON.stringify(trainingConfig));

    if (canManageColmapImages) {
      // Keep default + active run training caches in sync for base/shared workflows.
      localStorage.setItem(getTrainingConfigStorageKey(), JSON.stringify(trainingConfig));

      const sharedConfig = {
        images_resize_enabled: imagesResizeEnabled,
        images_max_size: imagesMaxSize,
        colmap: {
          max_image_size: colmapMaxImageSize,
          peak_threshold: colmapPeakThreshold,
          guided_matching: colmapGuidedMatching,
          camera_model: colmapCameraModel,
          single_camera: colmapSingleCamera,
          camera_params: colmapCameraParams,
          matching_type: colmapMatchingType,
          mapper_num_threads: colmapMapperThreads,
          mapper_min_num_matches: colmapMapperMinNumMatches,
          mapper_abs_pose_min_num_inliers: colmapMapperAbsPoseMinNumInliers,
          mapper_init_min_num_inliers: colmapMapperInitMinNumInliers,
          sift_matching_min_num_inliers: colmapSiftMatchingMinNumInliers,
          run_image_registrator: colmapRunImageRegistrator,
        },
      };
      localStorage.setItem(getSharedConfigStorageKey(), JSON.stringify(sharedConfig));
    }
  };

  const buildCurrentSharedConfigPayload = (): Record<string, any> => ({
    images_resize_enabled: imagesResizeEnabled,
    images_max_size: imagesMaxSize,
    colmap: {
      max_image_size: colmapMaxImageSize,
      peak_threshold: colmapPeakThreshold,
      guided_matching: colmapGuidedMatching,
      camera_model: colmapCameraModel,
      single_camera: colmapSingleCamera,
      camera_params: colmapCameraParams,
      matching_type: colmapMatchingType,
      mapper_num_threads: colmapMapperThreads,
      mapper_min_num_matches: colmapMapperMinNumMatches,
      mapper_abs_pose_min_num_inliers: colmapMapperAbsPoseMinNumInliers,
      mapper_init_min_num_inliers: colmapMapperInitMinNumInliers,
      sift_matching_min_num_inliers: colmapSiftMatchingMinNumInliers,
      run_image_registrator: colmapRunImageRegistrator,
    },
  });

  const handleSaveConfig = async () => {
    setIsSavingConfig(true);
    setError(null);

    try {
      const trainingParams = buildCurrentSessionTrainingParams();
      let targetRunId = selectedRunId;

      if (!targetRunId) {
        const seedName = buildDefaultRunName(projectDisplayName, projectId, projectRuns);
        const createRes = await api.post(`/projects/${projectId}/runs`, {
          run_name: seedName,
          resolved_params: trainingParams,
        });
        targetRunId = String(createRes.data?.run_id || seedName);

        const runsRes = await api.get(`/projects/${projectId}/runs`);
        const runs = Array.isArray(runsRes.data?.runs) ? (runsRes.data.runs as ProjectRunInfo[]) : [];
        setProjectRuns(runs);
        setBaseSessionId(typeof runsRes.data?.base_session_id === "string" ? runsRes.data.base_session_id : "");
        setSelectedRunId(targetRunId);
        setNewRunName(targetRunId);
      }

      await api.patch(`/projects/${projectId}/runs/${targetRunId}/config`, {
        requested_params: trainingParams,
        resolved_params: trainingParams,
      });

      if (canManageColmapImages) {
        await api.patch(`/projects/${projectId}/shared-config`, {
          run_id: targetRunId,
          shared: buildCurrentSharedConfigPayload(),
        });
      }

      persistCurrentConfigLocally();
      const toastText = canManageColmapImages
        ? "Saved training + shared config to backend"
        : "Saved training config to backend";
      setConfigSavedToast(toastText);
      if (saveToastTimeoutRef.current !== null) {
        window.clearTimeout(saveToastTimeoutRef.current);
      }
      saveToastTimeoutRef.current = window.setTimeout(() => {
        setConfigSavedToast("");
        saveToastTimeoutRef.current = null;
      }, 2600);
      setShowConfig(false);
    } catch (err) {
      setConfigSavedToast("");
      setError(err instanceof Error ? err.message : "Failed to save config");
    } finally {
      setIsSavingConfig(false);
    }
  };

  const handleCreateSessionDraft = async () => {
    if (!canCreateSessionDraft) {
      setError(createSessionDisabledReason || "Create Session is disabled until base COLMAP is ready.");
      return;
    }
    const draftName = (newSessionNameDraft || "").trim() || buildDefaultRunName(projectDisplayName, projectId, projectRuns);
    setIsCreatingSessionDraft(true);
    setError(null);
    try {
      let resolvedParamsForCreate: Record<string, any> | undefined = buildCurrentSessionTrainingParams();
      if (newSessionConfigSource === "defaults") {
        const defaults = getDefaultProcessConfig();
        const includeSessionControls =
          defaults.engine === "gsplat" &&
          defaults.mode === "modified" &&
          defaults.tune_scope === "core_ai_optimization";
        applyTrainingDefaults(defaults);
        resolvedParamsForCreate = {
          mode: defaults.engine === "gsplat" ? defaults.mode : "baseline",
          tune_start_step: defaults.tune_start_step,
          tune_min_improvement: defaults.tune_min_improvement,
          tune_end_step: defaults.tune_end_step,
          tune_interval: defaults.tune_interval,
          tune_scope: defaults.tune_scope,
          trend_scope:
            defaults.tune_scope === "core_ai_optimization" && !defaults.ai_input_mode
              ? (defaults.trend_scope === "phase" ? "phase" : "run")
              : undefined,
          ai_input_mode:
            defaults.tune_scope === "core_ai_optimization"
              ? (defaults.ai_input_mode || undefined)
              : undefined,
          ai_selector_strategy:
            defaults.tune_scope === "core_ai_optimization"
              ? defaults.ai_selector_strategy
              : undefined,
          baseline_session_id:
            defaults.tune_scope === "core_ai_optimization" && defaults.ai_input_mode
              ? (defaults.baseline_session_id || undefined)
              : undefined,
          warmup_at_start: includeSessionControls ? defaults.warmup_at_start : undefined,
          run_count: includeSessionControls ? defaults.run_count : undefined,
          run_jitter_mode: includeSessionControls ? defaults.run_jitter_mode : undefined,
          run_jitter_factor: includeSessionControls ? defaults.run_jitter_factor : undefined,
          run_jitter_min: includeSessionControls ? defaults.run_jitter_min : undefined,
          run_jitter_max: includeSessionControls ? defaults.run_jitter_max : undefined,
          continue_on_failure: includeSessionControls ? defaults.continue_on_failure : undefined,
          start_model_mode: includeSessionControls ? defaults.start_model_mode : undefined,
          project_model_name: includeSessionControls ? defaults.project_model_name || undefined : undefined,
          source_model_id:
            includeSessionControls && defaults.start_model_mode === "reuse"
              ? defaults.source_model_id || undefined
              : undefined,
          engine: defaults.engine,
          max_steps: defaults.maxSteps,
          log_interval: defaults.logInterval,
          splat_export_interval: defaults.splatInterval,
          best_splat_interval: defaults.bestSplatInterval,
          best_splat_start_step: defaults.bestSplatStartStep,
          save_best_splat: defaults.saveBestSplat,
          auto_early_stop: defaults.auto_early_stop,
          early_stop_monitor_interval: defaults.earlyStopMonitorInterval,
          early_stop_decision_points: defaults.earlyStopDecisionPoints,
          early_stop_min_eval_points: defaults.earlyStopMinEvalPoints,
          early_stop_min_step_ratio: defaults.earlyStopMinStepRatio,
          early_stop_monitor_min_relative_improvement: defaults.earlyStopMonitorMinRelativeImprovement,
          early_stop_eval_min_relative_improvement: defaults.earlyStopEvalMinRelativeImprovement,
          early_stop_max_volatility_ratio: defaults.earlyStopMaxVolatilityRatio,
          early_stop_ema_alpha: defaults.earlyStopEmaAlpha,
          png_export_interval: defaults.evalInterval,
          eval_interval: defaults.evalInterval,
          save_interval: defaults.saveInterval,
          densify_from_iter: defaults.densifyFromIter,
          densify_until_iter: defaults.densifyUntilIter,
          densification_interval: defaults.densificationInterval,
          densify_grad_threshold: defaults.densifyGradThreshold,
          opacity_threshold: defaults.opacityThreshold,
          lambda_dssim: defaults.lambdaDssim,
          sparse_preference: defaults.sparse_preference,
          sparse_merge_selection: defaults.sparse_preference === "merge_selected" ? defaults.sparse_merge_selection : undefined,
          litegs_target_primitives: defaults.litegs_target_primitives,
          litegs_alpha_shrink: defaults.litegs_alpha_shrink,
        };
      }

      const createRes = await api.post(`/projects/${projectId}/runs`, {
        run_name: draftName,
        resolved_params: resolvedParamsForCreate,
      });
      const createdRunId = String(createRes.data?.run_id || draftName);

      const runsRes = await api.get(`/projects/${projectId}/runs`);
      const runs = Array.isArray(runsRes.data?.runs) ? (runsRes.data.runs as ProjectRunInfo[]) : [];
      setProjectRuns(runs);
      setBaseSessionId(typeof runsRes.data?.base_session_id === "string" ? runsRes.data.base_session_id : "");

      setSelectedRunId(createdRunId);
      setNewRunName(createdRunId);
      setShowNewSessionModal(false);
      setShowConfig(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to prepare new session");
    } finally {
      setIsCreatingSessionDraft(false);
    }
  };

  const handleStopProcess = async () => {
    try {
      await api.post(`/projects/${projectId}/stop`);
      setIsStopping(true);
      setStoppingMessage("Will stop after current step completes...");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to request stop");
    }
  };

  // Calculate expected time remaining for overall progress
  const expectedTimeRemaining = (() => {
    // Only show if training is running and steps are available
    if (
      currentStageKey === 'training' &&
      typeof trainingCurrentStep === 'number' &&
      typeof trainingMaxSteps === 'number' &&
      !isNaN(trainingCurrentStep) &&
      !isNaN(trainingMaxSteps) &&
      trainingCurrentStep > 0 &&
      overallProgress > 0 &&
      processing
    ) {
      const stepsLeft = Math.max(0, trainingMaxSteps - trainingCurrentStep);
      if (stepsLeft <= 0) return null;

      let secondsLeft: number | null = null;
      const smoothedSecondsPerStep = etaSecondsPerStepRef.current;
      if (typeof smoothedSecondsPerStep === "number" && Number.isFinite(smoothedSecondsPerStep) && smoothedSecondsPerStep > 0) {
        secondsLeft = smoothedSecondsPerStep * stepsLeft;
      }

      if (secondsLeft !== null && Number.isFinite(secondsLeft) && secondsLeft > 0) {
        return `${formatDurationCompact(secondsLeft)} remaining`;
      }
    }
    return null;
  })();

  const fetchTelemetry = useCallback(async () => {
    const requestGeneration = telemetryGenerationRef.current;
    const runIdForTelemetry = selectedRunId || processingRunId || undefined;
    if (!runIdForTelemetry) {
      if (requestGeneration === telemetryGenerationRef.current) {
        setTelemetryData(null);
      }
      return;
    }
    try {
      setTelemetryLoading(true);
      setTelemetryError(null);
      const res = await api.get(`/projects/${projectId}/telemetry`, {
        params: {
          run_id: runIdForTelemetry,
          log_limit: 5000,
          eval_limit: 20,
          from_start: 1,
        },
      });
      if (
        requestGeneration !== telemetryGenerationRef.current ||
        !telemetryModalOpenRef.current
      ) {
        return;
      }
      setTelemetryData(res.data as TelemetryPayload);
    } catch (err) {
      if (
        requestGeneration !== telemetryGenerationRef.current ||
        !telemetryModalOpenRef.current
      ) {
        return;
      }
      setTelemetryError(err instanceof Error ? err.message : "Failed to load telemetry");
    } finally {
      if (requestGeneration === telemetryGenerationRef.current) {
        setTelemetryLoading(false);
      }
    }
  }, [projectId, selectedRunId, processingRunId]);

  const handleDownloadTelemetryJson = useCallback(async (evt?: { preventDefault?: () => void; stopPropagation?: () => void }) => {
    evt?.preventDefault?.();
    evt?.stopPropagation?.();

    const runIdForTelemetry = (telemetryData?.run_id || selectedRunId || processingRunId || "").trim();
    if (!runIdForTelemetry) {
      setTelemetryError("No session selected for telemetry export.");
      return;
    }

    try {
      setTelemetryDownloadBusy(true);
      setTelemetryError(null);

      const res = await api.get(`/projects/${projectId}/telemetry`, {
        params: {
          run_id: runIdForTelemetry,
          log_limit: 500,
          eval_limit: 100,
          from_start: 1,
        },
      });

      const telemetryPayload = res.data as TelemetryPayload;
      const exportedAt = new Date().toISOString();
      const exportDocument = {
        export_type: "process_tab_full_log",
        exported_at: exportedAt,
        project: {
          id: projectId,
          name: telemetryPayload.project_name || projectDisplayName || projectId,
        },
        run: {
          id: runIdForTelemetry,
          name: selectedRunMeta?.run_name || runIdForTelemetry,
          is_base: selectedRunMeta?.is_base ?? null,
          session_status: selectedRunMeta?.session_status ?? null,
          shared_config_version: selectedRunMeta?.shared_config_version ?? null,
          active_sparse_shared_version: selectedRunMeta?.active_sparse_shared_version ?? null,
          shared_outdated: selectedRunMeta?.shared_outdated ?? null,
        },
        telemetry: telemetryPayload,
      };

      const projectToken = sanitizeFilenameToken(telemetryPayload.project_name || projectDisplayName || projectId) || "project";
      const runToken = sanitizeFilenameToken(runIdForTelemetry) || "run";
      const stamp = exportedAt.replace(/[:.]/g, "-");
      const filename = `${projectToken}_${runToken}_full_log_${stamp}.json`;

      const blob = new Blob([JSON.stringify(exportDocument, null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      a.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      setTelemetryError(err instanceof Error ? err.message : "Failed to download telemetry JSON");
    } finally {
      setTelemetryDownloadBusy(false);
    }
  }, [telemetryData?.run_id, selectedRunId, processingRunId, projectId, projectDisplayName, selectedRunMeta]);

  const handleDownloadTelemetryPdf = useCallback(async (evt?: { preventDefault?: () => void; stopPropagation?: () => void }) => {
    evt?.preventDefault?.();
    evt?.stopPropagation?.();

    const runIdForTelemetry = (telemetryData?.run_id || selectedRunId || processingRunId || "").trim();
    if (!runIdForTelemetry) {
      setTelemetryError("No session selected for telemetry export.");
      return;
    }

    try {
      setTelemetryDownloadBusy(true);
      setTelemetryError(null);

      const res = await api.get(`/projects/${projectId}/telemetry`, {
        params: {
          run_id: runIdForTelemetry,
          log_limit: 500,
          eval_limit: 100,
          from_start: 1,
        },
      });

      const telemetryPayload = res.data as TelemetryPayload;
      const projectName = telemetryPayload.project_name || projectDisplayName || projectId;
      const runName = selectedRunMeta?.run_name || runIdForTelemetry;
      const exportedAt = new Date().toISOString();

      // Create PDF
      const pdf = new jsPDF({ format: "a4", compress: true });
      pdf.setFont("helvetica", "normal");
      let yPos = 15;
      const pageWidth = pdf.internal.pageSize.getWidth();
      const pageHeight = pdf.internal.pageSize.getHeight();
      const margin = 12;
      const contentWidth = pageWidth - 2 * margin;

      const addHeading = (text: string, size: number = 14, isBold: boolean = true) => {
        if (yPos + 8 > pageHeight - 10) {
          pdf.addPage();
          yPos = 15;
        }
        pdf.setFontSize(size);
        if (isBold) pdf.setFont("helvetica", "bold");
        pdf.text(text, margin, yPos);
        pdf.setFont("helvetica", "normal");
        yPos += size / 2.5 + 2;
      };

      const addText = (text: string, size: number = 10) => {
        if (yPos + 4 > pageHeight - 10) {
          pdf.addPage();
          yPos = 15;
        }
        pdf.setFontSize(size);
        const wrapped = pdf.splitTextToSize(text, contentWidth);
        pdf.text(wrapped, margin, yPos);
        yPos += wrapped.length * (size / 2.8) + 1;
      };

      const formatConfigValue = (value: any) => {
        if (value === null || value === undefined) return "-";
        if (typeof value === "boolean") return value ? "Yes" : "No";
        return String(value);
      };

      const addConfigTable = (title: string, entries: Array<[string, any]>) => {
        if (entries.length === 0) return;
        addSection(title);
        autoTable(pdf, {
          startY: yPos,
          margin: { left: margin, right: margin, top: margin, bottom: margin },
          head: [["Field", "Value"]],
          body: entries.map(([key, value]) => [key, formatConfigValue(value)]),
          columnStyles: {
            0: { cellWidth: 45 },
            1: { cellWidth: contentWidth - 45 },
          },
          headStyles: { fontSize: 8, textColor: [255, 255, 255], fillColor: [25, 55, 120] },
          bodyStyles: { fontSize: 7 },
        });
        yPos = ((pdf as any).lastAutoTable?.finalY ?? yPos) + 5;
      };

      const addSection = (title: string) => {
        if (yPos + 8 > pageHeight - 10) {
          pdf.addPage();
          yPos = 15;
        }
        pdf.setFontSize(11);
        pdf.setFont("helvetica", "bold");
        pdf.setTextColor(25, 55, 120);
        pdf.text(title, margin, yPos);
        pdf.setTextColor(0, 0, 0);
        pdf.setFont("helvetica", "normal");
        yPos += 6;
      };

      const addKeyValue = (key: string, value: any, size: number = 9) => {
        if (yPos + 4 > pageHeight - 10) {
          pdf.addPage();
          yPos = 15;
        }
        pdf.setFontSize(size);
        const displayValue = value === null || value === undefined ? "-" : String(value);
        pdf.text(`${key}: ${displayValue}`, margin + 2, yPos);
        yPos += 4;
      };

      // Title
      addHeading("Training Telemetry Report", 18, true);
      addText(`Exported: ${exportedAt}`, 8);
      yPos += 2;

      // Project & Run Info
      addSection("Project Information");
      addKeyValue("Project Name", projectName);
      addKeyValue("Project ID", projectId);
      yPos += 2;

      addSection("Session Information");
      addKeyValue("Session Name", runName);
      addKeyValue("Session ID", runIdForTelemetry);
      addKeyValue("Is Base", selectedRunMeta?.is_base ? "Yes" : "No");
      addKeyValue("Session Status", selectedRunMeta?.session_status || "-");
      yPos += 2;

      // Current Status
      if (telemetryPayload.status) {
        addSection("Current Status");
        addKeyValue("Stage", telemetryPayload.status.stage || "-");
        addKeyValue("Step", telemetryPayload.status.currentStep ? `${telemetryPayload.status.currentStep.toLocaleString()} / ${telemetryPayload.status.maxSteps ? telemetryPayload.status.maxSteps.toLocaleString() : "?"}` : "-");
        addKeyValue("Current Loss", typeof telemetryPayload.status.current_loss === "number" ? telemetryPayload.status.current_loss.toFixed(6) : "-");
        addKeyValue("Message", telemetryPayload.status.message || "-");
        yPos += 2;
      }

      const runConfig = telemetryPayload.run_config;
      const aiInsights = telemetryPayload.ai_insights;
      const resolvedConfig = runConfig?.resolved_params || {};
      const requestedConfig = runConfig?.requested_params || {};

      if (aiInsights && aiInsights.ai_input_mode) {
        addConfigTable("AI Input Insights", [
          ["mode", aiInsights.ai_input_mode],
          ["feature_source", aiInsights.feature_source],
          ["baseline_session_id", aiInsights.baseline_session_id],
          ["heuristic_preset", aiInsights.heuristic_preset],
          ["selected_preset", aiInsights.selected_preset],
          ["cache_used", aiInsights.cache_used],
          ["reward", aiInsights.reward],
          ["reward_positive", aiInsights.reward_positive],
          ["reward_label", aiInsights.reward_label],
          ["reward_mode", aiInsights.reward_mode],
          ["reward_preset", aiInsights.reward_preset],
        ]);
        addConfigTable(
          "AI Initial Parameters",
          Object.entries(aiInsights.initial_params || {}).map(([key, value]) => [key, value])
        );
        addConfigTable(
          "AI Input Features",
          Object.entries(aiInsights.feature_details || {}).map(([key, value]) => [key, value])
        );
      }

      addConfigTable("Run Configuration (resolved)", [
        ["mode", resolvedConfig.mode],
        ["engine", resolvedConfig.engine],
        ["stage", resolvedConfig.stage],
        ["tune_scope", resolvedConfig.tune_scope],
        ["trend_scope", resolvedConfig.trend_scope],
        ["max_steps", resolvedConfig.max_steps],
        ["log_interval", resolvedConfig.log_interval],
        ["eval_interval", resolvedConfig.eval_interval],
        ["save_interval", resolvedConfig.save_interval],
        ["splat_export_interval", resolvedConfig.splat_export_interval],
        ["best_splat_interval", resolvedConfig.best_splat_interval],
        ["best_splat_start_step", resolvedConfig.best_splat_start_step],
        ["densify_from_iter", resolvedConfig.densify_from_iter],
        ["densify_until_iter", resolvedConfig.densify_until_iter],
        ["densification_interval", resolvedConfig.densification_interval],
        ["batch_size", resolvedConfig.batch_size],
        ["tune_start_step", resolvedConfig.tune_start_step],
        ["tune_end_step", resolvedConfig.tune_end_step],
        ["tune_interval", resolvedConfig.tune_interval],
        ["tune_min_improvement", resolvedConfig.tune_min_improvement],
        ["ai_lr_up_multiplier", resolvedConfig.ai_lr_up_multiplier],
        ["ai_lr_down_multiplier", resolvedConfig.ai_lr_down_multiplier],
        ["ai_gate_alpha", resolvedConfig.ai_gate_alpha],
        ["ai_cooldown_intervals", resolvedConfig.ai_cooldown_intervals],
        ["ai_small_change_band", resolvedConfig.ai_small_change_band],
        ["ai_reward_step_weight", resolvedConfig.ai_reward_step_weight],
        ["ai_reward_trend_weight", resolvedConfig.ai_reward_trend_weight],
      ]);

      addConfigTable("Run Configuration (requested)", [
        ["mode", requestedConfig.mode],
        ["engine", requestedConfig.engine],
        ["stage", requestedConfig.stage],
        ["trend_scope", requestedConfig.trend_scope],
        ["tune_interval", requestedConfig.tune_interval],
        ["tune_min_improvement", requestedConfig.tune_min_improvement],
        ["ai_lr_up_multiplier", requestedConfig.ai_lr_up_multiplier],
        ["ai_lr_down_multiplier", requestedConfig.ai_lr_down_multiplier],
        ["ai_gate_alpha", requestedConfig.ai_gate_alpha],
        ["ai_cooldown_intervals", requestedConfig.ai_cooldown_intervals],
        ["ai_small_change_band", requestedConfig.ai_small_change_band],
        ["ai_reward_step_weight", requestedConfig.ai_reward_step_weight],
        ["ai_reward_trend_weight", requestedConfig.ai_reward_trend_weight],
      ]);

      // Latest Eval
      if (telemetryPayload.latest_eval) {
        const latestEval = telemetryPayload.latest_eval;
        addSection("Latest Evaluation Metrics");
        addKeyValue("Eval Step", latestEval.step ? latestEval.step.toLocaleString() : "-");
        addKeyValue("PSNR", typeof latestEval.psnr === "number" ? latestEval.psnr.toFixed(4) : "-");
        addKeyValue("LPIPS", typeof latestEval.lpips === "number" ? latestEval.lpips.toFixed(4) : "-");
        addKeyValue("SSIM", typeof latestEval.ssim === "number" ? latestEval.ssim.toFixed(4) : "-");
        addKeyValue("Gaussians", latestEval.num_gaussians ? latestEval.num_gaussians.toLocaleString() : "-");
        yPos += 2;
      }

      // Events
      if (telemetryPayload.event_rows && telemetryPayload.event_rows.length > 0) {
        addSection("Important Events");
        const eventRows = telemetryPayload.event_rows;
        pdf.setFontSize(8);
        const eventData = eventRows.map((row) => [
          row.timestamp ? row.timestamp.substring(0, 19) : "-",
          row.type || "-",
          row.step ? String(row.step) : "-",
          row.summary ? (row.summary.length > 120 ? row.summary.substring(0, 117) + "..." : row.summary) : "-",
        ]);
        autoTable(pdf, {
          startY: yPos,
          margin: { left: margin, right: margin, top: margin, bottom: margin },
          head: [["Time", "Type", "Step", "Summary"]],
          body: eventData,
          columnStyles: {
            0: { cellWidth: 24 },
            1: { cellWidth: 18 },
            2: { cellWidth: 10 },
            3: { cellWidth: contentWidth - 52 },
          },
          headStyles: { fontSize: 8, textColor: [255, 255, 255], fillColor: [25, 55, 120] },
          bodyStyles: { fontSize: 7 },
        });
        yPos = ((pdf as any).lastAutoTable?.finalY ?? yPos) + 5;
      }

      // Training Rows
      if (telemetryPayload.training_rows && telemetryPayload.training_rows.length > 0) {
        addSection("Training Log");
        const trainingRows = telemetryPayload.training_rows;
        pdf.setFontSize(8);
        const trainingData = trainingRows.map((row) => [
          row.timestamp ? row.timestamp.substring(0, 19) : "-",
          row.step ? String(row.step) : "-",
          row.max_steps ? `${row.step}/${row.max_steps}` : "-",
          typeof row.loss === "number" ? row.loss.toFixed(6) : "-",
          typeof row.elapsed_seconds === "number" ? row.elapsed_seconds.toFixed(1) + "s" : "-",
          row.eta || "-",
          row.speed || "-",
        ]);
        autoTable(pdf, {
          startY: yPos,
          margin: { left: margin, right: margin, top: margin, bottom: margin },
          head: [["Time", "Step", "Progress", "Loss", "Elapsed", "ETA", "Speed"]],
          body: trainingData,
          columnStyles: {
            0: { cellWidth: 25 },
            1: { cellWidth: 12 },
            2: { cellWidth: 15 },
            3: { cellWidth: 16 },
            4: { cellWidth: 14 },
            5: { cellWidth: 14 },
            6: { cellWidth: 12 },
          },
          headStyles: { fontSize: 8, textColor: [255, 255, 255], fillColor: [25, 55, 120] },
          bodyStyles: { fontSize: 7 },
        });
        yPos = ((pdf as any).lastAutoTable?.finalY ?? yPos) + 5;
      }

      // Eval Rows
      if (telemetryPayload.eval_rows && telemetryPayload.eval_rows.length > 0) {
        addSection("Evaluation Metrics (All)");
        pdf.setFontSize(8);
        const evalData = telemetryPayload.eval_rows.map((row) => [
          row.step ? String(row.step) : "-",
          typeof row.psnr === "number" ? row.psnr.toFixed(4) : "-",
          typeof row.lpips === "number" ? row.lpips.toFixed(4) : "-",
          typeof row.ssim === "number" ? row.ssim.toFixed(4) : "-",
          row.num_gaussians ? String(row.num_gaussians.toLocaleString()) : "-",
        ]);
        autoTable(pdf, {
          startY: yPos,
          margin: { left: margin, right: margin, top: margin, bottom: margin },
          head: [["Step", "PSNR", "LPIPS", "SSIM", "Gaussians"]],
          body: evalData,
          columnStyles: {
            0: { cellWidth: 20 },
            1: { cellWidth: 25 },
            2: { cellWidth: 25 },
            3: { cellWidth: 25 },
            4: { cellWidth: contentWidth - 95 },
          },
          headStyles: { fontSize: 8, textColor: [255, 255, 255], fillColor: [25, 55, 120] },
          bodyStyles: { fontSize: 7 },
        });
      }

      // Save PDF
      const projectToken = sanitizeFilenameToken(projectName) || "project";
      const runToken = sanitizeFilenameToken(runIdForTelemetry) || "run";
      const stamp = exportedAt.replace(/[:.]/g, "-").substring(0, 15);
      const filename = `${projectToken}_${runToken}_log_${stamp}.pdf`;

      pdf.save(filename);
    } catch (err) {
      setTelemetryError(err instanceof Error ? err.message : "Failed to download telemetry PDF");
    } finally {
      setTelemetryDownloadBusy(false);
    }
  }, [telemetryData?.run_id, selectedRunId, processingRunId, projectId, projectDisplayName, selectedRunMeta]);

  useEffect(() => {
    if (!showTelemetryModal) return;
    void fetchTelemetry();
    if (telemetryDownloadBusy) return;
    const pollId = setInterval(() => {
      void fetchTelemetry();
    }, 3000);
    return () => clearInterval(pollId);
  }, [showTelemetryModal, fetchTelemetry, telemetryDownloadBusy]);

  const engineOptions = Object.values(engineOutputMap).filter((bundle) => bundle.hasModel);
  const telemetryBestLoss = useMemo(() => {
    const summary = telemetryData?.training_summary;
    const bestLoss = typeof summary?.best_loss === "number" ? summary.best_loss : null;
    const bestStep = typeof summary?.best_loss_step === "number" ? summary.best_loss_step : null;
    return { bestStep, bestLoss };
  }, [telemetryData]);
  const telemetryBestTrackingStartStep = useMemo(() => {
    const resolved = telemetryData?.run_config?.resolved_params;
    if (!resolved || typeof resolved !== "object") return null;
    const value = (resolved as Record<string, unknown>)["best_splat_start_step"];
    return typeof value === "number" ? value : null;
  }, [telemetryData]);
  const hasEngineOutputs = Object.keys(engineOutputMap).length > 0;
  const showEngineDropdown = engineOptions.length > 1;
  const activeEngineBundle = selectedEngineName ? engineOutputMap[selectedEngineName] : null;
  const selectedLayerModelUrl = selectedModelLayer === "best"
    ? (activeEngineBundle?.bestModelUrl || activeEngineBundle?.finalModelUrl || null)
    : (activeEngineBundle?.finalModelUrl || activeEngineBundle?.bestModelUrl || null);
  const selectedLayerModelLabel = selectedModelLayer === "best" ? "Best Splat" : "Final Splat";
  const finalModelAvailable = activeEngineBundle ? activeEngineBundle.hasModel : has3DModel;
  const viewerModelAvailable = finalModelAvailable || modelSnapshots.length > 0 || Boolean(selectedModelSnapshot);
  const selectedEngineLabel = activeEngineBundle?.label;
  const selectedPngEntry = selectedPng ? pngFiles.find((file) => file.url === selectedPng) : undefined;
  const showFinalModelSection = hasEngineOutputs || has3DModel || pngFiles.length > 0 || modelSnapshots.length > 0;
  const engineSelectValue = selectedEngineName ?? engineOptions[0]?.name ?? "";

  return (
    <div className="max-w-7xl space-y-4">
      {configSavedToast && (
        <div className="fixed bottom-4 right-4 z-[1100] pointer-events-none">
          <div className="inline-flex items-center gap-2 rounded-lg border border-emerald-200 bg-emerald-50 px-3 py-2 text-xs font-semibold text-emerald-800 shadow-lg">
            <Check className="w-4 h-4 text-emerald-600" />
            <span>{configSavedToast}</span>
          </div>
        </div>
      )}
      {processInfoToast && (
        <div className={`fixed right-4 z-[1090] pointer-events-none ${configSavedToast ? "bottom-16" : "bottom-4"}`}>
          <div className="inline-flex max-w-[32rem] items-start gap-2 rounded-lg border border-sky-200 bg-sky-50 px-3 py-2 text-xs font-semibold text-sky-900 shadow-lg">
            <Info className="mt-0.5 w-4 h-4 text-sky-700" />
            <span>{processInfoToast}</span>
          </div>
        </div>
      )}
      {!gpuAvailable && (
        <div className="bg-yellow-50 border-l-4 border-yellow-400 p-3 rounded flex items-center gap-3 text-xs text-yellow-800">
          <svg className="h-4 w-4 text-yellow-500" viewBox="0 0 20 20" fill="currentColor">
            <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
          </svg>
          <div>
            <strong>No GPU detected.</strong> Training will run on CPU and be significantly slower.
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 xl:grid-cols-4 gap-4">
        {/* Left Sidebar - Basic Controls */}
        <div className="space-y-3">
          <div className="bg-white rounded-lg border border-slate-200 shadow-sm p-4 space-y-3">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs uppercase font-semibold text-slate-500">Pipeline</p>
                <h3 className="text-base font-bold text-slate-900">Stages</h3>
              </div>
              <div className="flex items-center gap-2">
                <button
                  onClick={openNewSessionModal}
                  disabled={!canCreateSessionDraft}
                  title={!canCreateSessionDraft ? createSessionDisabledReason : "Create a new session draft"}
                  className="inline-flex items-center gap-1 px-3 py-1.5 rounded-lg border border-blue-200 text-xs font-semibold text-blue-700 hover:bg-blue-50 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  + New Session
                </button>
                <button
                  onClick={() => setShowConfig(true)}
                  className="inline-flex items-center gap-1 px-3 py-1.5 rounded-lg border border-slate-200 text-xs font-semibold text-slate-700 hover:bg-slate-50"
                >
                  <Settings2 className="w-4 h-4" />
                  Config
                </button>
              </div>
            </div>

            <div className="rounded-lg border border-slate-200 bg-slate-50 p-2">
              <label className="block text-[11px] font-semibold text-slate-600 mb-1">Active Session Output</label>
              {projectRuns.length === 0 ? (
                <div className="w-full px-2 py-1.5 text-xs border border-slate-300 rounded-md bg-white text-slate-500">
                  Latest (no named sessions yet)
                </div>
              ) : (
                <details className="group relative">
                  <summary className="w-full px-2 py-1.5 text-xs border border-slate-300 rounded-md bg-white list-none cursor-pointer flex items-center justify-between gap-2">
                    <span className="truncate">
                      {selectedRunId
                        ? (projectRuns.find((r) => r.run_id === selectedRunId)?.run_name || selectedRunId)
                        : "Select session"}
                    </span>
                    <span className="inline-flex items-center gap-1 shrink-0">
                      {selectedRunIsProcessing ? (
                        <Clock className="w-3.5 h-3.5 text-blue-600 animate-pulse" />
                      ) : selectedRunMeta?.session_status === "completed" ? (
                        <Check className="w-3.5 h-3.5 text-green-600" />
                      ) : null}
                    </span>
                  </summary>
                  <div className="absolute z-20 mt-1 w-full max-h-64 overflow-auto rounded-md border border-slate-200 bg-white shadow-lg">
                    {projectRuns.map((run) => {
                      const isBase = run.run_id === baseSessionId || run.is_base;
                      const isSelected = run.run_id === selectedRunId;
                      const isProcessingRun = Boolean(processingRunId) && run.run_id === processingRunId;
                      return (
                        <button
                          key={run.run_id}
                          type="button"
                          onClick={(event) => {
                            selectedRunIdRef.current = run.run_id;
                            setSelectedRunId(run.run_id);
                            const detailsEl = event.currentTarget.closest("details") as HTMLDetailsElement | null;
                            if (detailsEl) detailsEl.open = false;
                          }}
                          className={`w-full px-2 py-2 text-left text-xs flex items-center justify-between gap-2 hover:bg-slate-50 ${isSelected ? "bg-blue-50" : ""}`}
                        >
                          <span className="truncate">
                            {run.run_name || run.run_id}
                            {isBase ? " [BASE]" : ""}
                          </span>
                          <span className="inline-flex items-center shrink-0">
                            {isProcessingRun ? (
                              <Clock className="w-3.5 h-3.5 text-blue-600 animate-pulse" />
                            ) : run.session_status === "completed" ? (
                              <Check className="w-3.5 h-3.5 text-green-600" />
                            ) : null}
                          </span>
                        </button>
                      );
                    })}
                  </div>
                </details>
              )}
              <div className="mt-2 flex items-center justify-between gap-2">
                <span className="text-[11px] text-slate-500 truncate">
                  {selectedRunId
                    ? (projectRuns.find((r) => r.run_id === selectedRunId)?.run_name || selectedRunId)
                    : "No session selected"}
                </span>
                {baseSessionId && selectedRunId === baseSessionId && (
                  <span className="px-1.5 py-0.5 text-[10px] font-semibold rounded bg-emerald-100 text-emerald-700">BASE</span>
                )}
                <div className="flex items-center gap-1">
                  <button
                    type="button"
                    onClick={openElevateModelModal}
                    disabled={!selectedRunId || isElevatingModel || processing || isStopping}
                    className="px-2 py-1 text-[11px] font-semibold rounded border border-violet-300 text-violet-700 hover:bg-violet-50 disabled:opacity-50 disabled:cursor-not-allowed"
                    title="Promote this session output to reusable model"
                  >
                    {isElevatingModel ? "Elevating..." : "Elevate"}
                  </button>
                  <button
                    type="button"
                    onClick={promptRenameCurrentSession}
                    disabled={!selectedRunId || isRenamingRun}
                    className="px-2 py-1 text-[11px] font-semibold rounded border border-slate-300 text-slate-700 hover:bg-slate-100 disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {isRenamingRun ? "Renaming..." : "Rename"}
                  </button>
                </div>
              </div>
            </div>

            {!canManageColmapImages && sharedImageSizeMismatch && baseColmapProfile && (
              <div className="rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-900">
                Shared image size changed from base session <strong>{baseColmapProfile.runName}</strong>
                {baseColmapProfile.resizeEnabled && typeof baseColmapProfile.imageSize === "number"
                  ? ` (${baseColmapProfile.imageSize}px)`
                  : " (original size)"}
                . Current shared value is {imagesResizeEnabled && typeof imagesMaxSize === "number" ? `${imagesMaxSize}px` : "original size"}.
                Re-run COLMAP on the base session before trusting training outputs in this session.
              </div>
            )}
            {selectedRunSharedOutdated && (
              <div className="rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-900">
                This session was created with an older base shared configuration. Re-run from the base session or create a fresh session to use the latest shared image/COLMAP settings.
              </div>
            )}

            <div className="space-y-2">
              {/* --- STAGE LABELS --- */}
              {canManageColmapImages && (
              <label className="flex items-center justify-between px-3 py-2 rounded-lg border border-slate-200 bg-slate-50">
                <div className="flex items-center gap-2">
                  <input type="checkbox" className="w-4 h-4 accent-blue-600" checked={runColmap} onChange={(e) => {
                    const val = e.target.checked;
                    setRunColmap(val);
                    // If user unchecks COLMAP and it's not already successful, disable downstream steps
                    if (!val && stageStatus.colmap !== 'success') {
                      setRunTraining(false);
                      setRunExport(false);
                    }
                  }} disabled={processing || isStopping} />
                  <span className={stageStatus.colmap === "success" ? "text-xs font-medium text-green-700" : "text-xs font-medium text-slate-800"}>COLMAP</span>
                </div>
                {stageStatus.colmap === "success" && <Check className="w-4 h-4 text-green-600" />}
                {stageStatus.colmap === "failed" && <X className="w-4 h-4 text-red-600" />}
                {stageStatus.colmap === "running" && <Clock className="w-4 h-4 text-blue-600 animate-pulse" />}
              </label>
              )}
              <label className="flex items-center justify-between px-3 py-2 rounded-lg border border-slate-200 bg-slate-50">
                <div className="flex items-center gap-2">
                  <input type="checkbox" className="w-4 h-4 accent-blue-600" checked={runTraining} 
                    onChange={(e) => {
                      const val = e.target.checked;
                      setRunTraining(val);
                      // If user unchecks TRAINING and it's not already successful, disable export
                      if (!val && stageStatus.training !== 'success') {
                        setRunExport(false);
                      }
                    }}
                    disabled={processing || isStopping || (canManageColmapImages && !runColmap && stageStatus.colmap !== "success")}
                  />
                  <span className={stageStatus.training === "success" ? "text-xs font-medium text-green-700" : "text-xs font-medium text-slate-800"}>Training</span>
                </div>
                {stageStatus.training === "success" && <Check className="w-4 h-4 text-green-600" />}
                {stageStatus.training === "failed" && <X className="w-4 h-4 text-red-600" />}
                {stageStatus.training === "running" && <Clock className="w-4 h-4 text-blue-600 animate-pulse" />}
              </label>
              <label className="flex items-center justify-between px-3 py-2 rounded-lg border border-slate-200 bg-slate-50">
                <div className="flex items-center gap-2">
                  <input type="checkbox" className="w-4 h-4 accent-blue-600" checked={runExport} 
                    onChange={(e) => setRunExport(e.target.checked)}
                    disabled={processing || isStopping || (!runTraining && stageStatus.training !== "success" && runExport) || (canManageColmapImages && !runColmap && stageStatus.colmap !== "success" && !runTraining)}
                  />
                  <span className={stageStatus.export === "success" ? "text-xs font-medium text-green-700" : "text-xs font-medium text-slate-800"}>Export</span>
                </div>
                {stageStatus.export === "success" && <Check className="w-4 h-4 text-green-600" />}
                {stageStatus.export === "failed" && <X className="w-4 h-4 text-red-600" />}
                {stageStatus.export === "running" && <Clock className="w-4 h-4 text-blue-600 animate-pulse" />}
              </label>
              {/* --- END STAGE LABELS --- */}

              <div className="grid grid-cols-1 gap-2 text-xs text-slate-700">
                {/* Auto early stop option moved to config modal */}
              </div>

              {error && (
                <div className="text-xs text-rose-700 bg-rose-50 border border-rose-200 rounded-lg px-3 py-2">
                  {error}
                </div>
              )}

              {isStopping && stoppingMessage && (
                <div className="text-xs text-orange-700 bg-orange-50 border border-orange-200 rounded-lg px-3 py-2">
                  <p className="font-semibold mb-0.5">Stopping...</p>
                  <p>{stoppingMessage}</p>
                </div>
              )}

              {densifyBlockedReason && !processing && (
                <div className="text-xs text-rose-700 bg-rose-50 border border-rose-200 rounded-lg px-3 py-2">
                  <p className="font-semibold mb-0.5">Adjust densification schedule</p>
                  <p>{densifyBlockedReason}</p>
                </div>
              )}

              {/* Start/Resume Buttons - Side by Side */}
              <div className="grid gap-2">
                {processing ? (
                  // Show only Stop button when processing
                  <button
                    onClick={handleStopProcess}
                    className="w-full inline-flex items-center justify-center gap-2 px-4 py-3 rounded-lg bg-red-600 hover:bg-red-700 text-white font-semibold shadow-sm"
                  >
                    <Square className="w-4 h-4" />
                    Stop Processing
                  </button>
                ) : (
                  // Show Start/Restart and Resume buttons when not processing
                  <>
                    <button
                      onClick={() => {
                        void handleProcess();
                      }}
                      disabled={processing || densifyScheduleBlocked || (canManageColmapImages && !runColmap && stageStatus.colmap !== "success") || (!runTraining && stageStatus.training !== "success" && runExport) || (!runColmap && !runTraining && !runExport)}
                      className="w-full inline-flex items-center justify-center gap-2 px-4 py-3 rounded-lg bg-green-600 hover:bg-green-700 text-white font-semibold shadow-sm disabled:bg-slate-300 disabled:cursor-not-allowed"
                    >
                      <Play className="w-4 h-4" />
                      {(pipelineDone || wasStopped)
                        ? ((showBatchActions || warmupAtStart) ? "Batch Restart" : "Restart Processing")
                        : ((showBatchActions || warmupAtStart) ? "Batch Start" : "Start Processing")}
                    </button>
                    {/* --- RESUME BUTTON LOGIC --- */}
                    {canResume && wasStopped ? (
                      // Show resume only when the stage that was stopped is selected
                      ((stoppedStage === 'colmap' && runColmap) || (stoppedStage === 'training' && runTraining) || (stoppedStage === 'export' && runExport) || (stoppedStage == null && canResume)) ? (
                        <button
                          className="w-full inline-flex items-center justify-center gap-2 px-4 py-3 rounded-lg bg-blue-600 hover:bg-blue-700 text-white font-semibold shadow-sm disabled:bg-slate-300 disabled:cursor-not-allowed"
                          onClick={() => {
                            // Ensure required stages are checked for resume
                            if (stoppedStage === 'colmap' && stageStatus.colmap !== "success") setRunColmap(true);
                            if (stoppedStage === 'training' && stageStatus.training !== "success") setRunTraining(true);
                            if (stoppedStage === 'export' && stageStatus.export !== "success") setRunExport(true);
                            if (!densifyScheduleBlocked) {
                              handleResumeProcess();
                            }
                          }}
                          disabled={densifyScheduleBlocked}
                        >
                          <Play className="w-4 h-4" />
                          {showBatchActions ? "Batch Continue" : "Resume Processing"}
                        </button>
                      ) : null
                    ) : null}
                    {/* --- END RESUME BUTTON LOGIC --- */}
                    {!processing && (selectedRunMeta?.batch_total || 0) > 1 && (
                      <button
                        className="w-full inline-flex items-center justify-center gap-2 px-4 py-3 rounded-lg bg-violet-600 hover:bg-violet-700 text-white font-semibold shadow-sm disabled:bg-slate-300 disabled:cursor-not-allowed"
                        onClick={() => {
                          if (!densifyScheduleBlocked) {
                            void handleContinueBatchFromSelected();
                          }
                        }}
                        disabled={densifyScheduleBlocked}
                      >
                        <Play className="w-4 h-4" />
                        Continue Batch Chain
                      </button>
                    )}
                  </>
                )}
              </div>
              
              {(processing || processingStatus) && (
                <div className="mt-3 space-y-3">
                  {/* Overall Pipeline Progress */}
                  {/* Always show overall progress bar with all selected steps */}
                  <div className="px-3 py-2 bg-indigo-50 border border-indigo-200 rounded-lg">
                    <div className="flex items-center justify-between text-xs text-indigo-700 mb-1">
                      <div className="flex items-center gap-2">
                        <span className="font-semibold">Overall Progress</span>
                        <button
                          type="button"
                          onClick={() => setShowTelemetryModal(true)}
                          className="text-[11px] font-semibold text-indigo-700 underline decoration-indigo-400 hover:text-indigo-900"
                        >
                          Full log
                        </button>
                      </div>
                      <span className="font-bold">{wasStopped ? `${overallProgress}% (stopped)` : `${overallProgress}%`}</span>
                    </div>
                    {batchTotal > 1 && (
                      <div className="mb-1.5 text-[11px] text-indigo-800 flex flex-wrap items-center gap-2">
                        <span className="px-2 py-0.5 rounded bg-indigo-100 border border-indigo-200 font-semibold">
                          Sessions: {Math.min(batchCompleted, batchTotal)}/{batchTotal} completed
                        </span>
                        <span className="px-2 py-0.5 rounded bg-indigo-100 border border-indigo-200">
                          Current session: {Math.min(Math.max(batchCurrentIndex, 1), batchTotal)}/{batchTotal}
                        </span>
                        {processingRunId && (
                          <span className="truncate">Active run: <span className="font-semibold">{processingRunId}</span></span>
                        )}
                      </div>
                    )}
                    {/* Stage labels above the segmented bar */}
                    <div className="flex justify-between mb-0.5 px-1">
                      {[runColmap && <span key="colmap" className="text-[9px] text-indigo-900 font-medium">COLMAP</span>,
                        runTraining && <span key="training" className="text-[9px] text-indigo-900 font-medium">Training</span>,
                        runExport && <span key="export" className="text-[9px] text-indigo-900 font-medium">Export</span>].filter(Boolean)}
                    </div>
                    {/* Segmented progress bar */}
                    <div className="w-full flex gap-1 mb-1">
                      {(() => {
                        const stages = [
                          { key: 'colmap', active: runColmap },
                          { key: 'training', active: runTraining },
                          { key: 'export', active: runExport },
                        ].filter(s => s.active);
                        const count = Math.max(stages.length, 1);
                        const segSize = 100 / count;
                        return stages.map((s, idx) => {
                          const segStart = idx * segSize;
                          const segEnd = segStart + segSize;
                          let fill = 0;
                          if (overallProgress >= segEnd) fill = 100;
                          else if (overallProgress > segStart) fill = ((overallProgress - segStart) / segSize) * 100;
                          return (
                            <div key={s.key} className="flex-1 bg-indigo-100 rounded-full overflow-hidden h-2 relative flex items-center justify-center">
                              <div className="bg-indigo-600 h-2 rounded-full transition-all duration-500 ease-out" style={{ width: `${Math.max(0, Math.min(fill, 100))}%` }} />
                            </div>
                          );
                        });
                      })()}
                    </div>
                    {pipelineDone ? (
                      <p className="text-xs text-green-600 mt-1 font-semibold">Completed</p>
                    ) : wasStopped ? (
                      stoppedStage === 'training' && typeof trainingCurrentStep === 'number' && trainingMaxSteps ? (
                        <p className="text-xs text-orange-600 mt-1 font-semibold">Stopped during Training at step {trainingCurrentStep} / {trainingMaxSteps}</p>
                      ) : stoppedStage ? (
                        <p className="text-xs text-orange-600 mt-1 font-semibold">Stopped during {stoppedStage.charAt(0).toUpperCase() + stoppedStage.slice(1)}</p>
                      ) : (
                        <p className="text-xs text-orange-600 mt-1 font-semibold">Stopped at: {currentStage || (stageStatus && (Object.entries(stageStatus).find(([, v]) => v === 'running') || [null])[0]) || 'Unknown'}</p>
                      )
                    ) : currentStage && (
                      <p className="text-xs text-indigo-600 mt-1">Current: {currentStage}</p>
                    )}
                    {expectedTimeRemaining && (
                      <p className="text-xs text-indigo-600 mt-1">Estimated time left: {expectedTimeRemaining}</p>
                    )}
                  </div>
                  {/* Hide Stage Status block completely when all stages are success */}
                  {!pipelineDone && processingStatus && !wasStopped && (
                    <div className="px-3 py-2 bg-blue-50 border border-blue-200 rounded-lg space-y-2">
                      <p className="text-xs font-semibold text-blue-700">Stage Status</p>
                      <p className="text-xs text-blue-900 whitespace-pre-line">{processingStatus}</p>
                      {/* Substep progress bar for current stage */}
                      {currentStageKey === 'training' &&
                        typeof trainingCurrentStep === 'number' &&
                        typeof trainingMaxSteps === 'number' &&
                        !isNaN(trainingCurrentStep) &&
                        !isNaN(trainingMaxSteps) &&
                        trainingMaxSteps > 0 && (
                          <div className="mt-2">
                            <div className="flex items-center justify-between text-xs text-blue-700 mb-1">
                              <span>Training Step {trainingCurrentStep.toLocaleString()} / {trainingMaxSteps.toLocaleString()}</span>
                              <span className="font-semibold">{((trainingCurrentStep / trainingMaxSteps) * 100).toFixed(1)}%</span>
                            </div>
                            <div className="w-full bg-blue-100 rounded-full h-2 overflow-hidden">
                              <div 
                                className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                                style={{ width: `${Math.min((trainingCurrentStep / trainingMaxSteps) * 100, 100)}%` }}
                              />
                            </div>
                          </div>
                      )}
                      {currentStageKey !== 'training' && typeof stageProgress === 'number' && (
                        <div className="mt-2">
                          <div className="flex items-center justify-between text-xs text-blue-700 mb-1">
                            <span>{currentStage || 'Current Stage'} Progress</span>
                            <span className="font-semibold">{stageProgress}%</span>
                          </div>
                          <div className="w-full bg-blue-100 rounded-full h-2 overflow-hidden">
                            <div 
                              className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                              style={{ width: `${Math.max(0, Math.min(stageProgress, 100))}%` }}
                            />
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Right Area - Layers & Map/Viewer Side by Side (stack on small screens) */}
        <div className="xl:col-span-3 flex flex-col xl:flex-row gap-3">
          {/* Layers Panel - Left Side */}
          <div className="w-full sm:w-64 flex-shrink-0 bg-white rounded-lg border border-slate-200 shadow-sm p-4">
            <div className="flex items-center gap-2 mb-3">
              <Layers className="w-4 h-4 text-slate-500" />
              <h3 className="text-sm font-bold text-slate-900">Layers</h3>
            </div>
            <div className="space-y-2 text-sm text-slate-700">
              <p className="text-xs uppercase font-semibold text-slate-500 mb-2">Images</p>
              <div className="space-y-2">
                <label className={`flex items-center gap-2 px-3 py-2 rounded-lg border ${showImagesLayer ? 'border-slate-200 bg-slate-50' : 'border-slate-100 bg-slate-50'}`}>
                  <input type="checkbox" className="w-4 h-4" checked={showImagesLayer} onChange={(e) => { setShowImagesLayer(e.target.checked); if (e.target.checked) setTopView('map'); }} />
                  <span>Image Locations</span>
                </label>
              </div>

              <div className="mt-4">
                <p className="text-xs uppercase font-semibold text-slate-500 mb-2">Point Clouds</p>
                <div className="space-y-2 px-1">
                  <label className={`flex items-center gap-2 px-2 py-2 rounded-lg border ${viewerOutput === 'pointcloud' ? 'border-blue-500 bg-blue-50' : 'border-slate-200 bg-slate-50'} ${!hasSparseCloud ? 'opacity-50 cursor-not-allowed' : ''}`}>
                    <input type="radio" name={`viewerOutput_${projectId}`} value="pointcloud" checked={viewerOutput === 'pointcloud'} onChange={() => { if (hasSparseCloud) { setViewerOutput('pointcloud'); setTopView('viewer'); } }} disabled={!hasSparseCloud} />
                    <span className="text-sm">View Sparse Cloud</span>
                  </label>
                </div>
              </div>

              {showEngineDropdown && (
                <div className="mt-4">
                  <p className="text-xs uppercase font-semibold text-slate-500 mb-2">Final Model Engine</p>
                  <select
                    value={engineSelectValue}
                    onChange={(event) => handleEngineSelection(event.target.value)}
                    className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                  >
                    {engineOptions.map((option) => (
                      <option key={option.name} value={option.name}>{option.label}</option>
                    ))}
                  </select>
                </div>
              )}

              {showFinalModelSection && (
                <>
                  <div className="mt-4">
                    <p className="text-xs uppercase font-semibold text-slate-500 mb-2">Final Model</p>
                    <label className={`flex items-center gap-2 px-2 py-2 rounded-lg border ${viewerOutput === 'model' ? 'border-blue-500 bg-blue-50' : 'border-slate-200 bg-slate-50'} ${!finalModelAvailable ? 'opacity-50 cursor-not-allowed' : ''}`}>
                      <input
                        type="radio"
                        name={`viewerOutput_${projectId}`}
                        value="model"
                        checked={viewerOutput === 'model'}
                        onChange={() => { if (finalModelAvailable) { setViewerOutput('model'); setTopView('viewer'); } }}
                        disabled={!finalModelAvailable}
                      />
                      <div className="flex flex-col">
                        <span className="text-sm">View Final Model{selectedEngineLabel ? ` (${selectedEngineLabel})` : ''}</span>
                        <span className="text-[11px] text-slate-500">Engine-specific outputs feed the 3D viewer</span>
                      </div>
                    </label>
                    {viewerOutput === 'model' && finalModelAvailable && activeEngineBundle && (
                      <div className="mt-2 space-y-1">
                        <label className={`flex items-center gap-2 px-3 py-1.5 rounded text-xs ${selectedModelLayer === 'final' ? 'bg-blue-50 border border-blue-200' : 'hover:bg-blue-50 border border-transparent'} ${!activeEngineBundle.finalModelUrl ? 'opacity-50 cursor-not-allowed' : ''}`}>
                          <input
                            type="radio"
                            name={`modelLayer_${projectId}`}
                            checked={selectedModelLayer === 'final'}
                            onChange={() => { if (activeEngineBundle.finalModelUrl) { setSelectedModelLayer('final'); setSelectedModelSnapshot(null); setTopView('viewer'); } }}
                            disabled={!activeEngineBundle.finalModelUrl}
                          />
                          <span className="truncate">Final Splat</span>
                        </label>
                        <label className={`flex items-center gap-2 px-3 py-1.5 rounded text-xs ${selectedModelLayer === 'best' ? 'bg-blue-50 border border-blue-200' : 'hover:bg-blue-50 border border-transparent'} ${!activeEngineBundle.bestModelUrl ? 'opacity-50 cursor-not-allowed' : ''}`}>
                          <input
                            type="radio"
                            name={`modelLayer_${projectId}`}
                            checked={selectedModelLayer === 'best'}
                            onChange={() => { if (activeEngineBundle.bestModelUrl) { setSelectedModelLayer('best'); setSelectedModelSnapshot(null); setTopView('viewer'); } }}
                            disabled={!activeEngineBundle.bestModelUrl}
                          />
                          <span className="truncate">Best Splat</span>
                        </label>
                      </div>
                    )}
                  </div>

                  <div className="border-t border-slate-200 pt-2 mt-2">
                    <p className="text-xs uppercase font-semibold text-slate-500 mb-2">PNG Previews ({pngFiles.length})</p>
                    <div className="space-y-1 h-56 overflow-y-auto">
                      {pngFiles.length === 0 && <div className="text-xs text-slate-500 px-3">No PNG previews available</div>}
                      {pngFiles.map((png) => (
                        <label key={png.url} className={`flex items-center gap-2 px-3 py-1.5 rounded text-xs hover:bg-blue-50 text-slate-700 truncate ${selectedPng === png.url ? 'bg-blue-50 border border-blue-200' : ''}`}>
                          <input
                            type="radio"
                            name={`pngPreview_${projectId}`}
                            checked={selectedPng === png.url}
                            onChange={() => { setSelectedPng(png.url); setTopView('png'); }}
                          />
                          <span className="truncate">{png.name}</span>
                        </label>
                      ))}
                    </div>
                  </div>

                  <div className="border-t border-slate-200 pt-2 mt-2">
                    <p className="text-xs uppercase font-semibold text-slate-500 mb-2">Model Snapshots ({modelSnapshots.length})</p>
                    <div className="space-y-1 h-56 overflow-y-auto">
                      {modelSnapshots.length === 0 && <div className="text-xs text-slate-500 px-3">No snapshots exported yet</div>}
                      {modelSnapshots.length > 0 && (
                        <label className={`flex items-center gap-2 px-3 py-1.5 rounded text-xs hover:bg-blue-50 text-slate-700 ${selectedModelSnapshot === null ? 'bg-blue-50 border border-blue-200' : ''}`}>
                          <input
                            type="radio"
                            name={`modelSnapshot_${projectId}`}
                            checked={selectedModelSnapshot === null}
                            onChange={() => { setSelectedModelSnapshot(null); setViewerOutput('model'); setTopView('viewer'); }}
                          />
                          <span className="truncate">Latest export</span>
                        </label>
                      )}
                      {modelSnapshots.map((snapshot) => (
                        <div key={snapshot.url} className={`flex items-center gap-2 px-2 py-1.5 rounded ${selectedModelSnapshot === snapshot.url ? 'bg-indigo-50 border border-indigo-200' : 'hover:bg-indigo-50'}`}>
                          <label className="flex items-center gap-2 flex-1">
                            <input
                              type="radio"
                              name={`modelSnapshot_${projectId}`}
                              checked={selectedModelSnapshot === snapshot.url}
                              onChange={() => { setSelectedModelSnapshot(snapshot.url); setViewerOutput('model'); setTopView('viewer'); }}
                            />
                            <div className="flex-1 overflow-hidden">
                              <p className="truncate text-xs font-semibold text-slate-800">
                                {snapshot.step ? `Step ${snapshot.step.toLocaleString()}` : snapshot.name}
                              </p>
                              <p className="text-[11px] text-slate-500 truncate">
                                {(snapshot.format || 'splat').toUpperCase()}{snapshot.size ? ` Â· ${formatBytes(snapshot.size)}` : ''}
                              </p>
                            </div>
                          </label>
                          <a
                            href={snapshot.url}
                            download
                            onClick={(e) => e.stopPropagation()}
                            className="text-slate-500 hover:text-slate-800"
                            title="Download snapshot"
                          >
                            <Download className="w-4 h-4" />
                          </a>
                        </div>
                      ))}
                    </div>
                  </div>
                </>
              )}
            </div>
          </div>

          {/* Map/Viewer Area */}
          <div className="flex-1 bg-white rounded-lg border border-slate-200 shadow-sm overflow-hidden">
            <div className={`sticky top-0 z-40 transition-all bg-slate-50 border-b border-slate-200 ${headerCompact ? 'py-1 px-3' : 'py-3 px-4'}`}>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2 text-sm font-semibold text-slate-800">
                  <MapIcon className={`transition-all ${headerCompact ? 'w-5 h-5' : 'w-4 h-4'}`} />
                  <span className={`${headerCompact ? 'text-sm font-semibold' : 'text-base font-bold'}`}>
                    {topView === "map" ? "Map" : topView === "viewer" ? "3D Viewer" : "PNG Viewer"}
                  </span>
                  {!headerCompact && locLoading && topView === "map" && <span className="text-xs text-slate-500">(loading)</span>}
                </div>
                <div className="flex items-center gap-2">
                  <button
                    title="Map"
                    className={`transition-all rounded-lg border ${topView === "map" ? "bg-white border-blue-500 text-blue-700" : "border-slate-200"} ${headerCompact ? 'p-1' : 'px-3 py-1.5'}`}
                    onClick={() => setTopView("map")}
                  >
                    {headerCompact ? <MapIcon className="w-4 h-4" /> : 'Map'}
                  </button>
                  <button
                    title="3D Viewer"
                    className={`transition-all rounded-lg border ${topView === "viewer" ? "bg-white border-blue-500 text-blue-700" : "border-slate-200"} ${headerCompact ? 'p-1' : 'px-3 py-1.5'}`}
                    onClick={() => setTopView("viewer")}
                  >
                    {headerCompact ? <Boxes className="w-4 h-4" /> : (<><Boxes className="w-4 h-4 inline" /> 3D Viewer</>)}
                  </button>
                  {/* Viewer output is selected in the Outputs panel; keep only the 3D Viewer tab here */}
                  {pngFiles.length > 0 && (
                    <button
                      title="PNG Viewer"
                      className={`transition-all rounded-lg border ${topView === "png" ? "bg-white border-blue-500 text-blue-700" : "border-slate-200"} ${headerCompact ? 'p-1' : 'px-3 py-1.5'}`}
                      onClick={() => setTopView("png")}
                    >
                      {headerCompact ? <span className="text-xs">PNG</span> : 'PNG Viewer'}
                    </button>
                  )}
                </div>
              </div>
            </div>

          {topView === "viewer" ? (
            <div className="h-[560px] bg-slate-50 flex items-center justify-center p-4">
              <div className="w-full h-full rounded-lg overflow-hidden border border-slate-200 bg-black relative">
                {viewerOutput === 'model' && viewerModelAvailable ? (
                  <ViewerTab
                    projectId={projectId}
                    snapshotUrl={selectedModelSnapshot}
                    engineOverride={selectedEngineName}
                    modelUrlOverride={selectedModelSnapshot ? null : selectedLayerModelUrl}
                    modelLabelOverride={selectedModelSnapshot ? "Snapshot" : selectedLayerModelLabel}
                  />
                ) : viewerOutput === 'pointcloud' && hasSparseCloud ? (
                  <SparseViewer projectId={projectId} focusTarget={focusTarget} />
                ) : (
                  <div className="h-[560px] flex items-center justify-center text-sm text-slate-400">No 3D model or sparse cloud available to view</div>
                )}
              </div>
            </div>
          ) : topView === "png" ? (
            <div className="h-[560px] bg-slate-900 flex items-center justify-center p-4 relative">
              
              {selectedPng ? (
                <img 
                  src={selectedPng} 
                  alt="Preview" 
                  className="max-h-full max-w-full object-contain"
                />
              ) : (
                <p className="text-slate-400">Select a PNG preview from the layers panel</p>
              )}
              {selectedPng && (
                <div className="absolute bottom-4 left-1/2 transform -translate-x-1/2 bg-white/90 backdrop-blur rounded-lg px-4 py-2 text-xs text-slate-700">
                  {selectedPngEntry?.name || selectedPng.split('/').pop()}
                </div>
              )}
            </div>
          ) : (
            <div className="relative h-[560px]">
              {!locations.length && !locLoading && (
                <div className="absolute inset-0 flex items-center justify-center text-sm text-slate-600 z-10 pointer-events-none">
                  No GPS positions yet. Upload geotagged images to see them here.
                </div>
              )}

              {/* Map overlays (rendered after Map so they don't block interaction) */}

              <Map
                ref={mapRef}
                mapLib={maplibregl as unknown as any}
                initialViewState={{
                  latitude: mapCenter[0],
                  longitude: mapCenter[1],
                  zoom: locations.length ? 12 : 2,
                  pitch: mapDim === "3d" ? 45 : 0,
                  bearing: mapDim === "3d" ? -20 : 0,
                }}
                /* Uncontrolled map: we use mapRef.fitBounds() for auto-fit and avoid
                   passing a controlled `viewState`/`onMove` which can cause update loops. */
                style={{ width: "100%", height: "100%" }}
                mapStyle={mapStyle as any}
                dragPan={true}
                dragRotate={true}
                scrollZoom={true}
                touchZoomRotate={true}
                doubleClickZoom={true}
              >
                <NavigationControl position="top-right" />
                {/* Image positions are rendered as a GeoJSON circle layer for reliable display. */}
              </Map>
                <div className="absolute inset-0 z-10 pointer-events-none">
                  <div className="absolute left-3 top-3 pointer-events-auto flex items-start">
                              <div className="bg-white/90 backdrop-blur rounded-lg border border-slate-200 p-3 w-36 text-left">
                      <div className="text-xs font-semibold text-slate-700 mb-2">Basemap</div>
                      <div className="flex flex-col gap-2 w-full text-left">
                        <button
                          className={`w-full flex justify-start items-center text-left text-xs px-2 py-1 rounded border ${basemap === "satellite" ? "border-blue-500 text-blue-700" : "border-slate-200 text-slate-700"}`}
                          onClick={() => setBasemap("satellite")}
                        >
                          <span className="block">Satellite</span>
                        </button>
                        <button
                          className={`w-full flex justify-start items-center text-left text-xs px-2 py-1 rounded border ${basemap === "osm" ? "border-blue-500 text-blue-700" : "border-slate-200 text-slate-700"}`}
                          onClick={() => setBasemap("osm")}
                        >
                          <span className="block">OSM</span>
                        </button>
                      </div>
                    </div>
                  </div>
                  <div className="absolute right-3 top-3 pointer-events-auto">
                    <div className="bg-white/90 backdrop-blur rounded-lg border border-slate-200 p-1 text-xs flex items-center">
                      {/* <div className="text-xs font-semibold text-slate-700 mr-2">View</div> */}
                      <div className="inline-flex rounded-md border overflow-hidden">
                        <button
                          className={`px-2 py-1 text-xs ${mapDim === "2d" ? "bg-blue-600 text-white" : "bg-white text-slate-700"}`}
                          onClick={() => setMapDim("2d")}
                        >
                          2D
                        </button>
                        <button
                          className={`px-2 py-1 text-xs ${mapDim === "3d" ? "bg-blue-600 text-white" : "bg-white text-slate-700"}`}
                          onClick={() => setMapDim("3d")}
                        >
                          3D
                        </button>
                      </div>
                    </div>
                  </div>
                </div>
          </div>
        )}
          </div>
        </div>
      </div>

      {showTelemetryModal && (
        <div className="fixed inset-0 z-[1200]">
          <div className="absolute inset-0 bg-black/50" onClick={() => setShowTelemetryModal(false)} />
          <div className="absolute inset-0 flex items-center justify-center p-4">
            <div className="w-[1080px] max-w-full max-h-[92vh] overflow-hidden bg-white rounded-xl shadow-2xl border border-slate-200">
              <div className="flex items-center justify-between px-4 py-2 border-b border-slate-200">
                <div>
                  <h3 className="text-base font-bold text-slate-900">Training Telemetry</h3>
                  <p className="text-xs text-slate-500">Run: {telemetryData?.run_id || processingRunId || selectedRunId || "-"}</p>
                  <p className="text-xs text-slate-600 mt-0.5">
                    Total elapsed: <span className="font-semibold">{formatDurationCompact(telemetryData?.training_summary?.total_elapsed_seconds)}</span>
                    {typeof telemetryData?.training_summary?.first_step === "number" && typeof telemetryData?.training_summary?.last_step === "number"
                      ? ` • Steps ${telemetryData.training_summary.first_step.toLocaleString()} to ${telemetryData.training_summary.last_step.toLocaleString()}`
                      : ""}
                  </p>
                </div>
                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    onClick={(e) => void handleDownloadTelemetryJson(e)}
                    disabled={telemetryDownloadBusy || telemetryLoading}
                    className="inline-flex items-center gap-1.5 rounded-md border border-slate-300 px-2.5 py-1.5 text-xs font-semibold text-slate-700 hover:bg-slate-50 disabled:opacity-50 disabled:cursor-not-allowed"
                    title="Download full telemetry as JSON"
                  >
                    <Download className="w-3.5 h-3.5" />
                    {telemetryDownloadBusy ? "Preparing..." : "Download JSON"}
                  </button>
                  <button
                    type="button"
                    onClick={(e) => void handleDownloadTelemetryPdf(e)}
                    disabled={telemetryDownloadBusy || telemetryLoading}
                    className="inline-flex items-center gap-1.5 rounded-md border border-slate-300 px-2.5 py-1.5 text-xs font-semibold text-slate-700 hover:bg-slate-50 disabled:opacity-50 disabled:cursor-not-allowed"
                    title="Download full telemetry as PDF"
                  >
                    <Download className="w-3.5 h-3.5" />
                    {telemetryDownloadBusy ? "Preparing..." : "Download PDF"}
                  </button>
                  <button type="button" className="text-sm text-slate-600" onClick={() => setShowTelemetryModal(false)}>Close</button>
                </div>
              </div>

              <div className="p-4 overflow-auto max-h-[84vh] space-y-4 text-sm">
                {telemetryLoading && <p className="text-xs text-slate-500">Loading telemetry...</p>}
                {telemetryError && <p className="text-xs text-red-600">{telemetryError}</p>}

                {telemetryData?.status && (
                  <div className="rounded-lg border border-slate-200 bg-slate-50 p-3">
                    <p className="text-xs font-semibold text-slate-700 mb-1">Current status</p>
                    <div className="grid grid-cols-1 md:grid-cols-6 gap-2 text-xs text-slate-700">
                      <div>Stage: <span className="font-semibold">{telemetryData.status.stage || "-"}</span></div>
                      <div>Step: <span className="font-semibold">{typeof telemetryData.status.currentStep === "number" ? telemetryData.status.currentStep.toLocaleString() : "-"}</span></div>
                      <div>Loss: <span className="font-semibold">{typeof telemetryData.status.current_loss === "number" ? telemetryData.status.current_loss.toFixed(6) : "-"}</span></div>
                      <div>Best tracking starts: <span className="font-semibold">{typeof telemetryBestTrackingStartStep === "number" ? telemetryBestTrackingStartStep.toLocaleString() : "-"}</span></div>
                      <div>Best loss step: <span className="font-semibold">{typeof telemetryBestLoss.bestStep === "number" ? telemetryBestLoss.bestStep.toLocaleString() : "-"}</span></div>
                      <div>Best loss value: <span className="font-semibold">{typeof telemetryBestLoss.bestLoss === "number" ? telemetryBestLoss.bestLoss.toFixed(6) : "-"}</span></div>
                    </div>
                  </div>
                )}

                {telemetryData?.ai_insights?.ai_input_mode && (
                  <div className="rounded-lg border border-sky-200 bg-sky-50/60 p-3 space-y-3">
                    <p className="text-xs font-semibold text-slate-700">AI mode extracted values</p>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-2 text-xs text-slate-700">
                      <div>Mode: <span className="font-semibold">{formatTelemetryScalar(telemetryData.ai_insights.ai_input_mode)}</span></div>
                      <div>Feature source: <span className="font-semibold">{formatTelemetryScalar(telemetryData.ai_insights.feature_source)}</span></div>
                      <div>Cache used: <span className="font-semibold">{formatTelemetryScalar(telemetryData.ai_insights.cache_used)}</span></div>
                      <div>Baseline session: <span className="font-semibold">{formatTelemetryScalar(telemetryData.ai_insights.baseline_session_id)}</span></div>
                      <div>Heuristic preset: <span className="font-semibold">{formatTelemetryScalar(telemetryData.ai_insights.heuristic_preset)}</span></div>
                      <div>Selected strategy/action: <span className="font-semibold">{formatTelemetryScalar(telemetryData.ai_insights.selected_preset)}</span></div>
                      <div>Reward value: <span className="font-semibold">{formatTelemetryScalar(telemetryData.ai_insights.reward)}</span></div>
                      <div>Rewarded: <span className="font-semibold">{formatTelemetryScalar(telemetryData.ai_insights.reward_positive)}</span></div>
                      <div>Reward label: <span className="font-semibold">{formatTelemetryScalar(telemetryData.ai_insights.reward_label)}</span></div>
                    </div>

                    <div>
                      <p className="text-xs font-semibold text-slate-700 mb-2">Initial parameters used</p>
                      <div className="max-h-36 overflow-auto border border-slate-200 rounded-lg bg-white">
                        <table className="w-full text-xs">
                          <thead className="bg-slate-50 text-slate-700">
                            <tr>
                              <th className="text-left px-3 py-2">Field</th>
                              <th className="text-left px-3 py-2">Value</th>
                            </tr>
                          </thead>
                          <tbody>
                            {Object.entries(telemetryData.ai_insights.initial_params || {}).length === 0 ? (
                              <tr>
                                <td className="px-3 py-2 text-slate-500" colSpan={2}>No initial parameters captured.</td>
                              </tr>
                            ) : (
                              Object.entries(telemetryData.ai_insights.initial_params || {}).map(([key, value]) => (
                                <tr key={`ai-init-${key}`} className="border-t border-slate-100">
                                  <td className="px-3 py-2 text-slate-700">{formatTelemetryFieldLabel(key)}</td>
                                  <td className="px-3 py-2 text-slate-900">
                                    <div className="flex items-center justify-between gap-2">
                                      <span>{formatTelemetryScalar(value)}</span>
                                      {LEARNABLE_AI_PARAM_KEYS.has(key) && (
                                        <span className="inline-flex items-center rounded-full bg-blue-100 text-blue-700 px-2 py-0.5 text-[10px] font-semibold">
                                          Learned
                                        </span>
                                      )}
                                    </div>
                                  </td>
                                </tr>
                              ))
                            )}
                          </tbody>
                        </table>
                      </div>
                    </div>

                    <div>
                      <p className="text-xs font-semibold text-slate-700 mb-2">Extracted features and missing flags</p>
                      <div className="max-h-44 overflow-auto border border-slate-200 rounded-lg bg-white">
                        <table className="w-full text-xs">
                          <thead className="bg-slate-50 text-slate-700">
                            <tr>
                              <th className="text-left px-3 py-2">Feature</th>
                              <th className="text-left px-3 py-2">Value</th>
                              <th className="text-left px-3 py-2">Status</th>
                              <th className="text-left px-3 py-2">Source</th>
                            </tr>
                          </thead>
                          <tbody>
                            {buildTelemetryFeatureRows(
                              (telemetryData.ai_insights.feature_details || {}) as Record<string, unknown>,
                              (telemetryData.ai_insights.missing_flags || {}) as Record<string, unknown>,
                              (telemetryData.ai_insights.feature_sources || {}) as Record<string, unknown>,
                            ).length === 0 ? (
                              <tr>
                                <td className="px-3 py-2 text-slate-500" colSpan={4}>No extracted feature details captured.</td>
                              </tr>
                            ) : (
                              buildTelemetryFeatureRows(
                                (telemetryData.ai_insights.feature_details || {}) as Record<string, unknown>,
                                (telemetryData.ai_insights.missing_flags || {}) as Record<string, unknown>,
                                (telemetryData.ai_insights.feature_sources || {}) as Record<string, unknown>,
                              ).map((row) => (
                                <tr key={`ai-feature-${row.key}`} className="border-t border-slate-100">
                                  <td className="px-3 py-2 text-slate-700">{formatTelemetryFieldLabel(row.key)}</td>
                                  <td className="px-3 py-2 text-slate-900">{formatTelemetryScalar(row.value)}</td>
                                  <td className="px-3 py-2 text-slate-900">
                                    {row.status === "defaulted" ? (
                                      <span className="inline-flex items-center rounded-full bg-amber-100 text-amber-700 px-2 py-0.5 text-[10px] font-semibold">
                                        Defaulted
                                      </span>
                                    ) : row.status === "present" ? (
                                      <span className="inline-flex items-center rounded-full bg-emerald-100 text-emerald-700 px-2 py-0.5 text-[10px] font-semibold">
                                        Present
                                      </span>
                                    ) : (
                                      <span className="inline-flex items-center rounded-full bg-slate-100 text-slate-600 px-2 py-0.5 text-[10px] font-semibold">
                                        Unknown
                                      </span>
                                    )}
                                  </td>
                                  <td className="px-3 py-2 text-slate-700">{row.source}</td>
                                </tr>
                              ))
                            )}
                          </tbody>
                        </table>
                      </div>
                    </div>

                    <div>
                      <p className="text-xs font-semibold text-slate-700 mb-2">Important events</p>
                      <div className="h-40 overflow-auto border border-slate-200 rounded-lg">
                        <table className="w-full text-xs">
                          <thead className="bg-slate-50 text-slate-700">
                            <tr>
                              <th className="text-left px-3 py-2">Type</th>
                              <th className="text-left px-3 py-2">Step</th>
                              <th className="text-left px-3 py-2">Summary</th>
                            </tr>
                          </thead>
                          <tbody>
                            {(telemetryData?.event_rows || []).length === 0 ? (
                              <tr>
                                <td className="px-3 py-2 text-slate-500" colSpan={3}>No important events recorded.</td>
                              </tr>
                            ) : (
                              (telemetryData?.event_rows || []).map((row, idx) => (
                                <tr key={`${row.type || "event"}-${row.step || "na"}-${idx}`} className="border-t border-slate-100">
                                  <td className="px-3 py-2 text-slate-900">{row.type || "-"}</td>
                                  <td className="px-3 py-2 text-slate-700">{typeof row.step === "number" ? row.step.toLocaleString() : "-"}</td>
                                  <td className="px-3 py-2 text-slate-700">{row.summary || "-"}</td>
                                </tr>
                              ))
                            )}
                          </tbody>
                        </table>
                      </div>
                    </div>

                  </div>
                )}

                <div>
                  <p className="text-xs font-semibold text-slate-700 mb-2">Log-interval snapshots</p>
                  <p className="text-[11px] text-slate-500 mb-2">
                    This table shows full training snapshots from step 1. Best-splat tracking starts at
                    {" "}
                    <span className="font-semibold text-slate-700">
                      {typeof telemetryBestTrackingStartStep === "number"
                        ? telemetryBestTrackingStartStep.toLocaleString()
                        : "configured start step"}
                    </span>
                    .
                  </p>
                  <div className="h-64 overflow-auto border border-slate-200 rounded-lg">
                    <table className="w-full text-xs">
                      <thead className="bg-slate-50 text-slate-700">
                        <tr>
                          <th className="text-left px-3 py-2">Time</th>
                          <th className="text-left px-3 py-2">Step</th>
                          <th className="text-left px-3 py-2">Loss</th>
                          <th className="text-left px-3 py-2">Elapsed</th>
                          <th className="text-left px-3 py-2">ETA</th>
                          <th className="text-left px-3 py-2">Speed</th>
                        </tr>
                      </thead>
                      <tbody>
                        {(telemetryData?.training_rows || []).length === 0 ? (
                          <tr>
                            <td className="px-3 py-2 text-slate-500" colSpan={6}>No snapshots yet.</td>
                          </tr>
                        ) : (
                          (telemetryData?.training_rows || []).map((row, idx) => (
                            <tr key={`${row.step || "na"}-${idx}`} className="border-t border-slate-100">
                              <td className="px-3 py-2 text-slate-700">{row.timestamp || "-"}</td>
                              <td className="px-3 py-2 text-slate-900">
                                {typeof row.step === "number" ? row.step.toLocaleString() : "-"}
                                {typeof row.max_steps === "number" ? ` / ${row.max_steps.toLocaleString()}` : ""}
                              </td>
                              <td className="px-3 py-2 text-slate-900">{typeof row.loss === "number" ? row.loss.toFixed(6) : "-"}</td>
                              <td className="px-3 py-2 text-slate-700">{typeof row.elapsed_seconds === "number" ? `${row.elapsed_seconds.toFixed(1)}s` : "-"}</td>
                              <td className="px-3 py-2 text-slate-700">{row.eta || "-"}</td>
                              <td className="px-3 py-2 text-slate-700">{row.speed || "-"}</td>
                            </tr>
                          ))
                        )}
                      </tbody>
                    </table>
                  </div>
                </div>

                <div>
                  <p className="text-xs font-semibold text-slate-700 mb-2">Latest eval metrics</p>
                  <div className="h-56 overflow-auto border border-slate-200 rounded-lg">
                    <table className="w-full text-xs">
                      <thead className="bg-slate-50 text-slate-700">
                        <tr>
                          <th className="text-left px-3 py-2">Eval step</th>
                          <th className="text-left px-3 py-2">PSNR</th>
                          <th className="text-left px-3 py-2">LPIPS</th>
                          <th className="text-left px-3 py-2">SSIM</th>
                          <th className="text-left px-3 py-2">Gaussians</th>
                        </tr>
                      </thead>
                      <tbody>
                        {(telemetryData?.eval_rows || []).length === 0 ? (
                          <tr>
                            <td className="px-3 py-2 text-slate-500" colSpan={5}>No eval entries yet.</td>
                          </tr>
                        ) : (
                          (telemetryData?.eval_rows || []).map((row, idx) => (
                            <tr key={`${row.step || "na"}-${idx}`} className="border-t border-slate-100">
                              <td className="px-3 py-2 text-slate-900">{typeof row.step === "number" ? row.step.toLocaleString() : "-"}</td>
                              <td className="px-3 py-2 text-slate-700">{typeof row.psnr === "number" ? row.psnr.toFixed(4) : "-"}</td>
                              <td className="px-3 py-2 text-slate-700">{typeof row.lpips === "number" ? row.lpips.toFixed(4) : "-"}</td>
                              <td className="px-3 py-2 text-slate-700">{typeof row.ssim === "number" ? row.ssim.toFixed(4) : "-"}</td>
                              <td className="px-3 py-2 text-slate-700">{typeof row.num_gaussians === "number" ? row.num_gaussians.toLocaleString() : "-"}</td>
                            </tr>
                          ))
                        )}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {showConfig && (
        <div className="fixed inset-0 z-50">
          <div className="absolute inset-0 bg-black/50" onClick={() => setShowConfig(false)} />
          <div className="absolute inset-0 flex items-center justify-center p-4">
            <div className="w-[820px] max-w-full bg-white rounded-xl shadow-2xl border border-slate-200 overflow-hidden">
              <div className="flex items-center justify-between px-3.5 py-0.5 border-b border-slate-200">
                <div>
                  {/* <p className="text-xs uppercase font-semibold text-slate-500">Advanced</p> */}
                  <h3 className="text-base font-bold leading-tight text-slate-900 pt-2">Processing Configuration</h3>
                  <p className="text-xs leading-none font-medium text-slate-500 mb-1">
                    Session: {selectedRunMeta?.run_name || selectedRunId || "latest"}
                  </p>
                </div>
                <button className="text-sm text-slate-600" onClick={() => setShowConfig(false)}>
                  Close
                </button>
              </div>

              <div className="px-3.5 pt-0.5 pb-2.5 text-sm overflow-auto max-h-[70vh]">
                {!canManageColmapImages && (
                  <div className="mb-1 rounded-lg border border-amber-200 bg-amber-50 px-2.5 py-1 text-xs text-amber-800">
                    Non-base session: Image and COLMAP settings are hidden. This session is training-only by default.
                  </div>
                )}
                {!canManageColmapImages && sharedImageSizeMismatch && baseColmapProfile && (
                  <div className="mb-1 rounded-lg border border-amber-200 bg-amber-50 px-2.5 py-1 text-xs text-amber-900">
                    Base COLMAP for {baseColmapProfile.runName} used {baseColmapProfile.resizeEnabled && typeof baseColmapProfile.imageSize === "number" ? `${baseColmapProfile.imageSize}px` : "original size"},
                    but shared image size is now {imagesResizeEnabled && typeof imagesMaxSize === "number" ? `${imagesMaxSize}px` : "original size"}.
                    Re-run base-session COLMAP to refresh shared sparse before continuing with this session.
                  </div>
                )}
                {selectedRunSharedOutdated && (
                  <div className="mb-1 rounded-lg border border-amber-200 bg-amber-50 px-2.5 py-1 text-xs text-amber-900">
                    This session references an older shared-config version from the base session. Create a new session (or re-run from base) to pick up latest shared image/COLMAP settings.
                  </div>
                )}
                <div className="grid grid-cols-12 gap-0">
                  <div className="col-span-2 flex flex-col gap-0.5">
                    {canManageColmapImages && (
                      <button
                        onClick={() => setConfigTab("images")}
                        className={`text-left text-sm px-2.5 py-1 rounded-md flex items-center justify-between ${configTab === "images" ? "bg-blue-600 text-white font-semibold shadow" : "bg-slate-50 text-slate-700 hover:bg-slate-100"}`}
                      >
                        <span>Images</span>
                        {configTab === "images" && <div className="w-2 h-5 bg-white/20 rounded ml-2" />}
                      </button>
                    )}
                    {canManageColmapImages && (
                      <button
                        onClick={() => setConfigTab("colmap")}
                        className={`text-left text-sm px-2.5 py-1 rounded-md flex items-center justify-between ${configTab === "colmap" ? "bg-blue-600 text-white font-semibold shadow" : "bg-slate-50 text-slate-700 hover:bg-slate-100"}`}
                      >
                        <span>COLMAP</span>
                        {configTab === "colmap" && <div className="w-2 h-5 bg-white/20 rounded ml-2" />}
                      </button>
                    )}
                    <button
                      onClick={() => setConfigTab("training")}
                      className={`text-left text-sm px-2.5 py-1 rounded-md flex items-center justify-between ${configTab === "training" ? "bg-blue-600 text-white font-semibold shadow" : "bg-slate-50 text-slate-700 hover:bg-slate-100"}`}
                    >
                      <span>Training</span>
                      {configTab === "training" && <div className="w-2 h-5 bg-white/20 rounded ml-2" />}
                    </button>
                  </div>
                  <div className="col-span-7 [&_input]:text-[15px] [&_select]:text-[15px]">
                    {configTab === "training" ? (
                      <div className="space-y-1 text-sm">
                        <div className="rounded-xl border border-slate-200 bg-slate-50/60 shadow-sm">
                          <div className="flex items-center justify-between px-3 py-1 border-b border-slate-100">
                            <div>
                              <p className="text-sm font-semibold text-slate-800">Shared controls</p>
                            </div>
                          </div>
                          <div className="grid grid-cols-1 md:grid-cols-2 gap-0 px-3 py-0.5">
                            <div className="md:col-span-2">
                              <label className="flex items-center justify-between text-[11px] font-medium text-slate-600 mb-0.5">
                                <span>Training Engine</span>
                                <button onClick={() => setSelectedInfoKey("engine")} className="p-1 text-slate-400 hover:text-slate-600"><Info /></button>
                              </label>
                              <select
                                value={engine}
                                onChange={(e) => setEngine(e.target.value as TrainingEngine)}
                                className="w-full px-2 py-1.5 text-xs border border-slate-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                              >
                                <option value="gsplat">gsplat (default pipeline)</option>
                                <option value="litegs">LiteGS (compact renderer)</option>
                              </select>
                            </div>
                            {engine === "gsplat" && (
                              <div className="md:col-span-2">
                                <label className="flex items-center justify-between text-[11px] font-medium text-slate-600 mb-0.5">
                                  <span>Training Profile</span>
                                  <button onClick={() => setSelectedInfoKey("mode")} className="p-1 text-slate-400 hover:text-slate-600"><Info /></button>
                                </label>
                                <select
                                  value={mode}
                                  onChange={(e) => setMode((e.target.value as "baseline" | "modified") || "baseline")}
                                  className="w-full px-2 py-1.5 text-xs border border-slate-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                                  disabled={processing || isStopping}
                                >
                                  <option value="baseline">Baseline</option>
                                  <option value="modified">Modified</option>
                                </select>
                              </div>
                            )}
                            {engine === "gsplat" && mode === "modified" && (
                              <div className="md:col-span-2">
                                <label className="flex items-center justify-between text-[11px] font-medium text-slate-600 mb-0.5">
                                  <span>Rule tuning scope</span>
                                  <button onClick={() => setSelectedInfoKey("tune_scope")} className="p-1 text-slate-400 hover:text-slate-600"><Info /></button>
                                </label>
                                <select
                                  value={tuneScopeDropdownValue}
                                  onChange={(e) => {
                                    const value = (e.target.value as TuneScopeDropdownValue) || "core_individual_plus_strategy";
                                    if (value === "core_ai_optimization") {
                                      setTuneScope("core_ai_optimization");
                                      setAiInputMode("");
                                      return;
                                    }
                                    if (value === "core_ai_optimization__exif_only") {
                                      setTuneScope("core_ai_optimization");
                                      setAiInputMode("exif_only");
                                      return;
                                    }
                                    if (value === "core_ai_optimization__exif_plus_flight_plan") {
                                      setTuneScope("core_ai_optimization");
                                      setAiInputMode("exif_plus_flight_plan");
                                      return;
                                    }
                                    if (value === "core_ai_optimization__exif_plus_flight_plan_plus_external") {
                                      setTuneScope("core_ai_optimization");
                                      setAiInputMode("exif_plus_flight_plan_plus_external");
                                      return;
                                    }
                                    setTuneScope((value as TuneScope) || "core_individual_plus_strategy");
                                  }}
                                  className="w-full px-2 py-1.5 text-xs border border-slate-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                                >
                                  <option value="core_individual">Core individual</option>
                                  <option value="core_only">Core only</option>
                                  <option value="core_ai_optimization">Core AI optimization (controller)</option>
                                  <option value="core_ai_optimization__exif_only">Core AI optimization (EXIF only)</option>
                                  <option value="core_ai_optimization__exif_plus_flight_plan">Core AI optimization (EXIF + flight-plan)</option>
                                  <option value="core_ai_optimization__exif_plus_flight_plan_plus_external">Core AI optimization (EXIF + flight-plan + external)</option>
                                  <option value="core_individual_plus_strategy">Core individual + strategy</option>
                                </select>
                              </div>
                            )}
                            {showCoreAiSessionControls && (
                              <div className="md:col-span-2 mt-1 rounded-lg border border-slate-300 bg-slate-100 p-2.5 space-y-2">
                                <div className="flex items-center justify-between">
                                  <p className="text-xs font-semibold text-blue-900">Session</p>
                                  <span className="text-[10px] text-blue-700">Core AI optimization only</span>
                                </div>
                                <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                                  <CoreAiSessionTestControls
                                    sessionExecutionMode={sessionExecutionMode}
                                    onSessionExecutionModeChange={setSessionExecutionMode}
                                    sourceModelId={sourceModelId}
                                    onSourceModelIdChange={setSourceModelId}
                                    reusableModels={modeCompatibleReusableModels}
                                    reusableModelsLoading={reusableModelsLoading}
                                    reusableModelsError={reusableModelsError}
                                    emptyStateLabel={modeModelEmptyLabel}
                                    onSelectInfoKey={setSelectedInfoKey}
                                  />
                                  {/* Always show trend scope and baseline session */}
                                  {hasLegacyControllerFlow && (
                                    <div className="md:col-span-2">
                                      <label className="flex items-center justify-between text-[11px] font-medium text-slate-600 mb-0.5">
                                        <span>Trend scope</span>
                                        <button onClick={() => setSelectedInfoKey("trend_scope")} className="p-1 text-slate-400 hover:text-slate-600"><Info /></button>
                                      </label>
                                      <select
                                        value={trendScope}
                                        onChange={(e) => setTrendScope((e.target.value as TrendScope) === "phase" ? "phase" : "run")}
                                        className="w-full px-2 py-1.5 text-xs border border-slate-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                                      >
                                        <option value="run">Whole current run trend</option>
                                        <option value="phase">Phase-wise trend</option>
                                      </select>
                                      <p className="mt-1 text-[10px] text-slate-500">Run = one trend across all steps. Phase = separate trend per phase.</p>
                                    </div>
                                  )}
                                  {hasAiInputModeTrainFlow && (
                                    <div className="md:col-span-2">
                                      <label className="flex items-center justify-between text-[11px] font-medium text-slate-600 mb-0.5">
                                        <span>AI selector strategy</span>
                                        <button onClick={() => setSelectedInfoKey("ai_selector_strategy")} className="p-1 text-slate-400 hover:text-slate-600"><Info /></button>
                                      </label>
                                      <select
                                        value={aiSelectorStrategy}
                                        onChange={(e) => setAiSelectorStrategy(e.target.value === "continuous_bandit_linear" ? "continuous_bandit_linear" : "preset_bias")}
                                        className="w-full px-2 py-1.5 text-xs border border-slate-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                                      >
                                        <option value="preset_bias">Preset bias</option>
                                        <option value="continuous_bandit_linear">Continuous bandit (linear)</option>
                                      </select>
                                      <p className="mt-1 text-[10px] text-slate-500">Continuous bandit uses bounded continuous multipliers instead of fixed preset templates.</p>
                                    </div>
                                  )}
                                  {hasAiInputModeCompareFlow && (
                                    <div className="md:col-span-2">
                                      <label className="flex items-center justify-between text-[11px] font-medium text-slate-600 mb-0.5">
                                        <span>{sessionExecutionMode === "train" ? "Baseline session (required)" : "Baseline session (optional)"}</span>
                                        <button onClick={() => setSelectedInfoKey("baseline_session_id")} className="p-1 text-slate-400 hover:text-slate-600"><Info /></button>
                                      </label>
                                      <select
                                        value={baselineSessionIdForAi}
                                        onChange={(e) => setBaselineSessionIdForAi(e.target.value || "")}
                                        className="w-full px-2 py-1.5 text-xs border border-slate-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                                      >
                                        {baselineCandidateRuns.length === 0 && <option value="">No completed baseline sessions found</option>}
                                        {baselineCandidateRuns.map((run) => (
                                          <option key={run.run_id} value={run.run_id}>
                                            {run.run_name || run.run_id} ({run.run_id})
                                          </option>
                                        ))}
                                      </select>
                                      <p className="mt-1 text-[10px] text-slate-500">
                                        {sessionExecutionMode === "train"
                                          ? "Used as reference for baseline-relative scoring."
                                          : "Optional in test mode. Leave empty to run without baseline-relative reward/base-S comparison."}
                                      </p>
                                    </div>
                                  )}
                                  {sessionExecutionMode === "train" && (
                                  <>
                                  <div className="md:col-span-2">
                                    <label className="inline-flex items-center gap-2 text-[11px] font-medium text-slate-700">
                                      <input
                                        type="checkbox"
                                        className="w-4 h-4"
                                        checked={warmupAtStart}
                                        onChange={(e) => {
                                          const enabled = e.target.checked;
                                          setWarmupAtStart(enabled);
                                          if (enabled) {
                                            setRunCount(30);
                                          }
                                        }}
                                      />
                                      <span className="flex items-center gap-1">
                                        Warmup at start (phased experiment)
                                        <button type="button" onClick={() => setSelectedInfoKey("warmup_at_start")} className="p-1 text-slate-400 hover:text-slate-600"><Info /></button>
                                      </span>
                                    </label>
                                    <p className="mt-1 text-[10px] text-slate-500">Uses this project base session config and runs a fixed A/B/C warmup schedule.</p>
                                  </div>
                                  <>
                                      <div>
                                        <label className="flex items-center justify-between text-[11px] font-medium text-slate-600 mb-0.5">
                                          <span>Run count</span>
                                          <button onClick={() => setSelectedInfoKey("run_count")} className="p-1 text-slate-400 hover:text-slate-600"><Info /></button>
                                        </label>
                                        <input
                                          type="number"
                                          min={warmupAtStart ? 10 : 1}
                                          step={1}
                                          value={runCount}
                                          onChange={(e) => {
                                            const parsed = parseInt(e.target.value || "1") || 1;
                                            const minRuns = warmupAtStart ? 10 : 1;
                                            setRunCount(Math.max(minRuns, parsed));
                                          }}
                                          className="w-full px-2 py-1.5 text-xs border border-slate-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                                        />
                                        <p className="mt-1 text-[10px] text-slate-500">{warmupAtStart ? `Total warmup runs across phases A/B/C: ${runCount} (minimum 10; default 30 when enabled).` : `Includes selected session as run 1; creates ${Math.max(0, runCount - 1)} additional sessions.`}</p>
                                      </div>
                                      <div>
                                        <label className="flex items-center justify-between text-[11px] font-medium text-slate-600 mb-0.5">
                                          <span>Jitter mode</span>
                                          <button onClick={() => setSelectedInfoKey("run_jitter_mode")} className="p-1 text-slate-400 hover:text-slate-600"><Info /></button>
                                        </label>
                                        <select
                                          value={runJitterMode}
                                          disabled={warmupAtStart}
                                          onChange={(e) => setRunJitterMode((e.target.value as RunJitterMode) === "random" ? "random" : "fixed")}
                                          className="w-full px-2 py-1.5 text-xs border border-slate-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-slate-100 disabled:text-slate-500"
                                        >
                                          <option value="fixed">Fixed (deterministic)</option>
                                          <option value="random">Random (bounded)</option>
                                        </select>
                                        <p className="mt-1 text-[10px] text-slate-500">Applied from run 2 onward. Run 1 uses the base values.</p>
                                      </div>
                                      <div>
                                        <label className="flex items-center justify-between text-[11px] font-medium text-slate-600 mb-0.5">
                                          <span>Jitter factor (fixed mode)</span>
                                          <button onClick={() => setSelectedInfoKey("run_jitter_factor")} className="p-1 text-slate-400 hover:text-slate-600"><Info /></button>
                                        </label>
                                        <input
                                          type="number"
                                          min={0.1}
                                          step={0.01}
                                          value={runJitterFactor}
                                          disabled={warmupAtStart || runJitterMode !== "fixed"}
                                          onChange={(e) => {
                                            const value = parseFloat(e.target.value);
                                            if (Number.isFinite(value)) {
                                              setRunJitterFactor(Math.max(0.1, value));
                                            }
                                          }}
                                          className="w-full px-2 py-1.5 text-xs border border-slate-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-slate-100 disabled:text-slate-500"
                                        />
                                      </div>
                                      <div>
                                        <label className="flex items-center justify-between text-[11px] font-medium text-slate-600 mb-0.5">
                                          <span>Random jitter min</span>
                                          <button onClick={() => setSelectedInfoKey("run_jitter_min")} className="p-1 text-slate-400 hover:text-slate-600"><Info /></button>
                                        </label>
                                        <input
                                          type="number"
                                          min={0.000001}
                                          step={0.01}
                                          value={runJitterMin}
                                          disabled={warmupAtStart || runJitterMode !== "random"}
                                          onChange={(e) => {
                                            const value = parseFloat(e.target.value);
                                            if (Number.isFinite(value)) {
                                              setRunJitterMin(Math.max(1e-6, value));
                                            }
                                          }}
                                          className="w-full px-2 py-1.5 text-xs border border-slate-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-slate-100 disabled:text-slate-500"
                                        />
                                      </div>
                                      <div>
                                        <label className="flex items-center justify-between text-[11px] font-medium text-slate-600 mb-0.5">
                                          <span>Random jitter max</span>
                                          <button onClick={() => setSelectedInfoKey("run_jitter_max")} className="p-1 text-slate-400 hover:text-slate-600"><Info /></button>
                                        </label>
                                        <input
                                          type="number"
                                          min={0.000001}
                                          step={0.01}
                                          value={runJitterMax}
                                          disabled={warmupAtStart || runJitterMode !== "random"}
                                          onChange={(e) => {
                                            const value = parseFloat(e.target.value);
                                            if (Number.isFinite(value)) {
                                              setRunJitterMax(Math.max(1e-6, value));
                                            }
                                          }}
                                          className="w-full px-2 py-1.5 text-xs border border-slate-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-slate-100 disabled:text-slate-500"
                                        />
                                      </div>
                                      <div className="md:col-span-2">
                                        <label className="inline-flex items-center gap-2 text-[11px] font-medium text-slate-700">
                                          <input
                                            type="checkbox"
                                            className="w-4 h-4"
                                            checked={continueOnFailure}
                                            disabled={warmupAtStart}
                                            onChange={(e) => setContinueOnFailure(e.target.checked)}
                                          />
                                          <span className="flex items-center gap-1">
                                            Continue remaining runs on failure
                                            <button type="button" onClick={() => setSelectedInfoKey("continue_on_failure")} className="p-1 text-slate-400 hover:text-slate-600"><Info /></button>
                                          </span>
                                        </label>
                                      </div>
                                      <div>
                                        <label className="flex items-center justify-between text-[11px] font-medium text-slate-600 mb-0.5">
                                          <span>Start mode</span>
                                          <button onClick={() => setSelectedInfoKey("start_model_mode")} className="p-1 text-slate-400 hover:text-slate-600"><Info /></button>
                                        </label>
                                        <select
                                          value={startModelMode}
                                          onChange={(e) => {
                                            const next = (e.target.value as StartModelMode) || "scratch";
                                            setStartModelMode(next);
                                            if (next !== "reuse") {
                                              setSourceModelId("");
                                            }
                                          }}
                                          className="w-full px-2 py-1.5 text-xs border border-slate-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                                        >
                                          <option value="scratch">Start from scratch</option>
                                          <option value="reuse">Warm-start from reusable model</option>
                                        </select>
                                      </div>
                                      <div>
                                        {!isReusableWarmStartSelected && (
                                          <>
                                            <label className="flex items-center justify-between text-[11px] font-medium text-slate-600 mb-0.5">
                                              <span>Project model name</span>
                                              <button onClick={() => setSelectedInfoKey("project_model_name")} className="p-1 text-slate-400 hover:text-slate-600"><Info /></button>
                                            </label>
                                            <input
                                              type="text"
                                              value={projectModelName}
                                              onChange={(e) => setProjectModelName(e.target.value)}
                                              placeholder="Project model name (optional)"
                                              className="mb-1 w-full px-2 py-1.5 text-xs border border-slate-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                                            />
                                          </>
                                        )}
                                        <label className="flex items-center justify-between text-[11px] font-medium text-slate-600 mb-0.5">
                                          <span>Global reusable model (optional)</span>
                                          <button onClick={() => setSelectedInfoKey("source_model_id")} className="p-1 text-slate-400 hover:text-slate-600"><Info /></button>
                                        </label>
                                        <select
                                          value={sourceModelId}
                                          onChange={(e) => setSourceModelId(e.target.value)}
                                          disabled={startModelMode !== "reuse" || reusableModelsLoading || modeCompatibleReusableModels.length === 0}
                                          className="w-full px-2 py-1.5 text-xs border border-slate-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-slate-100 disabled:text-slate-500"
                                        >
                                          <option value="">
                                            {reusableModelsLoading
                                              ? "Loading models..."
                                              : modeCompatibleReusableModels.length > 0
                                                ? "Select reusable model"
                                                : modeModelEmptyLabel}
                                          </option>
                                          {sourceModelId && !modeCompatibleReusableModels.some((item) => item.model_id === sourceModelId) && (
                                            <option value={sourceModelId}>{sourceModelId} (saved)</option>
                                          )}
                                          {modeCompatibleReusableModels.map((item) => (
                                            <option key={item.model_id} value={item.model_id}>
                                              {item.model_name || item.model_id}
                                            </option>
                                          ))}
                                        </select>
                                        {reusableModelsError && (
                                          <p className="mt-1 text-[10px] text-red-600">{reusableModelsError}</p>
                                        )}
                                      </div>
                                    </>
                                  </>
                                  )}
                                </div>
                              </div>
                            )}
                            {showManualModifiedTuneControls && (
                              <div>
                                <label className="flex items-center justify-between text-[11px] font-medium text-slate-600 mb-0.5">
                                  <span>Modified tuning start step</span>
                                  <button onClick={() => setSelectedInfoKey("tune_start_step")} className="p-1 text-slate-400 hover:text-slate-600"><Info /></button>
                                </label>
                                <input
                                  type="number"
                                  min={1}
                                  step={1}
                                  value={tuneStartStep}
                                  onChange={(e) => setTuneStartStep(Math.max(1, parseInt(e.target.value) || 100))}
                                  className="w-full px-2 py-1.5 text-xs border border-slate-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                                />
                              </div>
                            )}
                            {showManualModifiedTuneControls && (
                              <div>
                                <label className="flex items-center justify-between text-[11px] font-medium text-slate-600 mb-0.5">
                                  <span>Minimum improvement for update</span>
                                  <button onClick={() => setSelectedInfoKey("tune_min_improvement")} className="p-1 text-slate-400 hover:text-slate-600"><Info /></button>
                                </label>
                                <input
                                  type="number"
                                  min={0}
                                  max={1}
                                  step={0.001}
                                  value={tuneMinImprovement}
                                  onChange={(e) => {
                                    const value = parseFloat(e.target.value);
                                    if (Number.isFinite(value)) {
                                      setTuneMinImprovement(Math.max(0, Math.min(1, value)));
                                    }
                                  }}
                                  className="w-full px-2 py-1.5 text-xs border border-slate-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                                />
                              </div>
                            )}
                            {showManualModifiedTuneControls && (
                              <div>
                                <label className="flex items-center justify-between text-[11px] font-medium text-slate-600 mb-0.5">
                                  <span>Modified tuning end step</span>
                                  <button onClick={() => setSelectedInfoKey("tune_end_step")} className="p-1 text-slate-400 hover:text-slate-600"><Info /></button>
                                </label>
                                <input
                                  type="number"
                                  min={1}
                                  step={1}
                                  value={tuneEndStep}
                                  onChange={(e) => setTuneEndStep(Math.max(1, parseInt(e.target.value) || 15000))}
                                  className="w-full px-2 py-1.5 text-xs border border-slate-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                                />
                              </div>
                            )}
                            {showManualModifiedTuneControls && (
                              <div>
                                <label className="flex items-center justify-between text-[11px] font-medium text-slate-600 mb-0.5">
                                  <span>Modified tuning interval</span>
                                  <button onClick={() => setSelectedInfoKey("tune_interval")} className="p-1 text-slate-400 hover:text-slate-600"><Info /></button>
                                </label>
                                <input
                                  type="number"
                                  min={1}
                                  step={1}
                                  value={tuneInterval}
                                  onChange={(e) => setTuneInterval(Math.max(1, parseInt(e.target.value) || 100))}
                                  className="w-full px-2 py-1.5 text-xs border border-slate-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                                />
                              </div>
                            )}
                            <div className="md:col-span-2">
                              <label className="flex items-center justify-between text-[11px] font-medium text-slate-600 mb-0.5">
                                <span>Sparse reconstruction preference</span>
                                <button onClick={() => setSelectedInfoKey("sparse_preference")} className="p-1 text-slate-400 hover:text-slate-600"><Info /></button>
                              </label>
                              <select
                                value={sparsePreference}
                                onChange={(e) => setSparsePreference(e.target.value)}
                                className="w-full px-2 py-1.5 text-xs border border-slate-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                                disabled={sparseOptionsLoading}
                              >
                                {sparseOptions.map((opt) => (
                                  <option key={opt.value} value={opt.value}>{opt.label}</option>
                                ))}
                              </select>
                              {sparsePreference === "merge_selected" && (
                                <div className="mt-3 rounded-lg border border-slate-200 bg-slate-50 p-3 space-y-2">
                                  <div className="flex items-center justify-between gap-2">
                                    <p className="text-[11px] font-semibold text-slate-700">Select folders to merge</p>
                                    <div className="flex items-center gap-2">
                                      <button
                                        type="button"
                                        onClick={() => setSparseMergeSelection(sparseMergeCandidates.map((opt) => opt.value))}
                                        className="text-[11px] px-2 py-1 border border-slate-300 rounded text-slate-600 hover:bg-white"
                                      >
                                        Select all
                                      </button>
                                      <button
                                        type="button"
                                        disabled={sparseMergeBuildLoading || sparseMergeSelection.length < 2}
                                        onClick={buildSparseMergeNow}
                                        className="text-[11px] px-2 py-1 border border-blue-300 rounded text-blue-700 hover:bg-blue-50 disabled:opacity-50 disabled:cursor-not-allowed"
                                      >
                                        {sparseMergeBuildLoading ? "Building..." : "Build merged model now"}
                                      </button>
                                    </div>
                                  </div>
                                  {sparseMergeCandidates.length === 0 ? (
                                    <p className="text-[11px] text-slate-500">No sparse folders available yet.</p>
                                  ) : (
                                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                                      {sparseMergeCandidates.map((opt) => {
                                        const checked = sparseMergeSelection.includes(opt.value);
                                        return (
                                          <label key={`merge-${opt.value}`} className="flex items-start gap-2 text-[11px] text-slate-700 border border-slate-200 rounded bg-white px-2 py-2">
                                            <input
                                              type="checkbox"
                                              className="mt-0.5"
                                              checked={checked}
                                              onChange={() => toggleSparseMergeSelection(opt.value)}
                                            />
                                            <span>{opt.label}</span>
                                          </label>
                                        );
                                      })}
                                    </div>
                                  )}
                                  <p className={`text-[11px] ${sparseMergeSelection.length >= 2 ? "text-slate-500" : "text-amber-600"}`}>
                                    {sparseMergeSelection.length >= 2
                                      ? `${sparseMergeSelection.length} folders selected.`
                                      : "Pick at least two folders to enable merge mode."}
                                  </p>
                                  <p className="text-[11px] text-amber-600">
                                    Merge mode aligns selected folders to an anchor using overlapping registered cameras. Folders without enough overlap are skipped.
                                  </p>
                                  {sparseMergeBuildMessage && (
                                    <p className={`text-[11px] ${sparseMergeBuildMessage.toLowerCase().includes("ready") ? "text-emerald-700" : "text-rose-600"}`}>
                                      {sparseMergeBuildMessage}
                                    </p>
                                  )}
                                </div>
                              )}
                              {showMergeReportPanel && (
                                <div className="mt-3 rounded-lg border border-slate-200 bg-slate-50 p-3 space-y-2">
                                  <div className="flex items-center justify-between gap-2">
                                    <p className="text-[11px] font-semibold text-slate-700">Merge Report</p>
                                    <button
                                      type="button"
                                      onClick={async () => {
                                        setSparseMergeReportError(null);
                                        setSparseMergeReportLoading(true);
                                        try {
                                          const candidate = sparsePreference.startsWith("_merged/") ? sparsePreference : undefined;
                                          const res = await api.get(`/projects/${projectId}/sparse/merge-report`, {
                                            params: candidate ? { candidate } : {},
                                          });
                                          const data = res.data || {};
                                          if (!data.available || !data.report) {
                                            setSparseMergeReport(null);
                                            setSparseMergeReportCandidate(data.candidate ?? null);
                                          } else {
                                            setSparseMergeReport(data.report as SparseMergeReport);
                                            setSparseMergeReportCandidate(data.candidate ?? null);
                                          }
                                        } catch (err: any) {
                                          const msg = err?.response?.data?.detail || err?.message || "Failed to load merge report";
                                          setSparseMergeReportError(msg);
                                          setSparseMergeReport(null);
                                        } finally {
                                          setSparseMergeReportLoading(false);
                                        }
                                      }}
                                      className="text-[11px] px-2 py-1 border border-slate-300 rounded text-slate-600 hover:bg-white"
                                    >
                                      Refresh
                                    </button>
                                  </div>
                                  {sparseMergeReportLoading ? (
                                    <p className="text-[11px] text-slate-500">Loading merge metadata...</p>
                                  ) : sparseMergeReportError ? (
                                    <p className="text-[11px] text-rose-600">{sparseMergeReportError}</p>
                                  ) : !sparseMergeReport ? (
                                    <p className="text-[11px] text-slate-500">No cached merge metadata available yet.</p>
                                  ) : (
                                    <div className="space-y-2">
                                      <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-[11px] text-slate-700">
                                        <div className="rounded border border-slate-200 bg-white px-2 py-2">
                                          <span className="font-semibold">Candidate:</span> {sparseMergeReportCandidate ?? "unknown"}
                                        </div>
                                        <div className="rounded border border-slate-200 bg-white px-2 py-2">
                                          <span className="font-semibold">Created:</span> {formatMergeDate(sparseMergeReport.created_at)}
                                        </div>
                                        <div className="rounded border border-slate-200 bg-white px-2 py-2">
                                          <span className="font-semibold">Anchor:</span> {sparseMergeReport.anchor_relative_path ?? "unknown"}
                                        </div>
                                        <div className="rounded border border-slate-200 bg-white px-2 py-2">
                                          <span className="font-semibold">Merged points:</span> {(sparseMergeReport.merged_points ?? 0).toLocaleString()}
                                        </div>
                                      </div>
                                      {Array.isArray(sparseMergeReport.source_details) && sparseMergeReport.source_details.length > 0 && (
                                        <div className="rounded border border-slate-200 bg-white overflow-x-auto">
                                          <table className="min-w-full text-[11px]">
                                            <thead className="bg-slate-100 text-slate-700">
                                              <tr>
                                                <th className="px-2 py-1 text-left">Folder</th>
                                                <th className="px-2 py-1 text-left">Used</th>
                                                <th className="px-2 py-1 text-left">Overlap</th>
                                                <th className="px-2 py-1 text-left">Scale</th>
                                                <th className="px-2 py-1 text-left">Reason</th>
                                              </tr>
                                            </thead>
                                            <tbody>
                                              {sparseMergeReport.source_details.map((detail, idx) => (
                                                <tr key={`merge-detail-${idx}`} className="border-t border-slate-100 text-slate-700">
                                                  <td className="px-2 py-1">{detail.relative_path ?? "unknown"}</td>
                                                  <td className="px-2 py-1">{detail.used ? "yes" : "no"}</td>
                                                  <td className="px-2 py-1">{typeof detail.overlap_images === "number" ? detail.overlap_images : "-"}</td>
                                                  <td className="px-2 py-1">{typeof detail.scale === "number" ? detail.scale.toFixed(4) : "-"}</td>
                                                  <td className="px-2 py-1">{detail.reason ?? (detail.used ? "aligned" : "-")}</td>
                                                </tr>
                                              ))}
                                            </tbody>
                                          </table>
                                        </div>
                                      )}
                                    </div>
                                  )}
                                </div>
                              )}
                            </div>
                            <div>
                              <label className="flex items-center justify-between text-[11px] font-medium text-slate-600 mb-0.5">
                                <span>Max steps</span>
                                <button onClick={() => setSelectedInfoKey("maxSteps")} className="p-1 text-slate-400 hover:text-slate-600"><Info /></button>
                              </label>
                              <input
                                type="number"
                                value={maxSteps}
                                onChange={(e) => setMaxSteps(parseInt(e.target.value) || 15000)}
                                className="w-full px-2 py-1.5 text-xs border border-slate-300 rounded-md"
                                min={100}
                                max={50000}
                                step={100}
                              />
                            </div>
                            <div className="md:col-span-2 grid grid-cols-1 sm:grid-cols-2 gap-0.5">
                              {engine === "gsplat" && (
                                <>
                                  <div>
                                  <label className="flex items-center justify-between text-[11px] font-medium text-slate-600 mb-0.5">
                                    <span>Log interval</span>
                                    <button onClick={() => setSelectedInfoKey("logInterval")} className="p-1 text-slate-400 hover:text-slate-600"><Info /></button>
                                  </label>
                                  <input
                                    type="number"
                                    value={logInterval}
                                    onChange={(e) => setLogInterval(parseInt(e.target.value) || 100)}
                                    className="w-full px-2 py-1.5 text-xs border border-slate-300 rounded-md"
                                    min={10}
                                    step={10}
                                  />
                                  </div>
                                  <div>
                                  <label className="flex items-center justify-between text-[11px] font-medium text-slate-600 mb-0.5">
                                    <span>Eval interval</span>
                                    <button onClick={() => setSelectedInfoKey("evalInterval")} className="p-1 text-slate-400 hover:text-slate-600"><Info /></button>
                                  </label>
                                  <input
                                    type="number"
                                    value={evalInterval}
                                    onChange={(e) => setEvalInterval(parseInt(e.target.value) || 1000)}
                                    className="w-full px-2 py-1.5 text-xs border border-slate-300 rounded-md"
                                    min={50}
                                    step={50}
                                  />
                                  </div>
                                  <div>
                                    <label className="flex items-center justify-between text-[11px] font-medium text-slate-600 mb-0.5">
                                      <span>Splat export interval</span>
                                      <button onClick={() => setSelectedInfoKey("splatInterval")} className="p-1 text-slate-400 hover:text-slate-600"><Info /></button>
                                    </label>
                                    <input
                                      type="number"
                                      value={splatInterval}
                                      onChange={(e) => setSplatInterval(parseInt(e.target.value) || 31000)}
                                      className="w-full px-2 py-1.5 text-xs border border-slate-300 rounded-md"
                                      min={50}
                                      step={50}
                                    />
                                  </div>
                                  <div>
                                  <label className="flex items-center justify-between text-[11px] font-medium text-slate-600 mb-0.5">
                                    <span>Checkpoint interval</span>
                                    <button onClick={() => setSelectedInfoKey("saveInterval")} className="p-1 text-slate-400 hover:text-slate-600"><Info /></button>
                                  </label>
                                  <input
                                    type="number"
                                    value={saveInterval}
                                    onChange={(e) => setSaveInterval(parseInt(e.target.value) || 31000)}
                                    className="w-full px-2 py-1.5 text-xs border border-slate-300 rounded-md"
                                    min={50}
                                    step={50}
                                  />
                                  </div>
                                  <div className="flex items-center justify-between">
                                    <label className="flex items-center gap-1 text-[11px] font-medium text-slate-600 mb-0.5">
                                      <span>Save best splat</span>
                                      <button onClick={() => setSelectedInfoKey("saveBestSplat")} className="p-1 text-slate-400 hover:text-slate-600"><Info /></button>
                                    </label>
                                    <input
                                      type="checkbox"
                                      checked={saveBestSplat}
                                      onChange={e => setSaveBestSplat(e.target.checked)}
                                      className="ml-2"
                                    />
                                  </div>
                                  <div>
                                    <label className="flex items-center justify-between text-[11px] font-medium text-slate-600 mb-0.5">
                                      <span>Best splat interval</span>
                                      <button onClick={() => setSelectedInfoKey("bestSplatInterval")} className="p-1 text-slate-400 hover:text-slate-600"><Info /></button>
                                    </label>
                                    <input
                                      type="number"
                                      value={bestSplatInterval}
                                      onChange={(e) => setBestSplatInterval(parseInt(e.target.value) || 100)}
                                      className="w-full px-2 py-1.5 text-xs border border-slate-300 rounded-md"
                                      min={10}
                                      step={10}
                                    />
                                  </div>
                                  <div>
                                    <label className="flex items-center justify-between text-[11px] font-medium text-slate-600 mb-0.5">
                                      <span>Best splat start step</span>
                                      <button onClick={() => setSelectedInfoKey("bestSplatStartStep")} className="p-1 text-slate-400 hover:text-slate-600"><Info /></button>
                                    </label>
                                    <input
                                      type="number"
                                      value={bestSplatStartStep}
                                      onChange={(e) => setBestSplatStartStep(parseInt(e.target.value) || 2000)}
                                      className="w-full px-2 py-1.5 text-xs border border-slate-300 rounded-md"
                                      min={1}
                                      step={10}
                                    />
                                  </div>
                                </>
                              )}
                              {engine === "gsplat" && (
                              <>
                                <div className="sm:col-span-2 mt-1 rounded-md border border-slate-200 bg-white p-2">
                                  <label className="flex items-center justify-between text-[11px] font-medium text-slate-700 mb-1">
                                    <span>Enable Early Stop</span>
                                    <button onClick={() => setSelectedInfoKey("auto_early_stop")} className="p-1 text-slate-400 hover:text-slate-600"><Info /></button>
                                  </label>
                                  <div className="flex items-center gap-2">
                                    <input
                                      type="checkbox"
                                      checked={autoEarlyStop}
                                      onChange={(e) => setAutoEarlyStop(e.target.checked)}
                                      className="w-4 h-4"
                                    />
                                    <span className="text-[11px] text-slate-600">Monitor every N steps, confirm stop at eval points only</span>
                                  </div>
                                </div>
                                {autoEarlyStop && (
                                  <>
                                    <div>
                                      <label className="flex items-center justify-between text-[11px] font-medium text-slate-600 mb-0.5">
                                        <span>Monitor interval</span>
                                        <button onClick={() => setSelectedInfoKey("earlyStopMonitorInterval")} className="p-1 text-slate-400 hover:text-slate-600"><Info /></button>
                                      </label>
                                      <input
                                        type="number"
                                        value={earlyStopMonitorInterval}
                                        onChange={(e) => setEarlyStopMonitorInterval(parseInt(e.target.value) || 200)}
                                        className="w-full px-2 py-1.5 text-xs border border-slate-300 rounded-md"
                                        min={10}
                                        step={10}
                                      />
                                    </div>
                                    <div>
                                      <label className="flex items-center justify-between text-[11px] font-medium text-slate-600 mb-0.5">
                                        <span>Decision points</span>
                                        <button onClick={() => setSelectedInfoKey("earlyStopDecisionPoints")} className="p-1 text-slate-400 hover:text-slate-600"><Info /></button>
                                      </label>
                                      <input
                                        type="number"
                                        value={earlyStopDecisionPoints}
                                        onChange={(e) => setEarlyStopDecisionPoints(parseInt(e.target.value) || 10)}
                                        className="w-full px-2 py-1.5 text-xs border border-slate-300 rounded-md"
                                        min={3}
                                        step={1}
                                      />
                                    </div>
                                    <div>
                                      <label className="flex items-center justify-between text-[11px] font-medium text-slate-600 mb-0.5">
                                        <span>Min eval points</span>
                                        <button onClick={() => setSelectedInfoKey("earlyStopMinEvalPoints")} className="p-1 text-slate-400 hover:text-slate-600"><Info /></button>
                                      </label>
                                      <input
                                        type="number"
                                        value={earlyStopMinEvalPoints}
                                        onChange={(e) => setEarlyStopMinEvalPoints(parseInt(e.target.value) || 6)}
                                        className="w-full px-2 py-1.5 text-xs border border-slate-300 rounded-md"
                                        min={2}
                                        step={1}
                                      />
                                    </div>
                                    <div>
                                      <label className="flex items-center justify-between text-[11px] font-medium text-slate-600 mb-0.5">
                                        <span>Min step ratio</span>
                                        <button onClick={() => setSelectedInfoKey("earlyStopMinStepRatio")} className="p-1 text-slate-400 hover:text-slate-600"><Info /></button>
                                      </label>
                                      <input
                                        type="number"
                                        value={earlyStopMinStepRatio}
                                        onChange={(e) => setEarlyStopMinStepRatio(parseFloat(e.target.value) || 0.25)}
                                        className="w-full px-2 py-1.5 text-xs border border-slate-300 rounded-md"
                                        min={0}
                                        max={1}
                                        step={0.05}
                                      />
                                    </div>
                                    <div>
                                      <label className="flex items-center justify-between text-[11px] font-medium text-slate-600 mb-0.5">
                                        <span>Monitor min rel. improve</span>
                                        <button onClick={() => setSelectedInfoKey("earlyStopMonitorMinRelativeImprovement")} className="p-1 text-slate-400 hover:text-slate-600"><Info /></button>
                                      </label>
                                      <input
                                        type="number"
                                        value={earlyStopMonitorMinRelativeImprovement}
                                        onChange={(e) => setEarlyStopMonitorMinRelativeImprovement(parseFloat(e.target.value) || 0.0015)}
                                        className="w-full px-2 py-1.5 text-xs border border-slate-300 rounded-md"
                                        min={0}
                                        step={0.0001}
                                      />
                                    </div>
                                    <div>
                                      <label className="flex items-center justify-between text-[11px] font-medium text-slate-600 mb-0.5">
                                        <span>Eval min rel. improve</span>
                                        <button onClick={() => setSelectedInfoKey("earlyStopEvalMinRelativeImprovement")} className="p-1 text-slate-400 hover:text-slate-600"><Info /></button>
                                      </label>
                                      <input
                                        type="number"
                                        value={earlyStopEvalMinRelativeImprovement}
                                        onChange={(e) => setEarlyStopEvalMinRelativeImprovement(parseFloat(e.target.value) || 0.003)}
                                        className="w-full px-2 py-1.5 text-xs border border-slate-300 rounded-md"
                                        min={0}
                                        step={0.0001}
                                      />
                                    </div>
                                    <div>
                                      <label className="flex items-center justify-between text-[11px] font-medium text-slate-600 mb-0.5">
                                        <span>Max volatility ratio</span>
                                        <button onClick={() => setSelectedInfoKey("earlyStopMaxVolatilityRatio")} className="p-1 text-slate-400 hover:text-slate-600"><Info /></button>
                                      </label>
                                      <input
                                        type="number"
                                        value={earlyStopMaxVolatilityRatio}
                                        onChange={(e) => setEarlyStopMaxVolatilityRatio(parseFloat(e.target.value) || 0.01)}
                                        className="w-full px-2 py-1.5 text-xs border border-slate-300 rounded-md"
                                        min={0}
                                        step={0.001}
                                      />
                                    </div>
                                    <div>
                                      <label className="flex items-center justify-between text-[11px] font-medium text-slate-600 mb-0.5">
                                        <span>EMA alpha</span>
                                        <button onClick={() => setSelectedInfoKey("earlyStopEmaAlpha")} className="p-1 text-slate-400 hover:text-slate-600"><Info /></button>
                                      </label>
                                      <input
                                        type="number"
                                        value={earlyStopEmaAlpha}
                                        onChange={(e) => setEarlyStopEmaAlpha(parseFloat(e.target.value) || 0.1)}
                                        className="w-full px-2 py-1.5 text-xs border border-slate-300 rounded-md"
                                        min={0.001}
                                        max={1}
                                        step={0.01}
                                      />
                                    </div>
                                  </>
                                )}
                              </>
                              )}
                            </div>
                          </div>
                        </div>

                        {showManualDensificationControls && (
                          <div className="rounded-xl border border-slate-300 bg-slate-100 shadow-sm">
                            <div className="flex items-center justify-between px-3 py-2 border-b border-slate-100">
                              <div>
                                <p className="text-sm font-semibold text-slate-800">gsplat-only controls</p>
                                <p className="text-xs text-slate-500">Only applied when gsplat is selected.</p>
                              </div>
                              <span className="text-xs px-3 py-1 rounded-full bg-blue-100 text-blue-700">gsplat selected</span>
                            </div>
                            <div className="space-y-0.5 px-3 py-0.5 text-sm">
                              <div className="space-y-0.5 border-t border-slate-100 pt-0.5">
                                  <div>
                                    <p className="text-xs font-semibold text-slate-600 mb-2">Densification schedule</p>
                                    <div className="grid grid-cols-1 sm:grid-cols-3 gap-0">
                                      <div>
                                        <label className="flex items-center justify-between text-[11px] font-semibold text-slate-500 mb-1">
                                          <span>Start step</span>
                                          <button onClick={() => setSelectedInfoKey("densify_from_iter")} className="p-1 text-slate-400 hover:text-slate-600"><Info /></button>
                                        </label>
                                        <input
                                          type="number"
                                          value={densifyFromIter}
                                          onChange={(e) => setDensifyFromIter(parseInt(e.target.value) || 0)}
                                          className="w-full px-2 py-1.5 text-xs border border-slate-300 rounded-md"
                                          min={0}
                                          step={50}
                                        />
                                      </div>
                                      <div>
                                        <label className="flex items-center justify-between text-[11px] font-semibold text-slate-500 mb-1">
                                          <span>Stop step</span>
                                          <button onClick={() => setSelectedInfoKey("densify_until_iter")} className="p-1 text-slate-400 hover:text-slate-600"><Info /></button>
                                        </label>
                                        <input
                                          type="number"
                                          value={densifyUntilIter}
                                          onChange={(e) => setDensifyUntilIter(parseInt(e.target.value) || 0)}
                                          className="w-full px-2 py-1.5 text-xs border border-slate-300 rounded-md"
                                          min={0}
                                          step={100}
                                        />
                                      </div>
                                      <div>
                                        <label className="flex items-center justify-between text-[11px] font-semibold text-slate-500 mb-1">
                                          <span>Interval</span>
                                          <button onClick={() => setSelectedInfoKey("densification_interval")} className="p-1 text-slate-400 hover:text-slate-600"><Info /></button>
                                        </label>
                                        <input
                                          type="number"
                                          value={densificationInterval}
                                          onChange={(e) => setDensificationInterval(parseInt(e.target.value) || 1)}
                                          className="w-full px-2 py-1.5 text-xs border border-slate-300 rounded-md"
                                          min={10}
                                          step={10}
                                        />
                                      </div>
                                    </div>
                                  </div>
                                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-0">
                                    <div>
                                      <label className="flex items-center justify-between text-[11px] font-medium text-slate-600 mb-0.5">
                                        <span>Densify grad threshold</span>
                                        <button onClick={() => setSelectedInfoKey("densify_grad_threshold")} className="p-1 text-slate-400 hover:text-slate-600"><Info /></button>
                                      </label>
                                      <input
                                        type="number"
                                        step="0.00005"
                                        value={densifyGradThreshold}
                                        onChange={(e) => setDensifyGradThreshold(parseFloat(e.target.value) || 0.0002)}
                                        className="w-full px-2 py-1.5 text-xs border border-slate-300 rounded-md"
                                        min={0.00005}
                                      />
                                    </div>
                                    <div>
                                      <label className="flex items-center justify-between text-[11px] font-medium text-slate-600 mb-0.5">
                                        <span>Opacity threshold</span>
                                        <button onClick={() => setSelectedInfoKey("opacity_threshold")} className="p-1 text-slate-400 hover:text-slate-600"><Info /></button>
                                      </label>
                                      <input
                                        type="number"
                                        step="0.0005"
                                        value={opacityThreshold}
                                        onChange={(e) => setOpacityThreshold(parseFloat(e.target.value) || 0)}
                                        className="w-full px-2 py-1.5 text-xs border border-slate-300 rounded-md"
                                        min={0}
                                      />
                                    </div>
                                    <div>
                                      <label className="flex items-center justify-between text-[11px] font-medium text-slate-600 mb-0.5">
                                        <span>DSSIM weight</span>
                                        <button onClick={() => setSelectedInfoKey("lambda_dssim")} className="p-1 text-slate-400 hover:text-slate-600"><Info /></button>
                                      </label>
                                      <input
                                        type="number"
                                        step="0.05"
                                        value={lambdaDssim}
                                        onChange={(e) => setLambdaDssim(parseFloat(e.target.value) || 0)}
                                        className="w-full px-2 py-1.5 text-xs border border-slate-300 rounded-md"
                                        min={0}
                                      />
                                    </div>
                                  </div>
                                </div>
                              <div className="text-[11px] text-slate-500">Only controls that currently affect upstream gsplat runs are shown here.</div>
                            </div>
                          </div>
                        )}

                        {engine === "litegs" && (
                          <div className="rounded-xl border border-blue-200 bg-slate-50/60 shadow-sm">
                            <div className="flex items-center justify-between px-3 py-2 border-b border-slate-100">
                              <div>
                                <p className="text-sm font-semibold text-slate-800">LiteGS-only controls</p>
                                <p className="text-xs text-slate-500">Only applied when LiteGS is selected.</p>
                              </div>
                              <span className="text-xs px-3 py-1 rounded-full bg-blue-100 text-blue-700">LiteGS selected</span>
                            </div>
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-0 px-3 py-0.5 text-sm">
                              <div>
                                <label className="flex items-center justify-between text-[11px] font-medium text-slate-600 mb-0.5">
                                  <span>Target primitives</span>
                                  <button onClick={() => setSelectedInfoKey("litegs_target_primitives")} className="p-1 text-slate-400 hover:text-slate-600"><Info /></button>
                                </label>
                                <input
                                  type="number"
                                  value={litegsTargetPrimitives}
                                  onChange={(e) => setLitegsTargetPrimitives(parseInt(e.target.value || "0") || 0)}
                                  className="w-full px-2 py-1.5 text-xs border border-slate-300 rounded-md"
                                  min={5000}
                                  step={1000}
                                />
                              </div>
                              <div>
                                <label className="flex items-center justify-between text-[11px] font-medium text-slate-600 mb-0.5">
                                  <span>Alpha shrink factor</span>
                                  <button onClick={() => setSelectedInfoKey("litegs_alpha_shrink")} className="p-1 text-slate-400 hover:text-slate-600"><Info /></button>
                                </label>
                                <input
                                  type="number"
                                  step="0.01"
                                  min={0.5}
                                  max={1.0}
                                  value={litegsAlphaShrink}
                                  onChange={(e) => setLitegsAlphaShrink(parseFloat(e.target.value) || 0.95)}
                                  className="w-full px-2 py-1.5 text-xs border border-slate-300 rounded-md"
                                />
                              </div>
                            </div>
                          </div>
                        )}
                      </div>
                    ) : canManageColmapImages && configTab === "colmap" ? (
                      <div className="grid grid-cols-2 gap-0 text-sm">
                        <div>
                          <label className="flex items-center justify-between text-[11px] font-medium text-slate-600 mb-0.5">
                            <span>SIFT Max Image Size</span>
                            <button onClick={() => setSelectedInfoKey("max_image_size")} className="p-1 text-slate-400 hover:text-slate-600"><Info /></button>
                          </label>
                          <input
                            type="number"
                            value={colmapMaxImageSize ?? ""}
                            onChange={(e) => setColmapMaxImageSize(e.target.value ? parseInt(e.target.value) : undefined)}
                            className="w-full px-2 py-1.5 text-xs border border-slate-300 rounded-md"
                            min={256}
                            step={100}
                          />
                        </div>

                        <div>
                          <label className="flex items-center justify-between text-[11px] font-medium text-slate-600 mb-0.5">
                            <span>SIFT Peak Threshold</span>
                            <button onClick={() => setSelectedInfoKey("peak_threshold")} className="p-1 text-slate-400 hover:text-slate-600"><Info /></button>
                          </label>
                          <input
                            type="number"
                            value={colmapPeakThreshold ?? ""}
                            onChange={(e) => setColmapPeakThreshold(e.target.value ? parseFloat(e.target.value) : undefined)}
                            className="w-full px-2 py-1.5 text-xs border border-slate-300 rounded-md"
                            min={0.001}
                            step={0.001}
                          />
                        </div>

                        <div>
                          <label className="flex items-center justify-between text-[11px] font-medium text-slate-600 mb-0.5">
                            <span>Guided Matching</span>
                            <button onClick={() => setSelectedInfoKey("guided_matching")} className="p-1 text-slate-400 hover:text-slate-600"><Info /></button>
                          </label>
                          <label className="inline-flex items-center gap-2">
                            <input type="checkbox" className="w-4 h-4" checked={colmapGuidedMatching} onChange={e => setColmapGuidedMatching(e.target.checked)} />
                            <span className="text-sm text-slate-700">Enable guided matching</span>
                          </label>
                        </div>

                        <div>
                          <label className="flex items-center justify-between text-[11px] font-medium text-slate-600 mb-0.5">
                            <span>Camera Model</span>
                            <button onClick={() => setSelectedInfoKey("camera_model")} className="p-1 text-slate-400 hover:text-slate-600"><Info /></button>
                          </label>
                          <select value={colmapCameraModel} onChange={e => setColmapCameraModel(e.target.value)} className="w-full px-2 py-1.5 text-xs border border-slate-300 rounded-md">
                            {COLMAP_CAMERA_MODELS.map((model) => (
                              <option key={model} value={model}>{model}</option>
                            ))}
                          </select>
                        </div>

                        <div>
                          <label className="flex items-center justify-between text-[11px] font-medium text-slate-600 mb-0.5">
                            <span>Single Camera</span>
                            <button onClick={() => setSelectedInfoKey("single_camera")} className="p-1 text-slate-400 hover:text-slate-600"><Info /></button>
                          </label>
                          <label className="inline-flex items-center gap-2">
                            <input type="checkbox" className="w-4 h-4" checked={colmapSingleCamera} onChange={e => setColmapSingleCamera(e.target.checked)} />
                            <span className="text-sm text-slate-700">All images share one intrinsics set</span>
                          </label>
                        </div>

                        <div className="col-span-2">
                          <label className="flex items-center justify-between text-[11px] font-medium text-slate-600 mb-0.5">
                            <span>Camera Params (optional)</span>
                            <button onClick={() => setSelectedInfoKey("camera_params")} className="p-1 text-slate-400 hover:text-slate-600"><Info /></button>
                          </label>
                          <input
                            type="text"
                            value={colmapCameraParams}
                            onChange={(e) => setColmapCameraParams(e.target.value)}
                            className="w-full px-2 py-1.5 text-xs border border-slate-300 rounded-md"
                            placeholder="Example for OPENCV: fx,fy,cx,cy,k1,k2,p1,p2"
                          />
                        </div>

                        <div>
                          <label className="flex items-center justify-between text-[11px] font-medium text-slate-600 mb-0.5">
                            <span>Matching Strategy</span>
                            <button onClick={() => setSelectedInfoKey("matching_type")} className="p-1 text-slate-400 hover:text-slate-600"><Info /></button>
                          </label>
                          <select value={colmapMatchingType} onChange={e => setColmapMatchingType(e.target.value as any)} className="w-full px-2 py-1.5 text-xs border border-slate-300 rounded-md">
                            <option value="exhaustive">Exhaustive</option>
                            <option value="sequential">Sequential</option>
                          </select>
                        </div>

                        <div>
                          <label className="flex items-center justify-between text-[11px] font-medium text-slate-600 mb-0.5">
                            <span>Mapper Threads</span>
                            <button onClick={() => setSelectedInfoKey("mapper_num_threads")} className="p-1 text-slate-400 hover:text-slate-600"><Info /></button>
                          </label>
                          <input type="number" value={colmapMapperThreads ?? ""} onChange={e => setColmapMapperThreads(e.target.value ? parseInt(e.target.value) : undefined)} className="w-full px-2 py-1.5 text-xs border border-slate-300 rounded-md" min={1} max={128} />
                        </div>

                        <div>
                          <label className="flex items-center justify-between text-[11px] font-medium text-slate-600 mb-0.5">
                            <span>Mapper Min Matches</span>
                            <button onClick={() => setSelectedInfoKey("mapper_min_num_matches")} className="p-1 text-slate-400 hover:text-slate-600"><Info /></button>
                          </label>
                          <input type="number" value={colmapMapperMinNumMatches ?? ""} onChange={e => setColmapMapperMinNumMatches(e.target.value ? parseInt(e.target.value) : undefined)} className="w-full px-2 py-1.5 text-xs border border-slate-300 rounded-md" min={4} step={1} />
                        </div>

                        <div>
                          <label className="flex items-center justify-between text-[11px] font-medium text-slate-600 mb-0.5">
                            <span>Mapper AbsPose Min Inliers</span>
                            <button onClick={() => setSelectedInfoKey("mapper_abs_pose_min_num_inliers")} className="p-1 text-slate-400 hover:text-slate-600"><Info /></button>
                          </label>
                          <input type="number" value={colmapMapperAbsPoseMinNumInliers ?? ""} onChange={e => setColmapMapperAbsPoseMinNumInliers(e.target.value ? parseInt(e.target.value) : undefined)} className="w-full px-2 py-1.5 text-xs border border-slate-300 rounded-md" min={6} step={1} />
                        </div>

                        <div>
                          <label className="flex items-center justify-between text-[11px] font-medium text-slate-600 mb-0.5">
                            <span>Mapper Init Min Inliers</span>
                            <button onClick={() => setSelectedInfoKey("mapper_init_min_num_inliers")} className="p-1 text-slate-400 hover:text-slate-600"><Info /></button>
                          </label>
                          <input type="number" value={colmapMapperInitMinNumInliers ?? ""} onChange={e => setColmapMapperInitMinNumInliers(e.target.value ? parseInt(e.target.value) : undefined)} className="w-full px-2 py-1.5 text-xs border border-slate-300 rounded-md" min={10} step={1} />
                        </div>

                        <div>
                          <label className="flex items-center justify-between text-[11px] font-medium text-slate-600 mb-0.5">
                            <span>Match Min Inliers</span>
                            <button onClick={() => setSelectedInfoKey("sift_matching_min_num_inliers")} className="p-1 text-slate-400 hover:text-slate-600"><Info /></button>
                          </label>
                          <input type="number" value={colmapSiftMatchingMinNumInliers ?? ""} onChange={e => setColmapSiftMatchingMinNumInliers(e.target.value ? parseInt(e.target.value) : undefined)} className="w-full px-2 py-1.5 text-xs border border-slate-300 rounded-md" min={6} step={1} />
                        </div>

                        <div className="col-span-2">
                          <label className="flex items-center justify-between text-[11px] font-medium text-slate-600 mb-0.5">
                            <span>Post-Mapper Image Registration</span>
                            <button onClick={() => setSelectedInfoKey("run_image_registrator")} className="p-1 text-slate-400 hover:text-slate-600"><Info /></button>
                          </label>
                          <label className="inline-flex items-center gap-2">
                            <input type="checkbox" className="w-4 h-4" checked={colmapRunImageRegistrator} onChange={e => setColmapRunImageRegistrator(e.target.checked)} />
                            <span className="text-sm text-slate-700">Run extra registration + triangulation pass</span>
                          </label>
                        </div>
                        
                      </div>
                    ) : (
                      <div className="space-y-0.5 text-sm">
                        <div>
                          <label className="flex items-center justify-between text-[11px] font-medium text-slate-600 mb-0.5">
                            <span>Shared image set</span>
                            <button onClick={() => setSelectedInfoKey("resize_mode")} className="p-1 text-slate-400 hover:text-slate-600"><Info /></button>
                          </label>
                          <div className="space-y-0.5">
                            <label className={`flex items-center justify-between px-2.5 py-1.5 rounded-md border ${!imagesResizeEnabled ? 'border-blue-500 bg-blue-50 text-blue-700' : 'border-slate-200 bg-slate-50 text-slate-700'}`}>
                              <span className="flex flex-col text-left">
                                <span className="text-sm font-medium">Keep original uploads</span>
                                <span className="text-[11px] text-slate-500 font-normal">COLMAP + training read your untouched files (highest fidelity, more VRAM).</span>
                              </span>
                              <input
                                type="radio"
                                className="w-4 h-4"
                                checked={!imagesResizeEnabled}
                                onChange={() => setImagesResizeEnabled(false)}
                              />
                            </label>
                            <label className={`flex items-center justify-between px-2.5 py-1.5 rounded-md border ${imagesResizeEnabled ? 'border-blue-500 bg-blue-50 text-blue-700' : 'border-slate-200 bg-slate-50 text-slate-700'}`}>
                              <span className="flex flex-col text-left">
                                <span className="text-sm font-medium">Clone + downscale once</span>
                                <span className="text-[11px] text-slate-500 font-normal">Creates a resized working set reused by COLMAP and gsplat.</span>
                              </span>
                              <input
                                type="radio"
                                className="w-4 h-4"
                                checked={imagesResizeEnabled}
                                onChange={() => {
                                  setImagesResizeEnabled(true);
                                  if (!imagesMaxSize) setImagesMaxSize(1600);
                                }}
                              />
                            </label>
                          </div>
                        </div>
                        {imagesResizeEnabled ? (
                          <div className="space-y-0.5">
                            <label className="flex items-center justify-between text-[11px] font-medium text-slate-600 mb-0.5">
                              <span>Max dimension (px)</span>
                              <button onClick={() => setSelectedInfoKey("images_max_size")} className="p-1 text-slate-400 hover:text-slate-600"><Info /></button>
                            </label>
                            <input
                              type="number"
                              value={imagesMaxSize ?? ""}
                              onChange={(e) => setImagesMaxSize(e.target.value ? parseInt(e.target.value) : undefined)}
                              className="w-full px-2 py-1.5 text-xs border border-slate-300 rounded-md"
                              min={256}
                              step={50}
                            />
                            <p className="text-[11px] text-slate-500">Largest width or height applied to the cloned copies. Originals remain untouched on disk.</p>
                          </div>
                        ) : (
                          <p className="text-[11px] text-slate-500">Disabling downscaling means both stages ingest the original uploads. Expect longer COLMAP runtimes and higher VRAM usage.</p>
                        )}
                        <div className="text-[11px] text-slate-500 border border-slate-200 rounded-md bg-slate-50 px-2.5 py-1.5">
                          We always keep two folders: your uploads and (optionally) a resized mirror inside <code>images_resized</code>. COLMAP intrinsics + gsplat training both read from the same folder so rays line up perfectly.
                        </div>
                      </div>
                    )}
                  </div>
                    <div className="col-span-3">
                      <div className="p-2.5 border rounded-md h-full bg-slate-50">
                      <h4 className="font-semibold text-sm">Parameter Info</h4>
                      <div className="mt-1.5 text-xs text-slate-700 leading-relaxed">
                        {selectedInfoKey ? (
                          <div>{(configTab === "colmap"
                            ? (colmapInfo as any)[selectedInfoKey]
                            : configTab === "images"
                              ? (imagesInfo as any)[selectedInfoKey]
                              : (trainingInfo as any)[selectedInfoKey]) ?? "No information available."}</div>
                        ) : (
                          <div>Select a parameter's info icon to view details here.</div>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
              <div className="px-4 py-2.5 border-t border-slate-200 flex justify-end gap-1.5">
                <button
                  onClick={resetConfigToDefaults}
                  className="px-3 py-1.5 rounded-md border border-rose-200 text-rose-600 hover:bg-rose-50 text-[13px] font-semibold"
                >
                  Reset defaults
                </button>
                <button
                  onClick={() => setShowConfig(false)}
                  disabled={isSavingConfig}
                  className="px-3 py-1.5 rounded-md border border-slate-200 text-slate-700 hover:bg-slate-50 text-[13px] font-semibold"
                >
                  Close
                </button>
                <button
                  onClick={handleSaveConfig}
                  disabled={isSavingConfig}
                  className="px-3 py-1.5 rounded-md bg-blue-600 text-white hover:bg-blue-700 text-[13px] font-semibold"
                >
                  {isSavingConfig ? "Saving..." : "Save"}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {showNewSessionModal && (
        <div className="fixed inset-0 z-50">
          <div className="absolute inset-0 bg-black/50" onClick={() => setShowNewSessionModal(false)} />
          <div className="absolute inset-0 flex items-center justify-center p-4">
            <div className="w-[560px] max-w-full bg-white rounded-xl shadow-2xl border border-slate-200 overflow-hidden">
              <div className="flex items-center justify-between px-4 py-3 border-b border-slate-200">
                <div>
                  <p className="text-xs uppercase font-semibold text-slate-500">Session</p>
                  <h3 className="text-base font-bold text-slate-900">Create New Session</h3>
                </div>
                <button className="text-sm text-slate-600" onClick={() => setShowNewSessionModal(false)}>
                  Close
                </button>
              </div>

              <div className="p-4 space-y-4 text-sm">
                <div>
                  <label className="block text-xs font-semibold text-slate-600 mb-1">Session Name</label>
                  <input
                    value={newSessionNameDraft}
                    onChange={(e) => setNewSessionNameDraft(e.target.value)}
                    placeholder={buildDefaultRunName(projectDisplayName, projectId, projectRuns)}
                    className="w-full px-3 py-2 border border-slate-300 rounded-lg"
                  />
                </div>

                <div>
                  <label className="block text-xs font-semibold text-slate-600 mb-1">Config Source</label>
                  <select
                    value={newSessionConfigSource}
                    onChange={(e) => {
                      const nextSource = e.target.value as NewSessionConfigSource;
                      setNewSessionConfigSource(nextSource);
                    }}
                    className="w-full px-3 py-2 border border-slate-300 rounded-lg"
                  >
                    <option value="current">Copy current base config</option>
                    <option value="defaults">Training defaults only</option>
                  </select>
                </div>

                <div className="flex justify-end gap-2 pt-1">
                  <button
                    onClick={() => setShowNewSessionModal(false)}
                    className="px-3 py-2 rounded-lg border border-slate-300 text-slate-700"
                  >
                    Cancel
                  </button>
                  <button
                    onClick={handleCreateSessionDraft}
                    disabled={isCreatingSessionDraft}
                    className="px-3 py-2 rounded-lg bg-blue-600 text-white font-semibold disabled:bg-slate-300"
                  >
                    {isCreatingSessionDraft ? "Preparing..." : "Create Session Draft"}
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {showRenameSessionModal && (
        <div className="fixed inset-0 z-50">
          <div className="absolute inset-0 bg-black/50" onClick={() => !isRenamingRun && setShowRenameSessionModal(false)} />
          <div className="absolute inset-0 flex items-center justify-center p-4">
            <div className="w-[520px] max-w-full bg-white rounded-xl shadow-2xl border border-slate-200 overflow-hidden">
              <div className="flex items-center justify-between px-4 py-3 border-b border-slate-200">
                <div>
                  <p className="text-xs uppercase font-semibold text-slate-500">Session</p>
                  <h3 className="text-base font-bold text-slate-900">Rename Current Session</h3>
                </div>
                <button
                  className="text-sm text-slate-600 disabled:text-slate-300"
                  onClick={() => setShowRenameSessionModal(false)}
                  disabled={isRenamingRun}
                >
                  Close
                </button>
              </div>

              <div className="p-4 space-y-3 text-sm">
                <div>
                  <label className="block text-xs font-semibold text-slate-600 mb-1">Session Name</label>
                  <input
                    value={renameSessionDraft}
                    onChange={(e) => setRenameSessionDraft(e.target.value)}
                    placeholder="Enter session name"
                    className="w-full px-3 py-2 border border-slate-300 rounded-lg"
                    autoFocus
                  />
                </div>

                <div className="flex justify-end gap-2 pt-1">
                  <button
                    onClick={() => {
                      setShowRenameSessionModal(false);
                      setRenameSessionDraft("");
                    }}
                    disabled={isRenamingRun}
                    className="px-3 py-2 rounded-lg border border-slate-300 text-slate-700 disabled:text-slate-300 disabled:border-slate-200"
                  >
                    Cancel
                  </button>
                  <button
                    onClick={confirmRenameCurrentSession}
                    disabled={isRenamingRun}
                    className="px-3 py-2 rounded-lg bg-blue-600 text-white font-semibold disabled:bg-slate-300"
                  >
                    {isRenamingRun ? "Renaming..." : "Rename"}
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {showElevateModelModal && (
        <div className="fixed inset-0 z-50">
          <div className="absolute inset-0 bg-black/50" onClick={() => !isElevatingModel && setShowElevateModelModal(false)} />
          <div className="absolute inset-0 flex items-center justify-center p-4">
            <div className="w-[520px] max-w-full bg-white rounded-xl shadow-2xl border border-slate-200 overflow-hidden">
              <div className="flex items-center justify-between px-4 py-3 border-b border-slate-200">
                <div>
                  <p className="text-xs uppercase font-semibold text-slate-500">Model Registry</p>
                  <h3 className="text-base font-bold text-slate-900">Elevate Session Model</h3>
                </div>
                <button
                  className="text-sm text-slate-600 disabled:text-slate-300"
                  onClick={() => setShowElevateModelModal(false)}
                  disabled={isElevatingModel}
                >
                  Close
                </button>
              </div>

              <div className="p-4 space-y-3 text-sm">
                <div>
                  <label className="block text-xs font-semibold text-slate-600 mb-1">Reusable Model Name</label>
                  <input
                    value={elevateModelNameDraft}
                    onChange={(e) => setElevateModelNameDraft(e.target.value)}
                    placeholder="Enter model name"
                    className="w-full px-3 py-2 border border-slate-300 rounded-lg"
                    autoFocus
                  />
                  <p className="mt-1 text-[11px] text-slate-500">
                    This copies the selected session checkpoint into the global models registry for warm-start reuse.
                  </p>
                </div>

                <div className="flex justify-end gap-2 pt-1">
                  <button
                    onClick={() => {
                      setShowElevateModelModal(false);
                      setElevateModelNameDraft("");
                    }}
                    disabled={isElevatingModel}
                    className="px-3 py-2 rounded-lg border border-slate-300 text-slate-700 disabled:text-slate-300 disabled:border-slate-200"
                  >
                    Cancel
                  </button>
                  <button
                    onClick={confirmElevateModel}
                    disabled={isElevatingModel}
                    className="px-3 py-2 rounded-lg bg-violet-600 text-white font-semibold disabled:bg-slate-300"
                  >
                    {isElevatingModel ? "Elevating..." : "Elevate Model"}
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      <ConfirmModal
        open={showRestartConfirmModal}
        title="Restart Processing"
        message={
          <>
            Are you sure you want to restart processing? Existing generated model files and outputs in this session may be overridden. Create a new session first if you want to keep current outputs.
          </>
        }
        confirmLabel="Restart"
        cancelLabel="Cancel"
        tone="danger"
        busy={processing}
        onCancel={() => {
          if (!processing) setShowRestartConfirmModal(false);
        }}
        onConfirm={() => {
          setShowRestartConfirmModal(false);
          void handleProcess(true);
        }}
      />
    </div>
  );
}

