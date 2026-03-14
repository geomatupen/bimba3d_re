import { useMemo, useState, useEffect, useRef, useCallback } from "react";
import { Play, Settings2, Layers, Map as MapIcon, Boxes, Check, X, Clock, Square, Download, Info as LucideInfo } from "lucide-react";
import Map, { NavigationControl } from "react-map-gl/maplibre";
import maplibregl from "maplibre-gl";
import { api } from "../../api/client";
import ViewerTab from "./ViewerTab";
import SparseViewer from "../SparseViewer.tsx";

// Small Info wrapper: render a smaller info icon throughout the modal
const Info = (props: any) => <LucideInfo className={props.className ? props.className + " w-3 h-3" : "w-3 h-3"} {...props} />;

// Fix for window.__bimba3dTrainingStart type error
declare global {
  interface Window {
    __bimba3dTrainingStart?: number;
  }
}

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
  previews: PreviewFile[];
  snapshots: SnapshotEntry[];
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

type TrainingEngine = "gsplat" | "litegs";

const extractSnapshotStep = (name?: string): number | null => {
  if (!name) return null;
  const match = name.match(/(\d+)(?!.*\d)/);
  return match ? parseInt(match[1], 10) : null;
};

const formatEngineLabel = (name: string) =>
  name
    .replace(/_/g, " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());

const getDefaultProcessConfig = () => ({
  mode: "baseline" as "baseline" | "modified",
  tune_end_step: 200,
  tune_interval: 25,
  tune_scope: "with_strategy" as "core_only" | "with_strategy",
  engine: "gsplat" as TrainingEngine,
  maxSteps: 30000,
  logInterval: 100,
  splatInterval: 150,
  pngInterval: 50,
  evalInterval: 1000,
  saveInterval: 150,
  sparse_preference: "best",
  sparse_merge_selection: [] as string[],
  colmap: {
    max_image_size: 1600,
    peak_threshold: 0.02,
    guided_matching: true,
    matching_type: "sequential",
    mapper_num_threads: 4,
    mapper_min_num_matches: 12,
    mapper_abs_pose_min_num_inliers: 15,
    mapper_init_min_num_inliers: 60,
    sift_matching_min_num_inliers: 12,
    run_image_registrator: true,
  },
  images_max_size: 1600,
  images_resize_enabled: false,
  densifyFromIter: 500,
  densifyUntilIter: 15000,
  densificationInterval: 100,
  densifyGradThreshold: 0.0002,
  opacityThreshold: 0.005,
  lambdaDssim: 0.2,
  litegs_target_primitives: 50000,
  litegs_alpha_shrink: 0.95,
});

export default function ProcessTab({ projectId }: ProcessTabProps) {
  // Load config from localStorage or use defaults
  const loadConfig = () => {
    const defaults = getDefaultProcessConfig();
    try {
      const saved = localStorage.getItem(`processConfig_${projectId}`);
      if (saved) {
        const parsed = JSON.parse(saved);
        if (typeof parsed.images_resize_enabled === "undefined" && typeof parsed.training_image_resize_enabled !== "undefined") {
          parsed.images_resize_enabled = parsed.training_image_resize_enabled;
        }
        if (typeof parsed.images_max_size === "undefined" && typeof parsed.training_image_max_size === "number") {
          parsed.images_max_size = parsed.training_image_max_size;
        }
        return {
          ...defaults,
          ...parsed,
          colmap: { ...defaults.colmap, ...(parsed?.colmap || {}) },
        };
      }
    } catch (e) {
      console.error('Failed to load config:', e);
    }
    // Do not load user's persisted advanced toggle; always start hidden
    return { ...defaults };
  };
  
  const [selectedInfoKey, setSelectedInfoKey] = useState<string | null>(null);

  // Load persisted config and expose individual controls
  const cfg = loadConfig();
  const [mode, setMode] = useState<"baseline" | "modified">(cfg.mode ?? "baseline");
  const [tuneEndStep, setTuneEndStep] = useState<number>(cfg.tune_end_step ?? 200);
  const [tuneInterval, setTuneInterval] = useState<number>(cfg.tune_interval ?? 25);
  const [tuneScope, setTuneScope] = useState<"core_only" | "with_strategy">(cfg.tune_scope ?? "with_strategy");
  const [engine, setEngine] = useState<TrainingEngine>(cfg.engine ?? "gsplat");
  const [maxSteps, setMaxSteps] = useState<number>(cfg.maxSteps ?? 30000);
  const [logInterval, setLogInterval] = useState<number>(cfg.logInterval ?? 100);
  const [splatInterval, setSplatInterval] = useState<number>(cfg.splatInterval ?? 150);
  const [pngInterval, setPngInterval] = useState<number>(cfg.pngInterval ?? 50);
  const [evalInterval, setEvalInterval] = useState<number>(cfg.evalInterval ?? 1000);
  const [saveInterval, setSaveInterval] = useState<number>(cfg.saveInterval ?? 150);
  const [imagesMaxSize, setImagesMaxSize] = useState<number | undefined>(cfg.images_max_size ?? 1600);
  const [imagesResizeEnabled, setImagesResizeEnabled] = useState<boolean>(cfg.images_resize_enabled ?? false);
  const [, setShowAdvancedTraining] = useState<boolean>(cfg.showAdvancedTraining ?? false);

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
  const [densifyUntilIter, setDensifyUntilIter] = useState<number>(cfg.densifyUntilIter ?? 15000);
  const [densificationInterval, setDensificationInterval] = useState<number>(cfg.densificationInterval ?? 100);
  const [densifyGradThreshold, setDensifyGradThreshold] = useState<number>(cfg.densifyGradThreshold ?? 0.0002);
  const [opacityThreshold, setOpacityThreshold] = useState<number>(cfg.opacityThreshold ?? 0.005);
  const [lambdaDssim, setLambdaDssim] = useState<number>(cfg.lambdaDssim ?? 0.2);

  const densifySchedule = useMemo(() => {
    const normalizedStart = Math.max(0, densifyFromIter);
    const interval = Math.max(1, Math.abs(densificationInterval) || 1);
    const firstStep = normalizedStart <= 0 ? interval : normalizedStart;
    const previewSteps: number[] = [];
    for (let i = 0; i < 4; i += 1) {
      previewSteps.push(firstStep + i * interval);
    }
    return { interval, firstStep, previewSteps };
  }, [densifyFromIter, densificationInterval]);
  const firstDensifyStep = densifySchedule.firstStep;
  const upcomingDensifySteps = densifySchedule.previewSteps;
  const densifyStopRespected = useMemo(() => {
    if (densifyUntilIter <= 0) return true;
    return firstDensifyStep <= densifyUntilIter;
  }, [densifyUntilIter, firstDensifyStep]);
  const densifyScheduleBlocked = densificationInterval <= 0 || !densifyStopRespected;
  const densifyBlockedReason = useMemo(() => {
    if (densificationInterval <= 0) {
      return "Set a positive densification interval so gsplat can schedule refinements.";
    }
    if (!densifyStopRespected) {
      if (densifyFromIter > densifyUntilIter) {
        return `Start step (${densifyFromIter.toLocaleString()}) must be at or before the stop step (${densifyUntilIter.toLocaleString()}).`;
      }
      return `First densify pass would run at step ${firstDensifyStep.toLocaleString()}, which is after the stop step (${densifyUntilIter.toLocaleString()}).`;
    }
    return null;
  }, [densificationInterval, densifyFromIter, densifyStopRespected, densifyUntilIter, firstDensifyStep]);

  const [colmapMaxImageSize, setColmapMaxImageSize] = useState<number | undefined>(cfg.colmap?.max_image_size ?? 1600);
  const [colmapPeakThreshold, setColmapPeakThreshold] = useState<number | undefined>(cfg.colmap?.peak_threshold ?? undefined);
  const [colmapGuidedMatching, setColmapGuidedMatching] = useState<boolean>(cfg.colmap?.guided_matching ?? true);
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
    mode: 'Training profile. Baseline keeps default behavior; Modified applies deterministic tuning profile during training.',
    tune_end_step: 'For Modified mode, this is the last step where rule-based tuning updates are allowed. The worker keeps applying rule checks until this step, then continues normal training.',
    tune_interval: 'For Modified mode, worker evaluates and applies rule-based updates every N steps during the tuning window.',
    tune_scope: 'Rule tuning scope: Core only updates listed LR/threshold knobs; Core + strategy also updates additional strategy cadence/pruning controls.',
    // --- ORIGINAL KERBL PARAMETERS ---
    maxSteps: 'Total training iterations. This value is sent from frontend in both baseline and modified modes. [original]',
    logInterval: 'How often (in steps) to print consolidated training snapshots in worker logs. Lower values are more verbose. [custom]',
    splatInterval: 'How often (in steps) to export intermediate .splat/.ply files during training. [original]',
    pngInterval: 'Deprecated for gsplat: previews are generated on eval steps. Use eval interval to control preview cadence.',
    evalInterval: 'How often to run eval passes + metrics collection. This value is configurable from frontend in both modes. [original]',
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
  const [basemap, setBasemap] = useState<"satellite" | "osm">("satellite");
  const [showImagesLayer, setShowImagesLayer] = useState(true);
  const [_showSparseLayer, setShowSparseLayer] = useState(true);
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
  const [pipelineDone, setPipelineDone] = useState(false);

  const applyTrainingDefaults = (defaults: ReturnType<typeof getDefaultProcessConfig>) => {
    setMode(defaults.mode ?? "baseline");
    setTuneEndStep(defaults.tune_end_step ?? 200);
    setTuneInterval(defaults.tune_interval ?? 25);
    setTuneScope(defaults.tune_scope ?? "with_strategy");
    setEngine(defaults.engine ?? "gsplat");
    setMaxSteps(defaults.maxSteps);
    setLogInterval(defaults.logInterval ?? 100);
    setSplatInterval(defaults.splatInterval);
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

  const resetConfigToDefaults = () => {
    const defaults = getDefaultProcessConfig();
    if (configTab === "training") {
      applyTrainingDefaults(defaults);
    } else if (configTab === "colmap") {
      applyColmapDefaults(defaults);
    } else {
      applyImageDefaults(defaults);
    }
    localStorage.removeItem(`processConfig_${projectId}`);
  };

  // Auto-switch to viewer when 3D model layer is enabled
  

  // Save config to localStorage whenever it changes
  useEffect(() => {
    const config = {
      mode,
      tune_end_step: tuneEndStep,
      tune_interval: tuneInterval,
      tune_scope: tuneScope,
      engine,
      maxSteps,
      logInterval,
      splatInterval,
      pngInterval,
      evalInterval,
      saveInterval,
      sparse_preference: sparsePreference,
      sparse_merge_selection: sparseMergeSelection,
      images_resize_enabled: imagesResizeEnabled,
      images_max_size: imagesMaxSize,
      densifyFromIter,
      densifyUntilIter,
      densificationInterval,
      densifyGradThreshold,
      opacityThreshold,
      lambdaDssim,
      litegs_target_primitives: litegsTargetPrimitives,
      litegs_alpha_shrink: litegsAlphaShrink,
      colmap: {
        max_image_size: colmapMaxImageSize,
        peak_threshold: colmapPeakThreshold,
        guided_matching: colmapGuidedMatching,
        matching_type: colmapMatchingType,
        mapper_num_threads: colmapMapperThreads,
        mapper_min_num_matches: colmapMapperMinNumMatches,
        mapper_abs_pose_min_num_inliers: colmapMapperAbsPoseMinNumInliers,
        mapper_init_min_num_inliers: colmapMapperInitMinNumInliers,
        sift_matching_min_num_inliers: colmapSiftMatchingMinNumInliers,
        run_image_registrator: colmapRunImageRegistrator,
      }
    };
    localStorage.setItem(`processConfig_${projectId}`, JSON.stringify(config));
  }, [mode, tuneEndStep, tuneInterval, tuneScope, engine, maxSteps, logInterval, splatInterval, pngInterval, evalInterval, saveInterval, sparsePreference, sparseMergeSelection, imagesResizeEnabled, imagesMaxSize, densifyFromIter, densifyUntilIter, densificationInterval, densifyGradThreshold, opacityThreshold, lambdaDssim, projectId, colmapMaxImageSize, colmapPeakThreshold, colmapGuidedMatching, colmapMatchingType, colmapMapperThreads, colmapMapperMinNumMatches, colmapMapperAbsPoseMinNumInliers, colmapMapperInitMinNumInliers, colmapSiftMatchingMinNumInliers, colmapRunImageRegistrator, litegsTargetPrimitives, litegsAlphaShrink]);

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
    setSelectedModelSnapshot(null);
    setModelSnapshots([]);
  }, [projectId]);

  useEffect(() => {
    const checkOutputs = async () => {
      try {
        const [filesRes, statusRes] = await Promise.all([
          api.get(`/projects/${projectId}/files`),
          api.get(`/projects/${projectId}/status`).catch(() => ({ data: { stage: "idle", message: null, status: "pending" } }))
        ]);
        
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
          nextEngineMap[engineName] = {
            name: engineName,
            label: formatEngineLabel(engineName),
            hasModel: Boolean(bundle?.splats),
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
        
        // Store current step and max steps for progress bar
        setTrainingCurrentStep(status.currentStep);
        setTrainingMaxSteps(status.maxSteps);
        setStageProgress(status.stage_progress);

        // Use stopped_percentage when stopped, else 'percentage' or 'progress'
        let percent = undefined as number | undefined;
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
        const pctNum = typeof percent === 'number' ? percent : 0;
        setOverallProgress(Math.max(0, Math.min(100, isNaN(pctNum) ? 0 : pctNum)));

        // Current stage label
        let stageName = "" as string;
        let stageKey: "docker"|"colmap"|"training"|"export"|"" = "";
        // If stopped, prefer worker-provided stopped_stage for accurate location
        const effectiveStage = (status.status === 'stopped' && status.stopped_stage) ? status.stopped_stage : status.stage;
        // workerStoppedStage: the stage where the worker actually stopped (null if not stopped)
        const workerStoppedStage = status.status === 'stopped' ? (status.stopped_stage || status.stage) : null;
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
        const resumable = status.can_resume || sparse;
        setCanResume(resumable && status.status !== "processing" && status.status !== "stopping");
        // Track which stage the worker stopped at (prefer stopped_stage)
        setStoppedStage(workerStoppedStage);
        
        // Handle stopping state
        if (status.status === "stopping") {
          setIsStopping(true);
          setStoppingMessage(status.message || "Will stop after current step completes...");
        } else {
          setIsStopping(false);
          setStoppingMessage(null);
        }
        
        // Update processing flag based on status
        if (status.status === "processing" || status.status === "stopping") {
          setProcessing(true);
        } else {
          setProcessing(false);
        }
        
        // Build detailed status message
        let statusMsg: React.ReactNode = null;
        if (status.status === "failed") {
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
        
        // Detect if pipeline was stopped (status is 'stopped' or message contains 'stopped by user')
        const stopped = status.status === 'stopped' || (status.message && status.message.toLowerCase().includes('stopped by user'));
        setWasStopped(stopped);

        // Determine stage status based on actual pipeline state.
        // Mark previously completed stages as 'success' even when stopped,
        // but keep `pipelineDone` false when the pipeline was stopped.
        const newStatus = { colmap: "pending", training: "pending", export: "pending" };

        // COLMAP: consider it complete if sparse outputs exist, or if COLMAP reached 100% stage_progress,
        // or if the worker stopped after COLMAP (stoppedStage later than colmap), or overall status indicates completed.
        // Consider COLMAP complete only when there's an explicit sparse reconstruction present
        // OR the pipeline reports full completion for that stage (stage_progress >= 100 and overall status is completed).
        let colmapComplete = Boolean(sparse) || ((status.stage === 'colmap' || status.stage === 'colmap_only') && typeof status.stage_progress === 'number' && status.stage_progress >= 100 && status.status === 'completed') || workerStoppedStage === 'training' || workerStoppedStage === 'export' || (status.status === 'completed' && (status.stage === 'colmap' || status.stage === 'training' || status.stage === 'export'));
        // If the worker stopped at COLMAP but the COLMAP substep actually did NOT finish, do not treat as complete.
        if (workerStoppedStage === 'colmap' && !(Boolean(sparse) || ((status.stage === 'colmap' || status.stage === 'colmap_only') && typeof status.stage_progress === 'number' && status.stage_progress >= 100 && status.status === 'completed'))) {
          colmapComplete = false;
        }
        if (colmapComplete) {
          newStatus.colmap = 'success';
        } else if (status.stage === 'colmap' || status.stage === 'colmap_only') {
          // Show running when actively in COLMAP
          newStatus.colmap = (status.status === 'processing' || status.status === 'stopping') ? 'running' : 'pending';
        }

        // TRAINING: consider it complete if model outputs exist, or training reached 100% stage_progress,
        // or the worker stopped after training (stoppedStage === export) or overall completed.
        let trainingComplete = Boolean(model) || (status.stage === 'training' && typeof status.stage_progress === 'number' && status.stage_progress >= 100) || workerStoppedStage === 'export' || (status.status === 'completed' && (status.stage === 'training' || status.stage === 'export'));
        // If the worker stopped at training but training hadn't finished, do not mark as complete
        if (workerStoppedStage === 'training' && !(Boolean(model) || (status.stage === 'training' && typeof status.stage_progress === 'number' && status.stage_progress >= 100))) {
          trainingComplete = false;
        }
        if (trainingComplete) {
          newStatus.training = 'success';
        } else if (status.stage === 'training') {
          newStatus.training = (status.status === 'processing' || status.status === 'stopping') ? 'running' : 'pending';
        }

        // EXPORT: consider it complete if model outputs exist or overall status is completed
        let exportComplete = Boolean(model) || (status.status === 'completed' && status.stage === 'export');
        // If worker stopped during export and export did not finish, do not mark as complete
        if (workerStoppedStage === 'export' && !(Boolean(model) || (status.status === 'completed' && status.stage === 'export'))) {
          exportComplete = false;
        }
        if (exportComplete) {
          newStatus.export = 'success';
        } else if (status.stage === 'export' && status.status === 'processing') {
          newStatus.export = 'running';
        }

        setStageStatus(newStatus);

        // --- Show Completed and hide Stage Status if all are success and not stopped ---
        const allStagesSuccess = newStatus.colmap === "success" && newStatus.training === "success" && newStatus.export === "success" && !stopped;
        setPipelineDone(allStagesSuccess);

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
  }, [projectId, show3DModel]);

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
    if (!bytes || bytes <= 0) return "—";
    const units = ["B", "KB", "MB", "GB"];
    const order = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
    const value = bytes / Math.pow(1024, order);
    return `${order === 0 ? value.toFixed(0) : value.toFixed(1)} ${units[order]}`;
  };

  const [_mapViewState, setMapViewState] = useState<any | null>(null);
  const [hasAutoFitted, setHasAutoFitted] = useState(false);
  const mapRef = useRef<any | null>(null);
  const selectedEngineRef = useRef<string | null>(null);
  const isMountedRef = useRef(false);
  const prevProcessingRef = useRef<boolean>(false);
  const prevSparsePresenceRef = useRef<boolean>(false);

  useEffect(() => {
    isMountedRef.current = true;
    return () => {
      isMountedRef.current = false;
    };
  }, []);

  useEffect(() => {
    selectedEngineRef.current = selectedEngineName;
  }, [selectedEngineName]);

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
      const bestLabel = bestTarget ? `Auto (best → ${bestTarget})` : "Auto (best available)";

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
        attribution: "© Esri World Imagery",
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
            attribution: '© OpenStreetMap contributors'
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
        } catch (e) {
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
        const base = (window as any).__API_BASE__ || (api && api.defaults && api.defaults.baseURL) || 'http://localhost:8005';
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
            } catch (err) {
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
          } catch (e) {
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
      } catch (e) {
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
      try { if (map.getLayer && map.getLayer(layerId)) map.setLayoutProperty(layerId, 'visibility', 'none'); } catch (e) {}
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
        } catch (err) {
          if (retries > 0) setTimeout(() => ensureVisibility(retries - 1), 150);
        }
      };
      ensureVisibility();
    } catch (err) {
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
        } catch (e) {
          // ignore
        }
      } else if (showImagesLayer) {
        // Add layer if missing
        try {
          if (!map.getSource(srcId)) map.addSource(srcId, { type: 'geojson', data: buildGeo() });
          map.addLayer({ id: layerId, type: 'circle', source: srcId, paint: { 'circle-radius': 6, 'circle-color': '#10b981', 'circle-stroke-color': '#ffffff', 'circle-stroke-width': 2, 'circle-opacity': 0.95 } });
        } catch (e) {
          // ignore
        }
      }
    } catch (err) {
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
    setViewerOutput('model');
    setTopView('viewer');
  };

  const handleProcess = async () => {
    setProcessing(true);
    setError(null);

    if (sparsePreference === "merge_selected" && sparseMergeSelection.length < 2) {
      setError("Select at least two sparse folders when using merge mode.");
      setProcessing(false);
      return;
    }

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
      await api.post(`/projects/${projectId}/process`, {
        mode: effectiveMode,
        tune_end_step: effectiveMode === "modified" ? tuneEndStep : undefined,
        tune_interval: effectiveMode === "modified" ? tuneInterval : undefined,
        tune_scope: effectiveMode === "modified" ? tuneScope : undefined,
        stage,
        engine,
        max_steps: maxSteps,
        log_interval: logInterval,
        splat_export_interval: splatInterval,
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
      setError(err instanceof Error ? err.message : "Failed to start processing");
      setProcessing(false);
    }
  };

  const handleResumeProcess = async () => {
    setProcessing(true);
    setError(null);
    if (sparsePreference === "merge_selected" && sparseMergeSelection.length < 2) {
      setError("Select at least two sparse folders when using merge mode.");
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
      await api.post(`/projects/${projectId}/process`, {
        mode: effectiveMode,
        tune_end_step: effectiveMode === "modified" ? tuneEndStep : undefined,
        tune_interval: effectiveMode === "modified" ? tuneInterval : undefined,
        tune_scope: effectiveMode === "modified" ? tuneScope : undefined,
        stage: resumeStage,
        engine,
        max_steps: maxSteps,
        log_interval: logInterval,
        splat_export_interval: splatInterval,
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
      setError(err instanceof Error ? err.message : "Failed to resume processing");
      setProcessing(false);
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
      // Estimate time per step using previous steps
      const elapsed = window.__bimba3dTrainingStart ? (Date.now() - window.__bimba3dTrainingStart) / 1000 : null;
      if (!window.__bimba3dTrainingStart && processing) {
        window.__bimba3dTrainingStart = Date.now();
        return null;
      }
      if (elapsed && trainingCurrentStep > 0) {
        const avgStepTime = elapsed / trainingCurrentStep;
        const stepsLeft = trainingMaxSteps - trainingCurrentStep;
        const secondsLeft = avgStepTime * stepsLeft;
        if (secondsLeft > 0) {
          const h = Math.floor(secondsLeft / 3600);
          const m = Math.floor((secondsLeft % 3600) / 60);
          const s = Math.floor(secondsLeft % 60);
          return `${h > 0 ? h + 'h ' : ''}${m > 0 ? m + 'm ' : ''}${s}s remaining`;
        }
      }
    }
    return null;
  })();

  const engineOptions = Object.values(engineOutputMap).filter((bundle) => bundle.hasModel);
  const hasEngineOutputs = Object.keys(engineOutputMap).length > 0;
  const showEngineDropdown = engineOptions.length > 1;
  const activeEngineBundle = selectedEngineName ? engineOutputMap[selectedEngineName] : null;
  const finalModelAvailable = activeEngineBundle ? activeEngineBundle.hasModel : has3DModel;
  const viewerModelAvailable = finalModelAvailable || modelSnapshots.length > 0 || Boolean(selectedModelSnapshot);
  const selectedEngineLabel = activeEngineBundle?.label;
  const selectedPngEntry = selectedPng ? pngFiles.find((file) => file.url === selectedPng) : undefined;
  const showFinalModelSection = hasEngineOutputs || has3DModel || pngFiles.length > 0 || modelSnapshots.length > 0;
  const engineSelectValue = selectedEngineName ?? engineOptions[0]?.name ?? "";

  return (
    <div className="max-w-7xl space-y-4">
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
              <button
                onClick={() => setShowConfig(true)}
                className="inline-flex items-center gap-1 px-3 py-1.5 rounded-lg border border-slate-200 text-xs font-semibold text-slate-700 hover:bg-slate-50"
              >
                <Settings2 className="w-4 h-4" />
                Config
              </button>
            </div>

            <div className="space-y-2">
              {/* --- STAGE LABELS --- */}
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
                    disabled={processing || isStopping || (!runColmap && stageStatus.colmap !== "success")}
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
                    disabled={processing || isStopping || (!runTraining && stageStatus.training !== "success" && runExport) || (!runColmap && stageStatus.colmap !== "success" && !runTraining)}
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
                      onClick={handleProcess}
                      disabled={processing || densifyScheduleBlocked || (!runColmap && stageStatus.colmap !== "success") || (!runTraining && stageStatus.training !== "success" && runExport) || (!runColmap && !runTraining && !runExport)}
                      className="w-full inline-flex items-center justify-center gap-2 px-4 py-3 rounded-lg bg-green-600 hover:bg-green-700 text-white font-semibold shadow-sm disabled:bg-slate-300 disabled:cursor-not-allowed"
                    >
                      <Play className="w-4 h-4" />
                      {(pipelineDone || wasStopped) ? "Restart Processing" : "Start Processing"}
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
                          Resume Processing
                        </button>
                      ) : null
                    ) : null}
                    {/* --- END RESUME BUTTON LOGIC --- */}
                  </>
                )}
              </div>
              
              {(processing || processingStatus) && (
                <div className="mt-3 space-y-3">
                  {/* Overall Pipeline Progress */}
                  {/* Always show overall progress bar with all selected steps */}
                  <div className="px-3 py-2 bg-indigo-50 border border-indigo-200 rounded-lg">
                    <div className="flex items-center justify-between text-xs text-indigo-700 mb-1">
                      <span className="font-semibold">Overall Progress</span>
                      <span className="font-bold">{wasStopped ? `${overallProgress}% (stopped)` : `${overallProgress}%`}</span>
                    </div>
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
                        <p className="text-xs text-orange-600 mt-1 font-semibold">Stopped at: {currentStage || (stageStatus && (Object.entries(stageStatus).find(([_, v]) => v === 'running') || [null])[0]) || 'Unknown'}</p>
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
                                {(snapshot.format || 'splat').toUpperCase()}{snapshot.size ? ` · ${formatBytes(snapshot.size)}` : ''}
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
                  <ViewerTab projectId={projectId} snapshotUrl={selectedModelSnapshot} engineOverride={selectedEngineName} />
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

      {showConfig && (
        <div className="fixed inset-0 z-50">
          <div className="absolute inset-0 bg-black/50" onClick={() => setShowConfig(false)} />
          <div className="absolute inset-0 flex items-center justify-center p-4">
            <div className="w-[820px] max-w-full bg-white rounded-xl shadow-2xl border border-slate-200 overflow-hidden">
              <div className="flex items-center justify-between px-4 py-3 border-b border-slate-200">
                <div>
                  <p className="text-xs uppercase font-semibold text-slate-500">Advanced</p>
                  <h3 className="text-base font-bold text-slate-900">Processing Configuration</h3>
                </div>
                <button className="text-sm text-slate-600" onClick={() => setShowConfig(false)}>
                  Close
                </button>
              </div>
              <div className="px-4 py-4 text-sm overflow-auto max-h-[70vh]">
                <div className="grid grid-cols-12 gap-0">
                  <div className="col-span-2 flex flex-col gap-1">
                    <button
                      onClick={() => setConfigTab("images")}
                      className={`text-left px-3 py-2 rounded-lg flex items-center justify-between ${configTab === "images" ? "bg-blue-600 text-white font-semibold shadow" : "bg-slate-50 text-slate-700 hover:bg-slate-100"}`}
                    >
                      <span>Images</span>
                      {configTab === "images" && <div className="w-2 h-6 bg-white/20 rounded ml-2" />}
                    </button>
                    <button
                      onClick={() => setConfigTab("colmap")}
                      className={`text-left px-3 py-2 rounded-lg flex items-center justify-between ${configTab === "colmap" ? "bg-blue-600 text-white font-semibold shadow" : "bg-slate-50 text-slate-700 hover:bg-slate-100"}`}
                    >
                      <span>COLMAP</span>
                      {configTab === "colmap" && <div className="w-2 h-6 bg-white/20 rounded ml-2" />}
                    </button>
                    <button
                      onClick={() => setConfigTab("training")}
                      className={`text-left px-3 py-2 rounded-lg flex items-center justify-between ${configTab === "training" ? "bg-blue-600 text-white font-semibold shadow" : "bg-slate-50 text-slate-700 hover:bg-slate-100"}`}
                    >
                      <span>Training</span>
                      {configTab === "training" && <div className="w-2 h-6 bg-white/20 rounded ml-2" />}
                    </button>
                  </div>
                  <div className="col-span-7">
                    {configTab === "training" ? (
                      <div className="space-y-6">
                        <div className="rounded-xl border border-slate-200 bg-white shadow-sm">
                          <div className="flex items-center justify-between px-4 py-3 border-b border-slate-100">
                            <div>
                              <p className="text-sm font-semibold text-slate-800">Shared controls</p>
                              <p className="text-xs text-slate-500">Only knobs that both backends actually consume.</p>
                            </div>
                          </div>
                          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 px-4 py-4">
                            <div className="md:col-span-2">
                              <label className="flex items-center justify-between text-xs font-semibold text-slate-600 mb-1">
                                <span>Training Engine</span>
                                <button onClick={() => setSelectedInfoKey("engine")} className="p-1 text-slate-400 hover:text-slate-600"><Info /></button>
                              </label>
                              <select
                                value={engine}
                                onChange={(e) => setEngine(e.target.value as TrainingEngine)}
                                className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                              >
                                <option value="gsplat">gsplat (default pipeline)</option>
                                <option value="litegs">LiteGS (compact renderer)</option>
                              </select>
                              <p className="text-[11px] text-slate-500 mt-1">Switch engines to reveal the matching parameter card below.</p>
                            </div>
                            {engine === "gsplat" && (
                              <div className="md:col-span-2">
                                <label className="flex items-center justify-between text-xs font-semibold text-slate-600 mb-1">
                                  <span>Training Profile</span>
                                  <button onClick={() => setSelectedInfoKey("mode")} className="p-1 text-slate-400 hover:text-slate-600"><Info /></button>
                                </label>
                                <select
                                  value={mode}
                                  onChange={(e) => setMode((e.target.value as "baseline" | "modified") || "baseline")}
                                  className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                                  disabled={processing || isStopping}
                                >
                                  <option value="baseline">Baseline</option>
                                  <option value="modified">Modified</option>
                                </select>
                                <p className="text-[11px] text-slate-500 mt-1">Modified mode runs rule-based tuning updates through the configured end step, then continues normally.</p>
                              </div>
                            )}
                            {engine === "gsplat" && mode === "modified" && (
                              <div>
                                <label className="flex items-center justify-between text-xs font-semibold text-slate-600 mb-1">
                                  <span>Modified tuning end step</span>
                                  <button onClick={() => setSelectedInfoKey("tune_end_step")} className="p-1 text-slate-400 hover:text-slate-600"><Info /></button>
                                </label>
                                <input
                                  type="number"
                                  min={1}
                                  step={1}
                                  value={tuneEndStep}
                                  onChange={(e) => setTuneEndStep(Math.max(1, parseInt(e.target.value) || 200))}
                                  className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                                />
                                <p className="text-[11px] text-slate-500 mt-1">Rule-based updates run until this step (e.g., 300 or 500), then stop.</p>
                              </div>
                            )}
                            {engine === "gsplat" && mode === "modified" && (
                              <div>
                                <label className="flex items-center justify-between text-xs font-semibold text-slate-600 mb-1">
                                  <span>Modified tuning interval</span>
                                  <button onClick={() => setSelectedInfoKey("tune_interval")} className="p-1 text-slate-400 hover:text-slate-600"><Info /></button>
                                </label>
                                <input
                                  type="number"
                                  min={1}
                                  step={1}
                                  value={tuneInterval}
                                  onChange={(e) => setTuneInterval(Math.max(1, parseInt(e.target.value) || 25))}
                                  className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                                />
                                <p className="text-[11px] text-slate-500 mt-1">Rule-based checks are evaluated every N steps during the tuning window.</p>
                              </div>
                            )}
                            {engine === "gsplat" && mode === "modified" && (
                              <div>
                                <label className="flex items-center justify-between text-xs font-semibold text-slate-600 mb-1">
                                  <span>Rule tuning scope</span>
                                  <button onClick={() => setSelectedInfoKey("tune_scope")} className="p-1 text-slate-400 hover:text-slate-600"><Info /></button>
                                </label>
                                <select
                                  value={tuneScope}
                                  onChange={(e) => setTuneScope((e.target.value as "core_only" | "with_strategy") || "with_strategy")}
                                  className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                                >
                                  <option value="core_only">Core only</option>
                                  <option value="with_strategy">Core + strategy</option>
                                </select>
                                <p className="text-[11px] text-slate-500 mt-1">Core only tunes listed LR/threshold knobs. Core + strategy also tunes extra strategy controls.</p>
                              </div>
                            )}
                            <div className="md:col-span-2">
                              <label className="flex items-center justify-between text-xs font-semibold text-slate-600 mb-1">
                                <span>Sparse reconstruction preference</span>
                                <button onClick={() => setSelectedInfoKey("sparse_preference")} className="p-1 text-slate-400 hover:text-slate-600"><Info /></button>
                              </label>
                              <select
                                value={sparsePreference}
                                onChange={(e) => setSparsePreference(e.target.value)}
                                className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                                disabled={sparseOptionsLoading}
                              >
                                {sparseOptions.map((opt) => (
                                  <option key={opt.value} value={opt.value}>{opt.label}</option>
                                ))}
                              </select>
                              <p className="text-[11px] text-slate-500 mt-1">
                                {sparseOptionsLoading ? "Scanning for COLMAP runs..." : "Override the default whenever multiple reconstructions exist."}
                              </p>
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
                              <label className="flex items-center justify-between text-xs font-semibold text-slate-600 mb-1">
                                <span>Max steps</span>
                                <button onClick={() => setSelectedInfoKey("maxSteps")} className="p-1 text-slate-400 hover:text-slate-600"><Info /></button>
                              </label>
                              <input
                                type="number"
                                value={maxSteps}
                                onChange={(e) => setMaxSteps(parseInt(e.target.value) || 30000)}
                                className="w-full px-3 py-2 border border-slate-300 rounded-lg"
                                min={100}
                                max={50000}
                                step={100}
                              />
                            </div>
                            <div className="md:col-span-2 grid grid-cols-1 sm:grid-cols-2 gap-4">
                              <div>
                                <label className="flex items-center justify-between text-xs font-semibold text-slate-600 mb-1">
                                  <span>Preview cadence</span>
                                  <button onClick={() => setSelectedInfoKey("pngInterval")} className="p-1 text-slate-400 hover:text-slate-600"><Info /></button>
                                </label>
                                <div className="w-full px-3 py-2 border border-slate-200 bg-slate-50 rounded-lg text-xs text-slate-600">
                                  Generated automatically on each eval step.
                                </div>
                              </div>
                              <div>
                                <label className="flex items-center justify-between text-xs font-semibold text-slate-600 mb-1">
                                  <span>Splat export interval</span>
                                  <button onClick={() => setSelectedInfoKey("splatInterval")} className="p-1 text-slate-400 hover:text-slate-600"><Info /></button>
                                </label>
                                <input
                                  type="number"
                                  value={splatInterval}
                                  onChange={(e) => setSplatInterval(parseInt(e.target.value) || 150)}
                                  className="w-full px-3 py-2 border border-slate-300 rounded-lg"
                                  min={50}
                                  step={50}
                                />
                              </div>
                              {engine === "gsplat" && (
                              <>
                                <div>
                                <label className="flex items-center justify-between text-xs font-semibold text-slate-600 mb-1">
                                  <span>Log interval</span>
                                  <button onClick={() => setSelectedInfoKey("logInterval")} className="p-1 text-slate-400 hover:text-slate-600"><Info /></button>
                                </label>
                                <input
                                  type="number"
                                  value={logInterval}
                                  onChange={(e) => setLogInterval(parseInt(e.target.value) || 100)}
                                  className="w-full px-3 py-2 border border-slate-300 rounded-lg"
                                  min={10}
                                  step={10}
                                />
                                </div>
                                <div>
                                <label className="flex items-center justify-between text-xs font-semibold text-slate-600 mb-1">
                                  <span>Eval interval</span>
                                  <button onClick={() => setSelectedInfoKey("evalInterval")} className="p-1 text-slate-400 hover:text-slate-600"><Info /></button>
                                </label>
                                <input
                                  type="number"
                                  value={evalInterval}
                                  onChange={(e) => setEvalInterval(parseInt(e.target.value) || 1000)}
                                  className="w-full px-3 py-2 border border-slate-300 rounded-lg"
                                  min={50}
                                  step={50}
                                />
                                </div>
                                <div>
                                <label className="flex items-center justify-between text-xs font-semibold text-slate-600 mb-1">
                                  <span>Checkpoint interval</span>
                                  <button onClick={() => setSelectedInfoKey("saveInterval")} className="p-1 text-slate-400 hover:text-slate-600"><Info /></button>
                                </label>
                                <input
                                  type="number"
                                  value={saveInterval}
                                  onChange={(e) => setSaveInterval(parseInt(e.target.value) || 150)}
                                  className="w-full px-3 py-2 border border-slate-300 rounded-lg"
                                  min={50}
                                  step={50}
                                />
                                </div>
                              </>
                              )}
                            </div>
                          </div>
                        </div>

                        {engine === "gsplat" && (
                          <div className="rounded-xl border border-blue-200 bg-white shadow-sm">
                            <div className="flex items-center justify-between px-4 py-3 border-b border-slate-100">
                              <div>
                                <p className="text-sm font-semibold text-slate-800">gsplat-only controls</p>
                                <p className="text-xs text-slate-500">Only applied when gsplat is selected.</p>
                              </div>
                              <span className="text-xs px-3 py-1 rounded-full bg-blue-100 text-blue-700">gsplat selected</span>
                            </div>
                            <div className="space-y-5 px-4 py-4">
                                <div className="space-y-5 border-t border-slate-100 pt-4">
                                  <div>
                                    <p className="text-xs font-semibold text-slate-600 mb-2">Densification schedule</p>
                                    <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                                      <div>
                                        <label className="flex items-center justify-between text-[11px] font-semibold text-slate-500 mb-1">
                                          <span>Start step</span>
                                          <button onClick={() => setSelectedInfoKey("densify_from_iter")} className="p-1 text-slate-400 hover:text-slate-600"><Info /></button>
                                        </label>
                                        <input
                                          type="number"
                                          value={densifyFromIter}
                                          onChange={(e) => setDensifyFromIter(parseInt(e.target.value) || 0)}
                                          className="w-full px-3 py-2 border border-slate-300 rounded-lg"
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
                                          className="w-full px-3 py-2 border border-slate-300 rounded-lg"
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
                                          className="w-full px-3 py-2 border border-slate-300 rounded-lg"
                                          min={10}
                                          step={10}
                                        />
                                      </div>
                                    </div>
                                    <p className="text-[11px] text-slate-500 mt-1">Controls how aggressively new Gaussians are created throughout training.</p>
                                    <div className="mt-2 rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-[11px] text-slate-600 space-y-1">
                                      <p>
                                        gsplat runs densification whenever the iteration is at least the start step and $(iteration - start)$ is divisible by the interval.
                                        If the start is 0 or lower, the trainer waits until the first interval elapses to fire.
                                      </p>
                                      <p>
                                        With the current values the first pass would be at <span className="font-semibold text-slate-800">step {firstDensifyStep.toLocaleString()}</span>.
                                      </p>
                                      {upcomingDensifySteps.length > 0 && (
                                        <div>
                                          <p className="font-semibold text-slate-700 mt-1">Upcoming passes</p>
                                          <div className="flex flex-wrap gap-1 mt-1">
                                            {upcomingDensifySteps.map((step, idx) => (
                                              <span key={`densifyStep-${idx}`} className="px-2 py-1 rounded-full border border-slate-300 bg-white text-[11px] text-slate-700">
                                                {step.toLocaleString()}
                                              </span>
                                            ))}
                                          </div>
                                        </div>
                                      )}
                                      {!densifyStopRespected && (
                                        <p className="text-amber-600">This first pass is beyond the stop step of {densifyUntilIter.toLocaleString()}. Increase the stop step or lower the start/interval.</p>
                                      )}
                                    </div>
                                  </div>
                                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                                    <div>
                                      <label className="flex items-center justify-between text-xs font-semibold text-slate-600 mb-1">
                                        <span>Densify grad threshold</span>
                                        <button onClick={() => setSelectedInfoKey("densify_grad_threshold")} className="p-1 text-slate-400 hover:text-slate-600"><Info /></button>
                                      </label>
                                      <input
                                        type="number"
                                        step="0.00005"
                                        value={densifyGradThreshold}
                                        onChange={(e) => setDensifyGradThreshold(parseFloat(e.target.value) || 0.0002)}
                                        className="w-full px-3 py-2 border border-slate-300 rounded-lg"
                                        min={0.00005}
                                      />
                                    </div>
                                    <div>
                                      <label className="flex items-center justify-between text-xs font-semibold text-slate-600 mb-1">
                                        <span>Opacity threshold</span>
                                        <button onClick={() => setSelectedInfoKey("opacity_threshold")} className="p-1 text-slate-400 hover:text-slate-600"><Info /></button>
                                      </label>
                                      <input
                                        type="number"
                                        step="0.0005"
                                        value={opacityThreshold}
                                        onChange={(e) => setOpacityThreshold(parseFloat(e.target.value) || 0)}
                                        className="w-full px-3 py-2 border border-slate-300 rounded-lg"
                                        min={0}
                                      />
                                    </div>
                                    <div>
                                      <label className="flex items-center justify-between text-xs font-semibold text-slate-600 mb-1">
                                        <span>DSSIM weight</span>
                                        <button onClick={() => setSelectedInfoKey("lambda_dssim")} className="p-1 text-slate-400 hover:text-slate-600"><Info /></button>
                                      </label>
                                      <input
                                        type="number"
                                        step="0.05"
                                        value={lambdaDssim}
                                        onChange={(e) => setLambdaDssim(parseFloat(e.target.value) || 0)}
                                        className="w-full px-3 py-2 border border-slate-300 rounded-lg"
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
                          <div className="rounded-xl border border-blue-200 bg-white shadow-sm">
                            <div className="flex items-center justify-between px-4 py-3 border-b border-slate-100">
                              <div>
                                <p className="text-sm font-semibold text-slate-800">LiteGS-only controls</p>
                                <p className="text-xs text-slate-500">Only applied when LiteGS is selected.</p>
                              </div>
                              <span className="text-xs px-3 py-1 rounded-full bg-blue-100 text-blue-700">LiteGS selected</span>
                            </div>
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 px-4 py-4">
                              <div>
                                <label className="flex items-center justify-between text-xs font-semibold text-slate-600 mb-1">
                                  <span>Target primitives</span>
                                  <button onClick={() => setSelectedInfoKey("litegs_target_primitives")} className="p-1 text-slate-400 hover:text-slate-600"><Info /></button>
                                </label>
                                <input
                                  type="number"
                                  value={litegsTargetPrimitives}
                                  onChange={(e) => setLitegsTargetPrimitives(parseInt(e.target.value || "0") || 0)}
                                  className="w-full px-3 py-2 border border-slate-300 rounded-lg"
                                  min={5000}
                                  step={1000}
                                />
                                <div className="text-[11px] text-slate-500 mt-1">LiteGS grows Gaussians until it approaches this count.</div>
                              </div>
                              <div>
                                <label className="flex items-center justify-between text-xs font-semibold text-slate-600 mb-1">
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
                                  className="w-full px-3 py-2 border border-slate-300 rounded-lg"
                                />
                                <div className="text-[11px] text-slate-500 mt-1">Lower values tighten lobes faster; keep near 0.95 for stability.</div>
                              </div>
                            </div>
                          </div>
                        )}
                      </div>
                    ) : configTab === "colmap" ? (
                      <div className="grid grid-cols-2 gap-4">
                        <div>
                          <label className="flex items-center justify-between text-xs font-semibold text-slate-600 mb-1">
                            <span>SIFT Max Image Size</span>
                            <button onClick={() => setSelectedInfoKey("max_image_size")} className="p-1 text-slate-400 hover:text-slate-600"><Info /></button>
                          </label>
                          <input
                            type="number"
                            value={colmapMaxImageSize ?? ""}
                            onChange={(e) => setColmapMaxImageSize(e.target.value ? parseInt(e.target.value) : undefined)}
                            className="w-full px-3 py-2 border border-slate-300 rounded-lg"
                            min={256}
                            step={100}
                          />
                        </div>

                        <div>
                          <label className="flex items-center justify-between text-xs font-semibold text-slate-600 mb-1">
                            <span>SIFT Peak Threshold</span>
                            <button onClick={() => setSelectedInfoKey("peak_threshold")} className="p-1 text-slate-400 hover:text-slate-600"><Info /></button>
                          </label>
                          <input
                            type="number"
                            value={colmapPeakThreshold ?? ""}
                            onChange={(e) => setColmapPeakThreshold(e.target.value ? parseFloat(e.target.value) : undefined)}
                            className="w-full px-3 py-2 border border-slate-300 rounded-lg"
                            min={0.001}
                            step={0.001}
                          />
                        </div>

                        <div>
                          <label className="flex items-center justify-between text-xs font-semibold text-slate-600 mb-1">
                            <span>Guided Matching</span>
                            <button onClick={() => setSelectedInfoKey("guided_matching")} className="p-1 text-slate-400 hover:text-slate-600"><Info /></button>
                          </label>
                          <label className="inline-flex items-center gap-2">
                            <input type="checkbox" className="w-4 h-4" checked={colmapGuidedMatching} onChange={e => setColmapGuidedMatching(e.target.checked)} />
                            <span className="text-sm text-slate-700">Enable guided matching</span>
                          </label>
                        </div>

                        <div>
                          <label className="flex items-center justify-between text-xs font-semibold text-slate-600 mb-1">
                            <span>Matching Strategy</span>
                            <button onClick={() => setSelectedInfoKey("matching_type")} className="p-1 text-slate-400 hover:text-slate-600"><Info /></button>
                          </label>
                          <select value={colmapMatchingType} onChange={e => setColmapMatchingType(e.target.value as any)} className="w-full px-3 py-2 border border-slate-300 rounded-lg">
                            <option value="exhaustive">Exhaustive</option>
                            <option value="sequential">Sequential</option>
                          </select>
                        </div>

                        <div>
                          <label className="flex items-center justify-between text-xs font-semibold text-slate-600 mb-1">
                            <span>Mapper Threads</span>
                            <button onClick={() => setSelectedInfoKey("mapper_num_threads")} className="p-1 text-slate-400 hover:text-slate-600"><Info /></button>
                          </label>
                          <input type="number" value={colmapMapperThreads ?? ""} onChange={e => setColmapMapperThreads(e.target.value ? parseInt(e.target.value) : undefined)} className="w-full px-3 py-2 border border-slate-300 rounded-lg" min={1} max={128} />
                        </div>

                        <div>
                          <label className="flex items-center justify-between text-xs font-semibold text-slate-600 mb-1">
                            <span>Mapper Min Matches</span>
                            <button onClick={() => setSelectedInfoKey("mapper_min_num_matches")} className="p-1 text-slate-400 hover:text-slate-600"><Info /></button>
                          </label>
                          <input type="number" value={colmapMapperMinNumMatches ?? ""} onChange={e => setColmapMapperMinNumMatches(e.target.value ? parseInt(e.target.value) : undefined)} className="w-full px-3 py-2 border border-slate-300 rounded-lg" min={4} step={1} />
                        </div>

                        <div>
                          <label className="flex items-center justify-between text-xs font-semibold text-slate-600 mb-1">
                            <span>Mapper AbsPose Min Inliers</span>
                            <button onClick={() => setSelectedInfoKey("mapper_abs_pose_min_num_inliers")} className="p-1 text-slate-400 hover:text-slate-600"><Info /></button>
                          </label>
                          <input type="number" value={colmapMapperAbsPoseMinNumInliers ?? ""} onChange={e => setColmapMapperAbsPoseMinNumInliers(e.target.value ? parseInt(e.target.value) : undefined)} className="w-full px-3 py-2 border border-slate-300 rounded-lg" min={6} step={1} />
                        </div>

                        <div>
                          <label className="flex items-center justify-between text-xs font-semibold text-slate-600 mb-1">
                            <span>Mapper Init Min Inliers</span>
                            <button onClick={() => setSelectedInfoKey("mapper_init_min_num_inliers")} className="p-1 text-slate-400 hover:text-slate-600"><Info /></button>
                          </label>
                          <input type="number" value={colmapMapperInitMinNumInliers ?? ""} onChange={e => setColmapMapperInitMinNumInliers(e.target.value ? parseInt(e.target.value) : undefined)} className="w-full px-3 py-2 border border-slate-300 rounded-lg" min={10} step={1} />
                        </div>

                        <div>
                          <label className="flex items-center justify-between text-xs font-semibold text-slate-600 mb-1">
                            <span>Match Min Inliers</span>
                            <button onClick={() => setSelectedInfoKey("sift_matching_min_num_inliers")} className="p-1 text-slate-400 hover:text-slate-600"><Info /></button>
                          </label>
                          <input type="number" value={colmapSiftMatchingMinNumInliers ?? ""} onChange={e => setColmapSiftMatchingMinNumInliers(e.target.value ? parseInt(e.target.value) : undefined)} className="w-full px-3 py-2 border border-slate-300 rounded-lg" min={6} step={1} />
                        </div>

                        <div className="col-span-2">
                          <label className="flex items-center justify-between text-xs font-semibold text-slate-600 mb-1">
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
                      <div className="space-y-4">
                        <div>
                          <label className="flex items-center justify-between text-xs font-semibold text-slate-600 mb-1">
                            <span>Shared image set</span>
                            <button onClick={() => setSelectedInfoKey("resize_mode")} className="p-1 text-slate-400 hover:text-slate-600"><Info /></button>
                          </label>
                          <div className="space-y-2">
                            <label className={`flex items-center justify-between px-3 py-2 rounded-lg border ${!imagesResizeEnabled ? 'border-blue-500 bg-blue-50 text-blue-700' : 'border-slate-200 bg-slate-50 text-slate-700'}`}>
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
                            <label className={`flex items-center justify-between px-3 py-2 rounded-lg border ${imagesResizeEnabled ? 'border-blue-500 bg-blue-50 text-blue-700' : 'border-slate-200 bg-slate-50 text-slate-700'}`}>
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
                          <div className="space-y-2">
                            <label className="flex items-center justify-between text-xs font-semibold text-slate-600 mb-1">
                              <span>Max dimension (px)</span>
                              <button onClick={() => setSelectedInfoKey("images_max_size")} className="p-1 text-slate-400 hover:text-slate-600"><Info /></button>
                            </label>
                            <input
                              type="number"
                              value={imagesMaxSize ?? ""}
                              onChange={(e) => setImagesMaxSize(e.target.value ? parseInt(e.target.value) : undefined)}
                              className="w-full px-3 py-2 border border-slate-300 rounded-lg"
                              min={256}
                              step={50}
                            />
                            <p className="text-[11px] text-slate-500">Largest width or height applied to the cloned copies. Originals remain untouched on disk.</p>
                          </div>
                        ) : (
                          <p className="text-[11px] text-slate-500">Disabling downscaling means both stages ingest the original uploads. Expect longer COLMAP runtimes and higher VRAM usage.</p>
                        )}
                        <div className="text-[11px] text-slate-500 border border-slate-200 rounded-lg bg-slate-50 px-3 py-2">
                          We always keep two folders: your uploads and (optionally) a resized mirror inside <code>images_resized</code>. COLMAP intrinsics + gsplat training both read from the same folder so rays line up perfectly.
                        </div>
                      </div>
                    )}
                  </div>
                    <div className="col-span-3">
                      <div className="p-3 border rounded h-full bg-slate-50">
                      <h4 className="font-semibold text-sm">Parameter Info</h4>
                      <div className="mt-2 text-sm text-slate-700">
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
              <div className="px-4 py-3 border-t border-slate-200 flex justify-end gap-2">
                <button
                  onClick={resetConfigToDefaults}
                  className="px-4 py-2 rounded-lg border border-rose-200 text-rose-600 hover:bg-rose-50 text-sm font-semibold"
                >
                  Reset defaults
                </button>
                <button
                  onClick={() => setShowConfig(false)}
                  className="px-4 py-2 rounded-lg border border-slate-200 text-slate-700 hover:bg-slate-50 text-sm font-semibold"
                >
                  Close
                </button>
                <button
                  onClick={() => setShowConfig(false)}
                  className="px-4 py-2 rounded-lg bg-blue-600 text-white hover:bg-blue-700 text-sm font-semibold"
                >
                  Save
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
