import { useState, useEffect } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { ArrowLeft } from "lucide-react";
import axios from "axios";
import ConfirmModal from "../components/ConfirmModal";

const API_BASE = "http://localhost:8005";

interface DatasetInfo {
  name: string;
  path: string;
  image_count: number;
  size_mb: number;
  has_images: boolean;
  selected?: boolean;
  colmap_source_project_id?: string;
}

interface ExistingProject {
  id: string;
  name: string;
  has_colmap: boolean;
  dataset_path?: string;
  pipeline_name?: string;
}

interface PhaseConfig {
  phase_number: number;
  name: string;
  runs_per_project: number;
  passes: number;
  strategy_override?: string;
  preset_override?: string;
  update_model: boolean;
  context_jitter: boolean;  // Enable feature jittering for diverse exploration
  shuffle_order: boolean;
  session_execution_mode: string;
}


export default function TrainingPipelinePage() {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const editPipelineId = searchParams.get("edit");
  const [isEditMode, setIsEditMode] = useState(false);
  const [loadingPipeline, setLoadingPipeline] = useState(false);

  // Step 1: Dataset Selection
  const [baseDirectory, setBaseDirectory] = useState("E:\\Thesis\\Training Data");
  const [pipelineDirectory, setPipelineDirectory] = useState("E:\\Thesis\\PipelineProjects");
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [scanning, setScanning] = useState(false);
  const [existingProjects, setExistingProjects] = useState<ExistingProject[]>([]);

  // Step 2: Shared Configuration
  const [aiInputMode, setAiInputMode] = useState("exif_plus_flight_plan");
  const [aiSelectorStrategy, setAiSelectorStrategy] = useState("contextual_continuous");
  const [maxSteps, setMaxSteps] = useState(5000);
  const [evalInterval, setEvalInterval] = useState(1000);
  const [logInterval, setLogInterval] = useState(100);
  const [densifyUntil, setDensifyUntil] = useState(4000);
  const [imagesMaxSize, setImagesMaxSize] = useState(1600);

  // Storage Management
  const [saveEvalImages, setSaveEvalImages] = useState(true);
  const [replaceEvalImages, setReplaceEvalImages] = useState(true);  // Default: save storage
  const [saveCheckpoints, setSaveCheckpoints] = useState(true);
  const [replaceCheckpoints, setReplaceCheckpoints] = useState(true);  // Default: save storage
  const [saveFinalSplat, setSaveFinalSplat] = useState(true);  // Always recommended

  // Step 3: Training Schedule
  const [phases, setPhases] = useState<PhaseConfig[]>([
    {
      phase_number: 1,
      name: "Baseline Collection",
      runs_per_project: 1,
      passes: 1,
      strategy_override: "preset_bias",
      preset_override: "balanced",
      update_model: false,
      context_jitter: false,
      shuffle_order: false,
      session_execution_mode: "test",
    },
    {
      phase_number: 2,
      name: "Initial Exploration",
      runs_per_project: 1,
      passes: 1,
      update_model: true,
      context_jitter: false,
      shuffle_order: true,
      session_execution_mode: "train",
    },
    {
      phase_number: 3,
      name: "Multi-Pass Learning",
      runs_per_project: 1,
      passes: 5,
      update_model: true,
      context_jitter: true,
      shuffle_order: true,
      session_execution_mode: "train",
    },
  ]);

  // Step 4: Thermal Management
  const [thermalEnabled, setThermalEnabled] = useState(true);
  const [thermalStrategy, setThermalStrategy] = useState("fixed_interval");
  const [cooldownMinutes, setCooldownMinutes] = useState(10);

  // Step 5: Review
  const [pipelineName, setPipelineName] = useState(`training_${new Date().toISOString().split("T")[0]}`);
  const [creating, setCreating] = useState(false);

  // UI State
  const [currentStep, setCurrentStep] = useState(1);
  const [showCreateConfirm, setShowCreateConfirm] = useState(false);
  const [toast, setToast] = useState<{ message: string; type: "success" | "error" } | null>(null);

  const showToast = (message: string, type: "success" | "error" = "success") => {
    setToast({ message, type });
    window.setTimeout(() => setToast(null), 5000);
  };

  // Load pipeline for editing
  useEffect(() => {
    if (editPipelineId) {
      setIsEditMode(true);
      setLoadingPipeline(true);
      axios.get(`${API_BASE}/training-pipeline/${editPipelineId}`)
        .then((response) => {
          const pipeline = response.data;
          const config = pipeline.config;

          // Load configuration
          setPipelineName(pipeline.name);
          setBaseDirectory(config.base_directory || "");
          setPipelineDirectory(config.pipeline_directory || "");

          // Shared config
          const shared = config.shared_config || {};
          setAiInputMode(shared.ai_input_mode || "exif_plus_flight_plan");
          setAiSelectorStrategy(shared.ai_selector_strategy || "contextual_continuous");
          setMaxSteps(shared.max_steps || 5000);
          setEvalInterval(shared.eval_interval || 1000);
          setLogInterval(shared.log_interval || 100);
          setDensifyUntil(shared.densify_until_iter || 4000);
          setImagesMaxSize(shared.images_max_size || 1600);
          setSaveEvalImages(shared.save_eval_images !== false);
          setReplaceEvalImages(shared.replace_eval_images !== false);
          setSaveCheckpoints(shared.save_checkpoints !== false);
          setReplaceCheckpoints(shared.replace_checkpoints !== false);
          setSaveFinalSplat(shared.save_final_splat !== false);

          // Phases
          if (config.phases && config.phases.length > 0) {
            setPhases(config.phases);
          }

          // Thermal
          const thermal = config.thermal_management || {};
          setThermalEnabled(thermal.enabled !== false);
          setThermalStrategy(thermal.strategy || "fixed_interval");
          setCooldownMinutes(thermal.cooldown_minutes || 10);

          // Projects (datasets)
          if (config.projects && config.projects.length > 0) {
            const loadedDatasets = config.projects.map((p: any) => ({
              name: p.name,
              path: p.dataset_path,
              image_count: p.image_count || 0,
              size_mb: 0,
              has_images: true,
              selected: true,
              colmap_source_project_id: p.colmap_source_project_id,
            }));
            setDatasets(loadedDatasets);
          }

          showToast("Pipeline loaded for editing. Changes will require a restart to take effect.", "success");
          setCurrentStep(5); // Go to review step
        })
        .catch((error) => {
          console.error("Failed to load pipeline:", error);
          showToast(error.response?.data?.detail || "Failed to load pipeline", "error");
          navigate("/");
        })
        .finally(() => {
          setLoadingPipeline(false);
        });
    }
  }, [editPipelineId]);

  // Load existing projects with COLMAP
  const loadExistingProjects = async () => {
    try {
      const response = await axios.get(`${API_BASE}/projects`);
      const projects: ExistingProject[] = (response.data || []).map((p: any) => ({
        id: p.project_id,
        name: p.name || p.project_id,
        has_colmap: p.has_colmap || false,
        dataset_path: p.dataset_path,
        pipeline_name: p.pipeline_name,
      }));
      // Only show projects that have COLMAP outputs
      setExistingProjects(projects.filter(p => p.has_colmap));
    } catch (error: any) {
      console.error("Failed to load projects:", error);
      setExistingProjects([]);
    }
  };

  // Scan directory for datasets
  const handleScanDirectory = async () => {
    if (!baseDirectory.trim()) {
      alert("Please enter a directory path");
      return;
    }

    setScanning(true);
    try {
      const response = await axios.post(`${API_BASE}/training-pipeline/scan-directory`, {
        base_directory: baseDirectory,
      });

      const scannedDatasets = response.data.datasets.map((d: DatasetInfo) => ({
        ...d,
        selected: true, // Auto-select all by default
        colmap_source_project_id: undefined,
      }));

      setDatasets(scannedDatasets);

      // Load existing projects for COLMAP source selection
      await loadExistingProjects();
    } catch (error: any) {
      console.error("Failed to scan directory:", error);
      alert(`Failed to scan directory: ${error.response?.data?.detail || error.message}`);
    } finally {
      setScanning(false);
    }
  };

  // Calculate total runs
  const calculateTotalRuns = () => {
    const selectedCount = datasets.filter((d) => d.selected).length;
    let total = 0;
    for (const phase of phases) {
      total += phase.runs_per_project * phase.passes * selectedCount;
    }
    return total;
  };

  // Calculate estimated time
  const calculateEstimatedTime = () => {
    const totalRuns = calculateTotalRuns();
    const trainingMinutes = totalRuns * 8; // Assume 8 minutes per run
    const cooldownTime = thermalEnabled ? totalRuns * cooldownMinutes : 0;
    const totalMinutes = trainingMinutes + cooldownTime;
    const hours = Math.floor(totalMinutes / 60);
    const minutes = totalMinutes % 60;
    return { hours, minutes, totalMinutes };
  };

  // Show create confirmation
  const handleShowCreateConfirm = () => {
    const selectedDatasets = datasets.filter((d) => d.selected);

    if (selectedDatasets.length === 0) {
      alert("Please select at least one dataset");
      return;
    }

    setShowCreateConfirm(true);
  };

  // Create pipeline (after confirmation)
  const handleCreatePipeline = async () => {
    setShowCreateConfirm(false);
    setCreating(true);
    try {
      const selectedDatasets = datasets.filter((d) => d.selected);

      // Build configuration
      const config = {
        name: pipelineName,
        base_directory: baseDirectory,
        pipeline_directory: pipelineDirectory || null, // null = use default
        projects: selectedDatasets.map((d) => ({
          name: d.name,
          dataset_path: d.path,
          image_count: d.image_count,
          created: false,
          colmap_source_project_id: d.colmap_source_project_id || null,
        })),
        shared_config: {
          ai_input_mode: aiInputMode,
          ai_selector_strategy: aiSelectorStrategy,
          max_steps: maxSteps,
          eval_interval: evalInterval,
          log_interval: logInterval,
          densify_until_iter: densifyUntil,
          images_max_size: imagesMaxSize,
          // Storage management options
          save_eval_images: saveEvalImages,
          replace_eval_images: replaceEvalImages,
          save_checkpoints: saveCheckpoints,
          replace_checkpoints: replaceCheckpoints,
          save_final_splat: saveFinalSplat,
        },
        phases: phases,
        thermal_management: {
          enabled: thermalEnabled,
          strategy: thermalStrategy,
          cooldown_minutes: cooldownMinutes,
          gpu_temp_threshold: 70,
          check_interval_seconds: 30,
          max_wait_minutes: 30,
        },
        failure_handling: {
          continue_on_failure: true,
          max_retries_per_run: 1,
          skip_project_after_failures: 3,
        },
      };

      // Create or update pipeline
      if (isEditMode && editPipelineId) {
        await axios.put(`${API_BASE}/training-pipeline/${editPipelineId}/config`, config);
        showToast(`Pipeline "${pipelineName}" updated successfully! Restart to apply changes.`, "success");
        setTimeout(() => {
          navigate(`/pipelines/${editPipelineId}`);
        }, 1500);
      } else {
        await axios.post(`${API_BASE}/training-pipeline/create`, config);
        showToast(`Pipeline "${pipelineName}" created successfully!`, "success");
        setTimeout(() => {
          navigate("/");
        }, 1500);
      }

    } catch (error: any) {
      console.error("Failed to create pipeline:", error);
      showToast(`Failed to create pipeline: ${error.response?.data?.detail || error.message}`, "error");
    } finally {
      setCreating(false);
    }
  };

  if (loadingPipeline) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading pipeline configuration...</p>
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
                  <span className="text-xs font-medium text-white uppercase tracking-wider">
                    {isEditMode ? "Edit Training Pipeline" : "New Training Pipeline"}
                  </span>
                </div>
                <h1 className="text-2xl font-bold text-white mb-1">
                  {isEditMode ? "Edit Pipeline Configuration" : "Create Training Pipeline"}
                </h1>
                <p className="text-xs text-blue-100">
                  {isEditMode
                    ? "Update pipeline settings (requires restart to take effect)"
                    : "Configure automated multi-phase training across projects"}
                </p>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Edit Mode Warning */}
      {isEditMode && (
        <div className="bg-yellow-50 border-b-2 border-yellow-200">
          <div className="max-w-7xl mx-auto px-6 lg:px-8 py-4">
            <div className="flex items-start gap-3">
              <div className="flex-shrink-0">
                <svg className="w-6 h-6 text-yellow-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                </svg>
              </div>
              <div className="flex-1">
                <h3 className="text-sm font-semibold text-yellow-800 mb-1">
                  Configuration Changes Require Pipeline Restart
                </h3>
                <p className="text-xs text-yellow-700">
                  Any changes you make will only take effect after restarting the pipeline. Restarting will delete all training runs except the baseline, keeping only images, COLMAP data, and baseline splats.
                </p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Progress Indicator */}
      <div className="bg-white border-b border-gray-200 shadow-sm">
        <div className="max-w-7xl mx-auto px-6 lg:px-8">
          <div className="flex justify-between py-4">
            {[1, 2, 3, 4, 5].map((step) => (
              <div key={step} className="flex flex-col items-center flex-1">
                <div
                  className={`w-10 h-10 rounded-full flex items-center justify-center font-bold mb-2 transition-all ${
                    currentStep >= step
                      ? "bg-blue-600 text-white shadow-lg"
                      : "bg-gray-200 text-gray-400"
                  }`}
                >
                  {step}
                </div>
                <div className={`text-xs font-medium ${currentStep >= step ? "text-gray-900" : "text-gray-400"}`}>
                  {step === 1 && "Datasets"}
                  {step === 2 && "Config"}
                  {step === 3 && "Schedule"}
                  {step === 4 && "Thermal"}
                  {step === 5 && "Review"}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">

      {/* Step 1: Dataset Selection */}
      {currentStep === 1 && (
        <div style={{ border: "1px solid #ddd", padding: "20px", borderRadius: "4px", marginBottom: "20px" }}>
          <h2>Step 1: Dataset Selection</h2>

          <div style={{ marginBottom: "15px" }}>
            <label style={{ display: "block", marginBottom: "5px", fontWeight: "bold" }}>
              Pipeline Name:
            </label>
            <input
              type="text"
              value={pipelineName}
              onChange={(e) => setPipelineName(e.target.value)}
              placeholder="my_training_pipeline"
              style={{ width: "100%", padding: "8px" }}
            />
            <p style={{ fontSize: "12px", color: "#666", marginTop: "5px" }}>
              Name for this training pipeline (will be used as the folder name)
            </p>
          </div>

          <div style={{ marginBottom: "15px" }}>
            <label style={{ display: "block", marginBottom: "5px", fontWeight: "bold" }}>
              Source Data Directory (Read-Only):
            </label>
            <div style={{ display: "flex", gap: "10px" }}>
              <input
                type="text"
                value={baseDirectory}
                onChange={(e) => setBaseDirectory(e.target.value)}
                placeholder="E:/Thesis/exp_new_method"
                style={{ flex: 1, padding: "8px" }}
              />
              <button onClick={handleScanDirectory} disabled={scanning} style={{ padding: "8px 16px" }}>
                {scanning ? "Scanning..." : "Scan Directory"}
              </button>
            </div>
            <p style={{ fontSize: "12px", color: "#666", marginTop: "5px" }}>
              Directory containing dataset folders with images (will NOT be modified)
            </p>
          </div>

          <div style={{ marginBottom: "15px" }}>
            <label style={{ display: "block", marginBottom: "5px", fontWeight: "bold" }}>
              Pipeline Output Directory:
            </label>
            <input
              type="text"
              value={pipelineDirectory}
              onChange={(e) => setPipelineDirectory(e.target.value)}
              placeholder="Leave empty to use default projects directory"
              style={{ width: "100%", padding: "8px" }}
            />
            <p style={{ fontSize: "12px", color: "#666", marginTop: "5px" }}>
              Where to create the pipeline folder. Leave empty to use default location (same as manual projects).
              Pipeline will create: {pipelineDirectory || "[default]"}/{pipelineName}/
            </p>
          </div>

          {datasets.length > 0 && (
            <div>
              <h3>Discovered Datasets ({datasets.filter((d) => d.selected).length}/{datasets.length} selected):</h3>

              <div style={{ marginBottom: "10px" }}>
                <button onClick={() => setDatasets(datasets.map((d) => ({ ...d, selected: true })))} style={{ marginRight: "10px" }}>
                  Select All
                </button>
                <button onClick={() => setDatasets(datasets.map((d) => ({ ...d, selected: false })))}>
                  Deselect All
                </button>
              </div>

              <div style={{ maxHeight: "400px", overflowY: "auto", border: "1px solid #ddd", padding: "10px" }}>
                {datasets.map((dataset, idx) => (
                  <div key={idx} style={{ padding: "8px", borderBottom: "1px solid #eee" }}>
                    <div style={{ display: "flex", alignItems: "center", marginBottom: "8px" }}>
                      <input
                        type="checkbox"
                        checked={dataset.selected}
                        onChange={(e) => {
                          const updated = [...datasets];
                          updated[idx].selected = e.target.checked;
                          setDatasets(updated);
                        }}
                        style={{ marginRight: "10px" }}
                      />
                      <div style={{ flex: 1 }}>
                        <strong>{dataset.name}</strong>
                        <div style={{ fontSize: "12px", color: "#666" }}>
                          Images: {dataset.image_count} | Size: {dataset.size_mb.toFixed(1)} MB
                        </div>
                      </div>
                    </div>
                    {dataset.selected && (
                      <div style={{ marginLeft: "30px", marginTop: "4px" }}>
                        <label style={{ fontSize: "12px", color: "#555", display: "block", marginBottom: "4px" }}>
                          Copy COLMAP from existing project (optional):
                        </label>
                        <select
                          value={dataset.colmap_source_project_id || ""}
                          onChange={(e) => {
                            const updated = [...datasets];
                            updated[idx].colmap_source_project_id = e.target.value || undefined;
                            setDatasets(updated);
                          }}
                          style={{ width: "100%", padding: "4px", fontSize: "12px" }}
                        >
                          <option value="">-- Run COLMAP for this dataset --</option>
                          {existingProjects.map((proj) => (
                            <option key={proj.id} value={proj.id}>
                              {proj.name}{proj.pipeline_name ? `, ${proj.pipeline_name}` : ''}
                            </option>
                          ))}
                        </select>
                        {dataset.colmap_source_project_id && (() => {
                          const selectedProject = existingProjects.find(p => p.id === dataset.colmap_source_project_id);
                          const datasetName = dataset.name.toLowerCase();
                          const projectName = selectedProject?.name?.toLowerCase() || '';
                          // Check if names match (allowing for space/underscore differences)
                          const datasetNameNormalized = datasetName.replace(/\s+/g, '_');
                          const projectNameNormalized = projectName.replace(/\s+/g, '_');
                          const namesMatch = projectNameNormalized === datasetNameNormalized;

                          return (
                            <>
                              <div style={{ fontSize: "11px", color: "#0066cc", marginTop: "2px" }}>
                                ✓ Will copy COLMAP from selected project (saves ~15-30 min)
                              </div>
                              {!namesMatch && selectedProject && (
                                <div style={{ fontSize: "11px", color: "#ff6600", marginTop: "2px", fontWeight: "bold" }}>
                                  ⚠ Warning: Project name "{selectedProject.name}" does not match dataset name "{dataset.name}". COLMAP data may be from different images.
                                </div>
                              )}
                            </>
                          );
                        })()}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          <div style={{ marginTop: "20px", textAlign: "right" }}>
            <button onClick={() => setCurrentStep(2)} disabled={datasets.filter((d) => d.selected).length === 0} style={{ padding: "8px 24px" }}>
              Next
            </button>
          </div>
        </div>
      )}

      {/* Step 2: Shared Configuration */}
      {currentStep === 2 && (
        <div style={{ border: "1px solid #ddd", padding: "20px", borderRadius: "4px", marginBottom: "20px" }}>
          <h2>Step 2: Shared Training Configuration</h2>

          <div style={{ marginBottom: "15px" }}>
            <label style={{ display: "block", marginBottom: "5px" }}>AI Input Mode:</label>
            <select value={aiInputMode} onChange={(e) => setAiInputMode(e.target.value)} style={{ width: "100%", padding: "8px" }}>
              <option value="exif_only">EXIF Only</option>
              <option value="exif_plus_flight_plan">EXIF + Flight Plan</option>
              <option value="exif_plus_flight_plan_plus_external">EXIF + Flight Plan + External</option>
            </select>
          </div>

          <div style={{ marginBottom: "15px" }}>
            <label style={{ display: "block", marginBottom: "5px" }}>Selector Strategy:</label>
            <select value={aiSelectorStrategy} onChange={(e) => setAiSelectorStrategy(e.target.value)} style={{ width: "100%", padding: "8px" }}>
              <option value="contextual_continuous">Contextual Continuous (NEW)</option>
              <option value="continuous_bandit_linear">Continuous Bandit</option>
              <option value="preset_bias">Preset Bias</option>
            </select>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "15px" }}>
            <div>
              <label style={{ display: "block", marginBottom: "5px" }}>Max Steps:</label>
              <input type="number" value={maxSteps} onChange={(e) => setMaxSteps(Number(e.target.value))} style={{ width: "100%", padding: "8px" }} />
            </div>
            <div>
              <label style={{ display: "block", marginBottom: "5px" }}>
                Eval Interval:
                <span style={{ fontSize: "11px", color: "#666", fontWeight: "normal", marginLeft: "5px" }}>
                  (Quality evaluation frequency)
                </span>
              </label>
              <input
                type="number"
                value={evalInterval}
                onChange={(e) => setEvalInterval(Number(e.target.value))}
                style={{ width: "100%", padding: "8px" }}
                min={100}
                max={5000}
                step={100}
              />
              <p style={{ fontSize: "11px", color: "#888", marginTop: "3px" }}>
                More frequent = better quality tracking. Default: 1000
              </p>
            </div>
            <div>
              <label style={{ display: "block", marginBottom: "5px" }}>Log Interval:</label>
              <input type="number" value={logInterval} onChange={(e) => setLogInterval(Number(e.target.value))} style={{ width: "100%", padding: "8px" }} />
            </div>
            <div>
              <label style={{ display: "block", marginBottom: "5px" }}>Densify Until:</label>
              <input type="number" value={densifyUntil} onChange={(e) => setDensifyUntil(Number(e.target.value))} style={{ width: "100%", padding: "8px" }} />
            </div>
            <div>
              <label style={{ display: "block", marginBottom: "5px" }}>Images Max Size:</label>
              <input type="number" value={imagesMaxSize} onChange={(e) => setImagesMaxSize(Number(e.target.value))} style={{ width: "100%", padding: "8px" }} />
            </div>
          </div>

          {/* Storage Management Options */}
          <div style={{ marginTop: "25px", padding: "15px", background: "#fff8dc", border: "1px solid #daa520", borderRadius: "4px" }}>
            <h3 style={{ margin: "0 0 15px 0", fontSize: "16px", color: "#b8860b" }}>
              Storage Management
            </h3>
            <p style={{ fontSize: "12px", color: "#666", marginBottom: "15px" }}>
              Configure what gets saved to manage storage. Eval images at 200-500 step intervals can create massive storage requirements.
            </p>

            <div style={{ display: "grid", gridTemplateColumns: "1fr", gap: "12px" }}>
              {/* Save Eval Images */}
              <label style={{ display: "flex", alignItems: "center", gap: "8px", fontSize: "14px" }}>
                <input
                  type="checkbox"
                  checked={saveEvalImages}
                  onChange={(e) => setSaveEvalImages(e.target.checked)}
                  style={{ width: "18px", height: "18px" }}
                />
                <span>
                  <strong>Save Evaluation Images</strong>
                  <span style={{ fontSize: "11px", color: "#666", marginLeft: "8px" }}>
                    (Renders at each eval_interval. WARNING: ~5-50MB per eval × num_evals = GBs)
                  </span>
                </span>
              </label>

              {/* Replace Eval Images */}
              {saveEvalImages && (
                <label style={{ display: "flex", alignItems: "center", gap: "8px", fontSize: "14px", marginLeft: "30px" }}>
                  <input
                    type="checkbox"
                    checked={replaceEvalImages}
                    onChange={(e) => setReplaceEvalImages(e.target.checked)}
                    style={{ width: "18px", height: "18px" }}
                  />
                  <span>
                    <strong>Replace Eval Images</strong> (Keep only latest eval, delete previous)
                    <span style={{ fontSize: "11px", color: "#666", marginLeft: "8px" }}>
                      (Saves ~95% storage. Use for pipeline training, disable for final runs)
                    </span>
                  </span>
                </label>
              )}

              {/* Save Checkpoints */}
              <label style={{ display: "flex", alignItems: "center", gap: "8px", fontSize: "14px" }}>
                <input
                  type="checkbox"
                  checked={saveCheckpoints}
                  onChange={(e) => setSaveCheckpoints(e.target.checked)}
                  style={{ width: "18px", height: "18px" }}
                />
                <span>
                  <strong>Save Training Checkpoints</strong>
                  <span style={{ fontSize: "11px", color: "#666", marginLeft: "8px" }}>
                    (Model weights for resuming. ~100-500MB per checkpoint)
                  </span>
                </span>
              </label>

              {/* Replace Checkpoints */}
              {saveCheckpoints && (
                <label style={{ display: "flex", alignItems: "center", gap: "8px", fontSize: "14px", marginLeft: "30px" }}>
                  <input
                    type="checkbox"
                    checked={replaceCheckpoints}
                    onChange={(e) => setReplaceCheckpoints(e.target.checked)}
                    style={{ width: "18px", height: "18px" }}
                  />
                  <span>
                    <strong>Replace Checkpoints</strong> (Keep only latest checkpoint)
                    <span style={{ fontSize: "11px", color: "#666", marginLeft: "8px" }}>
                      (Recommended for pipelines. Keep only final model)
                    </span>
                  </span>
                </label>
              )}

              {/* Save Final Splat */}
              <label style={{ display: "flex", alignItems: "center", gap: "8px", fontSize: "14px" }}>
                <input
                  type="checkbox"
                  checked={saveFinalSplat}
                  onChange={(e) => setSaveFinalSplat(e.target.checked)}
                  style={{ width: "18px", height: "18px" }}
                />
                <span>
                  <strong>Save Final Splat Model</strong>
                  <span style={{ fontSize: "11px", color: "#666", marginLeft: "8px" }}>
                    (Always recommended. ~50-200MB. Needed for viewing results)
                  </span>
                </span>
              </label>
            </div>
          </div>

          <div style={{ marginTop: "20px", display: "flex", justifyContent: "space-between" }}>
            <button onClick={() => setCurrentStep(1)} style={{ padding: "8px 24px" }}>
              Back
            </button>
            <button onClick={() => setCurrentStep(3)} style={{ padding: "8px 24px" }}>
              Next
            </button>
          </div>
        </div>
      )}

      {/* Step 3: Training Schedule */}
      {currentStep === 3 && (
        <div style={{ border: "1px solid #ddd", padding: "20px", borderRadius: "4px", marginBottom: "20px" }}>
          <h2>Step 3: Training Schedule</h2>

          {phases.map((phase, idx) => (
            <div key={idx} style={{ marginBottom: "20px", padding: "15px", background: "#f9f9f9", borderRadius: "4px" }}>
              <h3>Phase {phase.phase_number}: {phase.name}</h3>

              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "10px" }}>
                <div>
                  <label style={{ display: "block", fontSize: "12px", marginBottom: "3px" }}>Runs per project:</label>
                  <input
                    type="number"
                    value={phase.runs_per_project}
                    onChange={(e) => {
                      const updated = [...phases];
                      updated[idx].runs_per_project = Number(e.target.value);
                      setPhases(updated);
                    }}
                    style={{ width: "100%", padding: "6px" }}
                  />
                </div>
                <div>
                  <label style={{ display: "block", fontSize: "12px", marginBottom: "3px" }}>Passes:</label>
                  <input
                    type="number"
                    value={phase.passes}
                    onChange={(e) => {
                      const updated = [...phases];
                      updated[idx].passes = Number(e.target.value);
                      setPhases(updated);
                    }}
                    style={{ width: "100%", padding: "6px" }}
                  />
                </div>
              </div>

              {/* Context Jitter and Shuffle Settings */}
              <div style={{ marginTop: "10px", padding: "10px", background: "#fff", border: "1px solid #ddd", borderRadius: "4px" }}>
                <div style={{ marginBottom: "8px" }}>
                  <label style={{ display: "flex", alignItems: "center", fontSize: "12px" }}>
                    <input
                      type="checkbox"
                      checked={phase.context_jitter}
                      onChange={(e) => {
                        const updated = [...phases];
                        updated[idx].context_jitter = e.target.checked;
                        setPhases(updated);
                      }}
                      style={{ marginRight: "8px" }}
                    />
                    <strong>Enable Context Jitter</strong>
                    <span style={{ marginLeft: "8px", color: "#666", fontWeight: "normal" }}>
                      (Randomize context features for diverse exploration)
                    </span>
                  </label>
                  {phase.context_jitter && (
                    <p style={{ fontSize: "10px", color: "#888", marginTop: "3px", marginLeft: "24px" }}>
                      Features will be randomly sampled from valid ranges to explore different scenarios
                    </p>
                  )}
                </div>

                <div style={{ marginTop: "8px" }}>
                  <label style={{ display: "flex", alignItems: "center", fontSize: "12px" }}>
                    <input
                      type="checkbox"
                      checked={phase.shuffle_order}
                      onChange={(e) => {
                        const updated = [...phases];
                        updated[idx].shuffle_order = e.target.checked;
                        setPhases(updated);
                      }}
                      style={{ marginRight: "8px" }}
                    />
                    <strong>Shuffle Project Order</strong>
                    <span style={{ marginLeft: "8px", color: "#666", fontWeight: "normal" }}>
                      (Randomize sequence each pass)
                    </span>
                  </label>
                </div>
              </div>

              <div style={{ marginTop: "10px", fontSize: "13px", color: "#666" }}>
                Total runs: {phase.runs_per_project * phase.passes * datasets.filter((d) => d.selected).length}
              </div>
            </div>
          ))}

          <div style={{ padding: "15px", background: "#e3f2fd", borderRadius: "4px" }}>
            <strong>Grand Total: {calculateTotalRuns()} runs</strong>
          </div>

          <div style={{ marginTop: "20px", display: "flex", justifyContent: "space-between" }}>
            <button onClick={() => setCurrentStep(2)} style={{ padding: "8px 24px" }}>
              Back
            </button>
            <button onClick={() => setCurrentStep(4)} style={{ padding: "8px 24px" }}>
              Next
            </button>
          </div>
        </div>
      )}

      {/* Step 4: Thermal Management */}
      {currentStep === 4 && (
        <div style={{ border: "1px solid #ddd", padding: "20px", borderRadius: "4px", marginBottom: "20px" }}>
          <h2>Step 4: Thermal Management</h2>

          <div style={{ marginBottom: "15px" }}>
            <label>
              <input type="checkbox" checked={thermalEnabled} onChange={(e) => setThermalEnabled(e.target.checked)} style={{ marginRight: "8px" }} />
              Enable cooldown periods between runs
            </label>
          </div>

          {thermalEnabled && (
            <div>
              <div style={{ marginBottom: "15px" }}>
                <label style={{ display: "block", marginBottom: "5px" }}>Cooldown Strategy:</label>
                <select value={thermalStrategy} onChange={(e) => setThermalStrategy(e.target.value)} style={{ width: "100%", padding: "8px" }}>
                  <option value="fixed_interval">Fixed Interval</option>
                  <option value="temperature_based">Temperature-based (requires GPU monitoring)</option>
                  <option value="time_scheduled">Time-of-day scheduling</option>
                </select>
              </div>

              {thermalStrategy === "fixed_interval" && (
                <div style={{ marginBottom: "15px" }}>
                  <label style={{ display: "block", marginBottom: "5px" }}>Wait time (minutes):</label>
                  <input
                    type="number"
                    value={cooldownMinutes}
                    onChange={(e) => setCooldownMinutes(Number(e.target.value))}
                    style={{ width: "100%", padding: "8px" }}
                  />
                </div>
              )}

              <div style={{ padding: "15px", background: "#fff3cd", borderRadius: "4px" }}>
                <h4>Estimated Total Time:</h4>
                <div>Training time: {calculateTotalRuns()} runs × 8 min ≈ {Math.floor((calculateTotalRuns() * 8) / 60)} hours</div>
                <div>Cooldown time: {calculateTotalRuns()} × {cooldownMinutes} min ≈ {Math.floor((calculateTotalRuns() * cooldownMinutes) / 60)} hours</div>
                <div style={{ fontWeight: "bold", marginTop: "10px" }}>
                  Total: ~{calculateEstimatedTime().hours}h {calculateEstimatedTime().minutes}m (~{(calculateEstimatedTime().totalMinutes / 1440).toFixed(1)} days)
                </div>
              </div>
            </div>
          )}

          <div style={{ marginTop: "20px", display: "flex", justifyContent: "space-between" }}>
            <button onClick={() => setCurrentStep(3)} style={{ padding: "8px 24px" }}>
              Back
            </button>
            <button onClick={() => setCurrentStep(5)} style={{ padding: "8px 24px" }}>
              Next
            </button>
          </div>
        </div>
      )}

      {/* Step 5: Review & Launch */}
      {currentStep === 5 && (
        <div style={{ border: "1px solid #ddd", padding: "20px", borderRadius: "4px", marginBottom: "20px" }}>
          <h2>Step 5: Review & Launch</h2>

          <div style={{ padding: "15px", background: "#f5f5f5", borderRadius: "4px", marginBottom: "15px" }}>
            <h3>Pipeline Summary:</h3>
            <ul>
              <li>Projects: {datasets.filter((d) => d.selected).length}</li>
              <li>Total runs: {calculateTotalRuns()}</li>
              <li>Strategy: {aiSelectorStrategy}</li>
              <li>Estimated duration: ~{calculateEstimatedTime().hours}h {calculateEstimatedTime().minutes}m</li>
            </ul>
          </div>

          <div style={{ marginBottom: "15px" }}>
            <label style={{ display: "block", marginBottom: "5px" }}>Pipeline Name:</label>
            <input
              type="text"
              value={pipelineName}
              onChange={(e) => setPipelineName(e.target.value)}
              style={{ width: "100%", padding: "8px" }}
            />
          </div>

          <div style={{ marginTop: "20px", display: "flex", justifyContent: "space-between" }}>
            <button onClick={() => setCurrentStep(4)} style={{ padding: "8px 24px" }}>
              Back
            </button>
            <button
              onClick={handleShowCreateConfirm}
              disabled={creating}
              style={{ padding: "8px 24px", background: "#4CAF50", color: "white", fontWeight: "bold", border: "none", borderRadius: "4px" }}
            >
              {creating
                ? (isEditMode ? "Updating..." : "Creating...")
                : (isEditMode ? "Save Changes" : "Create Pipeline")}
            </button>
          </div>
        </div>
      )}

      <ConfirmModal
        open={showCreateConfirm}
        title={isEditMode ? "Update Pipeline Configuration" : "Create Pipeline"}
        message={
          <>
            {isEditMode ? (
              <>
                You are about to update the pipeline configuration with:
                <ul className="list-disc ml-5 mt-2 mb-2">
                  <li><strong>{datasets.filter(d => d.selected).length}</strong> projects</li>
                  <li><strong>{calculateTotalRuns()}</strong> total training runs</li>
                  <li><strong>~{calculateEstimatedTime().hours}h {calculateEstimatedTime().minutes}m</strong> estimated duration</li>
                </ul>
                <strong className="text-yellow-700">⚠️ Warning:</strong> Changes will only take effect after restarting the pipeline.
                Restart will delete all non-baseline runs and reset the pipeline state.
                <br /><br />
                Do you want to save these changes?
              </>
            ) : (
              <>
                You are about to create a training pipeline with:
                <ul className="list-disc ml-5 mt-2 mb-2">
                  <li><strong>{datasets.filter(d => d.selected).length}</strong> projects</li>
                  <li><strong>{calculateTotalRuns()}</strong> total training runs</li>
                  <li><strong>~{calculateEstimatedTime().hours}h {calculateEstimatedTime().minutes}m</strong> estimated duration</li>
                </ul>
                The pipeline will be created with status "pending". You can start it manually from the pipeline list.
                <br /><br />
                Do you want to continue?
              </>
            )}
          </>
        }
        confirmLabel={isEditMode ? "Save Changes" : "Create Pipeline"}
        cancelLabel="Cancel"
        tone="default"
        busy={creating}
        onConfirm={handleCreatePipeline}
        onCancel={() => setShowCreateConfirm(false)}
      />

      {/* Toast Notification */}
      {toast && (
        <div
          style={{
            position: "fixed",
            bottom: "20px",
            right: "20px",
            padding: "12px 20px",
            background: toast.type === "success" ? "#4CAF50" : "#f44336",
            color: "white",
            borderRadius: "4px",
            boxShadow: "0 2px 8px rgba(0,0,0,0.2)",
            zIndex: 9999,
            maxWidth: "400px",
          }}
        >
          {toast.message}
        </div>
      )}
      </main>
    </div>
  );
}
