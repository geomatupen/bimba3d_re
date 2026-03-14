import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { Upload as UploadIcon, Cpu, AlertTriangle, ArrowLeft, Play, CheckCircle2 } from "lucide-react";
import { api } from "../api/client";

export default function Upload() {
  const [files, setFiles] = useState<File[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [name, setName] = useState("");
  const [gpuHealth, setGpuHealth] = useState<any | null>(null);

  // Training mode
  const [comparisonMode, setComparisonMode] = useState(false);
  const [trainingMode, setTrainingMode] = useState<"baseline" | "modified">("baseline");
  const [stage, setStage] = useState<"full" | "colmap_only" | "train_only">("full");

  // Processing parameters
  const [maxSteps, setMaxSteps] = useState(300);
  const [batchSize, setBatchSize] = useState(1);
  const [splatInterval, setSplatInterval] = useState<number | undefined>(150);
  const [pngInterval, setPngInterval] = useState<number | undefined>(50);
  const [autoEarlyStop, setAutoEarlyStop] = useState(false);
  const [resume, setResume] = useState(false);
  // Config tab state
  const [configTab, setConfigTab] = useState<"training" | "colmap">("training");

  // COLMAP parameters
  const [colmapMaxImageSize, setColmapMaxImageSize] = useState<number | undefined>(1600);
  const [colmapPeakThreshold, setColmapPeakThreshold] = useState<number | undefined>(0.01);
  const [colmapGuidedMatching, setColmapGuidedMatching] = useState(true);
  const [colmapMatchingType, setColmapMatchingType] = useState<"exhaustive" | "sequential">("exhaustive");
  const [colmapMapperThreads, setColmapMapperThreads] = useState<number | undefined>(2);
  // Gaussian Splatting init limit
  const [gsplatMaxGaussians, _setGsplatMaxGaussians] = useState<number | undefined>(20000);

  const navigate = useNavigate();

  useEffect(() => {
    const fetchGpuHealth = async () => {
      try {
        const res = await api.get("/health/gpu");
        setGpuHealth(res.data);
      } catch (e) {
        // If health check fails, default to not available
        setGpuHealth({ available: false });
      }
    };
    fetchGpuHealth();
  }, []);

  const handleUpload = async () => {
    if (files.length === 0) {
      setError("Please select at least one image");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      if (comparisonMode) {
        await runComparison();
      } else {
        await runSingle(trainingMode);
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : "Upload failed";
      setError(errorMessage);
      console.error("Upload error:", err);
    } finally {
      setLoading(false);
    }
  };

  const runSingle = async (mode: "baseline" | "modified") => {
    const projectRes = await api.post("/projects", {
      name: name.trim() || `${mode === "baseline" ? "Baseline" : "Optimized"} Run`,
      mode: mode,
    });
    const projectId = projectRes.data.project_id;

    const form = new FormData();
    files.forEach((f) => form.append("images", f));
    await api.post(`/projects/${projectId}/images`, form);

    await api.post(`/projects/${projectId}/process`, {
      mode: mode,
      stage: stage,
      gsplat_max_gaussians: gsplatMaxGaussians,
      max_steps: maxSteps,
      batch_size: batchSize,
      splat_export_interval: splatInterval,
      png_export_interval: pngInterval,
      auto_early_stop: autoEarlyStop,
      resume: resume,
      colmap: {
        max_image_size: colmapMaxImageSize,
        peak_threshold: colmapPeakThreshold,
        guided_matching: colmapGuidedMatching,
        matching_type: colmapMatchingType,
        mapper_num_threads: colmapMapperThreads,
      },
    });

    navigate(`/project/${projectId}`);
  };

  const runComparison = async () => {
    const compRes = await api.post("/projects/comparison", {
      name: name.trim() || "Baseline vs Optimized Comparison",
      max_steps: maxSteps,
      batch_size: batchSize,
    });
    const comparisonId = compRes.data.comparison_id;

    const form = new FormData();
    files.forEach((f) => form.append("images", f));
    await api.post(`/projects/comparison/${comparisonId}/images`, form);

    await api.post(`/projects/comparison/${comparisonId}/start`);

    navigate(`/comparison/${comparisonId}`);
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-white via-slate-50 to-white">
      <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-10 space-y-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <button
              onClick={() => navigate("/")}
              className="inline-flex items-center gap-2 text-slate-600 hover:text-slate-900 transition-colors"
            >
              <ArrowLeft className="w-5 h-5" />
              Back to Dashboard
            </button>
            <div>
              <p className="text-xs uppercase tracking-[0.14em] text-slate-500">Upload</p>
              <h1 className="text-2xl font-bold text-slate-900">Upload Images for 3D Reconstruction</h1>
            </div>
          </div>
        </div>

          {/* GPU Health */}
          {gpuHealth && (
            <div
              className={`flex items-start gap-3 rounded-xl border px-4 py-3 shadow-sm ${
                gpuHealth.available
                  ? "border-emerald-100 bg-emerald-50 text-emerald-800"
                  : "border-amber-100 bg-amber-50 text-amber-800"
              }`}
            >
              <div className="mt-0.5">
                {gpuHealth.available ? (
                  <Cpu className="w-5 h-5" />
                ) : (
                  <AlertTriangle className="w-5 h-5" />
                )}
              </div>
              <div className="text-sm">
                {gpuHealth.available ? (
                  <p>
                    <strong>GPU detected:</strong> {Array.isArray(gpuHealth.devices) && gpuHealth.devices.length > 0 ? gpuHealth.devices.join(", ") : "CUDA device"}
                    {gpuHealth.cuda_version ? ` (CUDA ${gpuHealth.cuda_version})` : ""}
                  </p>
                ) : (
                  <p>
                    <strong>No GPU detected.</strong> Training will run on CPU and be slower. Enable NVIDIA drivers and Docker GPU support.
                  </p>
                )}
              </div>
            </div>
          )}

          {/* Upload */}
          <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm space-y-5">
            <div className="flex items-start gap-3">
              <div className="h-11 w-11 rounded-xl bg-blue-50 border border-blue-100 flex items-center justify-center">
                <UploadIcon className="w-5 h-5 text-blue-600" />
              </div>
              <div className="flex-1">
                <label className="block text-sm font-semibold text-slate-800 mb-2">Project Name (optional)</label>
                <input
                  type="text"
                  placeholder="e.g., Cathedral Walkthrough"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  disabled={loading}
                  className="w-full rounded-lg border border-slate-300 px-4 py-3 text-slate-900 placeholder:text-slate-400 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
                <p className="text-sm text-slate-500 mt-1">Name helps you find runs quickly.</p>
              </div>
            </div>

            <div className="space-y-3">
              <label className="block text-sm font-semibold text-slate-800">Select Images</label>
              <input
                type="file"
                multiple
                accept="image/*"
                onChange={(e) => {
                  setFiles(Array.from(e.target.files || []));
                  setError(null);
                }}
                disabled={loading}
                className="w-full rounded-lg border border-dashed border-slate-300 bg-slate-50 px-4 py-10 text-center text-slate-500 focus:outline-none"
              />
              {files.length > 0 && (
                <div className="rounded-lg border border-slate-200 bg-slate-50 px-4 py-3 text-sm text-slate-700">
                  <p className="font-semibold mb-2">Selected files ({files.length})</p>
                  <ul className="space-y-1 max-h-40 overflow-auto">
                    {files.map((f) => (
                      <li key={f.name} className="flex items-center gap-2 text-slate-600">
                        <CheckCircle2 className="w-4 h-4 text-emerald-500" />
                        {f.name}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          </div>

          {/* Mode & Params */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm space-y-4">
              <div className="flex items-center justify-between">
                <h3 className="text-lg font-semibold text-slate-900">Training Mode</h3>
                <span className="text-xs uppercase tracking-wide text-slate-500">Control</span>
              </div>

              <label className="flex items-center gap-3 text-sm text-slate-800 cursor-pointer">
                <input
                  type="checkbox"
                  checked={comparisonMode}
                  onChange={(e) => setComparisonMode(e.target.checked)}
                  disabled={loading}
                  className="h-4 w-4 text-blue-600 rounded border-slate-300"
                />
                <span><strong>Run Comparison Mode</strong> (Baseline → Optimized → Auto-Compare)</span>
              </label>
              <p className="text-sm text-slate-500">
                {comparisonMode
                  ? "Runs baseline then optimized sequentially, then compares outputs."
                  : "Single training run with the selected mode."}
              </p>

              {!comparisonMode && (
                <div className="space-y-3">
                  <label className="flex items-center gap-3 text-sm text-slate-800 cursor-pointer">
                    <input
                      type="radio"
                      name="mode"
                      value="baseline"
                      checked={trainingMode === "baseline"}
                      onChange={() => setTrainingMode("baseline")}
                      disabled={loading}
                      className="h-4 w-4 text-blue-600 border-slate-300"
                    />
                    <span><strong>Baseline</strong> – standard gsplat training.</span>
                  </label>
                  <label className="flex items-center gap-3 text-sm text-slate-800 cursor-pointer">
                    <input
                      type="radio"
                      name="mode"
                      value="modified"
                      checked={trainingMode === "modified"}
                      onChange={() => setTrainingMode("modified")}
                      disabled={loading}
                      className="h-4 w-4 text-blue-600 border-slate-300"
                    />
                    <span><strong>Optimized</strong> – adaptive tuning (steps 50–300).</span>
                  </label>
                </div>
              )}
            </div>

            <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm space-y-4">
              <div className="flex items-center justify-between">
                <h3 className="text-lg font-semibold text-slate-900">Pipeline Stage</h3>
                <span className="text-xs uppercase tracking-wide text-slate-500">Stages</span>
              </div>

              <select
                value={stage}
                onChange={(e) => setStage(e.target.value as any)}
                disabled={loading}
                className="w-full rounded-lg border border-slate-300 px-4 py-3 text-slate-900 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              >
                <option value="full">Full Pipeline (COLMAP + Training)</option>
                <option value="colmap_only">COLMAP Only (Sparse reconstruction)</option>
                <option value="train_only">Training Only (requires existing sparse)</option>
              </select>
              <p className="text-sm text-slate-500">Run all steps or split the flow when resuming.</p>
            </div>
          </div>

          {/* Parameters - left: tabs, right: content */}
          <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm space-y-5">
            <div className="flex items-start justify-between">
              <h3 className="text-lg font-semibold text-slate-900">Process Configuration</h3>
              <span className="text-xs uppercase tracking-wide text-slate-500">COLMAP / Training</span>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-4 gap-4">
              <div className="col-span-1 flex flex-col gap-2">
                <button
                  onClick={() => setConfigTab("colmap")}
                  className={`text-left px-3 py-2 rounded-lg ${configTab === "colmap" ? "bg-slate-100 font-semibold" : "hover:bg-slate-50"}`}
                >COLMAP</button>
                <button
                  onClick={() => setConfigTab("training")}
                  className={`text-left px-3 py-2 rounded-lg ${configTab === "training" ? "bg-slate-100 font-semibold" : "hover:bg-slate-50"}`}
                >Training</button>
              </div>

              <div className="col-span-3">
                {configTab === "training" ? (
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    <div className="space-y-2">
                      <label className="text-sm font-medium text-slate-700">Max Training Steps</label>
                      <input
                        type="number"
                        value={maxSteps}
                        min={100}
                        max={50000}
                        step={100}
                        onChange={(e) => setMaxSteps(parseInt(e.target.value || "300"))}
                        disabled={loading}
                        className="w-full rounded-lg border border-slate-300 px-3 py-2 text-slate-900 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                      />
                      <p className="text-xs text-slate-500">Default 300; lower for quick tests.</p>
                    </div>

                    <div className="space-y-2">
                      <label className="text-sm font-medium text-slate-700">Batch Size</label>
                      <input
                        type="number"
                        value={batchSize}
                        min={1}
                        max={8}
                        onChange={(e) => setBatchSize(parseInt(e.target.value || "1"))}
                        disabled={loading}
                        className="w-full rounded-lg border border-slate-300 px-3 py-2 text-slate-900 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                      />
                      <p className="text-xs text-slate-500">Keep at 1 for single-GPU setups.</p>
                    </div>

                    <div className="space-y-2">
                      <label className="text-sm font-medium text-slate-700">Splat Export Interval (steps)</label>
                      <input
                        type="number"
                        value={splatInterval ?? ""}
                        min={100}
                        step={100}
                        onChange={(e) => setSplatInterval(e.target.value ? parseInt(e.target.value) : undefined)}
                        disabled={loading}
                        className="w-full rounded-lg border border-slate-300 px-3 py-2 text-slate-900 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                      />
                      <p className="text-xs text-slate-500">Exports .splat/.ply during training.</p>
                    </div>

                    <div className="space-y-2">
                      <label className="text-sm font-medium text-slate-700">Preview PNG Interval (steps)</label>
                      <input
                        type="number"
                        value={pngInterval ?? ""}
                        min={50}
                        step={50}
                        onChange={(e) => setPngInterval(e.target.value ? parseInt(e.target.value) : undefined)}
                        disabled={loading}
                        className="w-full rounded-lg border border-slate-300 px-3 py-2 text-slate-900 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                      />
                      <p className="text-xs text-slate-500">Saves preview images for progress.</p>
                    </div>

                    <div className="space-y-3">
                      <label className="flex items-center gap-3 text-sm text-slate-800 cursor-pointer">
                        <input
                          type="checkbox"
                          checked={autoEarlyStop}
                          onChange={(e) => setAutoEarlyStop(e.target.checked)}
                          disabled={loading}
                          className="h-4 w-4 text-blue-600 rounded border-slate-300"
                        />
                        <span><strong>Auto early stop</strong> on loss plateau</span>
                      </label>

                      <label className="flex items-center gap-3 text-sm text-slate-800 cursor-pointer">
                        <input
                          type="checkbox"
                          checked={resume}
                          onChange={(e) => setResume(e.target.checked)}
                          disabled={loading}
                          className="h-4 w-4 text-blue-600 rounded border-slate-300"
                        />
                        <span><strong>Resume from checkpoint</strong> when available</span>
                      </label>
                    </div>
                  </div>
                ) : (
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    <div className="space-y-2">
                      <label className="text-sm font-medium text-slate-700">SIFT Max Image Size</label>
                      <input
                        type="number"
                        value={colmapMaxImageSize ?? ""}
                        min={512}
                        step={100}
                        onChange={(e) => setColmapMaxImageSize(e.target.value ? parseInt(e.target.value) : undefined)}
                        disabled={loading}
                        className="w-full rounded-lg border border-slate-300 px-3 py-2 text-slate-900 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                      />
                      <p className="text-xs text-slate-500">Resize images for feature extraction (pixels).</p>
                    </div>

                    <div className="space-y-2">
                      <label className="text-sm font-medium text-slate-700">SIFT Peak Threshold</label>
                      <input
                        type="number"
                        value={colmapPeakThreshold ?? ""}
                        min={0.001}
                        step={0.001}
                        onChange={(e) => setColmapPeakThreshold(e.target.value ? parseFloat(e.target.value) : undefined)}
                        disabled={loading}
                        className="w-full rounded-lg border border-slate-300 px-3 py-2 text-slate-900 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                      />
                      <p className="text-xs text-slate-500">Higher values → fewer features (faster).</p>
                    </div>

                    <div className="space-y-2">
                      <label className="text-sm font-medium text-slate-700">Guided Matching</label>
                      <label className="flex items-center gap-3 text-sm text-slate-800 cursor-pointer">
                        <input
                          type="checkbox"
                          checked={colmapGuidedMatching}
                          onChange={(e) => setColmapGuidedMatching(e.target.checked)}
                          disabled={loading}
                          className="h-4 w-4 text-blue-600 rounded border-slate-300"
                        />
                        <span className="text-sm text-slate-700">Enable guided matching (slower but more accurate)</span>
                      </label>
                    </div>

                    <div className="space-y-2">
                      <label className="text-sm font-medium text-slate-700">Matching Strategy</label>
                      <select
                        value={colmapMatchingType}
                        onChange={(e) => setColmapMatchingType(e.target.value as any)}
                        disabled={loading}
                        className="w-full rounded-lg border border-slate-300 px-3 py-2 text-slate-900 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                      >
                        <option value="exhaustive">Exhaustive</option>
                        <option value="sequential">Sequential</option>
                      </select>
                      <p className="text-xs text-slate-500">Sequential is faster for ordered captures.</p>
                    </div>

                    <div className="space-y-2">
                      <label className="text-sm font-medium text-slate-700">Mapper Threads</label>
                      <input
                        type="number"
                        value={colmapMapperThreads ?? ""}
                        min={1}
                        max={32}
                        onChange={(e) => setColmapMapperThreads(e.target.value ? parseInt(e.target.value) : undefined)}
                        disabled={loading}
                        className="w-full rounded-lg border border-slate-300 px-3 py-2 text-slate-900 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                      />
                      <p className="text-xs text-slate-500">Increase to use more CPU cores for mapping.</p>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>

          {error && (
            <div className="rounded-lg border border-rose-200 bg-rose-50 px-4 py-3 text-rose-700">
              {error}
            </div>
          )}

          <div className="flex flex-col gap-2">
            <button
              onClick={handleUpload}
              disabled={loading || files.length === 0}
              className="w-full inline-flex items-center justify-center gap-2 rounded-xl bg-blue-600 hover:bg-blue-700 text-white font-semibold px-6 py-4 shadow-md disabled:bg-slate-300 disabled:cursor-not-allowed transition-colors"
            >
              {loading ? (
                comparisonMode ? "Running comparison..." : "Processing..."
              ) : (
                <>
                  <Play className="w-5 h-5" />
                  {comparisonMode ? "Start Comparison" : "Upload & Train"}
                </>
              )}
            </button>
            {comparisonMode && (
              <p className="text-sm text-slate-500 text-center">Comparison mode takes ~2x time (runs baseline and optimized).</p>
            )}
          </div>
        </div>
      </div>
    );
  }
