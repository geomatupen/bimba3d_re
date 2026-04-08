import { useEffect, useRef, useState } from "react";
import { Eye, Download, AlertCircle } from "lucide-react";
import ThreeDViewer from "../ThreeDViewer";
import { api } from "../../api/client";

interface ViewerTabProps {
  projectId: string;
  snapshotUrl?: string | null;
  engineOverride?: string | null;
  modelUrlOverride?: string | null;
  modelLabelOverride?: string | null;
}

interface EngineSource {
  name: string;
  label: string;
  finalViewerUrl: string | null;
  bestViewerUrl: string | null;
  format: string | null;
}

export default function ViewerTab({ projectId, snapshotUrl, engineOverride, modelUrlOverride, modelLabelOverride }: ViewerTabProps) {
  const [splatFile, setSplatFile] = useState<string | null>(null);
  const [baseFormat, setBaseFormat] = useState<string>("splat");
  const [engineSources, setEngineSources] = useState<EngineSource[]>([]);
  const [selectedEngine, setSelectedEngine] = useState<string | null>(null);
  const [selectedVariant, setSelectedVariant] = useState<"final" | "best">("final");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const allowLocalEngineSelection = !engineOverride;
  const engineOverrideRef = useRef<string | null>(engineOverride ?? null);

  useEffect(() => {
    engineOverrideRef.current = engineOverride ?? null;
  }, [engineOverride]);

  useEffect(() => {
    if (!engineOverride) return;
    if (engineSources.some((engine) => engine.name === engineOverride)) {
      setSelectedEngine(engineOverride);
    }
  }, [engineOverride, engineSources]);

  useEffect(() => {
    const fetchFiles = async () => {
      try {
        setError(null);
        const res = await api.get(`/projects/${projectId}/files`);
        const files = (res.data.files ?? {}) as Record<string, any>;
        const baseURL = (api.defaults.baseURL || "").replace(/\/$/, "");
        const absoluteUrl = (path?: string | null) => {
          if (!path) return null;
          if (/^https?:/i.test(path)) return path;
          return `${baseURL}${path}`;
        };

        const enginesData = (files.engines || {}) as Record<string, any>;
        const nextEngineSources: EngineSource[] = Object.entries(enginesData).map(([name, bundle]) => {
          const finalSplatUrl = absoluteUrl(bundle?.splats?.url);
          const bestSplatUrl = absoluteUrl(bundle?.best_splat?.url);
          return { 
            name,
            label: name
              .replace(/_/g, " ")
              .replace(/\b\w/g, (char: string) => char.toUpperCase()),
            finalViewerUrl: finalSplatUrl,
            bestViewerUrl: bestSplatUrl,
            format: bundle?.splats?.format || null,
          };
        }).filter((engine) => !!engine.finalViewerUrl || !!engine.bestViewerUrl);
        setEngineSources(nextEngineSources);
        const overrideTarget = engineOverrideRef.current;
        setSelectedEngine((prev) => {
          if (overrideTarget && nextEngineSources.some((engine) => engine.name === overrideTarget)) {
            return overrideTarget;
          }
          if (prev && nextEngineSources.some((engine) => engine.name === prev)) {
            return prev;
          }
          return nextEngineSources[0]?.name ?? null;
        });

        if (files.splats?.url) {
          setSplatFile(absoluteUrl(files.splats.url));
          setBaseFormat(files.splats.format || "splat");
        } else {
          setSplatFile(`${baseURL}/projects/${projectId}/download/splats`);
          setBaseFormat("splat");
        }

        setLoading(false);
      } catch (err) {
        console.error("Failed to fetch files:", err);
        setError("Failed to load viewer files");
        setLoading(false);
      }
    };

    fetchFiles();
  }, [projectId]);

  if (loading) {
    return (
      <div className="flex items-center justify-center py-20">
        <div className="animate-spin rounded-full h-12 w-12 border-4 border-blue-500 border-t-transparent"></div>
      </div>
    );
  }

  const selectedEngineSource = selectedEngine
    ? engineSources.find((engine) => engine.name === selectedEngine)
    : null;
  const activeEngineFinalUrl = selectedEngineSource?.finalViewerUrl || null;
  const activeEngineBestUrl = selectedEngineSource?.bestViewerUrl || null;
  const activeEngineUrl = selectedVariant === "best"
    ? (activeEngineBestUrl || activeEngineFinalUrl)
    : (activeEngineFinalUrl || activeEngineBestUrl);
  const activeViewerUrl = snapshotUrl || modelUrlOverride || activeEngineUrl || splatFile || "";
  const downloadLabel = selectedEngineSource?.format || baseFormat;
  const downloadUrl = modelUrlOverride || activeEngineUrl || splatFile;

  if (error) {
    return (
      <div className="bg-white rounded-xl shadow-md p-8 border border-gray-200">
        <div className="flex items-center gap-3 text-red-600">
          <AlertCircle className="w-6 h-6" />
          <p>{error}</p>
        </div>
      </div>
    );
  }

  if (!loading && !activeViewerUrl) {
    return (
      <div className="bg-white rounded-xl shadow-md p-8 border border-gray-200">
        <div className="flex items-center gap-3 text-gray-600">
          <AlertCircle className="w-6 h-6" />
          <p>No output files available for viewing yet.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-7xl">
      <div className="bg-white rounded-xl shadow-md border border-gray-200 overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-gray-200">
          <div className="flex items-center gap-3">
            <Eye className="w-6 h-6 text-gray-600" />
            <h2 className="text-xl font-bold text-gray-900">3D Viewer</h2>
          </div>
          <div className="flex items-center gap-3">
            {allowLocalEngineSelection && engineSources.length > 0 && (
              <div className="flex items-center gap-2">
                <label className="text-sm font-medium text-gray-600">Engine</label>
                <select
                  value={selectedEngine ?? ""}
                  onChange={(event) => setSelectedEngine(event.target.value || null)}
                  className="border border-gray-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  {engineSources.map((engine) => (
                    <option key={engine.name} value={engine.name}>
                      {engine.label}
                    </option>
                  ))}
                </select>
              </div>
            )}
            {!snapshotUrl && !modelUrlOverride && selectedEngineSource && (selectedEngineSource.bestViewerUrl || selectedEngineSource.finalViewerUrl) && (
              <div className="flex items-center gap-2">
                <label className="text-sm font-medium text-gray-600">Layer</label>
                <select
                  value={selectedVariant}
                  onChange={(event) => setSelectedVariant((event.target.value as "final" | "best") || "final")}
                  className="border border-gray-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="final">Final Splat</option>
                  {selectedEngineSource.bestViewerUrl && <option value="best">Best Splat</option>}
                </select>
              </div>
            )}
            {!allowLocalEngineSelection && selectedEngineSource && (
              <div className="text-xs font-medium text-gray-600">
                Engine: <span className="text-gray-900">{selectedEngineSource.label}</span>
              </div>
            )}
            {!snapshotUrl && modelLabelOverride && (
              <div className="text-xs font-medium text-gray-600">
                Layer: <span className="text-gray-900">{modelLabelOverride}</span>
              </div>
            )}
            {snapshotUrl && (
              <a
                href={snapshotUrl}
                download
                className="px-4 py-2 bg-amber-600 hover:bg-amber-700 text-white rounded-lg transition-colors flex items-center gap-2"
              >
                <Download className="w-4 h-4" />
                Snapshot
              </a>
            )}
            {downloadUrl && (
              <a
                href={downloadUrl}
                download
                className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors flex items-center gap-2"
              >
                <Download className="w-4 h-4" />
                Latest .{downloadLabel || "splat"}
              </a>
            )}
          </div>
        </div>

        {/* Viewer */}
        <div className="bg-gray-900 h-[700px]">
          <ThreeDViewer splatsUrl={activeViewerUrl} />
        </div>

        {/* Instructions */}
        <div className="bg-gray-50 p-4 text-sm text-gray-600">
          <strong>Controls:</strong> Left click + drag to rotate | Right click + drag to pan | Scroll to zoom
        </div>
      </div>
    </div>
  );
}
