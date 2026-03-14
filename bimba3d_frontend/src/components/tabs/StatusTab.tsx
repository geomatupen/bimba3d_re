import { useEffect, useState } from "react";
import { StopCircle, AlertCircle, CheckCircle, Clock, Cpu } from "lucide-react";
import { api } from "../../api/client";

interface StatusTabProps {
  projectId: string;
}

interface ProjectStatus {
  // removed can_resume and last_completed_step (resume is now automatic)
  currentStep?: number;
  maxSteps?: number;
  stage?: string;
  message?: string;
  device?: string;
  error?: string;
  stop_requested?: boolean;
  can_resume?: boolean;
  last_completed_step?: number;
  status?: string;
  progress?: number;
}

export default function StatusTab({ projectId }: StatusTabProps) {
  const [status, setStatus] = useState<ProjectStatus | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [stopSubmitting, setStopSubmitting] = useState(false);

  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const res = await api.get(`/projects/${projectId}/status`);
        setStatus(res.data);
      } catch (err) {
        console.error("Failed to fetch status:", err);
      }
    };

    fetchStatus();
    const interval = setInterval(fetchStatus, 3000);
    return () => clearInterval(interval);
  }, [projectId]);

  useEffect(() => {
    if (!status || status.status !== "processing") return;

    let urlHandle: string | null = null;
    let mounted = true;

    const fetchPreview = async () => {
      try {
        const res = await api.get(`/projects/${projectId}/preview`, {
          responseType: "blob",
          headers: { "Cache-Control": "no-store" },
        });
        if (!mounted) return;
        const objectUrl = URL.createObjectURL(res.data);
        if (urlHandle) URL.revokeObjectURL(urlHandle);
        urlHandle = objectUrl;
        setPreviewUrl(objectUrl + `#t=${Date.now()}`);
      } catch {
        // Preview not available yet
      }
    };

    fetchPreview();
    const interval = setInterval(fetchPreview, 5000);
    return () => {
      mounted = false;
      clearInterval(interval);
      if (urlHandle) URL.revokeObjectURL(urlHandle);
    };
  }, [projectId, status?.status]);

  const requestStop = async () => {
    setStopSubmitting(true);
    try {
      await api.post(`/projects/${projectId}/stop`);
    } catch (err) {
      console.error("Stop request failed:", err);
    } finally {
      setStopSubmitting(false);
    }
  };

  if (!status) {
    return (
      <div className="flex items-center justify-center py-20">
        <div className="animate-spin rounded-full h-12 w-12 border-4 border-blue-500 border-t-transparent"></div>
      </div>
    );
  }

  // Helper for ETA formatting
  function formatEta(seconds?: number) {
    if (!seconds || seconds < 1) return null;
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return m > 0 ? `${m}m ${s}s` : `${s}s`;
  }

  // Extract ETA if available
  const eta = (status as any)?.timing?.eta;

  return (
    <div className="max-w-4xl space-y-6">
      {/* Status Overview */}
      <div className="bg-white rounded-xl shadow-md p-8 border border-gray-200">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-xl font-bold text-gray-900">Processing Status</h2>
          <div className="flex items-center gap-2">
            {status.status === "completed" || status.status === "done" ? (
              <CheckCircle className="w-6 h-6 text-green-600" />
            ) : status.status === "processing" ? (
              <Clock className="w-6 h-6 text-blue-600 animate-pulse" />
            ) : status.status === "failed" ? (
              <AlertCircle className="w-6 h-6 text-red-600" />
            ) : null}
            <span className={`px-4 py-2 rounded-lg text-sm font-medium ${
              status.status === "completed" || status.status === "done"
                ? "bg-green-100 text-green-800"
                : status.status === "processing"
                ? "bg-blue-100 text-blue-800"
                : status.status === "failed"
                ? "bg-red-100 text-red-800"
                : status.status === "stopped" || status.status === "stopping"
                ? "bg-yellow-100 text-yellow-800"
                : "bg-gray-100 text-gray-800"
            }`}>
              {status.status}
            </span>
          </div>
        </div>

        {/* Main Progress Message (single line) */}
        {status.message && (
          <div className="mb-2 text-sm text-gray-700 font-medium">
            {status.message}
            {eta && (
              <span className="ml-2 text-xs text-gray-500">(Time remaining: {formatEta(eta)})</span>
            )}
          </div>
        )}

        {/* Progress Bar */}
        <div className="mb-6">
          <div className="flex justify-between text-sm text-gray-600 mb-2">
            <span>Progress</span>
            <span className="font-medium">{status.progress}%</span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-3">
            <div
              className="bg-blue-600 h-3 rounded-full transition-all duration-300"
              style={{ width: `${status.progress}%` }}
            ></div>
          </div>
        </div>

        {/* Device Info */}
        {status.device && (
          <div className="flex items-center gap-2 text-sm text-gray-600 mb-4">
            <Cpu className="w-4 h-4" />
            <span>Running on: <strong>{status.device}</strong></span>
          </div>
        )}

        {/* Error */}
        {status.error && (
          <div className="bg-red-50 border border-red-200 text-red-800 px-4 py-3 rounded-lg mb-4">
            <strong>Error:</strong> {status.error}
          </div>
        )}

        {/* Stop Requested Warning */}
        {status.stop_requested && (
          <div className="bg-yellow-50 border border-yellow-200 text-yellow-800 px-4 py-3 rounded-lg mb-4">
            Stop requested. Finalizing current step and exporting...
          </div>
        )}

        {/* Resume Info */}
        {(status.status === "completed" || status.status === "stopped") && status.can_resume && status.last_completed_step && (
          <div className="bg-blue-50 border border-blue-200 text-blue-800 px-4 py-3 rounded-lg mb-4">
            💾 Checkpoint available at step {status.last_completed_step}. You can resume training from the Process tab.
          </div>
        )}

        {/* Manual Stop Button */}
        {(status.status === "processing" || status.status === "stopping") && (
          <button
            onClick={requestStop}
            disabled={stopSubmitting || status.stop_requested}
            className="w-full px-6 py-3 bg-red-600 hover:bg-red-700 disabled:bg-gray-300 disabled:cursor-not-allowed text-white font-medium rounded-lg transition-colors shadow-md flex items-center justify-center gap-2"
          >
            <StopCircle className="w-5 h-5" />
            {stopSubmitting ? "Requesting Stop..." : "Stop Processing"}
          </button>
        )}
      </div>

      {/* Live Preview */}
      {previewUrl && (
        <div className="bg-white rounded-xl shadow-md p-8 border border-gray-200">
          <h3 className="text-lg font-bold text-gray-900 mb-4">Live Preview</h3>
          <img 
            src={previewUrl} 
            alt="Training Preview" 
            className="w-full rounded-lg shadow-md"
          />
          <p className="text-sm text-gray-500 mt-3 text-center">
            Auto-refreshes every 5 seconds during training
          </p>
        </div>
      )}
    </div>
  );
}
