import { useEffect, useState, useRef } from "react";
import { FileText, RefreshCw, Download } from "lucide-react";
import { api } from "../../api/client";

interface LogsTabProps {
  projectId: string;
}

export default function LogsTab({ projectId }: LogsTabProps) {
  const [logs, setLogs] = useState<string>("");
  const [loading, setLoading] = useState(true);
  const [autoScroll, setAutoScroll] = useState(true);
  const logsEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const fetchLogs = async () => {
      try {
        const res = await api.get(`/projects/${projectId}/logs?lines=500`);
        setLogs(res.data.logs || "No logs available yet.");
        setLoading(false);
      } catch (err) {
        console.error("Failed to fetch logs:", err);
        setLogs("Failed to load logs.");
        setLoading(false);
      }
    };

    fetchLogs();
    const interval = setInterval(fetchLogs, 5000);
    return () => clearInterval(interval);
  }, [projectId]);

  useEffect(() => {
    if (autoScroll && logsEndRef.current) {
      logsEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [logs, autoScroll]);

  const downloadLogs = () => {
    const blob = new Blob([logs], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${projectId}_logs.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const refreshLogs = async () => {
    setLoading(true);
    try {
      const res = await api.get(`/projects/${projectId}/logs?lines=500`);
      setLogs(res.data.logs || "No logs available yet.");
    } catch (err) {
      console.error("Failed to refresh logs:", err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-6xl">
      <div className="bg-white rounded-xl shadow-md border border-gray-200">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-gray-200">
          <div className="flex items-center gap-3">
            <FileText className="w-6 h-6 text-gray-600" />
            <h2 className="text-xl font-bold text-gray-900">Processing Logs</h2>
          </div>
          <div className="flex items-center gap-3">
            <label className="flex items-center gap-2 text-sm text-gray-600 cursor-pointer">
              <input
                type="checkbox"
                checked={autoScroll}
                onChange={(e) => setAutoScroll(e.target.checked)}
                className="w-4 h-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
              />
              Auto-scroll
            </label>
            <button
              onClick={refreshLogs}
              disabled={loading}
              className="px-4 py-2 bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-lg transition-colors flex items-center gap-2 disabled:opacity-50"
            >
              <RefreshCw className={`w-4 h-4 ${loading ? "animate-spin" : ""}`} />
              Refresh
            </button>
            <button
              onClick={downloadLogs}
              className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors flex items-center gap-2"
            >
              <Download className="w-4 h-4" />
              Download
            </button>
          </div>
        </div>

        {/* Logs Content */}
        <div className="p-0">
          <pre className="bg-gray-900 text-green-400 p-6 rounded-b-xl overflow-x-auto font-mono text-sm h-[600px] overflow-y-auto">
            {logs}
            <div ref={logsEndRef} />
          </pre>
        </div>
      </div>
    </div>
  );
}
