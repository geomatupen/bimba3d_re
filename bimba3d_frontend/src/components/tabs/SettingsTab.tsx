import { useState, useEffect } from "react";
import axios from "axios";
import { Settings, Info, Save } from "lucide-react";

const API_BASE = "http://localhost:8005";

interface SettingsTabProps {
  projectId: string;
}

interface SharedConfig {
  version?: number;
  base_run_id?: string;
  updated_at?: string;
  shared?: {
    ai_input_mode?: string;
    ai_selector_strategy?: string;
    max_steps?: number;
    eval_interval?: number;
    log_interval?: number;
    densify_until_iter?: number;
    images_max_size?: number;
    images_resize_enabled?: boolean;
    colmap_max_image_size?: number;
    // ... other shared settings
    [key: string]: any;
  };
}

export default function SettingsTab({ projectId }: SettingsTabProps) {
  const [config, setConfig] = useState<SharedConfig | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Editable state
  const [aiInputMode, setAiInputMode] = useState("");
  const [aiSelectorStrategy, setAiSelectorStrategy] = useState("");
  const [maxSteps, setMaxSteps] = useState(5000);
  const [evalInterval, setEvalInterval] = useState(1000);
  const [logInterval, setLogInterval] = useState(100);
  const [densifyUntil, setDensifyUntil] = useState(4000);
  const [imagesMaxSize, setImagesMaxSize] = useState(1600);

  const [hasChanges, setHasChanges] = useState(false);

  useEffect(() => {
    loadSharedConfig();
  }, [projectId]);

  const loadSharedConfig = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await axios.get(`${API_BASE}/projects/${projectId}/shared-config`);
      const data: SharedConfig = response.data;
      setConfig(data);

      // Populate form from loaded config
      const shared = data.shared || {};
      setAiInputMode(shared.ai_input_mode || "");
      setAiSelectorStrategy(shared.ai_selector_strategy || "");
      setMaxSteps(shared.max_steps || 5000);
      setEvalInterval(shared.eval_interval || 1000);
      setLogInterval(shared.log_interval || 100);
      setDensifyUntil(shared.densify_until_iter || 4000);
      setImagesMaxSize(shared.images_max_size || 1600);

      setHasChanges(false);
    } catch (err: any) {
      console.error("Failed to load shared config:", err);
      setError(err.response?.data?.detail || "Failed to load settings");
    } finally {
      setLoading(false);
    }
  };

  const handleSave = async () => {
    setSaving(true);
    setError(null);

    try {
      const payload = {
        shared: {
          ai_input_mode: aiInputMode || undefined,
          ai_selector_strategy: aiSelectorStrategy || undefined,
          max_steps: maxSteps,
          eval_interval: evalInterval,
          log_interval: logInterval,
          densify_until_iter: densifyUntil,
          images_max_size: imagesMaxSize,
        },
      };

      await axios.patch(`${API_BASE}/projects/${projectId}/shared-config`, payload);

      // Reload to get updated version
      await loadSharedConfig();

      alert("Settings saved successfully!");
    } catch (err: any) {
      console.error("Failed to save settings:", err);
      setError(err.response?.data?.detail || "Failed to save settings");
    } finally {
      setSaving(false);
    }
  };

  const markChanged = () => setHasChanges(true);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-gray-500">Loading settings...</div>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold flex items-center gap-2">
            <Settings className="w-6 h-6" />
            Project Settings
          </h2>
          <p className="text-sm text-gray-600 mt-1">
            Configure default training parameters for all runs in this project
          </p>
        </div>

        {config?.version && (
          <div className="text-sm text-gray-500">
            Version: {config.version}
            {config.updated_at && (
              <div className="text-xs">
                Updated: {new Date(config.updated_at).toLocaleString()}
              </div>
            )}
          </div>
        )}
      </div>

      {error && (
        <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded">
          {error}
        </div>
      )}

      {/* Info Banner */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 flex gap-3">
        <Info className="w-5 h-5 text-blue-600 flex-shrink-0 mt-0.5" />
        <div className="text-sm text-blue-800">
          <strong>Shared Configuration:</strong> These settings apply to all future training runs in this project.
          Existing runs will keep their original settings. Changes only affect new runs after saving.
        </div>
      </div>

      {/* Settings Form */}
      <div className="bg-white border border-gray-200 rounded-lg p-6 space-y-6">
        {/* AI Configuration */}
        <div>
          <h3 className="text-lg font-semibold mb-4 pb-2 border-b">AI Configuration</h3>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                AI Input Mode
              </label>
              <select
                value={aiInputMode}
                onChange={(e) => {
                  setAiInputMode(e.target.value);
                  markChanged();
                }}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="">Not set (use per-run default)</option>
                <option value="exif_only">EXIF Only</option>
                <option value="exif_plus_flight_plan">EXIF + Flight Plan</option>
                <option value="exif_plus_flight_plan_plus_external">EXIF + Flight Plan + External</option>
              </select>
              <p className="text-xs text-gray-500 mt-1">
                Feature set for contextual learning
              </p>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Selector Strategy
              </label>
              <select
                value={aiSelectorStrategy}
                onChange={(e) => {
                  setAiSelectorStrategy(e.target.value);
                  markChanged();
                }}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="">Not set (use per-run default)</option>
                <option value="contextual_continuous">Contextual Continuous</option>
                <option value="continuous_bandit_linear">Continuous Bandit</option>
                <option value="preset_bias">Preset Bias</option>
              </select>
              <p className="text-xs text-gray-500 mt-1">
                Parameter optimization strategy
              </p>
            </div>
          </div>
        </div>

        {/* Training Parameters */}
        <div>
          <h3 className="text-lg font-semibold mb-4 pb-2 border-b">Training Parameters</h3>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Max Steps
              </label>
              <input
                type="number"
                value={maxSteps}
                onChange={(e) => {
                  setMaxSteps(Number(e.target.value));
                  markChanged();
                }}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Eval Interval
              </label>
              <input
                type="number"
                value={evalInterval}
                onChange={(e) => {
                  setEvalInterval(Number(e.target.value));
                  markChanged();
                }}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Log Interval
              </label>
              <input
                type="number"
                value={logInterval}
                onChange={(e) => {
                  setLogInterval(Number(e.target.value));
                  markChanged();
                }}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Densify Until Iter
              </label>
              <input
                type="number"
                value={densifyUntil}
                onChange={(e) => {
                  setDensifyUntil(Number(e.target.value));
                  markChanged();
                }}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Images Max Size
              </label>
              <input
                type="number"
                value={imagesMaxSize}
                onChange={(e) => {
                  setImagesMaxSize(Number(e.target.value));
                  markChanged();
                }}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
          </div>
        </div>

        {/* Current Config Preview */}
        {config?.shared && (
          <div>
            <h3 className="text-lg font-semibold mb-4 pb-2 border-b">Current Configuration (JSON)</h3>
            <pre className="bg-gray-50 border border-gray-200 rounded p-4 text-xs overflow-auto max-h-64">
              {JSON.stringify(config.shared, null, 2)}
            </pre>
          </div>
        )}

        {/* Save Button */}
        <div className="flex items-center justify-end gap-4 pt-4 border-t">
          {hasChanges && (
            <span className="text-sm text-amber-600">
              You have unsaved changes
            </span>
          )}

          <button
            onClick={handleSave}
            disabled={saving || !hasChanges}
            className="flex items-center gap-2 px-6 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            <Save className="w-4 h-4" />
            {saving ? "Saving..." : "Save Settings"}
          </button>
        </div>
      </div>

      {/* Help Text */}
      <div className="text-sm text-gray-600 bg-gray-50 border border-gray-200 rounded-lg p-4">
        <strong>Note:</strong> If a setting is "Not set", the system will use per-run defaults or values
        specified when starting individual training runs. Setting values here establishes project-wide
        defaults that persist across all future runs.
      </div>
    </div>
  );
}
