import { Info as LucideInfo } from "lucide-react";

export type SessionExecutionMode = "train" | "test";

interface ReusableModelEntryOption {
  model_id: string;
  model_name?: string | null;
}

interface CoreAiSessionTestControlsProps {
  sessionExecutionMode: SessionExecutionMode;
  onSessionExecutionModeChange: (mode: SessionExecutionMode) => void;
  sourceModelId: string;
  onSourceModelIdChange: (modelId: string) => void;
  reusableModels: ReusableModelEntryOption[];
  reusableModelsLoading: boolean;
  reusableModelsError: string | null;
  emptyStateLabel?: string;
  onSelectInfoKey: (key: string) => void;
}

const Info = (props: React.ComponentProps<typeof LucideInfo>) => (
  <LucideInfo className={props.className ? `${props.className} w-3 h-3` : "w-3 h-3"} {...props} />
);

export default function CoreAiSessionTestControls({
  sessionExecutionMode,
  onSessionExecutionModeChange,
  sourceModelId,
  onSourceModelIdChange,
  reusableModels,
  reusableModelsLoading,
  reusableModelsError,
  emptyStateLabel,
  onSelectInfoKey,
}: CoreAiSessionTestControlsProps) {
  return (
    <>
      <div className="md:col-span-2">
        <label className="flex items-center justify-between text-[11px] font-medium text-slate-600 mb-0.5">
          <span>Session mode</span>
          <button
            type="button"
            onClick={() => onSelectInfoKey("session_execution_mode")}
            className="p-1 text-slate-400 hover:text-slate-600"
          >
            <Info />
          </button>
        </label>
        <div className="inline-flex items-center gap-2 rounded-md border border-slate-300 bg-white px-2 py-1">
          <span className={`text-[11px] font-medium ${sessionExecutionMode === "train" ? "text-blue-700" : "text-slate-500"}`}>
            Train
          </span>
          <button
            type="button"
            role="switch"
            aria-checked={sessionExecutionMode === "test"}
            aria-label="Toggle session mode between train and test"
            onClick={() => onSessionExecutionModeChange(sessionExecutionMode === "train" ? "test" : "train")}
            className={`relative inline-flex h-6 w-10 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500/60 ${
              sessionExecutionMode === "test" ? "bg-blue-600" : "bg-slate-300"
            }`}
          >
            <span
              className={`absolute left-1 top-1 h-4 w-4 rounded-full border border-slate-300 bg-white shadow-sm transition-transform ${
                sessionExecutionMode === "test" ? "translate-x-4" : "translate-x-0"
              }`}
            />
          </button>
          <span className={`text-[11px] font-medium ${sessionExecutionMode === "test" ? "text-blue-700" : "text-slate-500"}`}>
            Test
          </span>
        </div>
      </div>

      {sessionExecutionMode === "test" && (
        <div className="md:col-span-2 rounded-md border border-blue-200 bg-blue-50 p-2">
          <label className="flex items-center justify-between text-[11px] font-medium text-slate-700 mb-0.5">
            <span>Model for test (required)</span>
            <button
              type="button"
              onClick={() => onSelectInfoKey("source_model_id")}
              className="p-1 text-slate-400 hover:text-slate-600"
            >
              <Info />
            </button>
          </label>
          <select
            value={sourceModelId}
            onChange={(e) => onSourceModelIdChange(e.target.value)}
            disabled={reusableModelsLoading || reusableModels.length === 0}
            className="w-full px-2 py-1.5 text-xs border border-slate-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-slate-100 disabled:text-slate-500"
          >
            <option value="">
              {reusableModelsLoading
                ? "Loading models..."
                : reusableModels.length > 0
                  ? "Select reusable model"
                  : (emptyStateLabel || "No elevated models available")}
            </option>
            {sourceModelId && !reusableModels.some((item) => item.model_id === sourceModelId) && (
              <option value={sourceModelId}>{sourceModelId} (saved)</option>
            )}
            {reusableModels.map((item) => (
              <option key={item.model_id} value={item.model_id}>
                {item.model_name || item.model_id}
              </option>
            ))}
          </select>
          <p className="mt-1 text-[10px] text-slate-600">
            Test mode hides batch/warmup controls and always starts as reuse with the selected model.
          </p>
          {reusableModelsError && <p className="mt-1 text-[10px] text-red-600">{reusableModelsError}</p>}
        </div>
      )}
    </>
  );
}
