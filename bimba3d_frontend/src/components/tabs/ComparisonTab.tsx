import { useEffect, useMemo, useState } from "react";
import { api } from "../../api/client";

interface ComparisonTabProps {
  currentProjectId: string;
}

interface ProjectListItem {
  project_id: string;
  name?: string | null;
  status: string;
}

interface SummaryPayload {
  project_id: string;
  name?: string | null;
  status?: string;
  mode?: string;
  engine?: string | null;
  metrics?: Record<string, number | null | undefined>;
  tuning?: {
    initial?: Record<string, number | null | undefined>;
    final?: Record<string, number | null | undefined>;
    end_params?: Record<string, number | null | undefined>;
    end_step?: number | null;
    runs?: number | null;
    history_count?: number;
    history?: Array<{
      step?: number;
      adjustments?: string[];
      convergence?: Record<string, unknown>;
      instability?: Record<string, unknown>;
      params?: Record<string, number | null | undefined>;
    }>;
  };
  loss_milestones?: Record<string, number | null | undefined>;
  preview_url?: string | null;
  eval_points?: number;
}

const metricRows: Array<{ key: string; label: string; lowerIsBetter?: boolean }> = [
  { key: "convergence_speed", label: "Convergence Speed" },
  { key: "final_loss", label: "Final Loss", lowerIsBetter: true },
  { key: "lpips_mean", label: "LPIPS", lowerIsBetter: true },
  { key: "sharpness_mean", label: "Sharpness" },
  { key: "num_gaussians", label: "Gaussian Count" },
];

const tuningRows: Array<{ key: string; label: string }> = [
  { key: "lr_mult", label: "Learning Rate Mult" },
  { key: "opacity_lr_mult", label: "Opacity LR Mult" },
  { key: "sh_lr_mult", label: "SH LR Mult" },
  { key: "densify_threshold", label: "Densify Threshold" },
  { key: "position_lr_mult", label: "Position LR Mult" },
];

function fmt(v: unknown): string {
  if (typeof v !== "number" || Number.isNaN(v)) return "-";
  if (Math.abs(v) >= 1000) return v.toLocaleString();
  if (Math.abs(v) >= 1) return v.toFixed(4);
  return v.toPrecision(4);
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

function fmtMaybe(v: unknown): string {
  if (typeof v === "boolean") return v ? "true" : "false";
  if (typeof v === "number") return fmt(v);
  if (v === null || typeof v === "undefined") return "-";
  return String(v);
}

export default function ComparisonTab({ currentProjectId }: ComparisonTabProps) {
  const [projects, setProjects] = useState<ProjectListItem[]>([]);
  const [leftId, setLeftId] = useState<string>(currentProjectId);
  const [rightId, setRightId] = useState<string>("");
  const [leftSummary, setLeftSummary] = useState<SummaryPayload | null>(null);
  const [rightSummary, setRightSummary] = useState<SummaryPayload | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

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
    if (!leftId || !rightId) return;
    let mounted = true;
    const loadSummaries = async () => {
      try {
        setLoading(true);
        setError(null);
        const [leftRes, rightRes] = await Promise.all([
          api.get(`/projects/${leftId}/experiment-summary`),
          api.get(`/projects/${rightId}/experiment-summary`),
        ]);
        if (!mounted) return;
        setLeftSummary(leftRes.data as SummaryPayload);
        setRightSummary(rightRes.data as SummaryPayload);
      } catch (err) {
        if (!mounted) return;
        setError(err instanceof Error ? err.message : "Failed to load comparison summary");
      } finally {
        if (mounted) setLoading(false);
      }
    };
    loadSummaries();
    return () => {
      mounted = false;
    };
  }, [leftId, rightId]);

  const previewLeft = useMemo(() => {
    if (!leftSummary?.preview_url) return null;
    return `${api.defaults.baseURL}${leftSummary.preview_url}`;
  }, [leftSummary]);

  const previewRight = useMemo(() => {
    if (!rightSummary?.preview_url) return null;
    return `${api.defaults.baseURL}${rightSummary.preview_url}`;
  }, [rightSummary]);

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

  const leftHistory = leftSummary?.tuning?.history ?? [];
  const rightHistory = rightSummary?.tuning?.history ?? [];

  return (
    <div className="space-y-4">
      <div className="bg-white rounded-lg border border-slate-200 p-4">
        <h3 className="text-lg font-semibold text-slate-900 mb-3">Run Comparison</h3>
        <p className="text-sm text-slate-600 mb-4">Pick two projects and compare metrics, tuning values, and preview snapshots side-by-side.</p>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-xs font-semibold text-slate-600 mb-1">Left project</label>
            <select value={leftId} onChange={(e) => setLeftId(e.target.value)} className="w-full px-3 py-2 border border-slate-300 rounded-lg">
              {options.map((option) => (
                <option key={`left-${option.value}`} value={option.value}>{option.label}</option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-xs font-semibold text-slate-600 mb-1">Right project</label>
            <select value={rightId} onChange={(e) => setRightId(e.target.value)} className="w-full px-3 py-2 border border-slate-300 rounded-lg">
              <option value="">Select project</option>
              {options.map((option) => (
                <option key={`right-${option.value}`} value={option.value}>{option.label}</option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {error && <div className="bg-rose-50 border border-rose-200 text-rose-700 px-4 py-3 rounded-lg text-sm">{error}</div>}

      {loading && <div className="text-sm text-slate-500">Loading comparison data...</div>}

      {!loading && leftSummary && rightSummary && (
        <>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <div className="bg-white rounded-lg border border-slate-200 p-4">
              <p className="text-sm font-semibold text-slate-800">{leftSummary.name || leftSummary.project_id}</p>
              <p className="text-xs text-slate-500">mode: {leftSummary.mode || "-"} | engine: {leftSummary.engine || "-"}</p>
              <p className="text-xs text-slate-500">eval points: {leftSummary.eval_points ?? 0}</p>
              <p className="text-xs text-slate-500">tune end step: {leftSummary.tuning?.end_step ?? "-"}</p>
            </div>
            <div className="bg-white rounded-lg border border-slate-200 p-4">
              <p className="text-sm font-semibold text-slate-800">{rightSummary.name || rightSummary.project_id}</p>
              <p className="text-xs text-slate-500">mode: {rightSummary.mode || "-"} | engine: {rightSummary.engine || "-"}</p>
              <p className="text-xs text-slate-500">eval points: {rightSummary.eval_points ?? 0}</p>
              <p className="text-xs text-slate-500">tune end step: {rightSummary.tuning?.end_step ?? "-"}</p>
            </div>
          </div>

          <div className="bg-white rounded-lg border border-slate-200 overflow-hidden">
            <table className="w-full text-sm">
              <thead className="bg-slate-50 text-slate-700">
                <tr>
                  <th className="text-left px-4 py-3">Metric</th>
                  <th className="text-left px-4 py-3">Left</th>
                  <th className="text-left px-4 py-3">Right</th>
                  <th className="text-left px-4 py-3">Delta (Right - Left)</th>
                </tr>
              </thead>
              <tbody>
                {metricRows.map((row) => {
                  const leftVal = leftSummary.metrics?.[row.key] as number | undefined;
                  const rightVal = rightSummary.metrics?.[row.key] as number | undefined;
                  return (
                    <tr key={row.key} className="border-t border-slate-100">
                      <td className="px-4 py-2 text-slate-700">{row.label}</td>
                      <td className="px-4 py-2 text-slate-900">{fmt(leftVal)}</td>
                      <td className="px-4 py-2 text-slate-900">{fmt(rightVal)}</td>
                      <td className="px-4 py-2 text-slate-600">{deltaText(leftVal, rightVal, row.lowerIsBetter)}</td>
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
                    <th className="text-left px-4 py-3">Loss Milestone (500-step)</th>
                    <th className="text-left px-4 py-3">Left</th>
                    <th className="text-left px-4 py-3">Right</th>
                    <th className="text-left px-4 py-3">Delta (Right - Left)</th>
                  </tr>
                </thead>
                <tbody>
                  {milestoneKeys.map((key) => {
                    const leftVal = leftSummary.loss_milestones?.[key] as number | undefined;
                    const rightVal = rightSummary.loss_milestones?.[key] as number | undefined;
                    const stepLabel = key.replace("loss_at_", "");
                    return (
                      <tr key={key} className="border-t border-slate-100">
                        <td className="px-4 py-2 text-slate-700">Loss @ {stepLabel}</td>
                        <td className="px-4 py-2 text-slate-900">{fmt(leftVal)}</td>
                        <td className="px-4 py-2 text-slate-900">{fmt(rightVal)}</td>
                        <td className="px-4 py-2 text-slate-600">{deltaText(leftVal, rightVal, true)}</td>
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
                  <th className="text-left px-4 py-3">Tuning Param (tune-end)</th>
                  <th className="text-left px-4 py-3">Left</th>
                  <th className="text-left px-4 py-3">Right</th>
                  <th className="text-left px-4 py-3">Delta</th>
                </tr>
              </thead>
              <tbody>
                {tuningRows.map((row) => {
                  const leftVal = (leftSummary.tuning?.end_params?.[row.key] ?? leftSummary.tuning?.final?.[row.key]) as number | undefined;
                  const rightVal = (rightSummary.tuning?.end_params?.[row.key] ?? rightSummary.tuning?.final?.[row.key]) as number | undefined;
                  return (
                    <tr key={row.key} className="border-t border-slate-100">
                      <td className="px-4 py-2 text-slate-700">{row.label}</td>
                      <td className="px-4 py-2 text-slate-900">{fmt(leftVal)}</td>
                      <td className="px-4 py-2 text-slate-900">{fmt(rightVal)}</td>
                      <td className="px-4 py-2 text-slate-600">{deltaText(leftVal, rightVal, false)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <div className="bg-white rounded-lg border border-slate-200 p-4">
              <p className="text-sm font-semibold text-slate-800 mb-2">Left tuning history</p>
              {leftHistory.length === 0 ? (
                <p className="text-xs text-slate-500">No tuning history (likely baseline).</p>
              ) : (
                <div className="space-y-2 max-h-64 overflow-auto text-xs">
                  {leftHistory.map((item, idx) => (
                    <div key={`left-h-${idx}`} className="border border-slate-200 rounded p-2">
                      <div className="font-semibold text-slate-700">Step {item.step ?? "-"}</div>
                      <div className="text-slate-600">Adjustments: {(item.adjustments || []).join(", ") || "none"}</div>
                      <div className="text-slate-500">Reason: {item.instability?.has_issues ? "instability" : item.convergence?.has_issues ? "convergence" : "none"}</div>
                      <div className="text-slate-500">convergence_speed proxy (loss_trend): {fmtMaybe(item.convergence?.loss_trend)}</div>
                      <div className="text-slate-500">loss_variance: {fmtMaybe(item.convergence?.loss_variance)} | loss_plateau: {fmtMaybe(item.convergence?.loss_plateau)} | slow_convergence: {fmtMaybe(item.convergence?.slow_convergence)}</div>
                      <div className="text-slate-500">max_grad: {fmtMaybe(item.instability?.max_grad)} | loss_spikes: {fmtMaybe(item.instability?.loss_spikes)} | gradient_explosion: {fmtMaybe(item.instability?.gradient_explosion)}</div>
                      {item.params && (
                        <div className="text-slate-500">params: lr={fmtMaybe(item.params.lr_mult)}, opacity_lr={fmtMaybe(item.params.opacity_lr_mult)}, sh_lr={fmtMaybe(item.params.sh_lr_mult)}, pos_lr={fmtMaybe(item.params.position_lr_mult)}, densify_th={fmtMaybe(item.params.densify_threshold)}</div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
            <div className="bg-white rounded-lg border border-slate-200 p-4">
              <p className="text-sm font-semibold text-slate-800 mb-2">Right tuning history</p>
              {rightHistory.length === 0 ? (
                <p className="text-xs text-slate-500">No tuning history (likely baseline).</p>
              ) : (
                <div className="space-y-2 max-h-64 overflow-auto text-xs">
                  {rightHistory.map((item, idx) => (
                    <div key={`right-h-${idx}`} className="border border-slate-200 rounded p-2">
                      <div className="font-semibold text-slate-700">Step {item.step ?? "-"}</div>
                      <div className="text-slate-600">Adjustments: {(item.adjustments || []).join(", ") || "none"}</div>
                      <div className="text-slate-500">Reason: {item.instability?.has_issues ? "instability" : item.convergence?.has_issues ? "convergence" : "none"}</div>
                      <div className="text-slate-500">convergence_speed proxy (loss_trend): {fmtMaybe(item.convergence?.loss_trend)}</div>
                      <div className="text-slate-500">loss_variance: {fmtMaybe(item.convergence?.loss_variance)} | loss_plateau: {fmtMaybe(item.convergence?.loss_plateau)} | slow_convergence: {fmtMaybe(item.convergence?.slow_convergence)}</div>
                      <div className="text-slate-500">max_grad: {fmtMaybe(item.instability?.max_grad)} | loss_spikes: {fmtMaybe(item.instability?.loss_spikes)} | gradient_explosion: {fmtMaybe(item.instability?.gradient_explosion)}</div>
                      {item.params && (
                        <div className="text-slate-500">params: lr={fmtMaybe(item.params.lr_mult)}, opacity_lr={fmtMaybe(item.params.opacity_lr_mult)}, sh_lr={fmtMaybe(item.params.sh_lr_mult)}, pos_lr={fmtMaybe(item.params.position_lr_mult)}, densify_th={fmtMaybe(item.params.densify_threshold)}</div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <div className="bg-white rounded-lg border border-slate-200 p-3">
              <p className="text-xs font-semibold text-slate-600 mb-2">Left Preview</p>
              {previewLeft ? (
                <img src={previewLeft} alt="Left preview" className="w-full rounded border border-slate-200" />
              ) : (
                <p className="text-xs text-slate-500">No preview available.</p>
              )}
            </div>
            <div className="bg-white rounded-lg border border-slate-200 p-3">
              <p className="text-xs font-semibold text-slate-600 mb-2">Right Preview</p>
              {previewRight ? (
                <img src={previewRight} alt="Right preview" className="w-full rounded border border-slate-200" />
              ) : (
                <p className="text-xs text-slate-500">No preview available.</p>
              )}
            </div>
          </div>
        </>
      )}
    </div>
  );
}
