import { useEffect, useState } from "react";
import { api } from "../../api/client";
import ConfirmModal from "../ConfirmModal";

interface SessionsTabProps {
  projectId: string;
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
  adaptive_event_count?: number;
  has_run_config?: boolean;
  has_run_log?: boolean;
  is_base?: boolean;
}

const sanitizeRunToken = (value: string): string =>
  value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9_-]+/g, "-")
    .replace(/-+/g, "-")
    .replace(/^[-_]+|[-_]+$/g, "")
    .slice(0, 80);

const buildDefaultRunName = (projectId: string, runs: ProjectRunInfo[] = []): string => {
  const base = sanitizeRunToken(projectId || "project") || "project";
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

export default function SessionsTab({ projectId }: SessionsTabProps) {
  const [runs, setRuns] = useState<ProjectRunInfo[]>([]);
  const [baseSessionId, setBaseSessionId] = useState<string>("");
  const [canCreateSession, setCanCreateSession] = useState<boolean>(false);
  const [createSessionReason, setCreateSessionReason] = useState<string>("Complete COLMAP on the base session before creating new sessions.");
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [busyRunId, setBusyRunId] = useState<string | null>(null);
  const [pendingDeleteRun, setPendingDeleteRun] = useState<ProjectRunInfo | null>(null);
  const [pendingSetBaseRun, setPendingSetBaseRun] = useState<ProjectRunInfo | null>(null);
  const [pendingRenameRun, setPendingRenameRun] = useState<ProjectRunInfo | null>(null);
  const [renameDraft, setRenameDraft] = useState<string>("");
  const [showCreateSessionModal, setShowCreateSessionModal] = useState<boolean>(false);
  const [createSessionNameDraft, setCreateSessionNameDraft] = useState<string>("");
  const [createSessionSourceRunId, setCreateSessionSourceRunId] = useState<string>("");
  const [isCreatingSessionDraft, setIsCreatingSessionDraft] = useState<boolean>(false);

  const loadRuns = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await api.get(`/projects/${projectId}/runs`);
      const list = Array.isArray(res.data?.runs) ? (res.data.runs as ProjectRunInfo[]) : [];
      setRuns(list);
      setBaseSessionId(typeof res.data?.base_session_id === "string" ? res.data.base_session_id : "");
      setCanCreateSession(Boolean(res.data?.can_create_session));
      setCreateSessionReason(
        typeof res.data?.can_create_session_reason === "string" && res.data.can_create_session_reason.trim()
          ? res.data.can_create_session_reason
          : "Complete COLMAP on the base session before creating new sessions.",
      );

      if (list.length > 0) {
        const fallbackSource = (typeof res.data?.base_session_id === "string" && res.data.base_session_id) || list[0].run_id;
        setCreateSessionSourceRunId((prev) => prev || fallbackSource);
      } else {
        setCreateSessionSourceRunId("");
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load sessions");
      setCanCreateSession(false);
      setCreateSessionReason("Complete COLMAP on the base session before creating new sessions.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadRuns();
    const id = setInterval(loadRuns, 10000);
    return () => clearInterval(id);
  }, [projectId]);

  const renameRun = async (run: ProjectRunInfo, desired: string) => {
    if (!desired) return;
    setBusyRunId(run.run_id);
    setError(null);
    try {
      await api.patch(`/projects/${projectId}/runs/${run.run_id}`, { run_name: desired });
      setPendingRenameRun(null);
      setRenameDraft("");
      await loadRuns();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to rename session");
    } finally {
      setBusyRunId(null);
    }
  };

  const requestRenameRun = (run: ProjectRunInfo) => {
    setPendingRenameRun(run);
    setRenameDraft((run.run_name || run.run_id || "").trim());
  };

  const setBase = async (run: ProjectRunInfo) => {
    setBusyRunId(run.run_id);
    setError(null);
    try {
      await api.patch(`/projects/${projectId}/runs/${run.run_id}/set-base`);
      setPendingSetBaseRun(null);
      await loadRuns();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to set base session");
    } finally {
      setBusyRunId(null);
    }
  };

  const requestSetBaseRun = (run: ProjectRunInfo) => {
    setPendingSetBaseRun(run);
  };

  const requestDeleteRun = (run: ProjectRunInfo) => {
    setPendingDeleteRun(run);
  };

  const deleteRun = async () => {
    if (!pendingDeleteRun) return;
    const run = pendingDeleteRun;
    setBusyRunId(run.run_id);
    setError(null);
    try {
      await api.delete(`/projects/${projectId}/runs/${run.run_id}`);
      setPendingDeleteRun(null);
      await loadRuns();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to delete session");
    } finally {
      setBusyRunId(null);
    }
  };

  const openCreateSessionModal = () => {
    if (!canCreateSession) {
      setError(createSessionReason || "Create Session is disabled until base COLMAP is ready.");
      return;
    }
    setError(null);
    setCreateSessionNameDraft(buildDefaultRunName(projectId, runs));
    if (!createSessionSourceRunId) {
      setCreateSessionSourceRunId(baseSessionId || (runs.length > 0 ? runs[0].run_id : ""));
    }
    setShowCreateSessionModal(true);
  };

  const createSessionDraft = async () => {
    if (!canCreateSession) {
      setError(createSessionReason || "Create Session is disabled until base COLMAP is ready.");
      return;
    }
    const runName = (createSessionNameDraft || "").trim() || buildDefaultRunName(projectId, runs);
    const sourceRunId = (createSessionSourceRunId || "").trim();

    setIsCreatingSessionDraft(true);
    setError(null);
    try {
      await api.post(`/projects/${projectId}/runs`, {
        run_name: runName,
        source_run_id: sourceRunId || undefined,
      });
      setShowCreateSessionModal(false);
      setCreateSessionNameDraft("");
      await loadRuns();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to create session");
    } finally {
      setIsCreatingSessionDraft(false);
    }
  };

  return (
    <div className="space-y-4">
      <div className="bg-white rounded-lg border border-slate-200 p-4">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-lg font-semibold text-slate-900">Sessions</h3>
            <p className="text-xs text-slate-500">Manage run sessions: rename, delete, or mark base.</p>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={openCreateSessionModal}
              disabled={!canCreateSession}
              title={!canCreateSession ? createSessionReason : "Create a new session draft"}
              className="px-3 py-2 text-xs font-semibold rounded border border-blue-200 text-blue-700 hover:bg-blue-50 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              + Add Session
            </button>
            <button
              onClick={loadRuns}
              className="px-3 py-2 text-xs font-semibold rounded border border-slate-300 text-slate-700 hover:bg-slate-50"
            >
              Refresh
            </button>
          </div>
        </div>
        {!canCreateSession && (
          <p className="mt-2 text-[11px] text-amber-700">{createSessionReason}</p>
        )}
      </div>

      {loading && <div className="text-sm text-slate-500">Loading sessions...</div>}
      {error && <div className="text-sm text-rose-700 bg-rose-50 border border-rose-200 rounded px-3 py-2">{error}</div>}

      {!loading && runs.length === 0 && (
        <div className="bg-white rounded-lg border border-slate-200 p-4 text-sm text-slate-600">
          No sessions yet. Start processing to create the first session.
        </div>
      )}

      {!loading && runs.length > 0 && (
        <div className="bg-white rounded-lg border border-slate-200 overflow-hidden">
          <table className="w-full text-sm">
            <thead className="bg-slate-50 text-slate-700">
              <tr>
                <th className="text-left px-4 py-3">Session</th>
                <th className="text-left px-4 py-3">Saved</th>
                <th className="text-left px-4 py-3">Status</th>
                <th className="text-left px-4 py-3">Mode / Engine</th>
                <th className="text-left px-4 py-3">Actions</th>
              </tr>
            </thead>
            <tbody>
              {runs.map((run) => {
                const busy = busyRunId === run.run_id;
                const isBase = run.is_base || run.run_id === baseSessionId;
                return (
                  <tr key={run.run_id} className="border-t border-slate-100">
                    <td className="px-4 py-2">
                      <div className="flex items-center gap-2">
                        <span className="font-medium text-slate-900">{run.run_name || run.run_id}</span>
                        {isBase && <span className="px-1.5 py-0.5 text-[10px] font-semibold rounded bg-emerald-100 text-emerald-700">BASE</span>}
                      </div>
                      <p className="text-[11px] text-slate-500">{run.run_id}</p>
                    </td>
                    <td className="px-4 py-2 text-slate-600">{run.saved_at || "-"}</td>
                    <td className="px-4 py-2">
                      {run.session_status === "completed" ? (
                        <span className="px-2 py-0.5 text-[10px] font-semibold rounded bg-emerald-100 text-emerald-700">Completed</span>
                      ) : (
                        <span className="px-2 py-0.5 text-[10px] font-semibold rounded bg-amber-100 text-amber-700">Pending</span>
                      )}
                    </td>
                    <td className="px-4 py-2 text-slate-600">{`${run.mode || "-"} / ${run.engine || "-"}`}</td>
                    <td className="px-4 py-2">
                      <div className="flex items-center gap-2">
                        <button
                          disabled={busy}
                          onClick={() => requestRenameRun(run)}
                          className="px-2 py-1 text-xs rounded border border-slate-300 text-slate-700 hover:bg-slate-50 disabled:opacity-50"
                        >
                          Rename
                        </button>
                        {!isBase && (
                          <button
                            disabled={busy}
                            onClick={() => requestSetBaseRun(run)}
                            className="px-2 py-1 text-xs rounded border border-emerald-300 text-emerald-700 hover:bg-emerald-50 disabled:opacity-50"
                          >
                            Set Base
                          </button>
                        )}
                        <button
                          disabled={busy}
                          onClick={() => requestDeleteRun(run)}
                          className="px-2 py-1 text-xs rounded border border-rose-300 text-rose-700 hover:bg-rose-50 disabled:opacity-50"
                        >
                          Delete
                        </button>
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      <ConfirmModal
        open={Boolean(pendingRenameRun)}
        title="Rename Session"
        message={
          <div className="space-y-2">
            <p>
              Enter a new name for <span className="font-semibold text-slate-800">{pendingRenameRun?.run_name || pendingRenameRun?.run_id}</span>.
            </p>
            <input
              value={renameDraft}
              onChange={(e) => setRenameDraft(e.target.value)}
              className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm text-slate-800"
              placeholder="Session name"
              autoFocus
            />
          </div>
        }
        confirmLabel="Rename"
        cancelLabel="Cancel"
        tone="default"
        busy={pendingRenameRun ? busyRunId === pendingRenameRun.run_id : false}
        onCancel={() => {
          if (!busyRunId) {
            setPendingRenameRun(null);
            setRenameDraft("");
          }
        }}
        onConfirm={() => {
          if (pendingRenameRun) void renameRun(pendingRenameRun, renameDraft.trim());
        }}
      />

      <ConfirmModal
        open={Boolean(pendingSetBaseRun)}
        title="Set Base Session"
        message={
          <>
            Are you sure you want to set <span className="font-semibold text-slate-800">{pendingSetBaseRun?.run_name || pendingSetBaseRun?.run_id}</span> as base session?
          </>
        }
        confirmLabel="Set Base"
        cancelLabel="Cancel"
        tone="default"
        busy={pendingSetBaseRun ? busyRunId === pendingSetBaseRun.run_id : false}
        onCancel={() => {
          if (!busyRunId) setPendingSetBaseRun(null);
        }}
        onConfirm={() => {
          if (pendingSetBaseRun) void setBase(pendingSetBaseRun);
        }}
      />

      {pendingDeleteRun && (
        <div className="fixed inset-0 z-50">
          <div
            className="absolute inset-0 bg-black/50"
            onClick={() => {
              if (!busyRunId) setPendingDeleteRun(null);
            }}
          />
          <div className="absolute inset-0 flex items-center justify-center p-4">
            <div className="w-[520px] max-w-full bg-white rounded-xl shadow-2xl border border-slate-200 overflow-hidden">
              <div className="flex items-center justify-between px-4 py-3 border-b border-slate-200">
                <div>
                  <p className="text-xs uppercase font-semibold text-slate-500">Session</p>
                  <h3 className="text-base font-bold text-slate-900">Delete Session</h3>
                </div>
                <button
                  className="text-sm text-slate-600 disabled:text-slate-300"
                  onClick={() => setPendingDeleteRun(null)}
                  disabled={Boolean(busyRunId)}
                >
                  Close
                </button>
              </div>

              <div className="p-4 space-y-3 text-sm text-slate-700">
                <p>
                  Are you sure you want to delete <span className="font-semibold text-slate-900">{pendingDeleteRun.run_name || pendingDeleteRun.run_id}</span>? This cannot be undone.
                </p>

                <div className="flex justify-end gap-2 pt-1">
                  <button
                    onClick={() => setPendingDeleteRun(null)}
                    disabled={Boolean(busyRunId)}
                    className="px-3 py-2 rounded-lg border border-slate-300 text-slate-700 disabled:text-slate-300 disabled:border-slate-200"
                  >
                    Cancel
                  </button>
                  <button
                    onClick={deleteRun}
                    disabled={Boolean(busyRunId)}
                    className="px-3 py-2 rounded-lg bg-rose-600 text-white font-semibold hover:bg-rose-700 disabled:bg-slate-300"
                  >
                    {busyRunId === pendingDeleteRun.run_id ? "Deleting..." : "Delete"}
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {showCreateSessionModal && (
        <div className="fixed inset-0 z-50">
          <div className="absolute inset-0 bg-black/50" onClick={() => !isCreatingSessionDraft && setShowCreateSessionModal(false)} />
          <div className="absolute inset-0 flex items-center justify-center p-4">
            <div className="w-[540px] max-w-full bg-white rounded-xl shadow-2xl border border-slate-200 overflow-hidden">
              <div className="flex items-center justify-between px-4 py-3 border-b border-slate-200">
                <div>
                  <p className="text-xs uppercase font-semibold text-slate-500">Session</p>
                  <h3 className="text-base font-bold text-slate-900">Create New Session</h3>
                </div>
                <button
                  className="text-sm text-slate-600 disabled:text-slate-300"
                  onClick={() => setShowCreateSessionModal(false)}
                  disabled={isCreatingSessionDraft}
                >
                  Close
                </button>
              </div>

              <div className="p-4 space-y-4 text-sm">
                <div>
                  <label className="block text-xs font-semibold text-slate-600 mb-1">Session Name</label>
                  <input
                    value={createSessionNameDraft}
                    onChange={(e) => setCreateSessionNameDraft(e.target.value)}
                    placeholder={buildDefaultRunName(projectId, runs)}
                    className="w-full px-3 py-2 border border-slate-300 rounded-lg"
                  />
                </div>

                <div>
                  <label className="block text-xs font-semibold text-slate-600 mb-1">Copy Config From</label>
                  <select
                    value={createSessionSourceRunId}
                    onChange={(e) => setCreateSessionSourceRunId(e.target.value)}
                    className="w-full px-3 py-2 border border-slate-300 rounded-lg"
                  >
                    <option value="">No source (use defaults/current backend behavior)</option>
                    {runs.map((run) => (
                      <option key={run.run_id} value={run.run_id}>
                        {run.run_name || run.run_id}
                      </option>
                    ))}
                  </select>
                </div>

                <div className="flex justify-end gap-2 pt-1">
                  <button
                    onClick={() => setShowCreateSessionModal(false)}
                    disabled={isCreatingSessionDraft}
                    className="px-3 py-2 rounded-lg border border-slate-300 text-slate-700 disabled:text-slate-300"
                  >
                    Cancel
                  </button>
                  <button
                    onClick={createSessionDraft}
                    disabled={isCreatingSessionDraft}
                    className="px-3 py-2 rounded-lg bg-blue-600 text-white font-semibold hover:bg-blue-700 disabled:bg-slate-300"
                  >
                    {isCreatingSessionDraft ? "Creating..." : "Create Session Draft"}
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
