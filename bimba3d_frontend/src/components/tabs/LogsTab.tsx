import { useEffect, useState, useRef } from "react";
import { FileText, RefreshCw, Download } from "lucide-react";
import { jsPDF } from "jspdf";
import autoTable from "jspdf-autotable";
import { api } from "../../api/client";

interface LogsTabProps {
  projectId: string;
}

type LogsView = "processing" | "ai";
type LearningState = "learning" | "not_learning" | "neutral";

interface AILogEvent {
  time: number | null;
  step: number | null;
  loss: number | null;
  action: string;
  reason: string | null;
  relative_improvement: number | null;
  reward_from_previous: number | null;
  learning_state: LearningState;
  trend_scope: string | null;
  source: string;
}

interface AILogSummary {
  total: number;
  learning: number;
  not_learning: number;
  neutral: number;
}

interface ProjectRunInfo {
  run_id: string;
  run_name?: string | null;
  mode?: string | null;
  tune_scope?: string | null;
  adaptive_event_count?: number;
}

interface TelemetryEventRow {
  timestamp?: string | null;
  type?: string | null;
  step?: number | null;
  summary?: string | null;
  action?: string | null;
  reason?: string | null;
  loss?: number | null;
  relative_improvement?: number | null;
  reward_from_previous?: number | null;
}

interface TelemetryTrainingRow {
  step?: number | null;
  loss?: number | null;
}

interface ChartPoint {
  step: number;
  value: number;
  actionTaken: boolean;
  action?: string;
  reason?: string | null;
  reward?: number | null;
  relativeImprovement?: number | null;
}

interface ChartGeometry {
  polyline: string;
  min: number;
  max: number;
  minStep: number;
  maxStep: number;
  points: Array<{
    x: number;
    y: number;
    step: number;
    value: number;
    actionTaken: boolean;
    action?: string;
    reason?: string | null;
    reward?: number | null;
    relativeImprovement?: number | null;
  }>;
}

function buildPolylinePoints(points: ChartPoint[], width: number, height: number): ChartGeometry {
  if (points.length === 0) {
    return {
      polyline: "",
      min: 0,
      max: 0,
      minStep: 0,
      maxStep: 0,
      points: [],
    };
  }

  const minStep = points[0].step;
  const maxStep = points[points.length - 1].step;
  const minVal = Math.min(...points.map((p) => p.value));
  const maxVal = Math.max(...points.map((p) => p.value));

  const stepSpan = Math.max(maxStep - minStep, 1);
  const valSpan = Math.max(maxVal - minVal, 1e-12);

  const mapped = points.map((p) => {
    const x = ((p.step - minStep) / stepSpan) * width;
    const y = height - ((p.value - minVal) / valSpan) * height;
    return {
      x,
      y,
      step: p.step,
      value: p.value,
      actionTaken: p.actionTaken,
      action: p.action,
      reason: p.reason,
      reward: p.reward,
      relativeImprovement: p.relativeImprovement,
    };
  });

  const polyline = mapped.map((p) => `${p.x.toFixed(2)},${p.y.toFixed(2)}`).join(" ");
  return { polyline, min: minVal, max: maxVal, minStep, maxStep, points: mapped };
}

function niceNumber(value: number): number {
  if (!Number.isFinite(value) || value <= 0) {
    return 1;
  }
  const exponent = Math.floor(Math.log10(value));
  const fraction = value / Math.pow(10, exponent);

  let niceFraction = 1;
  if (fraction <= 1) {
    niceFraction = 1;
  } else if (fraction <= 2) {
    niceFraction = 2;
  } else if (fraction <= 5) {
    niceFraction = 5;
  } else {
    niceFraction = 10;
  }

  return niceFraction * Math.pow(10, exponent);
}

function formatTickValue(value: number, maxDecimals = 3): string {
  if (!Number.isFinite(value)) {
    return "0";
  }
  if (Math.abs(value) >= 1000) {
    return Math.round(value).toString();
  }
  const fixed = value.toFixed(maxDecimals);
  return fixed.replace(/\.0+$/, "").replace(/(\.\d*?)0+$/, "$1");
}

export default function LogsTab({ projectId }: LogsTabProps) {
  const [view, setView] = useState<LogsView>("processing");
  const [logs, setLogs] = useState<string>("");
  const [aiEvents, setAiEvents] = useState<AILogEvent[]>([]);
  const [aiSummary, setAiSummary] = useState<AILogSummary>({ total: 0, learning: 0, not_learning: 0, neutral: 0 });
  const [aiRunId, setAiRunId] = useState<string | null>(null);
  const [selectedAiRunId, setSelectedAiRunId] = useState<string>("");
  const [aiRunOptions, setAiRunOptions] = useState<ProjectRunInfo[]>([]);
  const [aiTrainingRows, setAiTrainingRows] = useState<TelemetryTrainingRow[]>([]);
  const [aiSource, setAiSource] = useState<string | null>(null);
  const [stateFilter, setStateFilter] = useState<"all" | LearningState>("all");
  const [nonKeepOnly, setNonKeepOnly] = useState(false);
  const [loading, setLoading] = useState(true);
  const [autoScroll, setAutoScroll] = useState(true);
  const [aiPdfDownloadBusy, setAiPdfDownloadBusy] = useState(false);
  const logsEndRef = useRef<HTMLDivElement>(null);

  const fetchProcessingLogs = async () => {
    const res = await api.get(`/projects/${projectId}/logs?lines=500`);
    setLogs(res.data.logs || "No logs available yet.");
  };

  const fetchAiLogs = async () => {
    const res = await api.get(`/projects/${projectId}/telemetry`, {
      params: {
        run_id: selectedAiRunId || undefined,
        log_limit: 500,
        eval_limit: 5,
      },
    });

    const trainingRows: TelemetryTrainingRow[] = Array.isArray(res.data.training_rows)
      ? res.data.training_rows
      : [];
    const eventRows: TelemetryEventRow[] = Array.isArray(res.data.event_rows)
      ? res.data.event_rows
      : [];
    setAiTrainingRows(trainingRows);

    const lossByStep = new Map<number, number>();
    for (const row of trainingRows) {
      if (typeof row.step === "number" && typeof row.loss === "number") {
        lossByStep.set(row.step, row.loss);
      }
    }

    const toLearningState = (action: string): LearningState => {
      const normalized = action.trim().toLowerCase();
      if (!normalized || normalized === "keep" || normalized === "ai_action_keep") {
        return "neutral";
      }
      if (normalized.includes("block") || normalized.includes("cooldown")) {
        return "not_learning";
      }
      return "learning";
    };

    const parseActionFromSummary = (summary: string | null | undefined): string => {
      if (!summary) {
        return "keep";
      }
      const marker = summary.lastIndexOf(":");
      if (marker >= 0 && marker < summary.length - 1) {
        return summary.slice(marker + 1).trim();
      }
      return "keep";
    };

    const events: AILogEvent[] = eventRows
      .filter((row) => String(row.type || "").toLowerCase().includes("ai"))
      .map((row) => {
        const action = String(row.action || "").trim() || parseActionFromSummary(row.summary);
        const step = typeof row.step === "number" ? row.step : null;
        return {
          time: null,
          step,
          loss: typeof row.loss === "number" ? row.loss : step !== null ? lossByStep.get(step) ?? null : null,
          action,
          reason: row.reason || row.summary || null,
          relative_improvement: typeof row.relative_improvement === "number" ? row.relative_improvement : null,
          reward_from_previous: typeof row.reward_from_previous === "number" ? row.reward_from_previous : null,
          learning_state: toLearningState(action),
          trend_scope: null,
          source: "telemetry",
        };
      });

    const summary: AILogSummary = {
      total: events.length,
      learning: events.filter((e) => e.learning_state === "learning").length,
      not_learning: events.filter((e) => e.learning_state === "not_learning").length,
      neutral: events.filter((e) => e.learning_state === "neutral").length,
    };

    setAiEvents(events);
    setAiSummary(summary);
    setAiRunId(res.data.run_id || selectedAiRunId || null);
    setAiSource("telemetry");
  };

  const fetchAiRuns = async () => {
    const res = await api.get(`/projects/${projectId}/runs`);
    const runs: ProjectRunInfo[] = Array.isArray(res.data?.runs) ? res.data.runs : [];
    const aiRuns = runs.filter((run) => {
      const mode = String(run.mode || "").toLowerCase();
      const scope = String(run.tune_scope || "").toLowerCase();
      const adaptiveEvents = Number(run.adaptive_event_count || 0);
      return adaptiveEvents > 0 || mode === "modified" || scope.includes("ai");
    });

    setAiRunOptions(aiRuns);

    if (aiRuns.length === 0) {
      setSelectedAiRunId("");
      return;
    }

    const stillValid = aiRuns.some((run) => run.run_id === selectedAiRunId);
    if (!stillValid) {
      setSelectedAiRunId(aiRuns[0].run_id);
    }
  };

  useEffect(() => {
    if (view !== "ai") {
      return;
    }

    fetchAiRuns().catch((err) => {
      console.error("Failed to fetch AI runs:", err);
      setAiRunOptions([]);
      setSelectedAiRunId("");
      setAiTrainingRows([]);
    });
  }, [projectId, view]);

  useEffect(() => {
    const fetchData = async () => {
      try {
        if (view === "processing") {
          await fetchProcessingLogs();
        } else {
          if (!selectedAiRunId) {
            setAiEvents([]);
            setAiSummary({ total: 0, learning: 0, not_learning: 0, neutral: 0 });
            setAiRunId(null);
            setAiSource(null);
            setAiTrainingRows([]);
          } else {
            await fetchAiLogs();
          }
        }
        setLoading(false);
      } catch (err) {
        console.error("Failed to fetch logs data:", err);
        if (view === "processing") {
          setLogs("Failed to load logs.");
        } else {
          setAiEvents([]);
        }
        setLoading(false);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 5000);
    return () => clearInterval(interval);
  }, [projectId, view, selectedAiRunId]);

  useEffect(() => {
    if (view === "processing" && autoScroll && logsEndRef.current) {
      logsEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [logs, autoScroll, view]);

  const filteredAiEvents = aiEvents.filter((event) => {
    if (stateFilter !== "all" && event.learning_state !== stateFilter) {
      return false;
    }
    if (nonKeepOnly && (event.action === "keep" || event.action === "ai_action_keep")) {
      return false;
    }
    return true;
  });

  const isActionTaken = (action: string | null | undefined) => {
    const normalized = (action || "").trim().toLowerCase();
    return normalized !== "keep" && normalized !== "ai_action_keep";
  };

  const chartEvents = [...filteredAiEvents]
    .filter((event) => typeof event.step === "number")
    .sort((a, b) => (a.step ?? 0) - (b.step ?? 0))
    .slice(-120);

  const aiEventsByStep = new Map<number, AILogEvent>();
  for (const event of filteredAiEvents) {
    if (typeof event.step === "number") {
      aiEventsByStep.set(event.step, event);
    }
  }

  const lossPoints: ChartPoint[] = [...aiTrainingRows]
    .filter((row) => typeof row.step === "number" && typeof row.loss === "number")
    .sort((a, b) => (a.step as number) - (b.step as number))
    .map((row) => {
      const step = row.step as number;
      const event = aiEventsByStep.get(step);
      return {
        step,
        value: row.loss as number,
        actionTaken: isActionTaken(event?.action),
        action: event?.action,
        reason: event?.reason,
        reward: event?.reward_from_previous,
        relativeImprovement: event?.relative_improvement,
      };
    });

  const rewardPoints: ChartPoint[] = chartEvents
    .filter((e) => typeof e.reward_from_previous === "number")
    .map((e) => ({
      step: e.step as number,
      value: e.reward_from_previous as number,
      actionTaken: isActionTaken(e.action),
      action: e.action,
      reason: e.reason,
      reward: e.reward_from_previous,
      relativeImprovement: e.relative_improvement,
    }));

  const lossChart = buildPolylinePoints(lossPoints, 300, 140);
  const rewardChart = buildPolylinePoints(rewardPoints, 300, 140);

  const buildXTicks = (min: number, max: number, plotWidth = 300) => {
    const span = Math.max(max - min, 1);
    const targetTicks = Math.max(4, Math.min(14, Math.floor(plotWidth / 36) + 1));
    const rawStep = span / Math.max(targetTicks - 1, 1);
    const step = Math.max(1, Math.round(niceNumber(rawStep)));
    const maxTickCount = Math.max(6, Math.min(24, Math.floor(plotWidth / 22) + 2));

    const start = Math.ceil(min / step) * step;
    const end = Math.floor(max / step) * step;
    const values: number[] = [];
    for (let v = start; v <= end; v += step) {
      values.push(v);
      if (values.length > maxTickCount) {
        break;
      }
    }

    if (values.length < 2) {
      const minRounded = Math.round(min / step) * step;
      const maxRounded = Math.round(max / step) * step;
      const fallbackA = Math.min(minRounded, maxRounded);
      const fallbackB = Math.max(minRounded, maxRounded || fallbackA + step);
      const spanSafe = Math.max(max - min, 1);
      return [
        { ratio: (fallbackA - min) / spanSafe, label: `${Math.round(fallbackA)}` },
        { ratio: (fallbackB - min) / spanSafe, label: `${Math.round(fallbackB)}` },
      ];
    }

    return values.map((value) => ({
      ratio: (value - min) / span,
      label: `${Math.round(value)}`,
    }));
  };

  const buildYTicks = (min: number, max: number, plotHeight = 140) => {
    const span = Math.max(max - min, 1e-12);
    const targetTicks = Math.max(4, Math.min(12, Math.floor(plotHeight / 18) + 1));
    const step = niceNumber(span / Math.max(targetTicks - 1, 1));
    const maxTickCount = Math.max(6, Math.min(18, Math.floor(plotHeight / 10) + 2));

    const start = Math.floor(min / step) * step;
    const end = Math.ceil(max / step) * step;
    const values: number[] = [];
    for (let v = start; v <= end + step * 0.5; v += step) {
      values.push(v);
      if (values.length > maxTickCount) {
        break;
      }
    }

    if (values.length < 2) {
      return [
        { ratio: 0, label: formatTickValue(max, 3) },
        { ratio: 1, label: formatTickValue(min, 3) },
      ];
    }

    const total = Math.max(end - start, 1e-12);
    return values
      .map((value) => ({
        ratio: 1 - (value - start) / total,
        label: formatTickValue(value, 3),
      }))
      .filter((tick) => tick.ratio >= -0.01 && tick.ratio <= 1.01);
  };

  const renderChart = (
    chart: ChartGeometry,
    color: string,
    yLabel: string,
    showZeroLine = false,
  ) => {
    const outerW = 360;
    const outerH = 190;
    const margin = { left: 44, right: 10, top: 10, bottom: 28 };
    const plotW = 300;
    const plotH = 140;
    const x0 = margin.left;
    const y0 = margin.top;

    const xTicks = buildXTicks(chart.minStep, chart.maxStep, plotW);
    const yTicks = buildYTicks(chart.min, chart.max, plotH);

    const plotPolyline = chart.points
      .map((p) => `${(x0 + p.x).toFixed(2)},${(y0 + p.y).toFixed(2)}`)
      .join(" ");

    let zeroY: number | null = null;
    if (showZeroLine && chart.min < 0 && chart.max > 0) {
      const ratio = (chart.max - 0) / Math.max(chart.max - chart.min, 1e-12);
      zeroY = y0 + ratio * plotH;
    }

    return (
      <svg viewBox={`0 0 ${outerW} ${outerH}`} className="w-full h-44 bg-slate-50 rounded border border-slate-100">
        {yTicks.map((tick, idx) => {
          const y = y0 + tick.ratio * plotH;
          return (
            <g key={`y-${idx}`}>
              <line x1={x0} y1={y} x2={x0 + plotW} y2={y} stroke="#e2e8f0" strokeWidth="1" />
              <text x={x0 - 6} y={y + 3} textAnchor="end" fontSize="10" fill="#64748b">
                {tick.label}
              </text>
            </g>
          );
        })}

        {xTicks.map((tick, idx) => {
          const x = x0 + tick.ratio * plotW;
          return (
            <g key={`x-${idx}`}>
              <line x1={x} y1={y0} x2={x} y2={y0 + plotH} stroke="#f1f5f9" strokeWidth="1" />
              <text x={x} y={y0 + plotH + 14} textAnchor="middle" fontSize="10" fill="#64748b">
                {tick.label}
              </text>
            </g>
          );
        })}

        <line x1={x0} y1={y0 + plotH} x2={x0 + plotW} y2={y0 + plotH} stroke="#94a3b8" strokeWidth="1" />
        <line x1={x0} y1={y0} x2={x0} y2={y0 + plotH} stroke="#94a3b8" strokeWidth="1" />

        {zeroY !== null && (
          <line x1={x0} y1={zeroY} x2={x0 + plotW} y2={zeroY} stroke="#cbd5e1" strokeWidth="1" strokeDasharray="4 4" />
        )}

        {chart.points.length >= 2 && (
          <polyline
            fill="none"
            stroke={color}
            strokeWidth="2"
            points={plotPolyline}
          />
        )}

        {chart.points
          .filter((p) => Boolean(p.action))
          .map((p, idx) => (
            <circle
              key={`action-dot-${idx}-${p.step}`}
              cx={x0 + p.x}
              cy={y0 + p.y}
              r={p.actionTaken ? "2.8" : "2.0"}
              fill={p.actionTaken ? "#2563eb" : "#94a3b8"}
              stroke="#ffffff"
              strokeWidth="0.8"
            >
              <title>{`Step: ${p.step}\nAction: ${p.action || "-"}\nReward: ${typeof p.reward === "number" ? p.reward.toFixed(6) : "-"}\nRel.Impr: ${typeof p.relativeImprovement === "number" ? p.relativeImprovement.toFixed(6) : "-"}\nReason: ${p.reason || "-"}`}</title>
            </circle>
          ))}

        <text x={x0 + plotW / 2} y={outerH - 4} textAnchor="middle" fontSize="10" fill="#64748b">
          Step
        </text>
        <text
          x={12}
          y={y0 + plotH / 2}
          textAnchor="middle"
          fontSize="10"
          fill="#64748b"
          transform={`rotate(-90 12 ${y0 + plotH / 2})`}
        >
          {yLabel}
        </text>
      </svg>
    );
  };

  const downloadLogs = () => {
    const isAiView = view === "ai";
    const content = isAiView ? JSON.stringify(filteredAiEvents, null, 2) : logs;
    const blob = new Blob([content], { type: isAiView ? "application/json" : "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = isAiView
      ? `${projectId}_ai_logs_${aiRunId || "latest"}.json`
      : `${projectId}_logs.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const refreshLogs = async () => {
    setLoading(true);
    try {
      if (view === "processing") {
        await fetchProcessingLogs();
      } else {
        if (!selectedAiRunId) {
          setAiEvents([]);
          setAiSummary({ total: 0, learning: 0, not_learning: 0, neutral: 0 });
          setAiRunId(null);
          setAiSource(null);
          setAiTrainingRows([]);
        } else {
          await fetchAiLogs();
        }
      }
    } catch (err) {
      console.error("Failed to refresh logs:", err);
    } finally {
      setLoading(false);
    }
  };

  const downloadAiLogsPdf = () => {
    if (view !== "ai") {
      return;
    }

    try {
      setAiPdfDownloadBusy(true);
      const pdf = new jsPDF({ format: "a4", compress: true });
      pdf.setFont("helvetica", "normal");

      const pageWidth = pdf.internal.pageSize.getWidth();
      const pageHeight = pdf.internal.pageSize.getHeight();
      const margin = 12;
      const contentWidth = pageWidth - margin * 2;
      let yPos = 15;

      const allEvents = [...aiEvents];
      const allEventsSorted = [...allEvents].sort((a, b) => (a.step ?? 0) - (b.step ?? 0));
      const exportRunId = aiRunId || selectedAiRunId || "-";
      const aiEventsByStepAll = new Map<number, AILogEvent>();
      for (const event of allEvents) {
        if (typeof event.step === "number") {
          aiEventsByStepAll.set(event.step, event);
        }
      }

      const lossPointsAll: ChartPoint[] = [...aiTrainingRows]
        .filter((row) => typeof row.step === "number" && typeof row.loss === "number")
        .sort((a, b) => (a.step as number) - (b.step as number))
        .map((row) => {
          const step = row.step as number;
          const event = aiEventsByStepAll.get(step);
          return {
            step,
            value: row.loss as number,
            actionTaken: isActionTaken(event?.action),
            action: event?.action,
            reason: event?.reason,
            reward: event?.reward_from_previous,
            relativeImprovement: event?.relative_improvement,
          };
        });

      const rewardPointsAll: ChartPoint[] = allEventsSorted
        .filter((e) => typeof e.step === "number" && typeof e.reward_from_previous === "number")
        .slice(-120)
        .map((e) => ({
          step: e.step as number,
          value: e.reward_from_previous as number,
          actionTaken: isActionTaken(e.action),
          action: e.action,
          reason: e.reason,
          reward: e.reward_from_previous,
          relativeImprovement: e.relative_improvement,
        }));

      const lossChartAll = buildPolylinePoints(lossPointsAll, 300, 140);
      const rewardChartAll = buildPolylinePoints(rewardPointsAll, 300, 140);

      const addHeading = (text: string, size = 14) => {
        pdf.setFont("helvetica", "bold");
        pdf.setFontSize(size);
        pdf.text(text, margin, yPos);
        pdf.setFont("helvetica", "normal");
        yPos += size / 2.5 + 2;
      };

      const drawChart = (
        title: string,
        chart: ChartGeometry,
        color: [number, number, number],
        yAxisLabel: string,
        showZeroLine = false,
      ) => {
        const chartH = 96;
        if (yPos + chartH + 20 > pageHeight - margin) {
          pdf.addPage();
          yPos = 15;
        }

        pdf.setFont("helvetica", "bold");
        pdf.setFontSize(11);
        pdf.text(title, margin, yPos);
        pdf.setFont("helvetica", "normal");

        const outerTop = yPos + 4;
        const outerLeft = margin;
        const outerWidth = contentWidth;
        const outerHeight = chartH;

        const plotLeft = outerLeft + 18;
        const plotTop = outerTop + 6;
        const plotWidth = outerWidth - 24;
        const plotHeight = outerHeight - 20;

        pdf.setDrawColor(226, 232, 240);
        pdf.rect(outerLeft, outerTop, outerWidth, outerHeight);

        // Use a denser virtual scale for PDF so we show more ticks within available space.
        const xTicks = buildXTicks(chart.minStep, chart.maxStep, plotWidth * 3.0);
        const yTicks = buildYTicks(chart.min, chart.max, plotHeight * 2.4);

        pdf.setFontSize(6);
        pdf.setTextColor(100, 116, 139);

        for (const tick of yTicks) {
          const y = plotTop + tick.ratio * plotHeight;
          pdf.setDrawColor(226, 232, 240);
          pdf.line(plotLeft, y, plotLeft + plotWidth, y);
          pdf.text(tick.label, plotLeft - 2, y + 1, { align: "right" });
        }

        for (const tick of xTicks) {
          const x = plotLeft + tick.ratio * plotWidth;
          pdf.setDrawColor(241, 245, 249);
          pdf.line(x, plotTop, x, plotTop + plotHeight);
          pdf.text(tick.label, x, plotTop + plotHeight + 4, { align: "center" });
        }

        // Axes
        pdf.setDrawColor(148, 163, 184);
        pdf.line(plotLeft, plotTop + plotHeight, plotLeft + plotWidth, plotTop + plotHeight);
        pdf.line(plotLeft, plotTop, plotLeft, plotTop + plotHeight);

        if (showZeroLine && chart.min < 0 && chart.max > 0) {
          const valSpan = Math.max(chart.max - chart.min, 1e-12);
          const zeroRatio = (chart.max - 0) / valSpan;
          const zeroY = plotTop + zeroRatio * plotHeight;
          pdf.setDrawColor(203, 213, 225);
          pdf.line(plotLeft, zeroY, plotLeft + plotWidth, zeroY);
        }

        // Polyline
        if (chart.points.length >= 2) {
          pdf.setDrawColor(color[0], color[1], color[2]);
          for (let i = 1; i < chart.points.length; i += 1) {
            const p0 = chart.points[i - 1];
            const p1 = chart.points[i];
            const x0 = plotLeft + (p0.x / 300) * plotWidth;
            const y0 = plotTop + (p0.y / 140) * plotHeight;
            const x1 = plotLeft + (p1.x / 300) * plotWidth;
            const y1 = plotTop + (p1.y / 140) * plotHeight;
            pdf.line(x0, y0, x1, y1);
          }
        }

        // Action points
        for (const p of chart.points.filter((pt) => Boolean(pt.action))) {
          const cx = plotLeft + (p.x / 300) * plotWidth;
          const cy = plotTop + (p.y / 140) * plotHeight;
          if (p.actionTaken) {
            pdf.setFillColor(37, 99, 235);
            pdf.circle(cx, cy, 1.0, "F");
          } else {
            pdf.setFillColor(148, 163, 184);
            pdf.circle(cx, cy, 0.8, "F");
          }
        }

        // Axis labels
        pdf.setFontSize(8);
        pdf.setTextColor(100, 116, 139);
        pdf.text("Step", plotLeft + plotWidth / 2, outerTop + outerHeight - 1, { align: "center" });
        pdf.text(yAxisLabel, outerLeft + 1, plotTop + plotHeight / 2, { angle: 90, align: "center" });
        pdf.setTextColor(0, 0, 0);
        yPos = outerTop + outerHeight + 6;
      };

      addHeading("AI Logs Charts", 16);
      pdf.setFontSize(9);
      pdf.setTextColor(71, 85, 105);
      pdf.text(`Project: ${projectId} | Run: ${exportRunId}`, margin, yPos);
      pdf.setTextColor(0, 0, 0);
      yPos += 5;
      drawChart("Loss vs Step", lossChartAll, [15, 118, 110], "Loss", false);
      drawChart("Reward vs Step", rewardChartAll, [124, 58, 237], "Reward", true);

      pdf.addPage();
      yPos = 15;
      addHeading("AI Logs Table", 16);
      pdf.setFontSize(9);
      pdf.setTextColor(71, 85, 105);
      pdf.text(`Project: ${projectId} | Run: ${exportRunId}`, margin, yPos);
      pdf.setTextColor(0, 0, 0);
      yPos += 5;

      const rows = allEventsSorted.map((event) => [
        event.step ?? "-",
        event.action || "-",
        event.reason || "-",
        typeof event.relative_improvement === "number" ? event.relative_improvement.toFixed(6) : "-",
        typeof event.reward_from_previous === "number" ? event.reward_from_previous.toFixed(6) : "-",
        event.learning_state,
      ]);

      autoTable(pdf, {
        startY: yPos,
        margin: { left: margin, right: margin, top: margin, bottom: margin },
        head: [["Step", "Action", "Reason", "Rel.Impr", "Reward", "State"]],
        body: rows.length > 0 ? rows : [["-", "-", "No AI log events available.", "-", "-", "-"]],
        columnStyles: {
          0: { cellWidth: 14 },
          1: { cellWidth: 34 },
          2: { cellWidth: contentWidth - 94 },
          3: { cellWidth: 16 },
          4: { cellWidth: 16 },
          5: { cellWidth: 14 },
        },
        headStyles: {
          fontSize: 8,
          textColor: [255, 255, 255],
          fillColor: [25, 55, 120],
          font: "helvetica",
          fontStyle: "bold",
        },
        bodyStyles: {
          fontSize: 7,
          font: "helvetica",
          overflow: "linebreak",
          cellPadding: 1.5,
        },
        styles: {
          valign: "top",
        },
      });

      const projectToken = (projectId || "project").replace(/[^a-zA-Z0-9_-]+/g, "-");
      const runToken = (exportRunId || "latest").replace(/[^a-zA-Z0-9_-]+/g, "-");
      const stamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
      pdf.save(`${projectToken}_ai_logs_${runToken}_${stamp}.pdf`);
    } catch (err) {
      console.error("Failed to download AI logs PDF:", err);
    } finally {
      setAiPdfDownloadBusy(false);
    }
  };

  return (
    <div className="max-w-6xl">
      <div className="bg-white rounded-xl shadow-md border border-gray-200">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-gray-200">
          <div className="flex items-center gap-3">
            <FileText className="w-6 h-6 text-gray-600" />
            <h2 className="text-xl font-bold text-gray-900">Logs</h2>
          </div>
          <div className="flex items-center gap-3">
            {view === "processing" && (
              <label className="flex items-center gap-2 text-sm text-gray-600 cursor-pointer">
                <input
                  type="checkbox"
                  checked={autoScroll}
                  onChange={(e) => setAutoScroll(e.target.checked)}
                  className="w-4 h-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
                />
                Auto-scroll
              </label>
            )}
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
              Download {view === "ai" ? "AI Logs" : "Logs"}
            </button>
            {view === "ai" && (
              <button
                onClick={downloadAiLogsPdf}
                disabled={aiPdfDownloadBusy}
                className="px-4 py-2 bg-indigo-600 hover:bg-indigo-700 text-white rounded-lg transition-colors flex items-center gap-2 disabled:opacity-60 disabled:cursor-not-allowed"
              >
                <Download className="w-4 h-4" />
                {aiPdfDownloadBusy ? "Preparing PDF..." : "Download AI Logs PDF"}
              </button>
            )}
          </div>
        </div>

        <div className="px-6 pt-4 pb-2 border-b border-gray-100 flex items-center justify-between">
          <div className="inline-flex rounded-lg border border-gray-200 overflow-hidden">
            <button
              className={`px-4 py-2 text-sm font-medium ${view === "processing" ? "bg-slate-900 text-white" : "bg-white text-slate-700 hover:bg-slate-50"}`}
              onClick={() => setView("processing")}
            >
              Processing Logs
            </button>
            <button
              className={`px-4 py-2 text-sm font-medium border-l border-gray-200 ${view === "ai" ? "bg-slate-900 text-white" : "bg-white text-slate-700 hover:bg-slate-50"}`}
              onClick={() => setView("ai")}
            >
              AI Logs
            </button>
          </div>
          {view === "ai" && (
            <p className="text-xs text-slate-500">
              Run: <span className="font-semibold text-slate-700">{aiRunId || selectedAiRunId || "-"}</span>
              {aiSource ? ` | Source: ${aiSource}` : ""}
            </p>
          )}
        </div>

        {/* Logs Content */}
        <div className="p-0">
          {view === "processing" ? (
            <pre className="bg-gray-900 text-green-400 p-6 rounded-b-xl overflow-x-auto font-mono text-sm h-[600px] overflow-y-auto">
              {logs}
              <div ref={logsEndRef} />
            </pre>
          ) : (
            <div className="h-[600px] overflow-y-auto rounded-b-xl bg-slate-50">
              <div className="px-6 py-4 border-b border-gray-200 bg-white">
                <div className="flex flex-wrap items-center gap-4">
                  <div className="flex items-center gap-2">
                    <label className="text-sm text-slate-600">AI Run</label>
                    <select
                      value={selectedAiRunId}
                      onChange={(e) => setSelectedAiRunId(e.target.value)}
                      className="px-2 py-1 border border-gray-300 rounded text-sm min-w-52"
                    >
                      {aiRunOptions.length === 0 ? (
                        <option value="">No AI runs found</option>
                      ) : (
                        aiRunOptions.map((run) => (
                          <option key={run.run_id} value={run.run_id}>
                            {run.run_name || run.run_id}
                          </option>
                        ))
                      )}
                    </select>
                  </div>
                  <div className="flex items-center gap-2">
                    <label className="text-sm text-slate-600">State</label>
                    <select
                      value={stateFilter}
                      onChange={(e) => setStateFilter(e.target.value as "all" | LearningState)}
                      className="px-2 py-1 border border-gray-300 rounded text-sm"
                    >
                      <option value="all">All</option>
                      <option value="learning">Learning</option>
                      <option value="not_learning">Not Learning</option>
                      <option value="neutral">Neutral</option>
                    </select>
                  </div>
                  <label className="flex items-center gap-2 text-sm text-slate-600 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={nonKeepOnly}
                      onChange={(e) => setNonKeepOnly(e.target.checked)}
                      className="w-4 h-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
                    />
                    Non-keep actions only
                  </label>
                  <div className="text-xs text-slate-500">
                    Total: {aiSummary.total} | Learning: {aiSummary.learning} | Not Learning: {aiSummary.not_learning} | Neutral: {aiSummary.neutral}
                  </div>
                </div>
              </div>

              <div className="px-4 py-4 grid grid-cols-1 lg:grid-cols-[2fr_1fr] gap-4">
                <div className="overflow-x-auto bg-white border border-gray-200 rounded-lg">
                  <table className="min-w-full text-sm">
                    <thead className="bg-slate-100 text-slate-700">
                      <tr>
                        <th className="px-3 py-2 text-left font-semibold">Step</th>
                        <th className="px-3 py-2 text-left font-semibold">Action</th>
                        <th className="px-3 py-2 text-left font-semibold">Reason</th>
                        <th className="px-3 py-2 text-left font-semibold">Rel.Impr</th>
                        <th className="px-3 py-2 text-left font-semibold">Reward</th>
                        <th className="px-3 py-2 text-left font-semibold">State</th>
                      </tr>
                    </thead>
                    <tbody>
                      {filteredAiEvents.length === 0 ? (
                        <tr>
                          <td className="px-3 py-6 text-slate-500" colSpan={6}>No AI log events available for this filter.</td>
                        </tr>
                      ) : (
                        filteredAiEvents.map((event, idx) => (
                          <tr key={`${event.step ?? "na"}-${idx}`} className="border-t border-gray-100">
                            <td className="px-3 py-2 text-slate-700">{event.step ?? "-"}</td>
                            <td className="px-3 py-2 font-mono text-xs text-slate-700">{event.action || "-"}</td>
                            <td className="px-3 py-2 text-slate-600">{event.reason || "-"}</td>
                            <td className="px-3 py-2 text-slate-700">{typeof event.relative_improvement === "number" ? event.relative_improvement.toFixed(6) : "-"}</td>
                            <td className="px-3 py-2 text-slate-700">{typeof event.reward_from_previous === "number" ? event.reward_from_previous.toFixed(6) : "-"}</td>
                            <td className="px-3 py-2">
                              <span
                                className={`inline-flex px-2 py-1 rounded-full text-xs font-semibold ${
                                  event.learning_state === "learning"
                                    ? "bg-emerald-100 text-emerald-700"
                                    : event.learning_state === "not_learning"
                                      ? "bg-rose-100 text-rose-700"
                                      : "bg-slate-200 text-slate-700"
                                }`}
                              >
                                {event.learning_state}
                              </span>
                            </td>
                          </tr>
                        ))
                      )}
                    </tbody>
                  </table>
                </div>

                <div className="space-y-4">
                  <div className="bg-white border border-gray-200 rounded-lg p-3">
                    <div className="flex items-center justify-between mb-2">
                      <h3 className="text-sm font-semibold text-slate-800">Loss vs Step</h3>
                      <span className="text-xs text-slate-500">last {lossPoints.length} points</span>
                    </div>
                    {lossPoints.length < 2 ? (
                      <p className="text-xs text-slate-500 py-8 text-center">Not enough data for loss chart.</p>
                    ) : (
                      <>
                        {renderChart(lossChart, "#0f766e", "Loss")}
                        <div className="mt-2 flex items-center justify-between text-[11px] text-slate-500">
                          <span>min {formatTickValue(lossChart.min, 3)}</span>
                          <span>max {formatTickValue(lossChart.max, 3)}</span>
                        </div>
                      </>
                    )}
                  </div>

                  <div className="bg-white border border-gray-200 rounded-lg p-3">
                    <div className="flex items-center justify-between mb-2">
                      <h3 className="text-sm font-semibold text-slate-800">Reward vs Step</h3>
                      <span className="text-xs text-slate-500">last {rewardPoints.length} points</span>
                    </div>
                    {rewardPoints.length < 2 ? (
                      <p className="text-xs text-slate-500 py-8 text-center">Not enough data for reward chart.</p>
                    ) : (
                      <>
                        {renderChart(rewardChart, "#7c3aed", "Reward", true)}
                        <div className="mt-2 flex items-center justify-between text-[11px] text-slate-500">
                          <span>min {formatTickValue(rewardChart.min, 3)}</span>
                          <span>max {formatTickValue(rewardChart.max, 3)}</span>
                        </div>
                      </>
                    )}
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
