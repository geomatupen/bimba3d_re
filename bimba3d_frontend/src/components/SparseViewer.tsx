import React, {
  Suspense,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import { BoxSelect, RotateCcw, Save, Sparkles, Trash2 } from "lucide-react";
import * as THREE from "three";

interface SparseViewerProps {
  projectId: string;
  focusTarget?: [number, number, number] | null;
}

interface SparseCandidateOption {
  value: string;
  label: string;
  relativePath?: string | null;
  images?: number | null;
  points?: number | null;
}

interface SelectionRect {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

type SaveStatus = "idle" | "success" | "error";
type OptimizeStatus = "idle" | "queued" | "error";
type WebGLSupportState = "checking" | "supported" | "unsupported";

interface CanvasErrorBoundaryProps {
  children: React.ReactNode;
  fallback: React.ReactNode;
  onError: (message: string) => void;
}

interface CanvasErrorBoundaryState {
  hasError: boolean;
}

interface ToolboxButtonProps {
  label: string;
  icon: React.ComponentType<{ size?: number; strokeWidth?: number }>;
  onClick?: () => void;
  disabled?: boolean;
  active?: boolean;
  animating?: boolean;
}

function ToolboxButton({ label, icon: Icon, onClick, disabled, active, animating }: ToolboxButtonProps) {
  return (
    <span
      className={`sparse-tool-icon ${active ? "sparse-tool-icon--active" : ""} ${disabled ? "sparse-tool-icon--disabled" : ""}`}
      onClick={disabled ? undefined : onClick}
      title={label}
      tabIndex={0}
      role="button"
      aria-pressed={active}
      style={{ cursor: disabled ? "not-allowed" : "pointer", display: "flex", alignItems: "center", justifyContent: "center", width: 20, height: 20, margin: 2, opacity: disabled ? 0.38 : 1 }}
    >
      {animating ? <span className="sparse-tool-spinner" /> : <Icon size={15} strokeWidth={2} />}
    </span>
  );
}

class CanvasErrorBoundary extends React.Component<CanvasErrorBoundaryProps, CanvasErrorBoundaryState> {
  constructor(props: CanvasErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(): CanvasErrorBoundaryState {
    return { hasError: true };
  }

  componentDidCatch(err: unknown) {
    const msg = err instanceof Error ? err.message : "Unknown canvas error";
    this.props.onError(msg);
  }

  render() {
    if (this.state.hasError) {
      return this.props.fallback;
    }
    return this.props.children;
  }
}

function canCreateWebGLContext(): boolean {
  if (typeof document === "undefined") return false;
  try {
    const canvas = document.createElement("canvas");
    const gl2 = canvas.getContext("webgl2");
    if (gl2) return true;
    const gl = canvas.getContext("webgl") || canvas.getContext("experimental-webgl");
    return !!gl;
  } catch {
    return false;
  }
}

function PointsMesh({ positions, colors, size }: { positions: Float32Array; colors: Float32Array; size: number }) {
  const pointsRef = React.useRef<THREE.Points>(null!);

  const geometry = React.useMemo(() => {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    g.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    g.computeBoundingSphere();
    return g;
  }, [positions, colors]);

  useFrame(() => {
    if (pointsRef.current) {
      pointsRef.current.rotation.y += 0.0002;
    }
  });

  return (
    <points ref={pointsRef} geometry={geometry} frustumCulled={true}>
      <pointsMaterial vertexColors size={size} sizeAttenuation depthWrite={false} />
    </points>
  );
}

function CameraBridge({ onCameraReady }: { onCameraReady: (camera: THREE.Camera) => void }) {
  const { camera } = useThree();
  useEffect(() => {
    onCameraReady(camera);
  }, [camera, onCameraReady]);
  return null;
}

function FocusController({ target, bboxSize }: { target?: [number, number, number] | null; bboxSize: number }) {
  const { camera } = useThree();

  useEffect(() => {
    if (!target) return;
    const [x, y, z] = target;
    const dir = new THREE.Vector3(0, 0, 1);
    const distance = Math.max(1.5, bboxSize * 1.5);
    const newPos = new THREE.Vector3(x, y, z).addScaledVector(dir, distance);
    const startPos = camera.position.clone();
    const endPos = newPos;
    const endTarget = new THREE.Vector3(x, y, z);
    const duration = 600;
    const startTime = performance.now();

    const step = (now: number) => {
      const t = Math.min(1, (now - startTime) / duration);
      camera.position.lerpVectors(startPos, endPos, t);
      camera.lookAt(endTarget);
      camera.updateProjectionMatrix?.();
      if (t < 1) requestAnimationFrame(step);
    };

    requestAnimationFrame(step);
  }, [target, camera, bboxSize]);

  return null;
}

function getApiBase() {
  const win = typeof window !== "undefined" ? (window as any) : {};
  return (
    win.__API_BASE__ ||
    (import.meta.env ? (import.meta.env.VITE_API_BASE as string) : null) ||
    "http://localhost:8005"
  ).replace(/\/$/, "");
}

export default function SparseViewer({ projectId, focusTarget }: SparseViewerProps) {
  const [pointPositions, setPointPositions] = useState<Float32Array | null>(null);
  const [pointColors, setPointColors] = useState<Float32Array | null>(null);
  const [pointIds, setPointIds] = useState<Uint32Array | null>(null);
  const baseColorsRef = useRef<Float32Array | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [pointSize, setPointSize] = useState<number>(0.03);
  const [candidateOptions, setCandidateOptions] = useState<SparseCandidateOption[]>([
    { value: "best", label: "Auto (best available)" },
  ]);
  const [selectedCandidate, setSelectedCandidate] = useState<string>("best");
  const [candidatesLoading, setCandidatesLoading] = useState<boolean>(false);
  const [selectionRect, setSelectionRect] = useState<SelectionRect | null>(null);
  const [isSelecting, setIsSelecting] = useState(false);
  const [selectionEnabled, setSelectionEnabled] = useState(false);
  const [selectedIndices, setSelectedIndices] = useState<Set<number>>(new Set());
  const [pendingDeletes, setPendingDeletes] = useState<Set<number>>(new Set());
  const [isSaving, setIsSaving] = useState(false);
  const [saveMessage, setSaveMessage] = useState<string | null>(null);
  const [saveStatus, setSaveStatus] = useState<SaveStatus>("idle");
  const [optimizeArmed, setOptimizeArmed] = useState(false);
  const [optimizeStatus, setOptimizeStatus] = useState<OptimizeStatus>("idle");
  const [optimizeMessage, setOptimizeMessage] = useState<string | null>(null);
  const overlayRef = useRef<HTMLDivElement>(null);
  const cameraRef = useRef<THREE.Camera | null>(null);
  const [reloadToken, setReloadToken] = useState(0);
  const [webglSupport, setWebglSupport] = useState<WebGLSupportState>("checking");
  const [canvasError, setCanvasError] = useState<string | null>(null);

  const apiBase = getApiBase();

  useEffect(() => {
    setWebglSupport(canCreateWebGLContext() ? "supported" : "unsupported");
  }, []);

  const activeCandidateOption = useMemo(() => {
    return candidateOptions.find((opt) => opt.value === selectedCandidate) || candidateOptions[0];
  }, [candidateOptions, selectedCandidate]);

  const candidateRelativePath = activeCandidateOption?.relativePath ?? (
    activeCandidateOption?.value === "best" ? "." : activeCandidateOption?.value ?? "."
  );

  useEffect(() => {
    let active = true;

    const formatPoints = (points?: number | null) => {
      if (typeof points !== "number" || points <= 0) return null;
      return `${points.toLocaleString()} point${points === 1 ? "" : "s"}`;
    };

    const describeCandidate = (entry: SparseCandidateOption, bestRel?: string | null) => {
      const baseName = entry.label || (entry.relativePath && entry.relativePath !== "." ? entry.relativePath : "root");
      const pointsLabel = formatPoints(entry.points);
      const resolved = pointsLabel ? `${baseName} (${pointsLabel})` : baseName;
      if (entry.value === "best") {
        if (bestRel) {
          return `Auto (best → ${resolved})`;
        }
        return "Auto (best available)";
      }
      return resolved;
    };

    const fetchCandidates = async () => {
      setCandidatesLoading(true);
      try {
        const res = await fetch(`${apiBase}/projects/${projectId}/sparse/candidates`);
        if (!res.ok) throw new Error(`Failed to load sparse candidates: ${res.status}`);
        const data = await res.json();
        const bestRel: string | null = data?.best_relative_path ?? null;
        const rawList: SparseCandidateOption[] = Array.isArray(data?.candidates)
          ? data.candidates.map((entry: any) => ({
              value: entry?.relative_path ?? ".",
              label: entry?.label ?? (entry?.relative_path ?? "root"),
              relativePath: entry?.relative_path ?? ".",
              images: entry?.images ?? null,
              points: entry?.points ?? null,
            }))
          : [];

        const bestEntry = bestRel ? rawList.find((entry) => entry.relativePath === bestRel) ?? null : null;
        const formatted: SparseCandidateOption[] = [];

        const bestOption: SparseCandidateOption = {
          value: "best",
          label: bestEntry?.label || (bestRel && bestRel !== "." ? bestRel : "root"),
          relativePath: bestRel ?? ".",
          images: bestEntry?.images ?? null,
          points: bestEntry?.points ?? null,
        };
        bestOption.label = describeCandidate(bestOption, bestRel);
        formatted.push(bestOption);

        rawList.forEach((entry) => {
          formatted.push({
            ...entry,
            label: describeCandidate(entry),
          });
        });

        if (active) {
          setCandidateOptions(formatted);
          const validValues = formatted.map((opt) => opt.value);
          if (!validValues.includes(selectedCandidate)) {
            setSelectedCandidate("best");
          }
        }
      } catch (err) {
        if (active) {
          setCandidateOptions([{ value: "best", label: "Auto (best available)" }]);
          if (selectedCandidate !== "best") {
            setSelectedCandidate("best");
          }
        }
      } finally {
        if (active) setCandidatesLoading(false);
      }
    };

    fetchCandidates();
    return () => {
      active = false;
    };
  }, [apiBase, projectId, selectedCandidate]);

  useEffect(() => {
    let mounted = true;
    setError(null);
    setSaveMessage(null);
    setSelectedIndices(new Set());
    setPendingDeletes(new Set());
    baseColorsRef.current = null;
    setPointPositions(null);
    setPointColors(null);
    setPointIds(null);
    setSelectionEnabled(false);
    setIsSelecting(false);
    setSelectionRect(null);
    setSaveMessage(null);
    setSaveStatus("idle");
    setOptimizeStatus("idle");
    setOptimizeMessage(null);
    setOptimizeArmed(false);
    setCanvasError(null);

    const fetchPoints = async () => {
      try {
        const params = new URLSearchParams();
        if (selectedCandidate) params.set("candidate", selectedCandidate);
        params.set("mode", "editable");
        const res = await fetch(
          `${apiBase}/projects/${projectId}/download/points.bin?${params.toString()}`
        );
        if (!res.ok) throw new Error(`Failed to fetch points.bin: ${res.status}`);
        const buf = await res.arrayBuffer();
        const dv = new DataView(buf);
        if (dv.byteLength < 4) throw new Error("points.bin too small");
        const count = dv.getUint32(0, true);
        const posOffset = 4;
        const posFloatCount = count * 3;
        const posSlice = new Float32Array(buf, posOffset, posFloatCount);
        const colorsOffset = posOffset + posFloatCount * 4;
        const colorFloatCount = count * 3;
        const colorSlice = new Float32Array(buf, colorsOffset, colorFloatCount);
        const idsOffset = colorsOffset + colorFloatCount * 4;
        const idsSlice = new Uint32Array(buf, idsOffset, count);

        const positionsCopy = new Float32Array(posSlice);
        const colorsCopy = new Float32Array(colorSlice);
        const idsCopy = new Uint32Array(idsSlice);

        if (!mounted) return;
        baseColorsRef.current = colorsCopy;
        setPointPositions(positionsCopy);
        setPointColors(colorsCopy);
        setPointIds(idsCopy);
      } catch (e: any) {
        if (!mounted) return;
        setError(e.message || String(e));
      }
    };

    fetchPoints();
    return () => {
      mounted = false;
    };
  }, [apiBase, projectId, selectedCandidate, reloadToken]);

  const bbox = useMemo(() => {
    if (!pointPositions || pointPositions.length === 0) {
      return { center: [0, 0, 0] as [number, number, number], size: 1 };
    }
    let minX = Infinity,
      minY = Infinity,
      minZ = Infinity,
      maxX = -Infinity,
      maxY = -Infinity,
      maxZ = -Infinity;
    for (let i = 0; i < pointPositions.length; i += 3) {
      const x = pointPositions[i];
      const y = pointPositions[i + 1];
      const z = pointPositions[i + 2];
      if (x < minX) minX = x;
      if (y < minY) minY = y;
      if (z < minZ) minZ = z;
      if (x > maxX) maxX = x;
      if (y > maxY) maxY = y;
      if (z > maxZ) maxZ = z;
    }
    return {
      center: [
        (minX + maxX) / 2,
        (minY + maxY) / 2,
        (minZ + maxZ) / 2,
      ] as [number, number, number],
      size: Math.max(maxX - minX, maxY - minY, maxZ - minZ) || 1,
    };
  }, [pointPositions]);

  const updateSelectionColors = useCallback(
    (indices: Set<number>, color: "yellow" | "red" = "yellow") => {
      const base = baseColorsRef.current;
      if (!base) return;
      const tinted = new Float32Array(base);
      if (indices.size === 0) {
        setPointColors(tinted);
        return;
      }
      indices.forEach((idx) => {
        const offset = idx * 3;
        if (offset + 2 >= tinted.length) return;
        if (color === "red") {
          // Highlight: red
          tinted[offset] = 1.0;
          tinted[offset + 1] = 0.2;
          tinted[offset + 2] = 0.2;
        } else {
          // Highlight: yellow
          tinted[offset] = 1.0;
          tinted[offset + 1] = 0.95;
          tinted[offset + 2] = 0.2;
        }
      });
      setPointColors(tinted);
    },
    []
  );

  const selectInsideRect = useCallback(
    (rect: SelectionRect) => {
      if (!pointPositions || !cameraRef.current || !overlayRef.current) return new Set<number>();
      const overlayBounds = overlayRef.current.getBoundingClientRect();
      const width = overlayBounds.width || 1;
      const height = overlayBounds.height || 1;
      const minX = Math.min(rect.x1, rect.x2);
      const maxX = Math.max(rect.x1, rect.x2);
      const minY = Math.min(rect.y1, rect.y2);
      const maxY = Math.max(rect.y1, rect.y2);
      const selected = new Set<number>();
      const projector = new THREE.Vector3();
      for (let i = 0; i < pointPositions.length; i += 3) {
        projector.set(pointPositions[i], pointPositions[i + 1], pointPositions[i + 2]);
        projector.project(cameraRef.current);
        const screenX = (projector.x + 1) * 0.5 * width;
        const screenY = (1 - projector.y) * 0.5 * height;
        if (screenX >= minX && screenX <= maxX && screenY >= minY && screenY <= maxY) {
          selected.add(i / 3);
        }
      }
      return selected;
    },
    [pointPositions]
  );

  const handlePointerDown = useCallback(
    (event: React.MouseEvent<HTMLDivElement>) => {
      if (!selectionEnabled) return;
      if (event.button !== 0) return;
      if (!pointPositions || !pointIds) return;
      const bounds = overlayRef.current?.getBoundingClientRect();
      if (!bounds) return;
      event.preventDefault();
      const x = event.clientX - bounds.left;
      const y = event.clientY - bounds.top;
      setIsSelecting(true);
      setSelectionRect({ x1: x, y1: y, x2: x, y2: y });
    },
    [pointPositions, pointIds, selectionEnabled]
  );

  const updateSelectionEnd = useCallback((event: React.MouseEvent<HTMLDivElement>) => {
    if (!isSelecting || !selectionRect) return;
    const bounds = overlayRef.current?.getBoundingClientRect();
    if (!bounds) return;
    const x = event.clientX - bounds.left;
    const y = event.clientY - bounds.top;
    setSelectionRect((prev) => (prev ? { ...prev, x2: x, y2: y } : prev));
  }, [isSelecting, selectionRect]);

  const finalizeSelection = useCallback(() => {
    if (!isSelecting || !selectionRect) {
      setSelectionRect(null);
      setIsSelecting(false);
      return;
    }
    const indices = selectInsideRect(selectionRect);
    setSelectedIndices(indices);
    updateSelectionColors(indices, "red");
    setSelectionRect(null);
    setIsSelecting(false);
  }, [isSelecting, selectionRect, selectInsideRect, updateSelectionColors]);

  const handleDeleteSelected = useCallback(() => {
    if (!pointIds || !pointPositions || !baseColorsRef.current) return;
    if (selectedIndices.size === 0) return;
    const idsToRemove = new Set<number>();
    selectedIndices.forEach((idx) => {
      if (idx >= 0 && idx < pointIds.length) {
        idsToRemove.add(pointIds[idx]);
      }
    });
    if (idsToRemove.size === 0) return;
    const keepCount = pointIds.length - idsToRemove.size;
    const nextPositions = new Float32Array(Math.max(keepCount, 0) * 3);
    const nextColors = new Float32Array(Math.max(keepCount, 0) * 3);
    const nextIds = new Uint32Array(Math.max(keepCount, 0));
    const baseColors = baseColorsRef.current;
    let writePtr = 0;
    for (let i = 0; i < (pointIds?.length ?? 0); i++) {
      const pid = pointIds[i];
      if (idsToRemove.has(pid)) continue;
      const srcOffset = i * 3;
      const dstOffset = writePtr * 3;
      nextPositions[dstOffset] = pointPositions[srcOffset];
      nextPositions[dstOffset + 1] = pointPositions[srcOffset + 1];
      nextPositions[dstOffset + 2] = pointPositions[srcOffset + 2];
      nextColors[dstOffset] = baseColors![srcOffset];
      nextColors[dstOffset + 1] = baseColors![srcOffset + 1];
      nextColors[dstOffset + 2] = baseColors![srcOffset + 2];
      nextIds[writePtr] = pid;
      writePtr += 1;
    }
    baseColorsRef.current = nextColors;
    setPointPositions(nextPositions);
    setPointColors(nextColors);
    setPointIds(nextIds);
    setSelectedIndices(new Set());
    updateSelectionColors(new Set());
    setPendingDeletes((prev) => {
      const next = new Set(prev);
      idsToRemove.forEach((id) => next.add(id));
      return next;
    });
  }, [pointIds, pointPositions, selectedIndices, updateSelectionColors]);

  const handleReset = useCallback(() => {
    if (isSaving) return;
    setPendingDeletes(new Set());
    setSelectedIndices(new Set());
    setSelectionRect(null);
    setSelectionEnabled(false);
    setIsSelecting(false);
    setReloadToken((token) => token + 1);
    setSaveMessage(null);
    setSaveStatus("idle");
    setOptimizeStatus("idle");
    setOptimizeMessage(null);
    setOptimizeArmed(false);
  }, [isSaving]);

  const handleSave = useCallback(async () => {
    if (!pendingDeletes.size || isSaving) return;
    setIsSaving(true);
    setSaveMessage(null);
    setSaveStatus("idle");
    if (optimizeArmed) {
      setOptimizeStatus("idle");
      setOptimizeMessage(null);
    }
    try {
      const body = {
        candidate: selectedCandidate,
        remove_point_ids: Array.from(pendingDeletes),
        reoptimize: optimizeArmed,
      };
      const res = await fetch(`${apiBase}/projects/${projectId}/sparse/edit`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const payload = await res.json().catch(() => ({}));
      if (!res.ok) {
        throw new Error(payload?.detail || `Failed to save edits (${res.status})`);
      }
      setPendingDeletes(new Set());
      setSelectedIndices(new Set());
      setSaveStatus("success");
      setSaveMessage("Edits saved successfully.");
      if (optimizeArmed) {
        if (payload?.reoptimize_started) {
          setOptimizeStatus("queued");
          setOptimizeMessage("Optimize: COLMAP bundle adjustment running in background...");
        } else {
          setOptimizeStatus("error");
          setOptimizeMessage("Optimize: unable to start bundle adjustment.");
        }
      } else {
        setOptimizeStatus("idle");
        setOptimizeMessage(null);
      }
      setReloadToken((token) => token + 1);
    } catch (err: any) {
      console.error(err);
      setSaveStatus("error");
      setSaveMessage(err?.message ? `Save failed: ${err.message}` : "Save failed");
      if (optimizeArmed) {
        setOptimizeStatus("error");
        setOptimizeMessage("Optimize: failed to start due to save error.");
      }
    } finally {
      setIsSaving(false);
    }
  }, [pendingDeletes, isSaving, optimizeArmed, apiBase, projectId, selectedCandidate]);

  const toggleSelectionMode = useCallback(() => {
    setSelectionEnabled((prev) => {
      const next = !prev;
      if (!next) {
        setSelectionRect(null);
        setIsSelecting(false);
        setSelectedIndices(new Set());
        updateSelectionColors(new Set());
      } else {
        // When activating selection, reset any previous selection color
        updateSelectionColors(selectedIndices, "yellow");
      }
      return next;
    });
  }, [updateSelectionColors, selectedIndices]);

  // ESC key disables selection mode
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape" && selectionEnabled) {
        setSelectionEnabled(false);
        setSelectionRect(null);
        setIsSelecting(false);
        setSelectedIndices(new Set());
        updateSelectionColors(new Set());
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [selectionEnabled, updateSelectionColors]);

  const toggleOptimize = useCallback(() => {
    setOptimizeArmed((prev) => {
      const next = !prev;
      if (!next) {
        setOptimizeStatus("idle");
        setOptimizeMessage(null);
      }
      return next;
    });
  }, []);

  const selectionOverlayActive = selectionEnabled || isSelecting;

  if (error) {
    return <div className="h-[560px] flex items-center justify-center text-sm text-rose-600">{error}</div>;
  }

  if (webglSupport === "unsupported") {
    return (
      <div className="h-[560px] flex flex-col gap-2 items-center justify-center text-sm text-amber-700 bg-amber-50 border border-amber-200 rounded-xl px-6 text-center">
        <div className="font-semibold text-amber-800">WebGL is unavailable in this browser environment.</div>
        <div>
          Sparse point-cloud rendering needs hardware-accelerated WebGL. Enable GPU acceleration or use a browser/device where WebGL is available.
        </div>
      </div>
    );
  }

  if (canvasError) {
    return (
      <div className="h-[560px] flex flex-col gap-2 items-center justify-center text-sm text-rose-700 bg-rose-50 border border-rose-200 rounded-xl px-6 text-center">
        <div className="font-semibold text-rose-800">Unable to initialize the 3D viewer.</div>
        <div>{canvasError}</div>
      </div>
    );
  }

  if (webglSupport === "checking" || !pointPositions || !pointColors || !pointIds) {
    return (
      <div className="h-[560px] flex items-center justify-center">
        <div className="animate-spin rounded-full h-10 w-10 border-4 border-blue-500 border-t-transparent" />
      </div>
    );
  }

  const overlayClassName = `absolute inset-0 z-10 ${selectionOverlayActive ? "cursor-crosshair" : "pointer-events-none"}`;

  const selectionBoxStyles = selectionRect
    ? {
        left: `${Math.min(selectionRect.x1, selectionRect.x2)}px`,
        top: `${Math.min(selectionRect.y1, selectionRect.y2)}px`,
        width: `${Math.abs(selectionRect.x2 - selectionRect.x1)}px`,
        height: `${Math.abs(selectionRect.y2 - selectionRect.y1)}px`,
      }
    : null;

  const pendingCount = pendingDeletes.size;
  const selectedCount = selectedIndices.size;

  return (
    <div className="space-y-4">
      <div className="relative h-[560px] w-full bg-black rounded-2xl overflow-hidden">
        {/* Toolbar: top-left, ultra-compact, all icons same size */}
        <div className="sparse-toolbox sparse-toolbox-row bg-white/90 rounded shadow p-0.5 flex flex-col gap-0.5 absolute left-2 top-2 z-30 min-w-0 min-h-0" style={{boxShadow:'0 1px 4px 0 rgba(0,0,0,0.04)'}}>
          <ToolboxButton label={selectionEnabled ? "Selection active" : "Activate selection"} icon={BoxSelect} onClick={toggleSelectionMode} active={selectionEnabled} />
          <ToolboxButton label="Delete selection" icon={Trash2} onClick={handleDeleteSelected} disabled={selectedCount === 0} />
          <ToolboxButton label="Reset edits" icon={RotateCcw} onClick={handleReset} disabled={isSaving} />
          <ToolboxButton label="Save edits" icon={Save} onClick={handleSave} disabled={pendingCount === 0 || isSaving} animating={isSaving} />
          <ToolboxButton label={optimizeArmed ? "Optimize armed" : "Toggle optimize"} icon={Sparkles} onClick={toggleOptimize} active={optimizeArmed} animating={isSaving && optimizeArmed} />
          {/* Zoom controls, now using ToolboxButton for consistency */}
          <ToolboxButton label="Zoom in" icon={() => <span style={{fontWeight:700,fontSize:15,lineHeight:1,display:'inline-block',width:15,textAlign:'center'}}>+</span>} onClick={() => {const cam=cameraRef.current;if(cam&&(cam instanceof (THREE.PerspectiveCamera)||cam instanceof (THREE.OrthographicCamera))){cam.zoom=Math.min((cam.zoom||1)*1.2,20);cam.updateProjectionMatrix();}}} />
          <ToolboxButton label="Zoom out" icon={() => <span style={{fontWeight:700,fontSize:15,lineHeight:1,display:'inline-block',width:15,textAlign:'center'}}>-</span>} onClick={() => {const cam=cameraRef.current;if(cam&&(cam instanceof (THREE.PerspectiveCamera)||cam instanceof (THREE.OrthographicCamera))){cam.zoom=Math.max((cam.zoom||1)/1.2,0.1);cam.updateProjectionMatrix();}}} />
        </div>
        {/* Controls: top-right, point size slider above dropdown, points count above dropdown */}
        <div className="absolute right-2 top-2 z-30 flex flex-col items-end gap-1 bg-white/95 rounded-lg shadow p-2 min-w-[170px]">
          <div className="flex flex-row items-center gap-2 w-full mb-1">
            <label htmlFor="point-size-slider" className="text-xs text-slate-600 whitespace-nowrap">Point size:</label>
            <input
              id="point-size-slider"
              type="range"
              min={0.01}
              max={0.15}
              step={0.01}
              value={pointSize}
              onChange={e => setPointSize(Number(e.target.value))}
              className="w-20 h-2 accent-blue-500"
            />
            <span className="text-xs text-slate-700 w-8 text-right">{pointSize.toFixed(2)}</span>
          </div>
          <div className="text-xs text-slate-600 w-full text-right mb-1">
            Sparse candidate: {activeCandidateOption?.points?.toLocaleString() ?? "?"} points
          </div>
          <div className="w-full">
            <select
              id="candidate-select"
              className="border border-slate-300 rounded px-1 py-0.5 text-xs bg-white focus:outline-none focus:ring-2 focus:ring-blue-400 w-full"
              value={selectedCandidate}
              onChange={e => setSelectedCandidate(e.target.value)}
              disabled={candidatesLoading}
            >
              {candidateOptions.map(opt => (
                <option key={opt.value} value={opt.value}>{opt.label}</option>
              ))}
            </select>
          </div>
        </div>
        <div
          ref={overlayRef}
          className={overlayClassName}
          onMouseDown={handlePointerDown}
          onMouseMove={(event) => {
            if (!isSelecting) return;
            updateSelectionEnd(event);
          }}
          onMouseUp={finalizeSelection}
          onMouseLeave={finalizeSelection}
        />
        {selectionBoxStyles && (
          <div
            className="absolute z-30 border border-blue-400 bg-blue-500/10"
            style={selectionBoxStyles}
          />
        )}
        <CanvasErrorBoundary
          onError={(message) => setCanvasError(message || "Error creating WebGL context")}
          fallback={
            <div className="h-full w-full flex items-center justify-center text-sm text-rose-200 bg-black/90">
              Error creating WebGL context.
            </div>
          }
        >
          <Canvas
            dpr={Math.min(2, window.devicePixelRatio || 1)}
            camera={{
              position: [
                bbox.center[0],
                bbox.center[1],
                bbox.center[2] + Math.max(1.5, bbox.size * 1.5),
              ],
              fov: 45,
            }}
            onCreated={() => {
              if (canvasError) setCanvasError(null);
            }}
          >
            <color attach="background" args={[0, 0, 0]} />
            <ambientLight intensity={0.6} />
            <pointLight position={[10, 10, 10]} intensity={0.8} />
            <Suspense fallback={null}>
              <PointsMesh positions={pointPositions} colors={pointColors} size={pointSize} />
              <OrbitControls makeDefault enableDamping dampingFactor={0.08} />
              <CameraBridge onCameraReady={(camera) => (cameraRef.current = camera)} />
              <FocusController target={focusTarget ?? null} bboxSize={bbox.size} />
            </Suspense>
          </Canvas>
        </CanvasErrorBoundary>
        {/* Show number of selected points */}
        {selectedCount > 0 && (
          <div className="absolute left-3 bottom-3 z-30 bg-white/90 rounded px-2 py-1 text-xs font-semibold text-red-700 border border-red-200 shadow">
            {selectedCount} point{selectedCount === 1 ? "" : "s"} selected
          </div>
        )}
      </div>
      <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-4 flex flex-col gap-2 text-sm">
        <div className="text-xs text-slate-600">
          <div>
            Editing candidate:
            <strong className="ml-1 text-slate-900">{candidateRelativePath}</strong>
          </div>
          <div>Selection tool: {selectionEnabled ? "active" : "off"}</div>
          <div>Optimize after save: {optimizeArmed ? "enabled" : "off"}</div>
        </div>
        <div className="flex flex-wrap gap-2">
          {saveMessage && (
            <span className={`sparse-tool-status ${saveStatus === "error" ? "error" : "success"}`}>
              {saveMessage}
            </span>
          )}
          {optimizeMessage && (
            <span className={`sparse-tool-status ${optimizeStatus === "error" ? "error" : "info"}`}>
              {optimizeMessage}
            </span>
          )}
        </div>
      </div>
      <div className="bg-gray-50 p-4 text-sm text-gray-600 rounded-xl border border-gray-200">
        <strong>Controls:</strong> Use the toolbar icons to activate selection, delete, reset, save, or arm optimization. When selection is active, drag to box-select points. Orbit/zoom controls remain available while selection is off.
      </div>

    </div>
  );
}
