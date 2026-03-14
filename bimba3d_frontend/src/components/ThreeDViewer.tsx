import { useEffect, useRef } from "react";
// @ts-ignore - no types available
import * as GaussianSplats3D from "@mkkellogg/gaussian-splats-3d";

const LARGE_FILE_PROGRESSIVE_THRESHOLD_BYTES = 35 * 1024 * 1024; // ~35 MB
const LARGE_SCENE_GPU_TUNING_THRESHOLD_BYTES = 28 * 1024 * 1024; // heuristic for GPU pressure
const LARGE_FILE_NON_ISOLATED_PLY_FALLBACK_BYTES = 80 * 1024 * 1024; // ~80 MB
const DEFAULT_LOAD_TIMEOUT_MS = 30_000;
const PROGRESSIVE_LOAD_TIMEOUT_MS = 120_000;

interface SceneTarget {
  url: string;
  format: number | null;
  sizeBytes?: number | null;
}

interface UrlProbeResult {
  exists: boolean;
  sizeBytes?: number | null;
}

interface ViewerProps {
  splatsUrl?: string;
  metadataUrl?: string;
  onLoaded?: () => void;
  onError?: (error: string) => void;
}

export default function ThreeDViewer({
  splatsUrl,
  onLoaded,
  onError,
}: ViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const viewerRef = useRef<any>(null);

  useEffect(() => {
    if (!splatsUrl) return;

    let cleanup: (() => void) | undefined;
    let aborted = false;

    (async () => {
      const target = await resolveSceneTarget(splatsUrl);
      if (aborted || !target) return;

      console.log(
        "ThreeDViewer: Attempting to load splats from",
        target.url,
        target.sizeBytes ? `(size≈${formatBytes(target.sizeBytes)})` : ""
      );
      cleanup = initSplatRenderer(target);
    })();

    return () => {
      aborted = true;
      cleanup?.();
    };
  }, [splatsUrl, onLoaded, onError]);

  function initSplatRenderer(target: SceneTarget) {
    if (!containerRef.current) return;

    console.log("ThreeDViewer: Initializing Gaussian Splats renderer");

    // Check WebGL support
    const canvas = document.createElement("canvas");
    const webglContext = canvas.getContext("webgl2") || canvas.getContext("webgl");
    if (!webglContext) {
      console.error("WebGL not supported in this environment");
      onError?.("WebGL not available - this environment may not support 3D rendering");
      return;
    }
    console.log("✓ WebGL context available:", webglContext.getParameter(webglContext.VENDOR));

    let loadTimeout: any;
    let isDisposed = false;

    const { url, format, sizeBytes } = target;
    const canUseSharedBuffers = typeof window !== "undefined" && window.crossOriginIsolated === true;
    const shouldProgressivelyLoad = shouldUseProgressiveLoading(sizeBytes);
    const largeSceneTweaks = shouldApplyLargeSceneTweaks(sizeBytes);
    const timeoutMs = shouldProgressivelyLoad ? PROGRESSIVE_LOAD_TIMEOUT_MS : DEFAULT_LOAD_TIMEOUT_MS;

    if (!canUseSharedBuffers) {
      console.warn(
        "ThreeDViewer: Cross-origin isolation disabled; falling back to copy-based worker. Large splats may load slower."
      );
    }

    try {
      // Create a separate div for the viewer to manage, so React doesn't interfere with its DOM
      const viewerContainer = document.createElement("div");
      viewerContainer.style.width = "100%";
      viewerContainer.style.height = "100%";
      containerRef.current.appendChild(viewerContainer);

      const viewer = new GaussianSplats3D.Viewer({
        rootElement: viewerContainer,
        cameraUp: [0, 1, 0],
        initialCameraPosition: [0, 0, 5],
        initialCameraLookAt: [0, 0, 0],
        sharedMemoryForWorkers: canUseSharedBuffers,
        gpuAcceleratedSort: canUseSharedBuffers && !largeSceneTweaks,
        integerBasedSort: !largeSceneTweaks,
        splatSortDistanceMapPrecision: largeSceneTweaks ? 24 : 16,
        ignoreDevicePixelRatio: largeSceneTweaks,
        freeIntermediateSplatData: largeSceneTweaks,
        inMemoryCompressionLevel: largeSceneTweaks ? 1 : 0,
        rendererParameters: {
          alpha: true,
          antialias: true,
          preserveDrawingBuffer: true,
          powerPreference: "high-performance",
          failIfMajorPerformanceCaveat: false,
          stencil: false,
          depth: true,
          logarithmicDepthBuffer: true,
        },
      });

      loadTimeout = setTimeout(() => {
        if (!isDisposed) {
          console.warn("Splat load timeout - giving up");
          try {
            viewer.dispose?.();
          } catch (e) {
            console.warn("Error during timeout disposal:", e);
          }
          isDisposed = true;
          onError?.("Load timeout");
        }
      }, timeoutMs);

      const viewerOptions: Record<string, any> = {
        splatAlphaRemovalThreshold: 0,
        showLoadingUI: true,
        position: [0, 0, 0],
        rotation: [0, 0, 0, 1],
        scale: [1, 1, 1],
      };
      if (format !== null) {
        viewerOptions.format = format;
      }
      if (shouldProgressivelyLoad) {
        viewerOptions.progressiveLoad = true;
        viewerOptions.sceneRevealMode = GaussianSplats3D?.SceneRevealMode?.Gradual;
      }

      const webglCanvas = viewer.renderer?.domElement as HTMLCanvasElement | undefined;
      const handleContextLost = (event: Event) => {
        event?.preventDefault?.();
        console.error("ThreeDViewer: WebGL context lost - likely out of GPU memory.");
        onError?.(
          "Browser ran out of GPU memory while loading this splat. Try reloading or generating a lighter export."
        );
      };
      webglCanvas?.addEventListener("webglcontextlost", handleContextLost, { once: true });

      viewer
        .addSplatScene(url, viewerOptions)
        .then(() => {
          clearTimeout(loadTimeout);
          if (isDisposed) return;

          console.log("✓ Gaussian splats loaded successfully");
          onLoaded?.();
        })
        .catch((error: Error) => {
          clearTimeout(loadTimeout);
          if (isDisposed) return;

          console.error("✗ Failed to load splats:", error);
          try {
            viewer.dispose?.();
          } catch (e) {
            console.warn("Error during error disposal:", e);
          }
          isDisposed = true;
          onError?.(error.message);
        });

      viewer.start?.();
      viewerRef.current = viewer;

      return () => {
        clearTimeout(loadTimeout);
        isDisposed = true;
        try {
          viewer.stop?.();
          viewer.dispose?.();
          webglCanvas?.removeEventListener("webglcontextlost", handleContextLost);
        } catch (e) {
          console.warn("Error during cleanup disposal:", e);
        }
        // Safely remove the viewer container if it's still in the DOM
        try {
          if (containerRef.current?.contains(viewerContainer)) {
            containerRef.current.removeChild(viewerContainer);
          }
        } catch (e) {
          console.warn("Error removing viewer container:", e);
        }
      };
    } catch (error) {
      clearTimeout(loadTimeout);
      console.error("Failed to initialize Gaussian Splats:", error);
      onError?.((error as Error).message);
      return;
    }
  }

  async function resolveSceneTarget(baseUrl: string): Promise<SceneTarget | null> {
    const explicitFormat = getSceneFormatFromUrl(baseUrl);

    if (explicitFormat !== null) {
      const probe = await probeUrl(baseUrl);
      return {
        url: baseUrl,
        format: explicitFormat,
        sizeBytes: probe.sizeBytes ?? null,
      };
    }

    const splatUrl = baseUrl.replace("/download/splats", "/download/splats.splat");
    const plyUrl = baseUrl.replace("/download/splats", "/download/splats.ply");

    const [splatProbe, plyProbe] = await Promise.all([probeUrl(splatUrl), probeUrl(plyUrl)]);

    // For very large scenes without cross-origin isolation, .ply is often more reliable
    // than copy-based .splat worker loading in browsers.
    const preferPlyForLargeNonIsolatedScene =
      splatProbe.exists &&
      plyProbe.exists &&
      (splatProbe.sizeBytes ?? 0) >= LARGE_FILE_NON_ISOLATED_PLY_FALLBACK_BYTES &&
      (typeof window !== "undefined" ? window.crossOriginIsolated !== true : true);

    if (preferPlyForLargeNonIsolatedScene) {
      return {
        url: plyUrl,
        format: getSceneFormatFromUrl(plyUrl),
        sizeBytes: plyProbe.sizeBytes ?? splatProbe.sizeBytes ?? null,
      };
    }

    if (splatProbe.exists) {
      return {
        url: splatUrl,
        format: getSceneFormatFromUrl(splatUrl),
        sizeBytes: splatProbe.sizeBytes ?? null,
      };
    }

    if (plyProbe.exists) {
      return {
        url: plyUrl,
        format: getSceneFormatFromUrl(plyUrl),
        sizeBytes: plyProbe.sizeBytes ?? null,
      };
    }

    // Fallback to original URL even if probes failed, so at least we attempt to render something
    return {
      url: baseUrl,
      format: getSceneFormatFromUrl(baseUrl),
      sizeBytes: splatProbe.sizeBytes ?? plyProbe.sizeBytes ?? null,
    };
  }

  async function probeUrl(url: string): Promise<UrlProbeResult> {
    const parseResult = (headers: Headers): number | null => {
      const contentLength = headers.get("content-length");
      if (contentLength) {
        const value = Number(contentLength);
        return Number.isFinite(value) ? value : null;
      }
      const contentRange = headers.get("content-range");
      if (contentRange) {
        const match = contentRange.match(/\/(\d+)$/);
        if (match?.[1]) {
          const value = Number(match[1]);
          return Number.isFinite(value) ? value : null;
        }
      }
      return null;
    };

    try {
      const head = await fetch(url, { method: "HEAD", cache: "no-store" });
      if (head.ok) {
        return { exists: true, sizeBytes: parseResult(head.headers) };
      }
      if (head.status === 405 || head.status === 501) {
        const tinyProbe = await fetch(url, {
          method: "GET",
          cache: "no-store",
          headers: { Range: "bytes=0-0" },
        });
        if (tinyProbe.ok) {
          return { exists: true, sizeBytes: parseResult(tinyProbe.headers) };
        }
      }
    } catch (err) {
      console.warn("HEAD probe failed for", url, err);
    }

    return { exists: false };
  }

  function getSceneFormatFromUrl(url: string | undefined | null): number | null {
    if (!url) return null;
    const sanitized = url.split(/[?#]/)[0] ?? url;
    const loaderUtils = GaussianSplats3D?.LoaderUtils;
    if (loaderUtils?.sceneFormatFromPath) {
      return loaderUtils.sceneFormatFromPath(sanitized) ?? null;
    }
    if (/\.splat$/i.test(sanitized)) return GaussianSplats3D?.SceneFormat?.Splat ?? null;
    if (/\.ply$/i.test(sanitized)) return GaussianSplats3D?.SceneFormat?.Ply ?? null;
    if (/\.ksplat$/i.test(sanitized)) return GaussianSplats3D?.SceneFormat?.KSplat ?? null;
    if (/\.spz$/i.test(sanitized)) return GaussianSplats3D?.SceneFormat?.Spz ?? null;
    return null;
  }

  function shouldUseProgressiveLoading(sizeBytes?: number | null) {
    if (!sizeBytes) return false;
    return sizeBytes >= LARGE_FILE_PROGRESSIVE_THRESHOLD_BYTES;
  }

  function shouldApplyLargeSceneTweaks(sizeBytes?: number | null) {
    if (!sizeBytes) return false;
    return sizeBytes >= LARGE_SCENE_GPU_TUNING_THRESHOLD_BYTES;
  }

  function formatBytes(size: number) {
    if (!Number.isFinite(size)) return `${size}`;
    const units = ["B", "KB", "MB", "GB"] as const;
    let current = size;
    let unitIndex = 0;
    while (current >= 1024 && unitIndex < units.length - 1) {
      current /= 1024;
      unitIndex += 1;
    }
    return `${current.toFixed(current >= 100 || unitIndex === 0 ? 0 : 1)}${units[unitIndex]}`;
  }

  return (
    <div
      ref={containerRef}
      style={{
        width: "100%",
        height: "600px",
        borderRadius: "8px",
        overflow: "hidden",
        backgroundColor: "#1a1a1a",
      }}
    />
  );
}
