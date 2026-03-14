import { useEffect, useState } from "react";

/**
 * Shows a banner if Vite HMR/WebSocket is disconnected (dev only).
 */
export function HMRStatusBanner() {
  const [hmrLost, setHmrLost] = useState(false);

  useEffect(() => {
    if (!import.meta.hot) return;
    // Vite HMR client attaches to window.__vite_ws or similar, but we can listen for disconnects
    // Listen for Vite's HMR websocket closing
    const checkHMR = () => {
      // @ts-ignore
      const viteSocket = window.__vite_ws || window.__vite_client_ws;
      if (viteSocket && viteSocket.readyState !== 1) {
        setHmrLost(true);
      } else {
        setHmrLost(false);
      }
    };
    // Poll every 2s
    const interval = setInterval(checkHMR, 2000);
    return () => clearInterval(interval);
  }, []);

  if (!hmrLost) return null;
  return (
    <div style={{
      position: "fixed",
      top: 0,
      left: 0,
      width: "100vw",
      zIndex: 9999,
      background: "#fee",
      color: "#b00",
      fontWeight: 600,
      padding: "8px 0",
      textAlign: "center",
      boxShadow: "0 2px 8px 0 rgba(0,0,0,0.08)",
    }}>
      Lost connection to dev server (HMR/WebSocket). Please refresh the page.
    </div>
  );
}
