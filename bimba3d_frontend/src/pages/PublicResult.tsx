import { useMemo } from "react";
import { useParams } from "react-router-dom";
import ThreeDViewer from "../components/ThreeDViewer";
import { api } from "../api/client";

export default function PublicResult() {
  const { id } = useParams<{ id: string }>();
  const splatUrl = useMemo(() => {
    if (!id) return "";
    const baseURL = (api.defaults.baseURL || "").replace(/\/$/, "");
    return `${baseURL}/projects/${id}/download/splats`;
  }, [id]);

  return (
    <div className="w-screen h-screen bg-black overflow-hidden">
      {id ? <ThreeDViewer splatsUrl={splatUrl} /> : null}
    </div>
  );
}
