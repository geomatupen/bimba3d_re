import { useMemo } from "react";
import { ArrowLeft } from "lucide-react";
import { useNavigate, useParams } from "react-router-dom";
import ThreeDViewer from "../components/ThreeDViewer";
import { api } from "../api/client";

export default function PublicResult() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const splatUrl = useMemo(() => {
    if (!id) return "";
    const baseURL = (api.defaults.baseURL || "").replace(/\/$/, "");
    return `${baseURL}/projects/${id}/download/splats`;
  }, [id]);

  const handleBack = () => {
    if (window.history.length > 1) {
      navigate(-1);
      return;
    }
    navigate("/");
  };

  return (
    <div className="w-screen h-screen bg-black overflow-hidden">
      <button
        onClick={handleBack}
        className="absolute top-4 left-4 z-50 inline-flex items-center gap-2 px-3 py-2 rounded-xl bg-black/55 hover:bg-black/70 backdrop-blur-sm border border-white/20 text-white text-sm font-medium transition-colors"
        aria-label="Go back"
      >
        <ArrowLeft className="w-4 h-4" />
        Back
      </button>
      {id ? <ThreeDViewer splatsUrl={splatUrl} /> : null}
    </div>
  );
}
