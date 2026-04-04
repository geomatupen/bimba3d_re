import { useParams, Link } from "react-router-dom";
import { useEffect, useState } from "react";
import { ArrowLeft, Images, Play, FileText, Columns2, Boxes } from "lucide-react";
import { api } from "../api/client";

// Tab components
import ImagesTab from "../components/tabs/ImagesTab";
import ProcessTab from "../components/tabs/ProcessTab";
import LogsTab from "../components/tabs/LogsTab";
import ComparisonTab from "../components/tabs/ComparisonTab";
import SessionsTab from "../components/tabs/SessionsTab";

type TabType = "images" | "process" | "logs" | "sessions" | "comparison";

interface ProjectStatus {
  project_id: string;
  status: string;
  progress: number;
  name?: string | null;
  created_at?: string | null;
}

export default function ProjectDetail() {
  const { id } = useParams<{ id: string }>();
  const [activeTab, setActiveTab] = useState<TabType>("images");
  const [projectStatus, setProjectStatus] = useState<ProjectStatus | null>(null);
  const [hasImages, setHasImages] = useState(false);
  const [initialTabChosen, setInitialTabChosen] = useState(false);

  useEffect(() => {
    if (!id) return;

    const fetchStatus = async () => {
      try {
        const res = await api.get(`/projects/${id}/status`);
        setProjectStatus(res.data);
        
        // Check if has images
        const filesRes = await api.get(`/projects/${id}/files`);
        const has = filesRes.data.files?.images?.length > 0;
        setHasImages(has);
        if (!initialTabChosen) {
          setActiveTab(has ? "process" : "images");
          setInitialTabChosen(true);
        }
        
        // Auto-switch to process tab if processing starts
        if (res.data.status === "processing" && activeTab === "images") {
          setActiveTab("process");
        }
      } catch (err) {
        console.error("Failed to fetch project:", err);
      }
    };

    fetchStatus();
    const interval = setInterval(fetchStatus, 3000);
    return () => clearInterval(interval);
  }, [id, activeTab]);

  const tabs = [
    { id: "images" as TabType, label: "Images", icon: Images, enabled: true },
    { id: "process" as TabType, label: "Process", icon: Play, enabled: hasImages },
    { id: "logs" as TabType, label: "Logs", icon: FileText, enabled: true },
    { id: "sessions" as TabType, label: "Sessions", icon: Boxes, enabled: true },
    { id: "comparison" as TabType, label: "Comparison", icon: Columns2, enabled: true },
    // viewer tab removed: viewer functionality is available inside the Process tab
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-slate-50">
      {/* Header */}
      <header className="bg-gradient-to-r from-blue-600 via-blue-700 to-indigo-700 shadow-xl">
        <div className="max-w-7xl mx-auto px-6 lg:px-8 py-7">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <Link
                to="/"
                className="inline-flex items-center gap-2 px-3 py-2 rounded-xl bg-white/10 hover:bg-white/20 backdrop-blur-sm border border-white/20 text-white text-sm font-medium transition-all duration-200 hover:scale-105"
              >
                <ArrowLeft className="w-4 h-4" />
                Back
              </Link>
              <div>
                <div className="inline-flex items-center gap-2 px-2 py-0.5 rounded-full bg-white/10 backdrop-blur-sm border border-white/20 mb-1">
                  <span className="text-xs font-medium text-white uppercase tracking-wider">Project</span>
                </div>
                <h1 className="text-2xl font-bold text-white mb-1">
                  {projectStatus?.name || `Project ${id?.slice(0, 8)}`}
                </h1>
                <p className="text-xs text-blue-100 font-mono">ID: {id}</p>
              </div>
            </div>
            <div className="flex items-center gap-3">
              {projectStatus && (
                <span className={`px-4 py-2 rounded-xl text-xs font-semibold shadow-lg backdrop-blur-sm border-2 ${
                  projectStatus.status === "completed" || projectStatus.status === "done"
                    ? "bg-emerald-50/90 text-emerald-700 border-emerald-200"
                    : projectStatus.status === "processing"
                    ? "bg-blue-50/90 text-blue-700 border-blue-200"
                    : projectStatus.status === "failed"
                    ? "bg-rose-50/90 text-rose-700 border-rose-200"
                    : "bg-white/90 text-slate-700 border-slate-200"
                }`}>
                  {projectStatus.status}
                </span>
              )}
            </div>
          </div>
        </div>
      </header>

      {/* Tabs Navigation */}
      <div className="bg-white border-b-2 border-slate-200 shadow-sm">
        <div className="max-w-7xl mx-auto px-6 lg:px-8">
          <nav className="flex space-x-1" aria-label="Tabs">
            {tabs.map((tab) => {
              const Icon = tab.icon;
              const isActive = activeTab === tab.id;
              const isDisabled = !tab.enabled;

              return (
                <button
                  key={tab.id}
                  onClick={() => !isDisabled && setActiveTab(tab.id)}
                  disabled={isDisabled}
                  className={`
                    flex items-center gap-2 px-6 py-4 text-sm font-semibold border-b-3 transition-all duration-200
                    ${isActive
                      ? "border-blue-600 text-blue-700 bg-blue-50/50"
                      : isDisabled
                      ? "border-transparent text-slate-400 cursor-not-allowed"
                      : "border-transparent text-slate-600 hover:text-slate-900 hover:bg-slate-50 hover:border-slate-200"
                    }
                  `}
                >
                  <Icon className="w-5 h-5" />
                  {tab.label}
                </button>
              );
            })}
          </nav>
        </div>
      </div>

      {/* Tab Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-2">
        {activeTab === "images" && (
          <ImagesTab
            projectId={id!}
            onUploaded={() => {
              setHasImages(true);
              setActiveTab("process");
            }}
          />
        )}
        {activeTab === "process" && <ProcessTab projectId={id!} />}
        {activeTab === "logs" && <LogsTab projectId={id!} />}
        {activeTab === "sessions" && <SessionsTab projectId={id!} />}
        {activeTab === "comparison" && <ComparisonTab currentProjectId={id!} />}
        {/* Viewer tab removed; viewer is available inside the Process tab */}
      </main>
    </div>
  );
}
