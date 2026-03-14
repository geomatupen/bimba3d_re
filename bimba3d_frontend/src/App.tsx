import { Routes, Route } from "react-router-dom";
import Dashboard from "./pages/Dashboard";
import CreateProject from "./pages/CreateProject";
import ProjectDetail from "./pages/ProjectDetail";
import Comparison from "./pages/Comparison";
import { HMRStatusBanner } from "./HMRStatusBanner";

function App() {
  return (
    <>
      {import.meta.env.DEV && <HMRStatusBanner />}
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/create" element={<CreateProject />} />
        <Route path="/project/:id" element={<ProjectDetail />} />
        <Route path="/comparison/:id" element={<Comparison />} />
      </Routes>
    </>
  );
}

export default App;
