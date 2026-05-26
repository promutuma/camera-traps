import { BrowserRouter, Routes, Route } from "react-router-dom";
import Sidebar from "./components/Layout/Sidebar";
import Upload from "./pages/Upload";
import Results from "./pages/Results";
import Statistics from "./pages/Statistics";
import History from "./pages/History";
import Diagnostics from "./pages/Diagnostics";
import Ecological from "./pages/Ecological";
import QC from "./pages/QC";
import Stations from "./pages/Stations";
import ReviewQueue from "./pages/ReviewQueue";
import Community from "./pages/Community";
import Spatial from "./pages/Spatial";
import SpeciesLibrary from "./pages/SpeciesLibrary";
import Corridor from "./pages/Corridor";
import ProjectConfig from "./pages/ProjectConfig";
import ArcGIS from "./pages/ArcGIS";

export default function App() {
  return (
    <BrowserRouter>
      <div className="flex h-screen overflow-hidden bg-slate-50">
        <Sidebar />
        <div className="flex flex-col flex-1 overflow-hidden">
          <main className="flex-1 overflow-y-auto p-8">
            <Routes>
              <Route path="/" element={<Upload />} />
              <Route path="/results" element={<Results />} />
              <Route path="/statistics" element={<Statistics />} />
              <Route path="/history" element={<History />} />
              <Route path="/diagnostics" element={<Diagnostics />} />
              <Route path="/ecological" element={<Ecological />} />
              <Route path="/qc" element={<QC />} />
              <Route path="/stations" element={<Stations />} />
              <Route path="/review-queue" element={<ReviewQueue />} />
              <Route path="/community" element={<Community />} />
              <Route path="/spatial" element={<Spatial />} />
              <Route path="/species" element={<SpeciesLibrary />} />
              <Route path="/corridor" element={<Corridor />} />
              <Route path="/project" element={<ProjectConfig />} />
              <Route path="/arcgis" element={<ArcGIS />} />
            </Routes>
          </main>
        </div>
      </div>
    </BrowserRouter>
  );
}
