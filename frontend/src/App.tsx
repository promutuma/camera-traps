import { BrowserRouter, Routes, Route } from "react-router-dom";
import Sidebar from "./components/Layout/Sidebar";
import ErrorBoundary from "./components/ErrorBoundary";
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

function wrap(label: string, element: React.ReactElement) {
  return <ErrorBoundary label={label}>{element}</ErrorBoundary>;
}

export default function App() {
  return (
    <BrowserRouter>
      <div className="flex h-screen overflow-hidden bg-slate-50 dark:bg-slate-950 text-slate-900 dark:text-slate-100 transition-colors duration-200">
        <Sidebar />
        <div className="flex flex-col flex-1 overflow-hidden">
          <main className="flex-1 overflow-y-auto p-8">
            <Routes>
              <Route path="/"             element={wrap("Upload",          <Upload />)} />
              <Route path="/results"      element={wrap("Results",         <Results />)} />
              <Route path="/statistics"   element={wrap("Statistics",      <Statistics />)} />
              <Route path="/history"      element={wrap("History",         <History />)} />
              <Route path="/diagnostics"  element={wrap("Diagnostics",     <Diagnostics />)} />
              <Route path="/ecological"   element={wrap("Ecological",      <Ecological />)} />
              <Route path="/qc"           element={wrap("QC",              <QC />)} />
              <Route path="/stations"     element={wrap("Stations",        <Stations />)} />
              <Route path="/review-queue" element={wrap("Review Queue",    <ReviewQueue />)} />
              <Route path="/community"    element={wrap("Community",       <Community />)} />
              <Route path="/spatial"      element={wrap("Spatial",         <Spatial />)} />
              <Route path="/species"      element={wrap("Species Library", <SpeciesLibrary />)} />
              <Route path="/corridor"     element={wrap("Corridor",        <Corridor />)} />
              <Route path="/project"      element={wrap("Project Config",  <ProjectConfig />)} />
              <Route path="/arcgis"       element={wrap("ArcGIS",          <ArcGIS />)} />
            </Routes>
          </main>
        </div>
      </div>
    </BrowserRouter>
  );
}
