import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import Dashboard from "./pages/Dashboard.jsx";
import Screening from "./pages/Screening.jsx";
import Analysis from "./pages/Analysis.jsx";
import Explainability from "./pages/Explainability.jsx";
import Assistant from "./pages/Assistant.jsx";
import Reports from "./pages/Reports.jsx";
import Tracking from "./pages/Tracking.jsx";
import Layout from "./components/Layout.jsx";
import { ScreeningProvider } from "./hooks/useScreening.js";

export default function App() {
  return <BrowserRouter><ScreeningProvider><Routes>
    <Route element={<Layout />}>
      <Route path="/" element={<Dashboard />} />
      <Route path="/screening" element={<Screening />} />
      <Route path="/analysis" element={<Analysis />} />
      <Route path="/explainability" element={<Explainability />} />
      <Route path="/assistant" element={<Assistant />} />
      <Route path="/reports" element={<Reports />} />
      <Route path="/tracking" element={<Tracking />} />
    </Route>
    <Route path="*" element={<Navigate to="/" replace />} />
  </Routes></ScreeningProvider></BrowserRouter>;
}
