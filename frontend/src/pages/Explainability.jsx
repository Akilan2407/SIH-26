import { useEffect, useRef, useState } from "react";
import { AlertCircle, BarChart2, Eye, Info, Sparkles } from "lucide-react";
import { getGradCAM, getSHAP } from "../api/explainabilityApi.js";
import { useScreening } from "../hooks/useScreening.js";

const SESSION_KEY = "dr_chatbot_session_id";
const CLASS_COLORS = ["#22c55e", "#84cc16", "#f59e0b", "#f97316", "#ef4444"];
const CLASS_ORDER = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"];

// ---------------------------------------------------------------------------
// Grad-CAM visual card
// ---------------------------------------------------------------------------
function Visual({ title, src, overlay }) {
  return (
    <div className="visual-card">
      <div className="visual-head">
        <span>{title}</span>
        <span className="visual-chip">{overlay ? "ATTENTION" : "INPUT"}</span>
      </div>
      <div className="visual">
        <img src={src} alt={title} />
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// SHAP horizontal bar — one feature row
// ---------------------------------------------------------------------------
function SHAPBar({ entry, maxAbs }) {
  const pct = maxAbs > 0 ? (Math.abs(entry.shap) / maxAbs) * 100 : 0;
  const isRisk = entry.direction === "increases_risk";
  const barColor = isRisk ? "var(--shap-risk)" : "var(--shap-protect)";

  return (
    <div className="shap-row" title={entry.imputed ? "Value estimated from cohort average" : ""}>
      <div className="shap-label">
        <span className="shap-feat-name">{entry.label}</span>
        {entry.imputed && <span className="shap-imputed-badge">estimated</span>}
      </div>
      <div className="shap-bar-track">
        <div
          className="shap-bar-fill"
          style={{ width: `${pct}%`, background: barColor }}
        />
      </div>
      <div className="shap-value-col">
        <span className="shap-val" style={{ color: barColor }}>
          {entry.shap >= 0 ? "+" : ""}
          {entry.shap.toFixed(3)}
        </span>
        <span className="shap-raw-val">({entry.value})</span>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// DR grade badge
// ---------------------------------------------------------------------------
function GradeBadge({ grade, label, confidence }) {
  const color = CLASS_COLORS[grade] ?? "#64748b";
  return (
    <div className="shap-grade-badge" style={{ borderColor: color }}>
      <span className="shap-grade-dot" style={{ background: color }} />
      <div>
        <strong style={{ color }}>{label}</strong>
        <span>{confidence != null ? `${(confidence * 100).toFixed(1)}% confidence` : ""}</span>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Class probability mini-bars
// ---------------------------------------------------------------------------
function ProbabilityBars({ probabilities }) {
  if (!probabilities) return null;
  return (
    <div className="shap-prob-section">
      <p className="shap-section-title">Class probabilities</p>
      {CLASS_ORDER.map((cls, i) => {
        const prob = probabilities[cls] ?? 0;
        return (
          <div className="shap-prob-row" key={cls}>
            <span className="shap-prob-label">{cls}</span>
            <div className="shap-prob-track">
              <div
                className="shap-prob-fill"
                style={{ width: `${prob * 100}%`, background: CLASS_COLORS[i] }}
              />
            </div>
            <span className="shap-prob-pct">{(prob * 100).toFixed(1)}%</span>
          </div>
        );
      })}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Full SHAP panel
// ---------------------------------------------------------------------------
function SHAPPanel({ data }) {
  const shapValues = data.shap_values || data.shap || [];
  const maxAbs = Math.max(...shapValues.map((e) => Math.abs(e.shap)), 0.001);

  return (
    <div className="shap-panel">
      {/* Header: grade + probabilities */}
      <div className="shap-header-row">
        <GradeBadge
          grade={data.predicted_grade}
          label={data.predicted_label}
          confidence={data.confidence}
        />
        <ProbabilityBars probabilities={data.probabilities} />
      </div>

      {/* Summary */}
      {data.summary && (
        <div className="shap-summary-box">
          <Sparkles size={15} />
          <p>{data.summary}</p>
        </div>
      )}

      {/* Feature contribution chart */}
      <div className="shap-chart-section">
        <p className="shap-section-title">
          <BarChart2 size={14} /> Feature contributions (|SHAP|)
        </p>
        <div className="shap-legend">
          <span className="shap-legend-dot risk" /> Increases risk
          <span className="shap-legend-dot protect" /> Decreases risk
        </div>
        <div className="shap-bars">
          {shapValues.map((entry) => (
            <SHAPBar key={entry.feature} entry={entry} maxAbs={maxAbs} />
          ))}
        </div>
      </div>

      {/* Imputed fields notice */}
      {data.imputed_fields?.length > 0 && (
        <div className="shap-imputed-notice">
          <Info size={14} />
          <span>
            <strong>Estimated fields</strong> — the following values were not
            collected during intake and were inferred from cohort averages:{" "}
            {data.imputed_fields.join(", ")}.
          </span>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main Explainability page
// ---------------------------------------------------------------------------
export default function Explainability() {
  const { result, image, file } = useScreening();

  const [tab, setTab] = useState("gradcam");
  const [gradCamData, setGradCamData] = useState(null);
  const [shapData, setShapData] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const label = result?.predictedLabel || result?.severity || "Awaiting model result";

  // Load Grad-CAM whenever a new fundus image is uploaded
  useEffect(() => {
    if (!file) return;
    setLoading(true);
    setError("");
    getGradCAM(file)
      .then(setGradCamData)
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, [file]);

  // Load SHAP when the user clicks the SHAP tab
  async function handleSHAPTab() {
    setTab("shap");
    if (shapData) return; // already loaded
    setLoading(true);
    setError("");
    try {
      const sessionId =
        localStorage.getItem(SESSION_KEY) || `anon_${Date.now()}`;
      // Pass structured_data from the chatbot session if available
      const storedSession = (() => {
        try {
          return null; // structured_data comes from the backend session lookup
        } catch {
          return null;
        }
      })();
      const data = await getSHAP(sessionId, storedSession);
      setShapData(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <>
      {/* Page heading + tab switcher */}
      <div className="page-heading compact">
        <div>
          <p className="eyebrow">EXPLAINABILITY</p>
          <h1>Why did the AI say that?</h1>
          <p className="lede">Transparent visual context for both patients and reviewers.</p>
        </div>
        <div className="tab-switch">
          <button
            className={tab === "gradcam" ? "selected" : ""}
            onClick={() => setTab("gradcam")}
          >
            <Eye size={15} /> Grad-CAM
          </button>
          <button
            className={tab === "shap" ? "selected" : ""}
            onClick={handleSHAPTab}
          >
            <Sparkles size={15} /> SHAP
          </button>
        </div>
      </div>

      {/* Grad-CAM tab */}
      {tab === "gradcam" && (
        <>
          <div className="explain-grid">
            <Visual
              title="Original fundus image"
              src={gradCamData?.originalImage || image || "/fundus-placeholder.svg"}
            />
            <Visual
              title="Grad-CAM heatmap"
              src={gradCamData?.gradCam || "/fundus-placeholder.svg"}
            />
            <Visual
              title="Overlay"
              src={gradCamData?.overlay || "/fundus-placeholder.svg"}
              overlay
            />
          </div>
          {loading && <div className="inline-success">Generating AI explanation…</div>}
          <div className="explain-caption">
            <Info size={17} />
            <div>
              <strong>Model-attention visualization</strong>
              <p>
                Grad-CAM highlights regions that contributed strongly to the model's
                prediction. This is an explanation of model attention, not a
                confirmed lesion map or clinical diagnosis.
              </p>
            </div>
          </div>
          {error && <div className="inline-error">{error}</div>}
        </>
      )}

      {/* SHAP tab */}
      {tab === "shap" && (
        <>
          {loading && (
            <div className="inline-success">
              Computing SHAP feature contributions… (first request trains the
              model — may take ~10 s)
            </div>
          )}
          {!loading && shapData && <SHAPPanel data={shapData} />}
          {!loading && !shapData && !error && (
            <div className="empty-state">
              <AlertCircle size={25} />
              <h2>SHAP explanation not available</h2>
              <p>
                Click the SHAP tab to load feature-importance values from the
                tabular LightGBM model. Make sure you have completed the AI
                assistant intake first so clinical values are available.
              </p>
              <span>Session: {localStorage.getItem(SESSION_KEY) || "none"}</span>
            </div>
          )}
          {error && <div className="inline-error">{error}</div>}
        </>
      )}

      {/* Mini result footer */}
      <div className="mini-result">
        <span className="eyebrow">RETINAL IMAGE MODEL</span>
        <strong>{label}</strong>
        <span>
          {result?.confidence == null
            ? "Awaiting model result"
            : `Model confidence: ${Math.round(result.confidence * 100)}%`}
        </span>
      </div>
    </>
  );
}