import React, { useState, useEffect } from "react";
import "./App.css";

const API_BASE = "http://localhost:8000";

const formatMetric = (value, digits = 4) =>
  typeof value === "number" && Number.isFinite(value) ? value.toFixed(digits) : "N/A";

const formatPercent = (value) =>
  typeof value === "number" && Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : "N/A";

export default function App() {
  const [isHealthy, setIsHealthy] = useState(null);
  const [events, setEvents] = useState([]);
  const [loadingEvents, setLoadingEvents] = useState(false);
  const [selectedEventId, setSelectedEventId] = useState(null);
  const [explanation, setExplanation] = useState(null);
  const [loadingExplain, setLoadingExplain] = useState(false);
  const [robustnessData, setRobustnessData] = useState(null);
  const [loadingRobustness, setLoadingRobustness] = useState(false);
  const [errorMsg, setErrorMsg] = useState(null);

  // 1. Health Check
  const checkHealth = async () => {
    try {
      const res = await fetch(`${API_BASE}/health`);
      if (res.ok) {
        const data = await res.json();
        setIsHealthy(data.status === "ok");
      } else {
        setIsHealthy(false);
      }
    } catch {
      setIsHealthy(false);
    }
  };

  // 2. Fetch Events
  const fetchEvents = async () => {
    setLoadingEvents(true);
    setErrorMsg(null);
    try {
      const res = await fetch(`${API_BASE}/events?limit=50&offset=0`);
      if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
      const data = await res.json();
      if (!Array.isArray(data.events)) throw new Error("Unexpected /events response shape");
      setEvents(data.events);
    } catch (err) {
      console.error("Failed to fetch events:", err);
      setErrorMsg("Failed to fetch events from backend service.");
    } finally {
      setLoadingEvents(false);
    }
  };

  // 3. Fetch Explanation for Selected Event
  const fetchExplanation = async (eventId) => {
    setSelectedEventId(eventId);
    setLoadingExplain(true);
    setExplanation(null);
    try {
      const res = await fetch(`${API_BASE}/explain`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ event_id: eventId }),
      });
      if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
      const data = await res.json();
      setExplanation(data);
    } catch (err) {
      console.error("Failed to fetch explanation:", err);
    } finally {
      setLoadingExplain(false);
    }
  };

  // 4. Fetch Robustness Data
  const fetchRobustness = async () => {
    setLoadingRobustness(true);
    try {
      const res = await fetch(`${API_BASE}/robustness`);
      if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
      const data = await res.json();
      if (!Array.isArray(data.results)) throw new Error("Unexpected /robustness response shape");
      setRobustnessData(data.results);
    } catch (err) {
      console.error("Failed to fetch robustness data:", err);
    } finally {
      setLoadingRobustness(false);
    }
  };

  useEffect(() => {
    checkHealth();
    fetchEvents();
    fetchRobustness();
  }, []);

  const getThreatClass = (score) => {
    if (typeof score !== "number" || !Number.isFinite(score)) return "";
    if (score >= 70) return "threat-high";
    if (score >= 40) return "threat-medium";
    return "threat-low";
  };

  return (
    <div className="dashboard-container">
      {/* 1. Header */}
      <div className="header">
        <h1>Network Security — Layered AI Defense Dashboard</h1>
        <div className="health-badge">
          Backend Status:
          <span className={`dot ${isHealthy ? "green" : "red"}`}></span>
          <span>{isHealthy === null ? "Checking..." : isHealthy ? "Online" : "Offline"}</span>
        </div>
      </div>

      {errorMsg && <div style={{ color: "red", marginBottom: "16px" }}>{errorMsg}</div>}

      {/* 2. Events Table */}
      <div className="section">
        <div className="section-header">
          <h2>Recent Flagged Events</h2>
          <button onClick={fetchEvents} disabled={loadingEvents}>
            {loadingEvents ? "Refreshing..." : "Refresh Events"}
          </button>
        </div>

        {loadingEvents && events.length === 0 ? (
          <p>Loading events...</p>
        ) : events.length === 0 ? (
          <p>No events found.</p>
        ) : (
          <table>
            <thead>
              <tr>
                <th>Timestamp</th>
                <th>Predicted Class</th>
                <th>Threat Score</th>
                <th>Transformer Conf</th>
                <th>VAE Anomaly Score</th>
                <th>DGA Probability</th>
              </tr>
            </thead>
            <tbody>
              {events.map((evt) => (
                <tr
                  key={evt.event_id || evt.timestamp}
                  className={`clickable ${selectedEventId === (evt.event_id || evt.timestamp) ? "selected" : ""}`}
                  onClick={() => fetchExplanation(evt.event_id || evt.timestamp)}
                >
                  <td>{evt.timestamp}</td>
                  <td>{evt.predicted_class ?? "N/A"}</td>
                  <td className={getThreatClass(evt.threat_score)}>{formatMetric(evt.threat_score, 1)}</td>
                  <td>{formatPercent(evt.transformer_confidence)}</td>
                  <td>{formatMetric(evt.vae_anomaly_score)}</td>
                  <td>{formatPercent(evt.dga_probability)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {/* 3. Explain Panel */}
      {selectedEventId && (
        <div className="explain-panel">
          <h3>Event Explanation (Event ID: {selectedEventId})</h3>
          {loadingExplain ? (
            <p>Generating dual-channel SHAP & attention explanation...</p>
          ) : explanation ? (
            <>
              <div>
                <strong>Predicted Class:</strong> {explanation.predicted_class ?? "N/A"} |{" "}
                <strong>Confidence:</strong> {explanation.confidence !== undefined ? (explanation.confidence * 100).toFixed(1) + "%" : "N/A"}
              </div>

              {explanation.plain_english_explanation && (
                <div className="explanation-text">
                  <strong>Plain-English Assessment:</strong>
                  <div>{explanation.plain_english_explanation}</div>
                </div>
              )}

              <div className="explain-grid">
                <div className="explain-box">
                  <h4>Top SHAP Features</h4>
                  {explanation.top_shap_features && explanation.top_shap_features.length > 0 ? (
                    <ul>
                      {explanation.top_shap_features.map((item, idx) => (
                        <li key={idx}>
                          <strong>{item.feature}</strong> (timestep {item.timestep}): {item.shap_value > 0 ? "+" : ""}
                          {item.shap_value.toFixed(4)}
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <p>No SHAP features available.</p>
                  )}
                </div>

                <div className="explain-box">
                  <h4>Top Attended Timesteps</h4>
                  {explanation.top_attended_timesteps && explanation.top_attended_timesteps.length > 0 ? (
                    <ul>
                      {explanation.top_attended_timesteps.map((item, idx) => (
                        <li key={idx}>
                          Timestep {Array.isArray(item) ? item[0] : item.timestep}: Weight{" "}
                          {(Array.isArray(item) ? item[1] : item.weight).toFixed(4)}
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <p>No attention weights available.</p>
                  )}
                </div>
              </div>
            </>
          ) : (
            <p>Failed to load explanation details.</p>
          )}
        </div>
      )}

      {/* 4. Adversarial Robustness Section */}
      <div className="section" style={{ marginTop: "32px" }}>
        <h2>Adversarial Robustness Study Comparison</h2>
        {loadingRobustness ? (
          <p>Loading robustness benchmark results...</p>
        ) : robustnessData?.length > 0 ? (
          <table>
            <thead>
              <tr>
                <th>Model Variant</th>
                <th>Clean Acc</th>
                <th>Clean F1</th>
                <th>FGSM Acc</th>
                <th>FGSM F1</th>
                <th>PGD Acc</th>
                <th>PGD F1</th>
              </tr>
            </thead>
            <tbody>
              {robustnessData.map((row) => (
                <tr key={row.run_id}>
                  <td>{row.variant || row.model_name || "N/A"}</td>
                  <td>{formatMetric(row.clean_acc)}</td>
                  <td>{formatMetric(row.clean_f1)}</td>
                  <td>{formatMetric(row.fgsm_acc)}</td>
                  <td>{formatMetric(row.fgsm_f1)}</td>
                  <td>{formatMetric(row.pgd_acc)}</td>
                  <td>{formatMetric(row.pgd_f1)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <p>No robustness benchmark data loaded yet.</p>
        )}
      </div>
    </div>
  );
}
