import { useCallback, useEffect, useState } from "react";
import { api, getAdminKey } from "../api/client";
import { AdminRequired } from "../components/AdminRequired";
import { LoadingButton } from "../components/LoadingButton";

function logClass(line: string) {
  if (line.includes("ERROR") || line.includes("Error")) return "error";
  if (line.includes("WARNING") || line.includes("Warning")) return "warn";
  return "info";
}

export function LogsPage() {
  const [lines, setLines] = useState<string[]>([]);
  const [exists, setExists] = useState(true);
  const [error, setError] = useState("");
  const [autoRefresh, setAutoRefresh] = useState(true);
  const hasAdminKey = !!getAdminKey();

  const load = useCallback(async () => {
    if (!hasAdminKey) return;
    try {
      const data = await api.logs(400);
      setLines(data.lines);
      setExists(data.exists);
      setError("");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load logs");
    }
  }, [hasAdminKey]);

  useEffect(() => {
    load();
  }, [load]);

  useEffect(() => {
    if (!autoRefresh || !hasAdminKey) return;
    const id = setInterval(load, 5000);
    return () => clearInterval(id);
  }, [autoRefresh, hasAdminKey, load]);

  if (!hasAdminKey) {
    return (
      <>
        <div className="page-header">
          <h2>Agent Logs</h2>
        </div>
        <AdminRequired what="Agent logs" />
      </>
    );
  }

  return (
    <>
      <div className="page-header">
        <h2>Agent Logs</h2>
        <div className="page-header-actions">
          <label
            style={{
              fontSize: "0.85rem",
              color: "var(--text-muted)",
              display: "flex",
              alignItems: "center",
              gap: "0.35rem",
              minHeight: 44,
            }}
          >
            <input
              type="checkbox"
              checked={autoRefresh}
              onChange={(e) => setAutoRefresh(e.target.checked)}
            />
            Auto-refresh (5s)
          </label>
          <LoadingButton onClick={load}>Refresh</LoadingButton>
        </div>
      </div>

      {error && <div className="error">{error}</div>}

      <div className="panel">
        <div className="panel-header">
          logs/agent.log
          <span className="mono" style={{ fontWeight: 400, color: "var(--text-muted)" }}>
            {lines.length} lines
          </span>
        </div>
        <div className="panel-body">
          {!exists ? (
            <div className="empty">
              Log file not found. Start the trading agent with{" "}
              <code className="mono">python -m src.main</code> to generate logs.
            </div>
          ) : (
            <div className="log-viewer">
              {lines.map((line, i) => (
                <div key={i} className={`log-line ${logClass(line)}`}>
                  {line}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </>
  );
}
