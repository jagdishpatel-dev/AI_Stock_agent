import { useCallback, useEffect, useState } from "react";
import { api, getAdminKey } from "../api/client";
import { AdminRequired } from "../components/AdminRequired";
import { DateFilter } from "../components/DateFilter";
import { PageSkeleton } from "../components/Skeleton";
import type { Event } from "../types";
import { fmtTs } from "../utils/format";

const EVENT_TYPES = [
  "",
  "premarket_briefing",
  "watchlist_rank",
  "watchlist_react_step",
  "watchlist_gate",
  "kill_switch",
];

export function EventsPage() {
  const [date, setDate] = useState("");
  const [eventType, setEventType] = useState("");
  const [events, setEvents] = useState<Event[]>([]);
  const [error, setError] = useState("");
  const hasAdminKey = !!getAdminKey();
  const [loading, setLoading] = useState(hasAdminKey);

  const load = useCallback(async () => {
    if (!hasAdminKey) return;
    setLoading(true);
    setError("");
    try {
      const e = await api.events({
        date: date || undefined,
        event_type: eventType || undefined,
      });
      setEvents(e);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load");
    } finally {
      setLoading(false);
    }
  }, [date, eventType, hasAdminKey]);

  useEffect(() => {
    load();
  }, [load]);

  if (!hasAdminKey) {
    return (
      <>
        <div className="page-header">
          <h2>Events &amp; Briefings</h2>
        </div>
        <AdminRequired what="Agent events and briefings" />
      </>
    );
  }

  if (loading && !events.length) return <PageSkeleton />;

  return (
    <>
      <div className="page-header">
        <h2>Events &amp; Briefings</h2>
        <DateFilter value={date} onChange={setDate} />
      </div>

      <div className="filter-row">
        <select
          value={eventType}
          onChange={(e) => setEventType(e.target.value)}
          aria-label="Filter by event type"
        >
          <option value="">All events</option>
          {EVENT_TYPES.filter(Boolean).map((t) => (
            <option key={t} value={t}>
              {t.replace(/_/g, " ")}
            </option>
          ))}
        </select>
      </div>

      {error && <div className="error">{error}</div>}

      <div className="panel">
        <div className="panel-header">
          Agent events
          <span className="mono" style={{ fontWeight: 400, color: "var(--text-muted)" }}>
            {events.length} items
          </span>
        </div>
        <div className="panel-body">
          {events.length === 0 ? (
            <div className="empty">No events logged for this date.</div>
          ) : (
            events.map((e) => (
              <div className="insight-card" key={e.id}>
                <div className="trip-header">
                  <span className="badge badge-event">{e.event_type}</span>
                  <span className="mono" style={{ color: "var(--text-muted)", fontSize: "0.8rem" }}>
                    {fmtTs(e.ts)}
                  </span>
                </div>
                <pre
                  style={{
                    margin: 0,
                    whiteSpace: "pre-wrap",
                    fontFamily: "var(--font)",
                    fontSize: "0.88rem",
                    color: "var(--text-muted)",
                  }}
                >
                  {e.message}
                </pre>
              </div>
            ))
          )}
        </div>
      </div>
    </>
  );
}
