import type {
  AccountSnapshot,
  DailySummary,
  Event,
  Overview,
  PortfolioSnapshot,
  Position,
  RoundTrip,
  Signal,
  Trade,
  WatchlistEntry,
} from "../types";

// The API requires a signed session cookie issued by /api/session. Fetch it once,
// and again if a request comes back 401 (cookie expired or server restarted).
let sessionPromise: Promise<void> | null = null;

function ensureSession(refresh = false): Promise<void> {
  if (refresh || !sessionPromise) {
    sessionPromise = fetch("/api/session")
      .then((res) => {
        if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
      })
      .catch((e) => {
        sessionPromise = null;
        throw e;
      });
  }
  return sessionPromise;
}

async function get<T>(path: string): Promise<T> {
  await ensureSession();
  let res = await fetch(path);
  if (res.status === 401) {
    await ensureSession(true);
    res = await fetch(path);
  }
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json() as Promise<T>;
}

const ADMIN_KEY_STORAGE = "trading_admin_key";

export function getAdminKey(): string {
  return sessionStorage.getItem(ADMIN_KEY_STORAGE) ?? "";
}

export function setAdminKey(key: string): void {
  sessionStorage.setItem(ADMIN_KEY_STORAGE, key);
}

export function clearAdminKey(): void {
  sessionStorage.removeItem(ADMIN_KEY_STORAGE);
}

async function adminFetch<T>(
  path: string,
  options: RequestInit = {},
): Promise<T> {
  const key = getAdminKey();
  const res = await fetch(path, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      "X-Admin-Key": key,
      ...(options.headers as Record<string, string>),
    },
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(body || `${res.status} ${res.statusText}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  overview: (date?: string) =>
    get<Overview>(`/api/overview${date ? `?date=${date}` : ""}`),
  account: () => get<AccountSnapshot>("/api/account"),
  positions: () => get<{ positions: Position[]; count: number }>("/api/positions"),
  portfolio: () => get<PortfolioSnapshot>("/api/portfolio"),
  trades: (params?: { date?: string; symbol?: string }) => {
    const q = new URLSearchParams();
    if (params?.date) q.set("date", params.date);
    if (params?.symbol) q.set("symbol", params.symbol);
    const qs = q.toString();
    return get<Trade[]>(`/api/trades${qs ? `?${qs}` : ""}`);
  },
  roundTrips: (date?: string) =>
    get<RoundTrip[]>(`/api/round-trips${date ? `?date=${date}` : ""}`),
  signals: (params?: { date?: string; signal_type?: string }) => {
    const q = new URLSearchParams();
    if (params?.date) q.set("date", params.date);
    if (params?.signal_type) q.set("signal_type", params.signal_type);
    const qs = q.toString();
    return get<Signal[]>(`/api/signals${qs ? `?${qs}` : ""}`);
  },
  events: (params?: { date?: string; event_type?: string }) => {
    const q = new URLSearchParams();
    if (params?.date) q.set("date", params.date);
    if (params?.event_type) q.set("event_type", params.event_type);
    const qs = q.toString();
    return adminFetch<Event[]>(`/api/events${qs ? `?${qs}` : ""}`);
  },
  dailyPnl: () => get<DailySummary[]>("/api/daily-pnl"),
  watchlist: (date?: string) =>
    get<{ date: string; entries: WatchlistEntry[] }>(
      `/api/watchlist${date ? `?date=${date}` : ""}`,
    ),
  watchlistDates: () => get<string[]>("/api/watchlist/dates"),
  logs: (lines = 200) =>
    adminFetch<{ lines: string[]; exists: boolean }>(`/api/logs?lines=${lines}`),
  config: () => adminFetch<Record<string, unknown>>("/api/config"),
  lastTradeDate: async () => {
    const o = await get<Overview>("/api/overview");
    return o.last_trade_date;
  },
  adminSettings: () =>
    adminFetch<Record<string, unknown>>("/api/admin/settings"),
  adminUpdateSettings: (updates: Record<string, unknown>) =>
    adminFetch<{ ok: boolean; applied: string[]; settings: Record<string, unknown> }>(
      "/api/admin/settings",
      { method: "PUT", body: JSON.stringify({ updates }) },
    ),
  adminAddAnchor: (symbol: string) =>
    adminFetch<{ ok: boolean; anchor_symbols: string[] }>("/api/admin/anchors", {
      method: "POST",
      body: JSON.stringify({ symbol }),
    }),
  adminRemoveAnchor: (symbol: string) =>
    adminFetch<{ ok: boolean; anchor_symbols: string[] }>(
      `/api/admin/anchors/${encodeURIComponent(symbol)}`,
      { method: "DELETE" },
    ),
};
