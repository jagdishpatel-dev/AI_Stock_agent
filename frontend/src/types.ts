export interface DailySummary {
  date: string;
  starting_equity: number;
  ending_equity: number;
  net_pnl: number;
  trade_count: number;
  win_count: number;
}

export interface Stats {
  trade_count: number;
  win_count: number;
  loss_count: number;
  win_rate: number;
  flat_count: number;
  avg_win: number | null;
  avg_loss: number | null;
  profit_factor: number | null;
  total_pnl: number;
  entry_signals: number;
  llm_vetoes: number;
  exit_signals: number;
}

export interface LifetimeStats {
  total_pnl: number;
  trade_count: number;
  win_count: number;
  loss_count: number;
  win_rate: number;
  flat_count: number;
  avg_win: number | null;
  avg_loss: number | null;
  profit_factor: number | null;
}

export interface AccountSnapshot {
  available?: boolean;
  equity?: number;
  buying_power?: number;
  cash?: number;
  portfolio_value?: number;
  last_equity?: number;
  previous_balance?: number | null;
  change?: number | null;
  change_pct?: number | null;
}

export interface Position {
  symbol: string;
  qty: number;
  side: string;
  avg_entry_price: number;
  current_price: number;
  market_value: number;
  cost_basis: number;
  unrealized_pl: number;
  unrealized_plpc: number;
  portfolio_pct?: number | null;
}

export interface WatchlistEntry {
  date: string;
  symbol: string;
  source: string;
  rank: number;
  metrics: Record<string, number>;
  reason: string;
}

export interface Trade {
  id: number;
  ts: string;
  symbol: string;
  side: string;
  qty: number;
  price: number;
  order_id: string;
  pnl: number | null;
  reason: string;
}

export interface Signal {
  id: number;
  ts: string;
  symbol: string;
  signal_type: string;
  details: string;
  llm_action: string | null;
  llm_confidence: number | null;
  llm_reason: string | null;
  rsi: number | null;
  vwap_dev: number | null;
  volume_ratio: number | null;
}

export interface Event {
  id: number;
  ts: string;
  event_type: string;
  message: string;
}

export interface RoundTrip {
  symbol: string;
  buy_ts: string;
  buy_price: number;
  buy_qty: number;
  entry_reason: string | null;
  llm_action: string | null;
  llm_confidence: number | null;
  llm_reason: string | null;
  rsi: number | null;
  vwap_dev: number | null;
  volume_ratio: number | null;
  sell_ts?: string;
  sell_price?: number;
  exit_reason?: string;
  pnl?: number | null;
}

export interface Overview {
  date: string;
  summary: DailySummary | null;
  stats: Stats;
  display_pnl?: number;
  watchlist: WatchlistEntry[];
  config: Record<string, unknown>;
  account: AccountSnapshot | null;
  positions: Position[];
  lifetime_stats: LifetimeStats;
  market_status: string;
  daily_pnl_history: DailySummary[];
  last_trade_date: string | null;
}

export interface PortfolioSnapshot {
  account: AccountSnapshot | null;
  positions: Position[];
  daily_pnl: DailySummary[];
  lifetime_stats: LifetimeStats;
  market_status: string;
  today: string;
}

export interface AppConfig {
  symbols: string[];
  strategy: Record<string, unknown>;
  swing?: Record<string, unknown>;
  session: Record<string, string>;
  effective_session_end?: string;
  display_llm?: string;
  risk: Record<string, number>;
  execution: Record<string, unknown>;
  briefing: Record<string, unknown>;
  journal_context: Record<string, unknown>;
  screener: Record<string, unknown>;
  llm: Record<string, unknown>;
}
