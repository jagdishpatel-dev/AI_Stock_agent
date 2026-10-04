import { useCallback, useEffect, useMemo, useState } from "react";
import { Briefcase } from "lucide-react";
import {
  Area,
  AreaChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api } from "../api/client";
import { ChartPanel, chartHeights } from "../components/ChartPanel";
import { DataTable } from "../components/DataTable";
import { DateFilter } from "../components/DateFilter";
import { EmptyState } from "../components/EmptyState";
import { HeroMetric } from "../components/HeroMetric";
import { HoldingCard } from "../components/HoldingCard";
import { LoadingButton } from "../components/LoadingButton";
import { OverviewSkeleton } from "../components/Skeleton";
import { StatCard } from "../components/StatCard";
import { StatusBanner } from "../components/StatusBanner";
import type { Overview } from "../types";
import { chartTooltipStyle, getChartColors } from "../utils/chartTheme";
import { fmtMoney, fmtPct, fmtPctNum, DISPLAY_TZ_LABEL } from "../utils/format";

function marketBanner(status: string) {
  switch (status) {
    case "weekend":
      return "Market closed (weekend) — showing account balance and recent activity.";
    case "pre_market":
      return "Pre-market — session opens at 8:30 AM CST.";
    case "closed":
      return "Market closed for today — showing account balance and recent activity.";
    default:
      return null;
  }
}

export function OverviewPage() {
  const [date, setDate] = useState("");
  const [data, setData] = useState<Overview | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const [chartColors, setChartColors] = useState(getChartColors());

  const load = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const o = await api.overview(date || undefined);
      setDate(o.date);
      setData(o);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load");
    } finally {
      setLoading(false);
    }
  }, [date]);

  useEffect(() => {
    load();
  }, [load]);

  useEffect(() => {
    setChartColors(getChartColors());
  }, [data]);

  const equityChart = useMemo(() => {
    if (!data) return [];
    const history = [...(data.daily_pnl_history ?? [])].reverse();
    const points = history.map((r) => ({
      date: r.date.slice(5),
      equity: r.ending_equity,
      pnl: r.net_pnl,
    }));
    if (data.account?.equity != null) {
      const todayLabel = data.date.slice(5);
      const last = points[points.length - 1];
      if (!last || last.date !== todayLabel) {
        points.push({
          date: todayLabel,
          equity: data.account.equity,
          pnl: data.display_pnl ?? 0,
        });
      } else {
        last.equity = data.account.equity;
      }
    }
    return points;
  }, [data]);

  if (loading && !data) return <OverviewSkeleton />;

  const stats = data?.stats;
  const account = data?.account;
  const lifetime = data?.lifetime_stats;
  const config = data?.config ?? {};
  const strategy = config.strategy as { mode?: string } | undefined;
  const swing = config.swing as {
    take_profit_pct?: number;
    stop_loss_pct?: number;
    max_hold_days?: number;
  } | undefined;
  const session = config.session as { start_time?: string } | undefined;
  const sessionEnd =
    (config.effective_session_end as string) ??
    (config.session as { end_time?: string })?.end_time ??
    "11:30";

  const dayPnl = data?.display_pnl ?? stats?.total_pnl;
  const hasDayActivity =
    (stats?.trade_count ?? 0) > 0 ||
    (stats?.entry_signals ?? 0) > 0 ||
    (dayPnl != null && dayPnl !== 0);

  const banner = data ? marketBanner(data.market_status) : null;
  const previousBalance =
    account?.previous_balance ??
    account?.last_equity ??
    data?.daily_pnl_history?.[0]?.ending_equity;

  const change =
    account?.change ??
    (previousBalance != null && account?.equity != null
      ? account.equity - previousBalance
      : null);
  const changePct = account?.change_pct;
  const changePositive = (change ?? 0) >= 0;

  const positionsTable = data?.positions?.length ? (
    <table className="data-table">
      <thead>
        <tr>
          <th>Symbol</th>
          <th>Qty</th>
          <th>Entry</th>
          <th>Current</th>
          <th>Unrealized</th>
          <th>% Port</th>
        </tr>
      </thead>
      <tbody>
        {data.positions.map((p) => (
          <tr key={p.symbol}>
            <td>
              <strong>{p.symbol}</strong>
            </td>
            <td className="mono">{p.qty}</td>
            <td className="mono">{fmtMoney(p.avg_entry_price)}</td>
            <td className="mono">{fmtMoney(p.current_price)}</td>
            <td
              className={`mono ${p.unrealized_pl >= 0 ? "positive" : "negative"}`}
            >
              {p.unrealized_pl >= 0 ? "+" : ""}
              {fmtMoney(p.unrealized_pl)} ({fmtPctNum(p.unrealized_plpc)})
            </td>
            <td className="mono">
              {p.portfolio_pct != null ? `${p.portfolio_pct}%` : "—"}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  ) : (
    <EmptyState
      icon={Briefcase}
      title="No open positions"
      description="Your agent hasn't opened any positions yet."
    />
  );

  const positionsMobile = data?.positions?.length ? (
    <div>
      {data.positions.map((p) => (
        <HoldingCard key={p.symbol} position={p} />
      ))}
    </div>
  ) : (
    <EmptyState
      icon={Briefcase}
      title="No open positions"
      description="Your agent hasn't opened any positions yet."
    />
  );

  return (
    <>
      <div className="page-header">
        <h2>Overview</h2>
        <div className="page-header-actions">
          <LoadingButton loading={loading} onClick={load}>
            Refresh
          </LoadingButton>
          <DateFilter value={date} onChange={setDate} />
        </div>
      </div>

      {error && <div className="error">{error}</div>}
      {banner && <StatusBanner message={banner} />}

      <HeroMetric
        label="Portfolio Value"
        value={account?.equity != null ? fmtMoney(account.equity) : "—"}
        change={change != null ? `${changePositive ? "+" : ""}${fmtMoney(change)}` : undefined}
        changePct={changePct != null ? fmtPctNum(changePct) : undefined}
        positive={changePositive}
        sub={
          account?.available === false
            ? "Alpaca unavailable"
            : `Buying power ${account?.buying_power != null ? fmtMoney(account.buying_power) : "—"}`
        }
      />

      <div className="metrics-scroll">
        <StatCard
          variant="compact"
          label={`P&L (${data?.date ?? "today"})`}
          value={
            hasDayActivity || dayPnl != null ? fmtMoney(dayPnl ?? 0) : "—"
          }
          className={(dayPnl ?? 0) >= 0 ? "positive" : "negative"}
          sub={
            hasDayActivity
              ? `${stats?.win_count ?? 0}W / ${stats?.loss_count ?? 0}L${stats?.flat_count ? ` / ${stats.flat_count} flat` : ""}`
              : "No activity"
          }
        />
        <StatCard
          variant="compact"
          label="All-time P&L"
          value={fmtMoney(lifetime?.total_pnl ?? 0)}
          className={(lifetime?.total_pnl ?? 0) >= 0 ? "positive" : "negative"}
          sub={`${lifetime?.trade_count ?? 0} round trips`}
        />
        <StatCard
          variant="compact"
          label="Trade Quality"
          value={
            lifetime?.profit_factor != null
              ? `${lifetime.profit_factor.toFixed(2)}× PF`
              : (lifetime?.trade_count ?? 0) > 0
                ? fmtPct(lifetime?.win_rate)
                : "—"
          }
          sub={
            (lifetime?.trade_count ?? 0) > 0
              ? [
                  lifetime?.avg_win != null && lifetime?.avg_loss != null
                    ? `Avg win ${fmtMoney(lifetime.avg_win)} / loss ${fmtMoney(lifetime.avg_loss)}`
                    : null,
                  `${fmtPct(lifetime?.win_rate)} win rate · ${lifetime?.win_count ?? 0}W / ${lifetime?.loss_count ?? 0}L${
                    lifetime?.flat_count ? ` / ${lifetime.flat_count} flat` : ""
                  }`,
                ]
                  .filter(Boolean)
                  .join(" · ")
              : "No closed trades"
          }
        />
        <StatCard
          variant="compact"
          label="Positions"
          value={String(data?.positions?.length ?? 0)}
          sub={
            data?.positions?.length
              ? data.positions.map((p) => p.symbol).join(", ")
              : "None open"
          }
        />
      </div>

      {equityChart.length > 0 && (
        <div className="panel">
          <div className="panel-header">Balance Over Time</div>
          <div className="panel-body chart-wrap chart-wrap--tall">
            <ChartPanel
              title="Balance Over Time"
              summary={`Equity trend across ${equityChart.length} days. Latest value ${account?.equity != null ? fmtMoney(account.equity) : "unavailable"}.`}
              height={chartHeights.tall}
            >
              <AreaChart data={equityChart}>
                <defs>
                  <linearGradient id="equityGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor={chartColors.positive} stopOpacity={0.25} />
                    <stop offset="100%" stopColor={chartColors.positive} stopOpacity={0} />
                  </linearGradient>
                </defs>
                <XAxis
                  dataKey="date"
                  stroke={chartColors.axis}
                  fontSize={12}
                  tickLine={false}
                  axisLine={false}
                />
                <YAxis
                  stroke={chartColors.axis}
                  fontSize={12}
                  domain={["auto", "auto"]}
                  tickLine={false}
                  axisLine={false}
                  tickFormatter={(v) => `$${(v / 1000).toFixed(1)}k`}
                  width={52}
                />
                <Tooltip
                  contentStyle={chartTooltipStyle()}
                  formatter={(v: number) => [fmtMoney(v), "Equity"]}
                />
                <Area
                  type="monotone"
                  dataKey="equity"
                  stroke={chartColors.positive}
                  strokeWidth={2}
                  fill="url(#equityGradient)"
                  dot={false}
                  activeDot={{ r: 4, fill: chartColors.positive }}
                />
              </AreaChart>
            </ChartPanel>
          </div>
        </div>
      )}

      <div className="grid-2">
        <div className="panel">
          <div className="panel-header">Open Positions</div>
          <div className="panel-body panel-body--flush data-table-wrap">
            <DataTable desktop={positionsTable} mobile={positionsMobile} />
          </div>
        </div>

        <div className="panel">
          <div className="panel-header">Today&apos;s Watchlist</div>
          <div className="panel-body data-table-wrap">
            {data?.watchlist.length ? (
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Symbol</th>
                    <th>Source</th>
                    <th>Reason</th>
                  </tr>
                </thead>
                <tbody>
                  {data.watchlist.map((w) => (
                    <tr key={w.symbol}>
                      <td>
                        <strong>{w.symbol}</strong>
                      </td>
                      <td>
                        <span
                          className={`badge badge-${w.source === "anchor" ? "anchor" : "screener"}`}
                        >
                          {w.source}
                        </span>
                      </td>
                      <td className="reason-cell">{w.reason || "—"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <div className="empty">No screener data for this date.</div>
            )}
          </div>
        </div>
      </div>

      <details className="config-details">
        <summary>Agent Config</summary>
        <div className="panel-body">
          <div className="trip-grid">
            <div>
              <dt>Mode</dt>
              <dd>{strategy?.mode ?? "—"}</dd>
            </div>
            <div>
              <dt>Session</dt>
              <dd>
                {session?.start_time}–{sessionEnd} {DISPLAY_TZ_LABEL}
              </dd>
            </div>
            <div>
              <dt>LLM</dt>
              <dd>
                {(config.display_llm as string) ??
                  (config.llm as { display_llm?: string })?.display_llm ??
                  "—"}
              </dd>
            </div>
            <div>
              <dt>Screener</dt>
              <dd>
                {(config.screener as { mode?: string })?.mode ?? "—"}
              </dd>
            </div>
            {strategy?.mode === "swing" && swing && (
              <div>
                <dt>Swing targets</dt>
                <dd>
                  TP {swing.take_profit_pct}% · stop {swing.stop_loss_pct}% · max{" "}
                  {swing.max_hold_days}d
                </dd>
              </div>
            )}
            <div>
              <dt>Risk / Trade</dt>
              <dd>
                {(config.risk as { max_risk_per_trade_pct?: number })
                  ?.max_risk_per_trade_pct ?? "—"}
                %
              </dd>
            </div>
          </div>
        </div>
      </details>
    </>
  );
}
