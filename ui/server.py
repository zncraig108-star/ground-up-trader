"""ui/server.py — FastAPI + WebSocket dashboard host.

Serves the single-file frontend at ``/``, exposes REST state/pool/deploy endpoints, and
broadcasts a JSON snapshot after each GA generation over the WebSocket at
``/ws/evolution``.

The ``EvolutionEngine`` is CPU-bound and synchronous, so it runs in a background thread;
each ``Snapshot`` is handed to the event loop via ``call_soon_threadsafe`` and pushed to
a queue that the broadcaster task fans out to all connected clients.

The deploy endpoint hands a winning genome (looked up by id in the deployment pool) to
an executor (the paper/live engine from ``live/executor.py``). Without an executor the
endpoint returns 503.
"""

from __future__ import annotations

import asyncio
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable, List, Optional

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel

from core.evolution import EvolutionEngine, Snapshot
from core.strategy import Genome

__all__ = ["app", "Dashboard", "create_dashboard", "run_server"]

_INDEX = Path(__file__).parent / "index.html"

class DeployRequest(BaseModel):
    genome_id: str


class Dashboard:
    """Holds the engine, executor, latest snapshot, and the broadcast fan-out."""

    def __init__(self, engine: Optional[EvolutionEngine] = None,
                 executor: Any = None, auto_start: bool = True,
                 n_generations: Optional[int] = None,
                 market_data: Optional[Any] = None,
                 training_year: Optional[int] = None,
                 init_fn: Optional[Callable[["Dashboard"], None]] = None) -> None:
        self.engine = engine
        self.executor = executor
        self.auto_start = auto_start
        self.n_generations = n_generations
        self.market_data = market_data
        self.training_year = training_year
        self.init_fn = init_fn
        if self.training_year is None and engine is not None and getattr(engine, "config", None):
            self.training_year = getattr(engine.config.data, "train_year", None)
        if self.training_year is None and market_data is not None and hasattr(market_data, "ohlcv_1m") and len(market_data.ohlcv_1m) > 0:
            self.training_year = int(market_data.ohlcv_1m.index[0].year)
        self._all_time_data: Optional[Any] = None
        self.latest_snapshot: Optional[dict] = None
        self.latest_progress: Optional[dict] = None
        self.lifecycle: Optional[dict] = None
        self.clients: set = set()
        self.queue: asyncio.Queue = asyncio.Queue()
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._task: Optional[asyncio.Task] = None
        self.on_engine_done: Optional[Any] = None  # called (thread) when the engine stops

    # ------------------------------------------------------------- lifecycle
    def start(self) -> None:
        """Start the broadcaster task and (optionally) the engine thread."""
        self.loop = asyncio.get_running_loop()
        self._task = asyncio.create_task(self._broadcast())
        if self.auto_start and (self.engine is not None or self.init_fn is not None):
            self._start_engine_thread()

    def _start_engine_thread(self) -> None:
        def run() -> None:
            if self.init_fn is not None:
                try:
                    self._push_lifecycle("loading_data", "Loading market data...")
                    self.init_fn(self)
                except Exception as exc:
                    self._push_lifecycle("failed", str(exc))
                    return
            if self.engine is None:
                return
            # None -> fall back to config; 0 -> run until cancelled (infinite).
            n = self.n_generations if self.n_generations is not None \
                else self.engine.config.ga.generations
            # While serving, keep cycling through every walk-forward window.
            if getattr(self.engine, "walk_forward", False):
                self.engine.walk_forward_repeat = True
            # Stream intra-generation progress (throttled by the engine).
            self.engine.progress_cb = self.push_progress
            try:
                self.engine.run(n, callback=self.push)
                self._push_lifecycle(self._terminal_status())
            except Exception as exc:
                self._push_lifecycle("failed", str(exc))
            finally:
                if self.on_engine_done is not None:
                    try:
                        self.on_engine_done()
                    except Exception:
                        pass

        self._thread = threading.Thread(target=run, name="evolution-engine", daemon=True)
        self._thread.start()

    async def stop(self) -> None:
        if self.engine is not None:
            self.engine.progress_cb = None
            self.engine.cancel()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        if self.engine is not None:
            self.engine.flush()  # persist final state on clean shutdown
        if self._task is not None:
            self._task.cancel()

    # ------------------------------------------------------------- broadcast
    def _enqueue(self, item: dict, coalesce: bool = False) -> None:
        """Schedule an item on the loop, replacing queued transient state if requested."""
        if self.loop is not None:
            self.loop.call_soon_threadsafe(self._enqueue_on_loop, item, coalesce)

    def _enqueue_on_loop(self, item: dict, coalesce: bool) -> None:
        """Enqueue from the event-loop thread without allowing stale state to accumulate."""
        if coalesce:
            retained = []
            while True:
                try:
                    queued = self.queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if queued["type"] != item["type"] or queued.get("final"):
                    retained.append(queued)
            for queued in retained:
                self.queue.put_nowait(queued)
        self.queue.put_nowait(item)

    def push(self, snap: Snapshot) -> None:
        """Thread-safe enqueue of a full snapshot from the (non-async) engine thread."""
        data = snap.to_dict()
        # The engine emits one authoritative snapshot after final persistence. Retain it
        # in queue order so it cannot be replaced by a transient generation snapshot.
        is_final = data.get("run_status") in {"completed", "cancelled", "failed"}
        item = {"type": "snapshot", "data": data}
        if is_final:
            item["final"] = True
        self._enqueue(item, coalesce=not is_final)

    def push_progress(self, progress: dict) -> None:
        """Thread-safe enqueue of an intra-generation progress event."""
        self._enqueue({"type": "progress", "data": progress}, coalesce=True)

    def _terminal_status(self) -> str:
        """Return the terminal lifecycle status reported by the engine snapshot."""
        if self.latest_snapshot is not None:
            return str(self.latest_snapshot.get("run_status", "completed"))
        return str(getattr(self.engine, "run_status", "completed"))

    def _push_lifecycle(self, status: str, message: Optional[str] = None) -> None:
        """Publish a non-droppable engine lifecycle transition."""
        data = {"status": status}
        if message:
            data["message"] = message
        self._enqueue({"type": "lifecycle", "data": data}, coalesce=False)

    async def _broadcast(self) -> None:
        while True:
            item = await self.queue.get()
            if item["type"] == "snapshot":
                self.latest_snapshot = item["data"]
            elif item["type"] == "progress":
                self.latest_progress = item["data"]
            elif item["type"] == "lifecycle":
                self.lifecycle = item["data"]
            dead = []
            for ws in list(self.clients):
                try:
                    await ws.send_json(item)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                self.clients.discard(ws)

    # ------------------------------------------------------------- deploy
    async def deploy(self, genome_id: str) -> dict:
        if self.executor is None:
            raise HTTPException(status_code=503, detail="paper executor not configured")
        entry = self._find_deployable_genome(genome_id)
        if entry is None:
            candidate = self._find_candidate(genome_id)
            if candidate is None:
                raise HTTPException(status_code=404, detail=f"unknown genome id {genome_id}")
            status = candidate.get("status")
            if status:
                reason = f"candidate status {status} is not qualified for deployment"
            else:
                reason = "candidate lacks six-period validation"
            raise HTTPException(status_code=409, detail=reason)
        genome = Genome.decode(entry["genome"])
        await self.executor.deploy(genome)
        return {"status": "deployed", "id": genome_id, "params": entry["params"]}

    def _find_deployable_genome(self, genome_id: str) -> Optional[dict]:
        if self.engine is None:
            return None
        for e in self._deployment_pool():
            if e.get("id") == genome_id:
                return e
        return None

    def _deployment_pool(self) -> List[dict]:
        """Return only candidates that passed validation and may be deployed."""
        if self.engine is None:
            return []
        return [
            entry for entry in getattr(self.engine, "deployment_pool", [])
            if isinstance(entry, dict)
            and entry.get("status") == "qualified"
            and entry.get("deployable") is True
        ]

    def _find_candidate(self, genome_id: str) -> Optional[dict]:
        """Find a known candidate without consulting engine-private registries."""
        if self.engine is None:
            return None
        sources = [getattr(self.engine, "deployment_pool", [])]
        for name in ("champion_candidates", "candidate_registry"):
            registry = getattr(self.engine, name, None)
            if isinstance(registry, dict):
                sources.append(registry.values())
            elif registry is not None:
                sources.append(registry)
        for candidates in sources:
            for candidate in candidates:
                if isinstance(candidate, dict) and candidate.get("id") == genome_id:
                    return candidate
        return None

    def get_available_years(self) -> List[int]:
        from core.data import available_years_for_data
        return available_years_for_data(self.market_data)

    def _get_all_time_data(self) -> Any:
        if self._all_time_data is None:
            from core.data import get_or_create_all_time_data
            cfg_data = self.engine.config.data if self.engine else None
            seed = cfg_data.global_seed if cfg_data else 42
            self._all_time_data = get_or_create_all_time_data(source_data=self.market_data, seed=seed, cfg=cfg_data)
        return self._all_time_data

    def get_all_time_range(self) -> dict:
        yd = self._get_all_time_data()
        if hasattr(yd, "ohlcv_1m") and len(yd.ohlcv_1m) > 0:
            s_dt = yd.ohlcv_1m.index[0]
            e_dt = yd.ohlcv_1m.index[-1]
            return {
                "start": str(s_dt),
                "end": str(e_dt),
                "start_formatted": s_dt.strftime("%b %Y"),
                "end_formatted": e_dt.strftime("%b %Y"),
                "start_date": s_dt.strftime("%Y-%m-%d"),
                "end_date": e_dt.strftime("%Y-%m-%d"),
                "start_year": int(s_dt.year),
                "end_year": int(e_dt.year),
                "bars_1m": len(yd.ohlcv_1m),
            }
        return {}

    def backtest_year(self, year: Any, genome_id: Optional[str] = None) -> dict:
        """Run a zero-lookahead backtest of the current champion (or genome_id) on a calendar year or all-time."""
        import numpy as np
        from config import InstrumentSpec
        from core.backtester import backtest
        from core.data import get_or_create_year_data
        from core.features import compute_features
        from core.strategy import GENE_INDEX, Genome, SignalFeatures, compute_signals
        from core.xp import asnumpy

        # 1. Resolve genome
        genome_dict = None
        if genome_id and self.engine:
            gid_key = int(genome_id) if str(genome_id).isdigit() else genome_id
            genome_dict = self.engine.candidate_registry.get(gid_key)
            if genome_dict is None:
                for cand in self.engine.deployment_pool:
                    if str(cand.get("id")) == str(genome_id):
                        genome_dict = cand
                        break

        if genome_dict is None and self.engine:
            champ_id = getattr(self.engine, "champion_genome_id", None)
            if champ_id is not None and champ_id in getattr(self.engine, "candidate_registry", {}):
                genome_dict = self.engine.candidate_registry[champ_id]
            elif getattr(self.engine, "deployment_pool", None):
                genome_dict = self.engine.deployment_pool[0]
            elif self.latest_snapshot and self.latest_snapshot.get("top_genomes"):
                genome_dict = self.latest_snapshot["top_genomes"][0]

        if genome_dict is None:
            g_vec = np.zeros((1, 15))
            g_vec[0, GENE_INDEX["ema_fast"]] = 12
            g_vec[0, GENE_INDEX["ema_slow"]] = 26
            g_vec[0, GENE_INDEX["st_period"]] = 10
            g_vec[0, GENE_INDEX["st_mult"]] = 3.0
            g_vec[0, GENE_INDEX["atr_period"]] = 14
            g_vec[0, GENE_INDEX["entry_offset_atr_mult"]] = 1.0
            g_vec[0, GENE_INDEX["tp_atr_mult"]] = 3.0
            g_vec[0, GENE_INDEX["sl_atr_mult"]] = 2.0
            g_vec[0, GENE_INDEX["max_hold_bars"]] = 48
            g_vec[0, GENE_INDEX["target_leverage"]] = 1.0
            g_vec[0, GENE_INDEX["risk_per_trade_pct"]] = 0.015
            g_vec[0, GENE_INDEX["trailing_atr_mult"]] = 2.0
            g_vec[0, GENE_INDEX["funding_filter_z"]] = 2.0
            g_vec[0, GENE_INDEX["volume_filter_z"]] = -1.0
            gid = "default"
        else:
            gid = str(genome_dict.get("id", "champion"))
            if "genome" in genome_dict:
                g_vec = np.asarray(genome_dict["genome"], dtype=np.float64)[None, :]
            elif "params" in genome_dict:
                g_vec = Genome(**genome_dict["params"]).encode()[None, :]
            else:
                g_vec = np.zeros((1, 15))

        # 2. Get market data (single calendar year, year range, or all-time)
        from core.data import parse_year_spec
        cfg_data = self.engine.config.data if self.engine else None
        seed = cfg_data.global_seed if cfg_data else 42
        y1, y2 = parse_year_spec(year)
        is_all_time = (y1 is None)
        if is_all_time:
            yd = self._get_all_time_data()
            display_year = "all"
        else:
            yd = get_or_create_year_data(year, source_data=self.market_data, seed=seed, cfg=cfg_data)
            display_year = str(y1) if y1 == y2 else f"{y1}-{y2}"

        # Enforce strict whole-hour alignment (len(ohlcv_1m) % 60 == 0)
        m_len = (len(yd.ohlcv_1m) // 60) * 60
        if m_len < len(yd.ohlcv_1m):
            from core.data import _resample_signals, SyntheticData
            yd_1m = yd.ohlcv_1m.iloc[:m_len]
            h_len = m_len // 60
            yd_1h = yd.ohlcv_1h.iloc[:h_len] if len(yd.ohlcv_1h) >= h_len else _resample_signals(yd_1m, "1h")
            yd_fund = yd.funding.reindex(yd_1h.index).ffill().fillna(0.0)
            yd = SyntheticData(ohlcv_1m=yd_1m, ohlcv_1h=yd_1h, funding=yd_fund)

        # 3. Compute 1h rolling features and signals
        feats_1h = compute_features(yd.ohlcv_1h, yd.funding)
        sig_feat = SignalFeatures(
            open=yd.ohlcv_1h["open"].to_numpy(),
            high=yd.ohlcv_1h["high"].to_numpy(),
            low=yd.ohlcv_1h["low"].to_numpy(),
            close=yd.ohlcv_1h["close"].to_numpy(),
            volume_zscore=asnumpy(feats_1h.volume_zscore),
            funding_zscore=asnumpy(feats_1h.funding_zscore),
        )
        sig = compute_signals(sig_feat, g_vec)

        # 4. Run backtest with Numba JIT acceleration
        inst = self.engine.instrument if self.engine else InstrumentSpec()
        r = backtest(yd.ohlcv_1m, yd.ohlcv_1h, yd.funding, sig, g_vec, inst, use_numba=True)

        # 5. Extract equity curve and downsample for smooth UI charting (hourly or multi-hour for all-time)
        eq_raw = asnumpy(r.equity_curve)[0]
        m_len = len(yd.ohlcv_1m)
        step = max(60, (m_len // 7500 // 60) * 60) if is_all_time else max(1, 60)
        indices = np.arange(0, m_len, step)
        if indices[-1] != m_len - 1:
            indices = np.append(indices, m_len - 1)

        ts_index = yd.ohlcv_1m.index[indices]
        ts_ms = (ts_index.astype("int64") // 10**6).tolist()
        eq_sampled = eq_raw[indices].tolist()

        # SOL buy-and-hold benchmark normalized to $10,000
        sol_close = yd.ohlcv_1m["close"].iloc[indices].to_numpy()
        sol_ret = float((sol_close[-1] - sol_close[0]) / sol_close[0]) if len(sol_close) > 0 else 0.0
        sol_norm = (sol_close / sol_close[0] * 10000.0).tolist()

        eq_pts = [[t, round(v, 2)] for t, v in zip(ts_ms, eq_sampled)]
        sol_pts = [[t, round(v, 2)] for t, v in zip(ts_ms, sol_norm)]

        # 6. Compute robust metrics for the year or all-time
        trades_count = int(asnumpy(r.num_trades)[0])
        tot_ret = float((eq_raw[-1] - 10000.0) / 10000.0)
        peaks = np.maximum.accumulate(eq_raw)
        drawdowns = (peaks - eq_raw) / peaks
        max_dd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

        eq_hourly = eq_raw[::60]
        if len(eq_hourly) > 1:
            h_ret = np.diff(eq_hourly) / eq_hourly[:-1]
            ann_factor = np.sqrt(8760.0)
            ret_mean = np.mean(h_ret)
            ret_std = np.std(h_ret)
            sharpe = float(ret_mean / ret_std * ann_factor) if ret_std > 1e-9 else 0.0
            downside = h_ret[h_ret < 0]
            down_std = np.std(downside) if len(downside) > 1 else ret_std
            sortino = float(ret_mean / down_std * ann_factor) if down_std > 1e-9 else 0.0
        else:
            sharpe = 0.0
            sortino = 0.0

        win_rate = 0.0
        if r.trades and r.trades[0]:
            wins = sum(1 for t in r.trades[0] if t.pnl > 0)
            win_rate = float(wins / len(r.trades[0]))
        elif trades_count > 0:
            win_rate = max(0.0, min(1.0, 0.5 + (tot_ret / (trades_count * 0.02 + 1e-6))))

        tot_fees = float(asnumpy(r.total_fees)[0])
        tot_funding = float(asnumpy(r.total_funding)[0])

        start_dt = yd.ohlcv_1m.index[0]
        end_dt = yd.ohlcv_1m.index[-1]
        start_str = start_dt.strftime("%b %d, %Y")
        end_str = end_dt.strftime("%b %d, %Y")
        if is_all_time:
            title = f"All-Time Evaluation ({start_str} – {end_str})"
        elif y1 != y2:
            title = f"{display_year} Multi-Year Backtest ({start_str} – {end_str})"
        else:
            title = f"{display_year} Full-Year Backtest ({start_str} – {end_str})"

        return {
            "year": display_year,
            "is_all_time": is_all_time,
            "title": title,
            "training_year": self.training_year,
            "is_out_of_distribution": (self.training_year is not None and not is_all_time and str(display_year) != str(self.training_year)),
            "genome_id": gid,
            "metrics": {
                "total_return": tot_ret,
                "sharpe": sharpe,
                "sortino": sortino,
                "max_drawdown": max_dd,
                "trades": trades_count,
                "win_rate": win_rate,
                "fees": tot_fees,
                "funding": tot_funding,
            },
            "equity_curve": eq_pts,
            "benchmark_equity_curve": sol_pts,
            "benchmark_curve": sol_pts,
            "benchmark_return": sol_ret,
            "candle_count": len(yd.ohlcv_1m),
            "coverage": {
                "start": str(start_dt),
                "end": str(end_dt),
                "start_formatted": start_str,
                "end_formatted": end_str,
                "bars_1m": len(yd.ohlcv_1m),
                "bars_1h": len(yd.ohlcv_1h),
            }
        }


app = FastAPI(title="SOL-PERP Evolutionary Trading System", lifespan=None)


@asynccontextmanager
async def _lifespan(app_: FastAPI):
    dash: Dashboard = app_.state.dashboard
    if dash is not None:
        dash.start()
    try:
        yield
    finally:
        if dash is not None:
            await dash.stop()


# Set the lifespan after definition so Dashboard wiring is explicit.
app.router.lifespan_context = _lifespan


def _dash(request: Request) -> Dashboard:
    return request.app.state.dashboard


@app.get("/")
async def index() -> FileResponse:
    # no-store so the browser always loads the current dashboard (avoids stale UI).
    return FileResponse(_INDEX, headers={"Cache-Control": "no-store, max-age=0"})


@app.get("/api/state")
async def state(request: Request) -> dict:
    dash = _dash(request)
    if dash.latest_snapshot is None:
        return {"generation": -1}
    return dash.latest_snapshot


@app.get("/api/pool")
async def pool(request: Request) -> List[dict]:
    dash = _dash(request)
    return dash._deployment_pool()


@app.post("/api/deploy")
async def deploy(req: DeployRequest, request: Request) -> dict:
    dash = _dash(request)
    return await dash.deploy(req.genome_id)


@app.post("/api/deploy_ensemble")
async def deploy_ensemble(request: Request) -> dict:
    dash = _dash(request)
    if dash.executor is None:
        raise HTTPException(status_code=503, detail="paper executor not configured")
    if dash.engine is None:
        raise HTTPException(status_code=503, detail="evolution engine not available")
    members = dash.engine.deployment_ensemble()
    if not members and dash.engine.deployment_pool:
        from core.strategy import Genome
        pool = dash.engine.deployment_pool[:5]
        w = 1.0 / len(pool)
        members = [{
            "id": m["id"],
            "genome": Genome.decode(m["genome"]),
            "weight": w,
            "score": float(m.get("score", 0.0)),
            "params": m.get("params", Genome.decode(m["genome"]).__dict__)
        } for m in pool]
    if not members:
        raise HTTPException(status_code=400, detail="no qualified members in deployment ensemble")
    await dash.executor.deploy_ensemble(members)
    return {"status": "deployed_ensemble", "members": members}


@app.get("/api/executor")
async def executor_status(request: Request) -> dict:
    dash = _dash(request)
    if dash.executor is None:
        return {"status": "inactive", "mode": "none"}
    telemetry = dash.executor.get_telemetry()
    return {"status": "active", **telemetry}


@app.get("/api/available_years")
async def available_years(request: Request) -> dict:
    dash = _dash(request)
    return {
        "years": dash.get_available_years(),
        "training_year": dash.training_year,
        "has_all_time": True,
        "all_time_range": dash.get_all_time_range(),
    }


@app.get("/api/backtest_year")
async def backtest_year(year: str, request: Request, genome_id: Optional[str] = None) -> dict:
    dash = _dash(request)
    return dash.backtest_year(year, genome_id)


# ---------------------------------------------------------------------------
# Multi-Asset Macro CTA Portfolio Endpoints (100x Engine)
# ---------------------------------------------------------------------------

@app.get("/api/portfolio/summary")
async def portfolio_summary(mode: str = "conservative") -> dict:
    from ui.portfolio_service import PortfolioService
    return PortfolioService.get_instance().get_summary(mode=mode)


@app.get("/api/portfolio/oos")
async def portfolio_oos(mode: str = "conservative") -> dict:
    from ui.portfolio_service import PortfolioService
    return PortfolioService.get_instance().get_oos_report(mode=mode)


@app.get("/api/portfolio/equity")
async def portfolio_equity(mode: str = "conservative", max_points: int = 600) -> dict:
    from ui.portfolio_service import PortfolioService
    return PortfolioService.get_instance().get_equity_curve(mode=mode, max_points=max_points)


@app.get("/api/portfolio/trades")
async def portfolio_trades(mode: str = "conservative", limit: int = 150) -> List[dict]:
    from ui.portfolio_service import PortfolioService
    return PortfolioService.get_instance().get_trades(mode=mode, limit=limit)


@app.websocket("/ws/evolution")
async def ws_evolution(websocket: WebSocket) -> None:
    dash: Dashboard = websocket.app.state.dashboard
    await websocket.accept()
    dash.clients.add(websocket)
    # Send the latest snapshot immediately if we already have one.
    if dash.latest_snapshot is not None:
        await websocket.send_json({"type": "snapshot", "data": dash.latest_snapshot})
    if dash.lifecycle is not None:
        await websocket.send_json({"type": "lifecycle", "data": dash.lifecycle})
    if dash.executor is not None:
        await websocket.send_json({"type": "executor", "data": dash.executor.get_telemetry()})
    try:
        while True:
            # Keep the socket alive; inbound messages are ignored (we only broadcast).
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        dash.clients.discard(websocket)


def create_dashboard(engine: Optional[EvolutionEngine] = None, executor: Any = None,
                     auto_start: bool = True,
                     n_generations: Optional[int] = None,
                     market_data: Optional[Any] = None,
                     training_year: Optional[int] = None,
                     init_fn: Optional[Callable[["Dashboard"], None]] = None) -> Dashboard:
    """Build a Dashboard and attach it to the app's state."""
    dash = Dashboard(engine=engine, executor=executor, auto_start=auto_start,
                     n_generations=n_generations,
                     market_data=market_data,
                     training_year=training_year,
                     init_fn=init_fn)
    app.state.dashboard = dash
    return dash


def run_server(host: str = "127.0.0.1", port: int = 8000,
                dashboard: Optional[Dashboard] = None) -> None:
    """Run uvicorn (blocking).

    The dashboard stays available after a finite evolution run finishes so its final
    walk-forward curves and deployment pool remain inspectable. Stop it with Ctrl+C.
    """
    import uvicorn

    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)
    server.run()
