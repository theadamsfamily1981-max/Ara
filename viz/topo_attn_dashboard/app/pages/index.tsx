import { useEffect, useState } from "react";
import PDPlot from "../components/PDPlot";
import AttnHeat from "../components/AttnHeat";
import SparsityGauge from "../components/SparsityGauge";
import MetricsPanel from "../components/MetricsPanel";

interface PDPoint {
  b: number;
  d: number;
}

interface Snapshot {
  ts: number;
  step: number;
  sparsity: number;
  kept_idx: number[];
  attn_block: number[][];
  pd: PDPoint[];
  epr_cv?: number;
  fdt_lr_delta?: number;
  spike_rate?: number;
  vfe?: number;
}

interface Message {
  type: "update" | "history" | "pong";
  snapshot?: Snapshot;
  snapshots?: Snapshot[];
}

export default function Home() {
  const [data, setData] = useState<Snapshot | null>(null);
  const [connected, setConnected] = useState(false);
  const [wsUrl, setWsUrl] = useState("ws://localhost:8765");
  const [history, setHistory] = useState<Snapshot[]>([]);

  useEffect(() => {
    let ws: WebSocket | null = null;
    let reconnectTimeout: NodeJS.Timeout;

    const connect = () => {
      try {
        ws = new WebSocket(wsUrl);

        ws.onopen = () => {
          console.log("WebSocket connected");
          setConnected(true);
        };

        ws.onmessage = (e) => {
          try {
            const msg: Message = JSON.parse(e.data);

            if (msg.type === "update" && msg.snapshot) {
              setData(msg.snapshot);
              setHistory(prev => [...prev.slice(-99), msg.snapshot!]);
            } else if (msg.type === "history" && msg.snapshots) {
              setHistory(msg.snapshots);
              if (msg.snapshots.length > 0) {
                setData(msg.snapshots[msg.snapshots.length - 1]);
              }
            }
          } catch (err) {
            console.error("Failed to parse message:", err);
          }
        };

        ws.onerror = (err) => {
          console.error("WebSocket error:", err);
        };

        ws.onclose = () => {
          console.log("WebSocket disconnected");
          setConnected(false);
          // Attempt reconnect after 3s
          reconnectTimeout = setTimeout(connect, 3000);
        };
      } catch (err) {
        console.error("Failed to connect:", err);
        reconnectTimeout = setTimeout(connect, 3000);
      }
    };

    connect();

    return () => {
      if (ws) {
        ws.close();
      }
      if (reconnectTimeout) {
        clearTimeout(reconnectTimeout);
      }
    };
  }, [wsUrl]);

  const exportRunCard = () => {
    if (!data) return;

    const blob = new Blob([JSON.dumps(data, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `run_card_step_${data.step}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const exportHistoryCSV = () => {
    if (history.length === 0) return;

    const headers = ["step", "sparsity", "epr_cv", "vfe", "spike_rate"];
    const rows = history.map(h => [
      h.step,
      h.sparsity,
      h.epr_cv || "",
      h.vfe || "",
      h.spike_rate || ""
    ]);

    const csv = [
      headers.join(","),
      ...rows.map(r => r.join(","))
    ].join("\n");

    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `metrics_history.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <main className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <header className="mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">
            🔬 Topo-Attention Glass
          </h1>
          <p className="text-gray-600">
            Real-time visualization of TF-A-N topology and attention dynamics
          </p>

          <div className="mt-4 flex items-center gap-4">
            <div className="flex items-center gap-2">
              <div className={`w-3 h-3 rounded-full ${connected ? 'bg-green-500' : 'bg-red-500'}`} />
              <span className="text-sm text-gray-600">
                {connected ? 'Connected' : 'Disconnected'}
              </span>
            </div>

            <input
              type="text"
              value={wsUrl}
              onChange={(e) => setWsUrl(e.target.value)}
              className="px-3 py-1 border border-gray-300 rounded text-sm"
              placeholder="WebSocket URL"
            />

            <button
              onClick={exportRunCard}
              disabled={!data}
              className="px-4 py-1 bg-blue-600 text-white rounded text-sm hover:bg-blue-700 disabled:bg-gray-300"
            >
              Export Run-Card
            </button>

            <button
              onClick={exportHistoryCSV}
              disabled={history.length === 0}
              className="px-4 py-1 bg-green-600 text-white rounded text-sm hover:bg-green-700 disabled:bg-gray-300"
            >
              Export History (CSV)
            </button>
          </div>
        </header>

        {/* Content */}
        {!data ? (
          <div className="flex items-center justify-center h-64">
            <div className="text-center">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
              <p className="text-gray-600">Waiting for stream...</p>
              <p className="text-sm text-gray-400 mt-2">
                Run: <code className="bg-gray-100 px-2 py-1 rounded">python scripts/emit_demo_metrics.py</code>
              </p>
            </div>
          </div>
        ) : (
          <>
            {/* Metrics Summary */}
            <MetricsPanel data={data} />

            {/* Main Visualizations */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-6">
              {/* Persistence Diagram */}
              <div className="bg-white rounded-lg shadow p-6">
                <h2 className="text-xl font-semibold mb-4">Persistence Diagram</h2>
                <PDPlot pd={data.pd} />
              </div>

              {/* Attention Heatmap */}
              <div className="bg-white rounded-lg shadow p-6">
                <h2 className="text-xl font-semibold mb-4">Attention Pattern</h2>
                <AttnHeat mat={data.attn_block} />
              </div>
            </div>

            {/* Sparsity Gauge */}
            <div className="mt-6">
              <SparsityGauge
                value={1 - data.sparsity}
                keptIndices={data.kept_idx}
                totalLength={data.attn_block.length}
              />
            </div>

            {/* Step Info */}
            <div className="mt-6 text-center text-sm text-gray-500">
              Step: {data.step} | Last updated: {new Date(data.ts * 1000).toLocaleTimeString()}
            </div>
          </>
        )}
      </div>
    </main>
  );
}
