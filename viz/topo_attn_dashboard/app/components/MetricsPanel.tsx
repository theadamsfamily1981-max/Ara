interface Snapshot {
  step: number;
  sparsity: number;
  epr_cv?: number;
  fdt_lr_delta?: number;
  spike_rate?: number;
  vfe?: number;
  pd: any[];
  // Antifragility metrics
  delta_p99?: number;
  delta_p99_percent?: number;
  antifragility_score?: number;
  baseline_p99?: number;
  adaptive_p99?: number;
  // CLV metrics
  clv_instability?: number;
  clv_resource?: number;
  clv_structural?: number;
  risk_level?: string;
}

interface MetricsPanelProps {
  data: Snapshot;
}

// Helper to determine color based on antifragility value
function getAntifragilityColor(delta?: number): string {
  if (delta === undefined) return "gray";
  if (delta > 5) return "green";    // Strong advantage
  if (delta > 0) return "emerald";  // Positive advantage
  if (delta > -2) return "yellow";  // Marginal
  return "red";                      // Worse than baseline
}

// Helper to determine color based on risk level
function getRiskColor(level?: string): string {
  switch (level?.toLowerCase()) {
    case "nominal": return "green";
    case "elevated": return "yellow";
    case "warning": return "orange";
    case "critical": return "red";
    case "emergency": return "red";
    default: return "gray";
  }
}

export default function MetricsPanel({ data }: MetricsPanelProps) {
  // Core metrics
  const coreMetrics = [
    {
      name: "Step",
      value: data.step.toLocaleString(),
      color: "gray",
      icon: "📊",
    },
    {
      name: "EPR-CV",
      value: data.epr_cv?.toFixed(4) || "N/A",
      color: data.epr_cv && data.epr_cv <= 0.15 ? "green" : "yellow",
      icon: "🎯",
      tooltip: "Epistemic uncertainty (target ≤ 0.15)",
    },
    {
      name: "VFE",
      value: data.vfe?.toFixed(3) || "N/A",
      color: "blue",
      icon: "⚡",
      tooltip: "Variational Free Energy",
    },
    {
      name: "Spike Rate",
      value: data.spike_rate ? `${(data.spike_rate * 100).toFixed(1)}%` : "N/A",
      color: "purple",
      icon: "🔋",
      tooltip: "SNN spike rate",
    },
    {
      name: "FDT LR Δ",
      value: data.fdt_lr_delta?.toExponential(2) || "N/A",
      color: "orange",
      icon: "🎛️",
      tooltip: "Learning rate adjustment from FDT",
    },
    {
      name: "PD Features",
      value: data.pd.length.toLocaleString(),
      color: "indigo",
      icon: "🔍",
      tooltip: "Persistence diagram feature count",
    },
  ];

  // Antifragility metrics (primary KPI for certification)
  const antifragilityMetrics = [
    {
      name: "Δp99 Latency",
      value: data.delta_p99 !== undefined
        ? `${data.delta_p99 > 0 ? "+" : ""}${data.delta_p99.toFixed(1)}ms`
        : "N/A",
      color: getAntifragilityColor(data.delta_p99),
      icon: "🛡️",
      tooltip: "Latency advantage vs baseline under burst load (positive = better)",
      highlight: true,
    },
    {
      name: "Δp99 %",
      value: data.delta_p99_percent !== undefined
        ? `${data.delta_p99_percent > 0 ? "+" : ""}${data.delta_p99_percent.toFixed(1)}%`
        : "N/A",
      color: getAntifragilityColor(data.delta_p99_percent),
      icon: "📈",
      tooltip: "Percentage improvement in p99 latency",
    },
    {
      name: "Antifragility",
      value: data.antifragility_score?.toFixed(2) || "N/A",
      color: data.antifragility_score && data.antifragility_score > 1.0 ? "green" : "yellow",
      icon: "💪",
      tooltip: "Score > 1.0 = adaptive system degrades less under stress",
    },
  ];

  // CLV metrics (Cognitive Load Vector)
  const clvMetrics = [
    {
      name: "CLV Risk",
      value: data.risk_level || "N/A",
      color: getRiskColor(data.risk_level),
      icon: "⚠️",
      tooltip: "Cognitive Load Vector risk level",
    },
    {
      name: "Instability",
      value: data.clv_instability?.toFixed(3) || "N/A",
      color: data.clv_instability !== undefined && data.clv_instability < 0.3 ? "green" : "orange",
      icon: "🌊",
      tooltip: "CLV instability component (EPR-CV + topo_gap)",
    },
    {
      name: "Resource",
      value: data.clv_resource?.toFixed(3) || "N/A",
      color: data.clv_resource !== undefined && data.clv_resource < 0.5 ? "green" : "yellow",
      icon: "💻",
      tooltip: "CLV resource component (jerk + latency)",
    },
  ];

  // Combine all metrics
  const metrics = [...coreMetrics, ...antifragilityMetrics, ...clvMetrics];

  const colorClasses: Record<string, string> = {
    gray: "bg-gray-100 border-gray-300 text-gray-900",
    green: "bg-green-100 border-green-300 text-green-900",
    emerald: "bg-emerald-100 border-emerald-300 text-emerald-900",
    yellow: "bg-yellow-100 border-yellow-300 text-yellow-900",
    blue: "bg-blue-100 border-blue-300 text-blue-900",
    purple: "bg-purple-100 border-purple-300 text-purple-900",
    orange: "bg-orange-100 border-orange-300 text-orange-900",
    red: "bg-red-100 border-red-300 text-red-900",
    indigo: "bg-indigo-100 border-indigo-300 text-indigo-900",
  };

  return (
    <div className="space-y-6">
      {/* Primary KPI: Antifragility Delta */}
      {data.delta_p99 !== undefined && (
        <div className="bg-gradient-to-r from-emerald-500 to-green-600 rounded-xl p-6 text-white shadow-lg">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-lg font-medium opacity-90">Antifragility Delta (Δp99)</h3>
              <p className="text-4xl font-bold mt-2">
                {data.delta_p99 > 0 ? "+" : ""}{data.delta_p99.toFixed(1)}ms
              </p>
              <p className="text-sm opacity-75 mt-1">
                {data.delta_p99_percent !== undefined && (
                  <span>{data.delta_p99_percent > 0 ? "+" : ""}{data.delta_p99_percent.toFixed(1)}% vs baseline</span>
                )}
              </p>
            </div>
            <div className="text-6xl opacity-80">🛡️</div>
          </div>
          <div className="mt-4 flex items-center gap-4 text-sm">
            <div className="bg-white/20 rounded-lg px-3 py-1">
              Baseline: {data.baseline_p99?.toFixed(1) || "N/A"}ms
            </div>
            <div className="bg-white/20 rounded-lg px-3 py-1">
              Adaptive: {data.adaptive_p99?.toFixed(1) || "N/A"}ms
            </div>
            <div className="bg-white/20 rounded-lg px-3 py-1">
              Score: {data.antifragility_score?.toFixed(2) || "N/A"}×
            </div>
          </div>
        </div>
      )}

      {/* Core Metrics Grid */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
        {metrics.map((metric) => (
          <div
            key={metric.name}
            className={`p-4 rounded-lg border-2 ${
              colorClasses[metric.color] || colorClasses.gray
            }`}
            title={metric.tooltip}
          >
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-semibold">{metric.name}</span>
              <span className="text-lg">{metric.icon}</span>
            </div>
            <div className="text-2xl font-bold">{metric.value}</div>
          </div>
        ))}
      </div>
    </div>
  );
}
