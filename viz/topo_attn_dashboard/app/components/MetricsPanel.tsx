interface Snapshot {
  step: number;
  sparsity: number;
  epr_cv?: number;
  fdt_lr_delta?: number;
  spike_rate?: number;
  vfe?: number;
  pd: any[];
}

interface MetricsPanelProps {
  data: Snapshot;
}

export default function MetricsPanel({ data }: MetricsPanelProps) {
  const metrics = [
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

  const colorClasses = {
    gray: "bg-gray-100 border-gray-300 text-gray-900",
    green: "bg-green-100 border-green-300 text-green-900",
    yellow: "bg-yellow-100 border-yellow-300 text-yellow-900",
    blue: "bg-blue-100 border-blue-300 text-blue-900",
    purple: "bg-purple-100 border-purple-300 text-purple-900",
    orange: "bg-orange-100 border-orange-300 text-orange-900",
    indigo: "bg-indigo-100 border-indigo-300 text-indigo-900",
  };

  return (
    <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
      {metrics.map((metric) => (
        <div
          key={metric.name}
          className={`p-4 rounded-lg border-2 ${
            colorClasses[metric.color as keyof typeof colorClasses]
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
  );
}
