interface PDPoint {
  b: number;
  d: number;
}

interface PDPlotProps {
  pd: PDPoint[];
  width?: number;
  height?: number;
}

export default function PDPlot({ pd, width = 500, height = 400 }: PDPlotProps) {
  if (!pd || pd.length === 0) {
    return (
      <div className="flex items-center justify-center h-96 bg-gray-50 rounded">
        <p className="text-gray-400">No persistence diagram data</p>
      </div>
    );
  }

  const margin = { top: 20, right: 20, bottom: 50, left: 60 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;

  // Find max value for scaling
  const maxVal = Math.max(
    ...pd.map(p => Math.max(p.b, p.d)),
    0.1
  );

  const scale = (v: number) => (v / maxVal) * Math.min(plotWidth, plotHeight);

  // Color by persistence
  const getColor = (p: PDPoint) => {
    const persistence = p.d - p.b;
    const normalized = persistence / maxVal;

    if (normalized > 0.5) return "rgb(220, 38, 38)"; // High persistence - red
    if (normalized > 0.2) return "rgb(251, 146, 60)"; // Medium - orange
    return "rgb(59, 130, 246)"; // Low - blue
  };

  return (
    <svg
      width={width}
      height={height}
      className="border border-gray-200 rounded bg-white"
    >
      <g transform={`translate(${margin.left},${margin.top})`}>
        {/* Diagonal line (birth = death) */}
        <line
          x1={0}
          y1={plotHeight}
          x2={Math.min(plotWidth, plotHeight)}
          y2={plotHeight - Math.min(plotWidth, plotHeight)}
          stroke="#e5e7eb"
          strokeWidth={2}
          strokeDasharray="5,5"
        />

        {/* X axis */}
        <line
          x1={0}
          y1={plotHeight}
          x2={plotWidth}
          y2={plotHeight}
          stroke="#374151"
          strokeWidth={2}
        />
        <text
          x={plotWidth / 2}
          y={plotHeight + 35}
          textAnchor="middle"
          className="text-sm fill-gray-700"
        >
          Birth
        </text>

        {/* Y axis */}
        <line
          x1={0}
          y1={0}
          x2={0}
          y2={plotHeight}
          stroke="#374151"
          strokeWidth={2}
        />
        <text
          x={-plotHeight / 2}
          y={-40}
          textAnchor="middle"
          transform={`rotate(-90, ${-plotHeight / 2}, -40)`}
          className="text-sm fill-gray-700"
        >
          Death
        </text>

        {/* Points */}
        {pd.map((p, i) => {
          const x = scale(p.b);
          const y = plotHeight - scale(p.d);
          const persistence = p.d - p.b;

          return (
            <g key={i}>
              <circle
                cx={x}
                cy={y}
                r={3 + Math.sqrt(persistence / maxVal) * 4}
                fill={getColor(p)}
                opacity={0.7}
                className="hover:opacity-100 transition-opacity"
              >
                <title>
                  Birth: {p.b.toFixed(4)}, Death: {p.d.toFixed(4)}, Persistence: {persistence.toFixed(4)}
                </title>
              </circle>
            </g>
          );
        })}

        {/* Axis ticks */}
        {[0, 0.25, 0.5, 0.75, 1.0].map((frac) => {
          const val = frac * maxVal;
          const pos = scale(val);

          return (
            <g key={frac}>
              {/* X tick */}
              <line
                x1={pos}
                y1={plotHeight}
                x2={pos}
                y2={plotHeight + 5}
                stroke="#374151"
              />
              <text
                x={pos}
                y={plotHeight + 20}
                textAnchor="middle"
                className="text-xs fill-gray-600"
              >
                {val.toFixed(2)}
              </text>

              {/* Y tick */}
              <line
                x1={0}
                y1={plotHeight - pos}
                x2={-5}
                y2={plotHeight - pos}
                stroke="#374151"
              />
              <text
                x={-10}
                y={plotHeight - pos + 4}
                textAnchor="end"
                className="text-xs fill-gray-600"
              >
                {val.toFixed(2)}
              </text>
            </g>
          );
        })}
      </g>

      {/* Legend */}
      <g transform={`translate(${width - 120}, 30)`}>
        <text className="text-xs fill-gray-700 font-semibold" y={0}>
          Persistence
        </text>
        <circle cx={10} cy={15} r={4} fill="rgb(220, 38, 38)" />
        <text x={20} y={19} className="text-xs fill-gray-600">
          High
        </text>
        <circle cx={10} cy={30} r={4} fill="rgb(251, 146, 60)" />
        <text x={20} y={34} className="text-xs fill-gray-600">
          Medium
        </text>
        <circle cx={10} cy={45} r={4} fill="rgb(59, 130, 246)" />
        <text x={20} y={49} className="text-xs fill-gray-600">
          Low
        </text>
      </g>
    </svg>
  );
}
