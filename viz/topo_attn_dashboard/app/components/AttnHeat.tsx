interface AttnHeatProps {
  mat: number[][];
  width?: number;
  height?: number;
}

export default function AttnHeat({ mat, width = 500, height = 500 }: AttnHeatProps) {
  if (!mat || mat.length === 0) {
    return (
      <div className="flex items-center justify-center h-96 bg-gray-50 rounded">
        <p className="text-gray-400">No attention data</p>
      </div>
    );
  }

  const rows = mat.length;
  const cols = mat[0]?.length || 0;

  if (cols === 0) {
    return (
      <div className="flex items-center justify-center h-96 bg-gray-50 rounded">
        <p className="text-gray-400">Invalid attention matrix</p>
      </div>
    );
  }

  const cellWidth = width / cols;
  const cellHeight = height / rows;

  // Find max value for normalization
  const maxVal = Math.max(...mat.flat(), 0.001);

  // Color scale: white -> blue
  const getColor = (value: number) => {
    const normalized = value / maxVal;
    const intensity = Math.floor(normalized * 255);

    // Blue gradient
    return `rgb(${255 - intensity}, ${255 - intensity}, 255)`;
  };

  return (
    <div className="relative">
      <svg
        width={width}
        height={height}
        className="border border-gray-200 rounded bg-white"
      >
        {/* Heatmap cells */}
        {mat.map((row, r) =>
          row.map((val, c) => (
            <rect
              key={`${r}-${c}`}
              x={c * cellWidth}
              y={r * cellHeight}
              width={cellWidth}
              height={cellHeight}
              fill={getColor(val)}
              stroke="#f3f4f6"
              strokeWidth={0.5}
            >
              <title>
                Row {r}, Col {c}: {val.toFixed(4)}
              </title>
            </rect>
          ))
        )}

        {/* Causal mask indicator (diagonal) */}
        <line
          x1={0}
          y1={0}
          x2={width}
          y2={height}
          stroke="#ef4444"
          strokeWidth={1}
          strokeDasharray="4,4"
          opacity={0.3}
        />
      </svg>

      {/* Labels */}
      <div className="mt-2 flex justify-between text-xs text-gray-600">
        <span>← Query</span>
        <span>Key →</span>
      </div>

      {/* Color scale legend */}
      <div className="mt-4 flex items-center gap-2">
        <span className="text-xs text-gray-600">Attention weight:</span>
        <div className="flex-1 h-4 rounded overflow-hidden flex">
          {[...Array(10)].map((_, i) => {
            const val = (i / 10) * maxVal;
            return (
              <div
                key={i}
                className="flex-1"
                style={{ backgroundColor: getColor(val) }}
                title={val.toFixed(4)}
              />
            );
          })}
        </div>
        <span className="text-xs text-gray-600">
          0.00 — {maxVal.toFixed(3)}
        </span>
      </div>

      {/* Stats */}
      <div className="mt-3 grid grid-cols-3 gap-2 text-xs">
        <div className="bg-gray-50 p-2 rounded">
          <div className="text-gray-500">Shape</div>
          <div className="font-semibold">{rows} × {cols}</div>
        </div>
        <div className="bg-gray-50 p-2 rounded">
          <div className="text-gray-500">Max Weight</div>
          <div className="font-semibold">{maxVal.toFixed(4)}</div>
        </div>
        <div className="bg-gray-50 p-2 rounded">
          <div className="text-gray-500">Avg Weight</div>
          <div className="font-semibold">
            {(mat.flat().reduce((a, b) => a + b, 0) / (rows * cols)).toFixed(4)}
          </div>
        </div>
      </div>
    </div>
  );
}
