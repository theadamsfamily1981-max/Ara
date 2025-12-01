interface SparsityGaugeProps {
  value: number; // Fraction kept (0-1)
  keptIndices: number[];
  totalLength: number;
}

export default function SparsityGauge({
  value,
  keptIndices,
  totalLength,
}: SparsityGaugeProps) {
  const pct = Math.round(value * 100);
  const dropped = totalLength - keptIndices.length;
  const droppedPct = Math.round((dropped / totalLength) * 100);

  // Visualization: sequence with kept/dropped tokens
  const maxVizTokens = 256;
  const vizTokens = Math.min(totalLength, maxVizTokens);
  const keptSet = new Set(keptIndices);

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <h2 className="text-xl font-semibold mb-4">Sparsity Pattern</h2>

      {/* Gauge */}
      <div className="mb-6">
        <div className="flex justify-between mb-2">
          <span className="text-sm text-gray-600">Tokens Kept</span>
          <span className="text-2xl font-bold text-blue-600">{pct}%</span>
        </div>

        <div className="w-full h-8 bg-gray-200 rounded-full overflow-hidden">
          <div
            className="h-full bg-blue-600 transition-all duration-500"
            style={{ width: `${pct}%` }}
          />
        </div>

        <div className="flex justify-between mt-2 text-xs text-gray-500">
          <span>
            {keptIndices.length.toLocaleString()} kept
          </span>
          <span>
            {dropped.toLocaleString()} dropped ({droppedPct}%)
          </span>
        </div>
      </div>

      {/* Token visualization */}
      <div className="mb-4">
        <h3 className="text-sm font-semibold text-gray-700 mb-2">
          Token Selection Pattern
        </h3>
        <div className="bg-gray-50 p-4 rounded border border-gray-200">
          <div className="flex flex-wrap gap-0.5">
            {[...Array(vizTokens)].map((_, i) => {
              const actualIdx = Math.floor((i / vizTokens) * totalLength);
              const isKept = keptSet.has(actualIdx);

              return (
                <div
                  key={i}
                  className={`w-2 h-4 rounded-sm ${
                    isKept ? "bg-blue-500" : "bg-gray-300"
                  }`}
                  title={`Token ${actualIdx}: ${isKept ? "kept" : "dropped"}`}
                />
              );
            })}
          </div>

          <div className="flex items-center justify-between mt-3 text-xs text-gray-500">
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-1">
                <div className="w-3 h-3 bg-blue-500 rounded-sm" />
                <span>Kept</span>
              </div>
              <div className="flex items-center gap-1">
                <div className="w-3 h-3 bg-gray-300 rounded-sm" />
                <span>Dropped</span>
              </div>
            </div>
            {vizTokens < totalLength && (
              <span>Showing {vizTokens} of {totalLength.toLocaleString()} tokens</span>
            )}
          </div>
        </div>
      </div>

      {/* Distribution stats */}
      <div className="grid grid-cols-3 gap-3 text-xs">
        <div className="bg-blue-50 p-3 rounded border border-blue-200">
          <div className="text-blue-600 font-semibold mb-1">Sparsity</div>
          <div className="text-2xl font-bold text-blue-900">{droppedPct}%</div>
        </div>

        <div className="bg-green-50 p-3 rounded border border-green-200">
          <div className="text-green-600 font-semibold mb-1">Efficiency</div>
          <div className="text-2xl font-bold text-green-900">
            {(1 / (value || 0.01)).toFixed(1)}×
          </div>
          <div className="text-xs text-green-600">vs dense</div>
        </div>

        <div className="bg-purple-50 p-3 rounded border border-purple-200">
          <div className="text-purple-600 font-semibold mb-1">Total Tokens</div>
          <div className="text-2xl font-bold text-purple-900">
            {totalLength.toLocaleString()}
          </div>
        </div>
      </div>
    </div>
  );
}
