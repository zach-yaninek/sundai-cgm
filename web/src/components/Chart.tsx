/**
 * One small SVG line chart, used for the glucose curve and the learning curve.
 *
 * Hand-rolled rather than pulled from a library: both series are under 30 points,
 * a charting package is ~500 KB for that, and drawing it ourselves means the
 * confidence band can change how the chart *looks* — which a stock component
 * fights rather than helps.
 */
interface Point {
  x: number;
  y: number;
}

interface Props {
  points: Point[];
  height?: number;
  /** Lower opacity and a dashed line for estimates the model is unsure about. */
  uncertain?: boolean;
  xLabel?: string;
  yLabel?: string;
  /** Optional horizontal reference, e.g. the 140 mg/dL threshold. */
  threshold?: { y: number; label: string };
  /**
   * Anchor the y-axis at zero. Right for a glucose excursion, wrong for the
   * learning curve — forcing zero there squashes a 28.7 -> 23.4 fall into a flat
   * line and hides the entire effect the chart exists to show.
   */
  zeroBaseline?: boolean;
  ariaLabel: string;
}

const W = 560;
const PAD = { top: 26, right: 14, bottom: 26, left: 42 };

export default function Chart({
  points, height = 180, uncertain = false, xLabel, yLabel, threshold,
  zeroBaseline = true, ariaLabel,
}: Props) {
  if (points.length < 2) return null;

  const xs = points.map((p) => p.x);
  const ys = points.map((p) => p.y);
  if (threshold) ys.push(threshold.y);

  const xMin = Math.min(...xs);
  const xMax = Math.max(...xs);
  const lo = Math.min(...ys);
  const hi = Math.max(...ys) || 1;
  // A little headroom either side so the line never sits on the frame.
  const pad = (hi - lo) * 0.12 || 1;
  const yMin = zeroBaseline ? Math.min(0, lo) : lo - pad;
  const yMax = zeroBaseline ? hi : hi + pad;

  const px = (x: number) =>
    PAD.left + ((x - xMin) / (xMax - xMin || 1)) * (W - PAD.left - PAD.right);
  const py = (y: number) =>
    height - PAD.bottom - ((y - yMin) / (yMax - yMin || 1)) * (height - PAD.top - PAD.bottom);

  const line = points.map((p, i) => `${i ? "L" : "M"}${px(p.x)},${py(p.y)}`).join(" ");
  const area =
    `${line} L${px(xMax)},${py(yMin)} L${px(xMin)},${py(yMin)} Z`;

  const ticks = [yMin, (yMin + yMax) / 2, yMax];

  return (
    <svg
      className={`chart${uncertain ? " chart-uncertain" : ""}`}
      viewBox={`0 0 ${W} ${height}`}
      role="img"
      aria-label={ariaLabel}
      preserveAspectRatio="none"
    >
      {ticks.map((t) => (
        <g key={t}>
          <line className="chart-grid" x1={PAD.left} x2={W - PAD.right} y1={py(t)} y2={py(t)} />
          <text className="chart-tick" x={PAD.left - 6} y={py(t) + 3} textAnchor="end">
            {Math.round(t)}
          </text>
        </g>
      ))}

      {threshold && (
        <g>
          <line
            className="chart-threshold"
            x1={PAD.left}
            x2={W - PAD.right}
            y1={py(threshold.y)}
            y2={py(threshold.y)}
          />
          <text className="chart-threshold-label" x={W - PAD.right} y={py(threshold.y) - 5} textAnchor="end">
            {threshold.label}
          </text>
        </g>
      )}

      <path className="chart-area" d={area} />
      <path className="chart-line" d={line} />
      <circle className="chart-end" cx={px(points[points.length - 1]!.x)} cy={py(points[points.length - 1]!.y)} r={3.5} />

      {xLabel && (
        <text className="chart-axis" x={(W + PAD.left) / 2} y={height - 6} textAnchor="middle">
          {xLabel}
        </text>
      )}
      {yLabel && (
        <text className="chart-axis" x={2} y={11}>
          {yLabel}
        </text>
      )}
    </svg>
  );
}
