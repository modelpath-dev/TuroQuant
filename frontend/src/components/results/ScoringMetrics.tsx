"use client";

import { MetricCard } from "@/components/common/MetricCard";
import type { ScoringResult, DedupScoring } from "@/types";

interface ScoringMetricsProps {
  scoring: ScoringResult | DedupScoring;
  isDeduplicated?: boolean;
  rawDetections?: number;
}

export function ScoringMetrics({ scoring, isDeduplicated, rawDetections }: ScoringMetricsProps) {
  const pct =
    typeof scoring.percentPos === "number"
      ? `${scoring.percentPos.toFixed(1)}%`
      : "0%";

  return (
    <div className="space-y-2">
      {isDeduplicated && rawDetections !== undefined && (
        <p className="text-xs text-muted-foreground">
          Deduplicated from {rawDetections} raw detections
        </p>
      )}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
        <MetricCard label="Total Cells" value={scoring.numTotal} />
        <MetricCard
          label="Positive"
          value={scoring.numPos}
          className="border-red-200/50"
        />
        <MetricCard
          label="Negative"
          value={scoring.numNeg}
          className="border-blue-200/50"
        />
        <MetricCard label="% Positive" value={pct} />
      </div>
    </div>
  );
}
