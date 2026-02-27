"use client";

import { MetricCard } from "@/components/common/MetricCard";
import type { ErPrScore, DedupScoring } from "@/types";

interface ErPrMetricsProps {
  scoring: ErPrScore | DedupScoring;
}

export function ErPrMetrics({ scoring }: ErPrMetricsProps) {
  const h = scoring.hScore ?? 0;
  const hl = scoring.hScoreLabel ?? "";
  const al = scoring.allredScore ?? 0;
  const alLabel = scoring.allredLabel ?? "";
  const ps = scoring.allredProportion ?? 0;
  const is_ = scoring.allredIntensity ?? 0;
  const dist = scoring.intensityDistribution;

  return (
    <div className="space-y-2">
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
        <MetricCard label={`H-Score (${hl})`} value={h} />
        <MetricCard label={`Allred (${alLabel})`} value={`${al}/8`} />
        <MetricCard label="Proportion" value={`${ps}/5`} />
        <MetricCard label="Intensity" value={`${is_}/3`} />
      </div>
      {dist && (
        <p className="text-xs text-muted-foreground text-center">
          Intensity distribution — Neg: {dist[0]} | 1+: {dist[1]} | 2+:{" "}
          {dist[2]} | 3+: {dist[3]}
        </p>
      )}
    </div>
  );
}
