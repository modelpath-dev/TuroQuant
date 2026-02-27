"use client";

import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { CheckCircle2, RotateCcw } from "lucide-react";
import { ScoringMetrics } from "./ScoringMetrics";
import { ErPrMetrics } from "./ErPrMetrics";
import { OverlayImage } from "./OverlayImage";
import { ChannelImages } from "./ChannelImages";
import { CopyResultsButton } from "./CopyResultsButton";
import { DownloadSingleButton, DownloadFramesButton } from "./DownloadButton";
import type { FrameResult, VideoResult, StainType, ErPrScore, DedupScoring } from "@/types";
import { isErPrScore } from "@/types";

interface SingleResultProps {
  result: FrameResult;
  stain: StainType;
  onReset: () => void;
}

export function SingleResultView({ result, stain, onReset }: SingleResultProps) {
  const showErPr = (stain === "ER" || stain === "PR") && isErPrScore(result.scoring);

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <CheckCircle2 className="h-5 w-5 text-green-500" />
          <h3 className="font-semibold text-sm">Analysis Complete</h3>
        </div>
        <Button variant="ghost" size="sm" className="text-xs" onClick={onReset}>
          <RotateCcw className="h-3.5 w-3.5 mr-1.5" />
          New Sample
        </Button>
      </div>

      <ScoringMetrics scoring={result.scoring} />

      {showErPr && (
        <ErPrMetrics scoring={result.scoring as ErPrScore} />
      )}

      {result.overlay && <OverlayImage src={result.overlay} />}

      <div>
        <h4 className="text-xs font-medium text-muted-foreground mb-2">
          Output Channels
        </h4>
        <ChannelImages images={result.images} />
      </div>

      <Separator />

      <div className="flex gap-2 flex-wrap">
        <CopyResultsButton scoring={result.scoring} />
        <DownloadSingleButton
          images={result.images}
          scoring={result.scoring}
          overlay={result.overlay}
          stain={stain}
        />
      </div>

      {result.notes.length > 0 && (
        <div className="space-y-1">
          {result.notes.map((n, i) => (
            <p key={i} className="text-xs text-muted-foreground">
              {n}
            </p>
          ))}
        </div>
      )}
    </div>
  );
}

interface VideoResultProps {
  videoResult: VideoResult;
  stain: StainType;
  onReset: () => void;
}

export function VideoResultView({ videoResult, stain, onReset }: VideoResultProps) {
  const { frameResults, dedupScoring } = videoResult;
  const frameCount = Object.keys(frameResults).length;
  const showErPr = (stain === "ER" || stain === "PR") && dedupScoring?.hScore !== undefined;

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <CheckCircle2 className="h-5 w-5 text-green-500" />
          <h3 className="font-semibold text-sm">
            {frameCount} frame(s) processed
          </h3>
        </div>
        <Button variant="ghost" size="sm" className="text-xs" onClick={onReset}>
          <RotateCcw className="h-3.5 w-3.5 mr-1.5" />
          New Sample
        </Button>
      </div>

      {/* Deduplicated scoring */}
      {dedupScoring && dedupScoring.numTotal > 0 && (
        <>
          <h4 className="text-xs font-medium text-muted-foreground">
            Deduplicated Cell Counts (final index: frame {frameCount})
          </h4>
          <ScoringMetrics
            scoring={dedupScoring}
            isDeduplicated
            rawDetections={dedupScoring.rawDetections}
          />
          {showErPr && <ErPrMetrics scoring={dedupScoring as DedupScoring} />}
          {dedupScoring.note && (
            <p className="text-xs text-muted-foreground">{dedupScoring.note}</p>
          )}
        </>
      )}

      {/* Per-frame results (collapsible) */}
      <details className="group">
        <summary className="text-xs font-medium text-muted-foreground cursor-pointer hover:text-foreground">
          View individual frames ({frameCount})
        </summary>
        <div className="mt-3 space-y-4 pl-2 border-l-2 border-muted">
          {Object.entries(frameResults).map(([name, res]) => (
            <div key={name} className="space-y-2">
              <p className="text-xs font-medium">{name}</p>
              <ScoringMetrics scoring={res.scoring} />
              <ChannelImages images={res.images} />
            </div>
          ))}
        </div>
      </details>

      <Separator />

      <div className="flex gap-2 flex-wrap">
        {dedupScoring && <CopyResultsButton scoring={dedupScoring} />}
        <DownloadFramesButton
          frameResults={frameResults}
          dedupScoring={dedupScoring}
        />
      </div>
    </div>
  );
}
