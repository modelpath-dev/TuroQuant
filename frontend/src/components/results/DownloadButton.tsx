"use client";

import { Button } from "@/components/ui/button";
import { Download } from "lucide-react";
import { saveAs } from "file-saver";
import { buildResultsZip, buildFramesZip } from "@/lib/image/zipBuilder";
import type { ScoringResult, FrameResult, DedupScoring } from "@/types";

interface DownloadSingleProps {
  images: Record<string, string>;
  scoring: ScoringResult;
  overlay?: string;
  stain: string;
}

export function DownloadSingleButton({ images, scoring, overlay, stain }: DownloadSingleProps) {
  const handleDownload = async () => {
    const blob = await buildResultsZip(images, scoring as unknown as Record<string, unknown>, overlay);
    saveAs(blob, `turoquant_${stain.toLowerCase()}_results.zip`);
  };

  return (
    <Button variant="outline" size="sm" className="text-xs" onClick={handleDownload}>
      <Download className="h-3.5 w-3.5 mr-1.5" />
      Download ZIP
    </Button>
  );
}

interface DownloadFramesProps {
  frameResults: Record<string, FrameResult>;
  dedupScoring?: DedupScoring;
}

export function DownloadFramesButton({ frameResults, dedupScoring }: DownloadFramesProps) {
  const handleDownload = async () => {
    const formatted = Object.fromEntries(
      Object.entries(frameResults).map(([k, v]) => [
        k,
        { images: v.images, scoring: v.scoring as unknown as Record<string, unknown> },
      ]),
    );
    const blob = await buildFramesZip(
      formatted,
      dedupScoring as unknown as Record<string, unknown> | undefined,
    );
    saveAs(blob, "turoquant_frames.zip");
  };

  return (
    <Button variant="outline" size="sm" className="text-xs" onClick={handleDownload}>
      <Download className="h-3.5 w-3.5 mr-1.5" />
      Download All Frames
    </Button>
  );
}
