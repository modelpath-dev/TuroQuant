"use client";

import { useRef, useState, useCallback, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { Checkbox } from "@/components/ui/checkbox";
import { Separator } from "@/components/ui/separator";
import { Camera, Plus, Play, Trash2, Video, VideoOff } from "lucide-react";
import { LoadingSpinner } from "@/components/common/LoadingSpinner";
import { ScoringMetrics } from "@/components/results/ScoringMetrics";
import { ErPrMetrics } from "@/components/results/ErPrMetrics";
import { OverlayImage } from "@/components/results/OverlayImage";
import { CopyResultsButton } from "@/components/results/CopyResultsButton";
import type { Settings, StainType, ErPrScore, DedupScoring } from "@/types";
import { isErPrScore } from "@/types";
import { useCamera } from "@/hooks/useCamera";

interface CameraCaptureProps {
  settings: Settings;
  stain: StainType;
}

export function CameraCapture({ settings, stain }: CameraCaptureProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const camera = useCamera();
  const [useRoi, setUseRoi] = useState(false);
  const [roi, setRoi] = useState({ x: 25, y: 25, w: 50, h: 50 }); // percentages

  const handleStart = async () => {
    if (videoRef.current) {
      await camera.startCamera(videoRef.current);
    }
  };

  const captureWithRoi = useCallback(() => {
    if (!videoRef.current) return null;
    const vw = videoRef.current.videoWidth;
    const vh = videoRef.current.videoHeight;

    if (useRoi) {
      return camera.captureFrame({
        x: Math.round((roi.x / 100) * vw),
        y: Math.round((roi.y / 100) * vh),
        w: Math.round((roi.w / 100) * vw),
        h: Math.round((roi.h / 100) * vh),
      });
    }
    return camera.captureFrame();
  }, [camera, useRoi, roi]);

  const handleAnalyze = async () => {
    const frame = captureWithRoi();
    if (frame) await camera.analyzeFrame(frame, settings);
  };

  const handleAddBatch = () => {
    const frame = captureWithRoi();
    if (frame) camera.addToBatch(frame);
  };

  const showErPr =
    stain !== "KI67" &&
    camera.singleResult &&
    isErPrScore(camera.singleResult.scoring);

  const showBatchErPr =
    stain !== "KI67" &&
    camera.batchResult?.dedupScoring?.hScore !== undefined;

  return (
    <div className="space-y-4">
      {/* Camera preview */}
      <div className="relative rounded-lg overflow-hidden bg-black aspect-video">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className={`w-full h-full object-cover ${camera.isStreaming ? "" : "hidden"}`}
        />
        {!camera.isStreaming && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-3">
            <Video className="h-10 w-10 text-muted-foreground" />
            <Button onClick={handleStart} size="sm">
              <Camera className="h-4 w-4 mr-2" />
              Start Camera
            </Button>
          </div>
        )}

        {/* ROI overlay */}
        {camera.isStreaming && useRoi && (
          <div
            className="absolute border-2 border-primary/80 bg-primary/10 pointer-events-none"
            style={{
              left: `${roi.x}%`,
              top: `${roi.y}%`,
              width: `${roi.w}%`,
              height: `${roi.h}%`,
            }}
          />
        )}
      </div>

      {camera.isStreaming && (
        <>
          {/* ROI controls */}
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <Checkbox
                id="useRoi"
                checked={useRoi}
                onCheckedChange={(v) => setUseRoi(!!v)}
              />
              <label htmlFor="useRoi" className="text-xs">
                Select ROI (region of interest)
              </label>
            </div>

            {useRoi && (
              <div className="grid grid-cols-2 gap-3">
                <div className="space-y-1">
                  <label className="text-xs text-muted-foreground">Left: {roi.x}%</label>
                  <Slider
                    value={[roi.x]}
                    min={0}
                    max={100 - roi.w}
                    step={1}
                    onValueChange={([v]) => setRoi((r) => ({ ...r, x: v }))}
                  />
                </div>
                <div className="space-y-1">
                  <label className="text-xs text-muted-foreground">Top: {roi.y}%</label>
                  <Slider
                    value={[roi.y]}
                    min={0}
                    max={100 - roi.h}
                    step={1}
                    onValueChange={([v]) => setRoi((r) => ({ ...r, y: v }))}
                  />
                </div>
                <div className="space-y-1">
                  <label className="text-xs text-muted-foreground">Width: {roi.w}%</label>
                  <Slider
                    value={[roi.w]}
                    min={10}
                    max={100 - roi.x}
                    step={1}
                    onValueChange={([v]) => setRoi((r) => ({ ...r, w: v }))}
                  />
                </div>
                <div className="space-y-1">
                  <label className="text-xs text-muted-foreground">Height: {roi.h}%</label>
                  <Slider
                    value={[roi.h]}
                    min={10}
                    max={100 - roi.y}
                    step={1}
                    onValueChange={([v]) => setRoi((r) => ({ ...r, h: v }))}
                  />
                </div>
              </div>
            )}
          </div>

          <Separator />

          {/* Action buttons */}
          <div className="grid grid-cols-3 gap-2">
            <Button
              onClick={handleAnalyze}
              disabled={camera.processing}
              size="sm"
              className="text-xs"
            >
              <Play className="h-3.5 w-3.5 mr-1" />
              Analyse
            </Button>
            <Button
              variant="outline"
              onClick={handleAddBatch}
              disabled={camera.processing}
              size="sm"
              className="text-xs"
            >
              <Plus className="h-3.5 w-3.5 mr-1" />
              Add to Batch
            </Button>
            <Button
              variant="outline"
              onClick={() => camera.processBatch(settings)}
              disabled={camera.processing || camera.batchFrames.length === 0}
              size="sm"
              className="text-xs"
            >
              Batch ({camera.batchFrames.length})
            </Button>
          </div>

          {camera.batchFrames.length > 0 && (
            <Button
              variant="ghost"
              size="sm"
              className="text-xs w-full"
              onClick={camera.clearBatch}
            >
              <Trash2 className="h-3.5 w-3.5 mr-1" />
              Clear batch ({camera.batchFrames.length} frames)
            </Button>
          )}

          {/* Stop camera */}
          <Button
            variant="ghost"
            size="sm"
            className="text-xs w-full text-destructive"
            onClick={camera.stopCamera}
          >
            <VideoOff className="h-3.5 w-3.5 mr-1" />
            Stop Camera
          </Button>
        </>
      )}

      {/* Processing state */}
      {camera.processing && (
        <LoadingSpinner
          text={
            camera.batchProgress.total > 0
              ? `Processing frame ${camera.batchProgress.current} of ${camera.batchProgress.total}...`
              : "Sending to DeepLIIF..."
          }
          progress={
            camera.batchProgress.total > 0
              ? (camera.batchProgress.current / camera.batchProgress.total) * 100
              : undefined
          }
        />
      )}

      {/* Single frame result */}
      {camera.singleResult && (
        <div className="space-y-3">
          <Separator />
          <h4 className="text-sm font-medium">Analysis Result</h4>
          <ScoringMetrics scoring={camera.singleResult.scoring} />
          {showErPr && (
            <ErPrMetrics scoring={camera.singleResult.scoring as ErPrScore} />
          )}
          {camera.singleResult.overlay && (
            <OverlayImage src={camera.singleResult.overlay} />
          )}
          <div className="flex gap-2">
            <CopyResultsButton scoring={camera.singleResult.scoring} />
            <Button
              variant="ghost"
              size="sm"
              className="text-xs"
              onClick={camera.clearSingleResult}
            >
              Clear
            </Button>
          </div>
        </div>
      )}

      {/* Batch result */}
      {camera.batchResult && (
        <div className="space-y-3">
          <Separator />
          <h4 className="text-sm font-medium">Batch Result</h4>
          <ScoringMetrics
            scoring={camera.batchResult.dedupScoring}
            isDeduplicated
            rawDetections={camera.batchResult.dedupScoring.rawDetections}
          />
          {showBatchErPr && (
            <ErPrMetrics scoring={camera.batchResult.dedupScoring as DedupScoring} />
          )}
          {camera.batchResult.overlay && (
            <OverlayImage src={camera.batchResult.overlay} />
          )}
          <div className="flex gap-2">
            <CopyResultsButton scoring={camera.batchResult.dedupScoring} />
            <Button
              variant="ghost"
              size="sm"
              className="text-xs"
              onClick={camera.clearBatchResult}
            >
              Clear
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}
