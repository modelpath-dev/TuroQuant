"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import type { Settings, FrameResult, DedupScoring, IntensityDistribution } from "@/types";
import { isErPrScore } from "@/types";
import { inferImage } from "@/lib/api/deepliif";
import { base64ToImageData, imageDataToDataUrl } from "@/lib/image/canvasUtils";
import { buildOverlay } from "@/lib/image/overlayBuilder";
import { scoreImage, computeGlobalErPrScore } from "@/lib/scoring";
import { gridDedup } from "@/lib/dedup/gridDedup";

export interface CameraBatchResult {
  dedupScoring: DedupScoring;
  overlay?: string;
}

export function useCamera() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [batchFrames, setBatchFrames] = useState<ImageData[]>([]);
  const [singleResult, setSingleResult] = useState<FrameResult | null>(null);
  const [batchResult, setBatchResult] = useState<CameraBatchResult | null>(null);
  const [processing, setProcessing] = useState(false);
  const [batchProgress, setBatchProgress] = useState({ current: 0, total: 0 });

  const startCamera = useCallback(async (videoElement: HTMLVideoElement) => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment", width: { ideal: 1280 }, height: { ideal: 720 } },
      });
      videoElement.srcObject = stream;
      await videoElement.play();
      videoRef.current = videoElement;
      streamRef.current = stream;
      setIsStreaming(true);
    } catch (err) {
      console.error("Camera access denied:", err);
      throw err;
    }
  }, []);

  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    setIsStreaming(false);
  }, []);

  const captureFrame = useCallback(
    (roi?: { x: number; y: number; w: number; h: number }): ImageData | null => {
      const video = videoRef.current;
      if (!video) return null;

      const canvas = document.createElement("canvas");
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext("2d")!;
      ctx.drawImage(video, 0, 0);

      if (roi) {
        return ctx.getImageData(roi.x, roi.y, roi.w, roi.h);
      }
      return ctx.getImageData(0, 0, canvas.width, canvas.height);
    },
    [],
  );

  const addToBatch = useCallback((frame: ImageData) => {
    setBatchFrames((prev) => [...prev, frame]);
  }, []);

  const clearBatch = useCallback(() => {
    setBatchFrames([]);
    setBatchResult(null);
  }, []);

  const analyzeFrame = useCallback(
    async (frame: ImageData, settings: Settings) => {
      setProcessing(true);
      try {
        const blob = await imageDataToPng(frame);
        const response = await inferImage(blob, settings);

        const segKey = Object.keys(response.images).find(
          (k) => k.toLowerCase() === "seg",
        );
        const markerKey = Object.keys(response.images).find(
          (k) => k.toLowerCase() === "marker",
        );

        if (!segKey) throw new Error("No Seg image returned");

        const segImageData = await base64ToImageData(response.images[segKey]);
        const markerImageData = markerKey
          ? await base64ToImageData(response.images[markerKey])
          : null;

        const scoring = scoreImage(segImageData, markerImageData, settings.stain);
        const overlayData = buildOverlay(frame, segImageData);

        const result: FrameResult = {
          images: response.images,
          scoring,
          overlay: imageDataToDataUrl(overlayData),
          cells: scoring.cells,
          notes: response.notes || [],
        };

        setSingleResult(result);
      } finally {
        setProcessing(false);
      }
    },
    [],
  );

  const processBatch = useCallback(
    async (settings: Settings) => {
      if (batchFrames.length === 0) return;
      setProcessing(true);
      setBatchProgress({ current: 0, total: batchFrames.length });

      try {
        const frameResults: FrameResult[] = [];

        for (let i = 0; i < batchFrames.length; i++) {
          const blob = await imageDataToPng(batchFrames[i]);
          const response = await inferImage(blob, settings);

          const segKey = Object.keys(response.images).find(
            (k) => k.toLowerCase() === "seg",
          );
          const markerKey = Object.keys(response.images).find(
            (k) => k.toLowerCase() === "marker",
          );

          if (segKey) {
            const segImageData = await base64ToImageData(response.images[segKey]);
            const markerImageData = markerKey
              ? await base64ToImageData(response.images[markerKey])
              : null;

            const scoring = scoreImage(segImageData, markerImageData, settings.stain);
            const overlayData = buildOverlay(batchFrames[i], segImageData);

            frameResults.push({
              images: response.images,
              scoring,
              overlay: imageDataToDataUrl(overlayData),
              cells: scoring.cells,
              notes: response.notes || [],
            });
          }

          setBatchProgress({ current: i + 1, total: batchFrames.length });
        }

        if (frameResults.length > 0) {
          // Deduplicate
          const allCells = frameResults.map((r) => r.cells);
          const deduped = gridDedup(allCells, settings.dedupRadius);

          // For ER/PR, compute global scores
          if (settings.stain !== "KI67") {
            const distributions: IntensityDistribution[] = [];
            for (const r of frameResults) {
              if (isErPrScore(r.scoring)) {
                distributions.push(r.scoring.intensityDistribution);
              }
            }
            if (distributions.length > 0) {
              const totalDist: IntensityDistribution = { 0: 0, 1: 0, 2: 0, 3: 0 };
              for (const dist of distributions) {
                totalDist[0] += dist[0];
                totalDist[1] += dist[1];
                totalDist[2] += dist[2];
                totalDist[3] += dist[3];
              }
              const globalScores = computeGlobalErPrScore(
                totalDist,
                deduped.numPos,
                deduped.numNeg,
              );
              Object.assign(deduped, globalScores);
            }
          }

          setBatchResult({
            dedupScoring: deduped,
            overlay: frameResults[frameResults.length - 1].overlay,
          });
          setBatchFrames([]);
        }
      } finally {
        setProcessing(false);
      }
    },
    [batchFrames],
  );

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
      }
    };
  }, []);

  return {
    isStreaming,
    batchFrames,
    singleResult,
    batchResult,
    processing,
    batchProgress,
    startCamera,
    stopCamera,
    captureFrame,
    addToBatch,
    clearBatch,
    analyzeFrame,
    processBatch,
    clearSingleResult: () => setSingleResult(null),
    clearBatchResult: () => setBatchResult(null),
  };
}

function imageDataToPng(imageData: ImageData): Promise<Blob> {
  const canvas = document.createElement("canvas");
  canvas.width = imageData.width;
  canvas.height = imageData.height;
  const ctx = canvas.getContext("2d")!;
  ctx.putImageData(imageData, 0, 0);
  return new Promise((resolve, reject) => {
    canvas.toBlob(
      (blob) => (blob ? resolve(blob) : reject(new Error("toBlob failed"))),
      "image/png",
    );
  });
}
