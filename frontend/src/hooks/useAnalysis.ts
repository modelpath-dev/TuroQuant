"use client";

import { useState, useCallback } from "react";
import type { AnalysisState, Settings, FrameResult, FileType, IntensityDistribution } from "@/types";
import { isErPrScore } from "@/types";
import { inferImage, imageDataToBlob } from "@/lib/api/deepliif";
import { base64ToImageData, fileToImageData, imageDataToDataUrl } from "@/lib/image/canvasUtils";
import { buildOverlay } from "@/lib/image/overlayBuilder";
import { scoreImage, computeGlobalErPrScore } from "@/lib/scoring";
import { extractVideoFrames } from "@/lib/video/frameExtractor";
import { parseTiffPages } from "@/lib/tiff/tiffParser";
import { deduplicateVideoCells, aggregateIntensityDistributions } from "@/lib/dedup/videoCellDedup";
import { VIDEO_FORMATS, TIFF_FORMATS } from "@/lib/constants";

function getFileType(filename: string): FileType {
  const ext = filename.split(".").pop()?.toLowerCase() || "";
  if (VIDEO_FORMATS.includes(ext as typeof VIDEO_FORMATS[number])) return "video";
  if (TIFF_FORMATS.includes(ext as typeof TIFF_FORMATS[number])) return "tiff";
  return "image";
}

export function useAnalysis() {
  const [state, setState] = useState<AnalysisState>({
    status: "idle",
    progress: 0,
    progressText: "",
  });

  const reset = useCallback(() => {
    setState({ status: "idle", progress: 0, progressText: "" });
  }, []);

  const analyzeImage = useCallback(
    async (file: File, settings: Settings) => {
      const fileType = getFileType(file.name);
      setState({ status: "processing", progress: 0, progressText: "Sending to DeepLIIF...", fileType });

      try {
        if (fileType === "image") {
          await processSingleImage(file, settings, setState);
        } else if (fileType === "video") {
          await processVideo(file, settings, setState);
        } else if (fileType === "tiff") {
          await processTiff(file, settings, setState);
        }
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        setState((prev) => ({ ...prev, status: "error", error: message }));
      }
    },
    [],
  );

  return { state, analyzeImage, reset };
}

async function processSingleImage(
  file: File,
  settings: Settings,
  setState: React.Dispatch<React.SetStateAction<AnalysisState>>,
) {
  const response = await inferImage(file, settings);
  setState((prev) => ({ ...prev, progress: 50, progressText: "Analyzing cells..." }));

  const segKey = Object.keys(response.images).find(
    (k) => k.toLowerCase() === "seg",
  );
  const markerKey = Object.keys(response.images).find(
    (k) => k.toLowerCase() === "marker",
  );

  if (!segKey) throw new Error("DeepLIIF did not return a Seg image.");

  const segImageData = await base64ToImageData(response.images[segKey]);
  const markerImageData = markerKey
    ? await base64ToImageData(response.images[markerKey])
    : null;

  const scoring = scoreImage(segImageData, markerImageData, settings.stain);

  // Build overlay
  const origImageData = await fileToImageData(file);
  const overlayData = buildOverlay(origImageData, segImageData);
  const overlayUrl = imageDataToDataUrl(overlayData);

  const result: FrameResult = {
    images: response.images,
    scoring,
    overlay: overlayUrl,
    cells: scoring.cells,
    notes: response.notes || [],
  };

  setState({
    status: "done",
    progress: 100,
    progressText: "Complete",
    fileType: "image",
    result,
  });
}

async function processVideo(
  file: File,
  settings: Settings,
  setState: React.Dispatch<React.SetStateAction<AnalysisState>>,
) {
  // Extract frames
  setState((prev) => ({
    ...prev,
    status: "extracting",
    progressText: "Extracting video frames...",
  }));

  const frames = await extractVideoFrames(file, settings.everyNSec, (current, total) => {
    setState((prev) => ({
      ...prev,
      progress: Math.round((current / total) * 20),
      progressText: `Extracting frame ${current} of ${total}...`,
    }));
  });

  await processFrames(frames, settings, setState, "video");
}

async function processTiff(
  file: File,
  settings: Settings,
  setState: React.Dispatch<React.SetStateAction<AnalysisState>>,
) {
  setState((prev) => ({
    ...prev,
    status: "extracting",
    progressText: "Reading TIFF pages...",
  }));

  const buffer = await file.arrayBuffer();
  const pages = await parseTiffPages(buffer);

  // Convert ImageData to frames array
  await processFrames(pages, settings, setState, "tiff");
}

async function processFrames(
  frames: ImageData[],
  settings: Settings,
  setState: React.Dispatch<React.SetStateAction<AnalysisState>>,
  fileType: FileType,
) {
  const totalFrames = frames.length;
  const frameResults: Record<string, FrameResult> = {};

  setState((prev) => ({
    ...prev,
    status: "processing",
    progressText: `Processing ${totalFrames} frame(s)...`,
    progress: 20,
  }));

  // Process frames sequentially (API rate limiting)
  for (let i = 0; i < totalFrames; i++) {
    const frameName = fileType === "tiff" ? `page_${String(i).padStart(4, "0")}` : `frame_${String(i).padStart(4, "0")}`;

    try {
      const blob = await imageDataToPngBlob(frames[i]);
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

        frameResults[frameName] = {
          images: response.images,
          scoring,
          cells: scoring.cells,
          notes: response.notes || [],
        };
      }
    } catch (err) {
      console.warn(`${frameName}: ${err}`);
    }

    const progress = 20 + Math.round(((i + 1) / totalFrames) * 60);
    setState((prev) => ({
      ...prev,
      progress,
      progressText: `Frame ${i + 1} of ${totalFrames}`,
    }));
  }

  // Deduplication for video
  if (fileType === "video" && Object.keys(frameResults).length > 0) {
    setState((prev) => ({
      ...prev,
      status: "deduplicating",
      progress: 85,
      progressText: "Deduplicating cells...",
    }));

    const frameData = Object.values(frameResults).map((r) => ({
      cells: r.cells,
      segBase64: r.images.Seg || r.images.seg || "",
    }));

    const dedupScoring = await deduplicateVideoCells(frameData, settings.dedupRadius);

    // For ER/PR, compute global scores
    if (settings.stain !== "KI67") {
      const distributions: IntensityDistribution[] = [];
      for (const r of Object.values(frameResults)) {
        if (isErPrScore(r.scoring)) {
          distributions.push(r.scoring.intensityDistribution);
        }
      }
      if (distributions.length > 0) {
        const totalDist = aggregateIntensityDistributions(distributions);
        const globalScores = computeGlobalErPrScore(
          totalDist,
          dedupScoring.numPos,
          dedupScoring.numNeg,
        );
        Object.assign(dedupScoring, globalScores);
      }
    }

    setState({
      status: "done",
      progress: 100,
      progressText: "Complete",
      fileType,
      videoResult: { frameResults, dedupScoring },
    });
  } else {
    setState({
      status: "done",
      progress: 100,
      progressText: "Complete",
      fileType,
      videoResult: { frameResults },
    });
  }
}

function imageDataToPngBlob(imageData: ImageData): Promise<Blob> {
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
