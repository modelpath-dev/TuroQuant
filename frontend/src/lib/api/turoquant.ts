import type { Settings, TuroQuantResponse } from "@/types";
import { INFER_API_URL } from "../constants";

/**
 * Send an image to the TuroQuant API proxy and return the response.
 * Supports retry with exponential backoff.
 */
export async function inferImage(
  imageBlob: Blob,
  settings: Settings,
  retries = 2,
): Promise<TuroQuantResponse> {
  const formData = new FormData();
  formData.append("img", imageBlob, "image.png");
  formData.append("resolution", settings.resolution);
  formData.append("prob_thresh", String(settings.probThresh));
  if (settings.nopost) formData.append("nopost", "true");
  if (settings.slim) formData.append("slim", "true");
  if (settings.usePil) formData.append("pil", "true");

  let lastError: Error | null = null;

  for (let attempt = 0; attempt <= retries; attempt++) {
    if (attempt > 0) {
      await new Promise((r) => setTimeout(r, Math.pow(2, attempt) * 1000));
    }

    try {
      const res = await fetch(INFER_API_URL, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const data = await res.json().catch(() => ({ error: `HTTP ${res.status}` }));
        if (data.retryWithNopost) {
          // Retry with nopost
          formData.set("nopost", "true");
          continue;
        }
        throw new Error(data.error || `HTTP ${res.status}`);
      }

      const data = await res.json();

      // Normalize response structure
      const images = data.images || {};
      const scoring =
        data.scoring || data.scores || data.cell_scoring || data.score || data.results || {};
      const notes: string[] = [];

      return { images, scoring, notes };
    } catch (err) {
      lastError = err instanceof Error ? err : new Error(String(err));
    }
  }

  throw lastError || new Error("Failed to infer image");
}

/**
 * Convert an ImageData to a PNG Blob for API submission.
 */
export function imageDataToBlob(imageData: ImageData): Promise<Blob> {
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
