import type { Settings, TuroQuantResponse } from "@/types";
import { INFER_API_URL } from "../constants";

/**
 * Send an image to the TuroQuant API proxy and return the response.
 * Server-side handles retries; client just sends once.
 */
export async function inferImage(
  imageBlob: Blob,
  settings: Settings,
): Promise<TuroQuantResponse> {
  const formData = new FormData();
  formData.append("img", imageBlob, "image.png");
  formData.append("resolution", settings.resolution);
  formData.append("prob_thresh", String(settings.probThresh));
  if (settings.nopost) formData.append("nopost", "true");
  if (settings.slim) formData.append("slim", "true");
  if (settings.usePil) formData.append("pil", "true");

  const res = await fetch(INFER_API_URL, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const data = await res.json().catch(() => ({ error: `HTTP ${res.status}` }));
    throw new Error(data.error || `HTTP ${res.status}`);
  }

  const data = await res.json();

  // Normalize response structure
  const images = data.images || {};
  const scoring =
    data.scoring || data.scores || data.cell_scoring || data.score || data.results || {};
  const notes: string[] = [];

  return { images, scoring, notes };
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
