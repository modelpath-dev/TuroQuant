/**
 * Extract frames from a video file at specified intervals using HTMLVideoElement + canvas.
 * Browser-based equivalent of OpenCV's VideoCapture.
 */
export async function extractVideoFrames(
  file: File,
  everyNSec: number,
  onProgress?: (current: number, total: number) => void,
): Promise<ImageData[]> {
  const url = URL.createObjectURL(file);
  const video = document.createElement("video");
  video.src = url;
  video.muted = true;
  video.preload = "auto";

  await new Promise<void>((resolve, reject) => {
    video.onloadedmetadata = () => resolve();
    video.onerror = () => reject(new Error("Failed to load video"));
  });

  const duration = video.duration;
  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext("2d")!;

  const frames: ImageData[] = [];
  const timestamps: number[] = [];
  for (let t = 0; t < duration; t += everyNSec) {
    timestamps.push(t);
  }

  for (let i = 0; i < timestamps.length; i++) {
    const t = timestamps[i];
    await seekToTime(video, t);
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    frames.push(ctx.getImageData(0, 0, canvas.width, canvas.height));
    onProgress?.(i + 1, timestamps.length);
  }

  URL.revokeObjectURL(url);
  return frames;
}

function seekToTime(video: HTMLVideoElement, time: number): Promise<void> {
  return new Promise((resolve) => {
    const onSeeked = () => {
      video.removeEventListener("seeked", onSeeked);
      resolve();
    };
    video.addEventListener("seeked", onSeeked);
    video.currentTime = time;
  });
}

/**
 * Get video metadata without extracting frames.
 */
export async function getVideoInfo(file: File): Promise<{
  duration: number;
  width: number;
  height: number;
  estimatedFrames: number;
}> {
  const url = URL.createObjectURL(file);
  const video = document.createElement("video");
  video.src = url;
  video.preload = "auto";

  await new Promise<void>((resolve, reject) => {
    video.onloadedmetadata = () => resolve();
    video.onerror = () => reject(new Error("Failed to load video"));
  });

  const info = {
    duration: video.duration,
    width: video.videoWidth,
    height: video.videoHeight,
    estimatedFrames: Math.ceil(video.duration), // rough estimate at 1fps
  };

  URL.revokeObjectURL(url);
  return info;
}
