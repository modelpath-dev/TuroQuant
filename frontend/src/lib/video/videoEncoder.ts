/**
 * Encode ImageData frames into an MP4/WebM video using MediaRecorder.
 */
export async function framesToVideo(
  frames: ImageData[],
  fps: number,
): Promise<Blob> {
  if (frames.length === 0) return new Blob();

  const canvas = document.createElement("canvas");
  canvas.width = frames[0].width;
  canvas.height = frames[0].height;
  const ctx = canvas.getContext("2d")!;

  const stream = canvas.captureStream(0);
  const mediaRecorder = new MediaRecorder(stream, {
    mimeType: getSupportedMimeType(),
    videoBitsPerSecond: 2_500_000,
  });

  const chunks: Blob[] = [];
  mediaRecorder.ondataavailable = (e) => {
    if (e.data.size > 0) chunks.push(e.data);
  };

  const done = new Promise<Blob>((resolve) => {
    mediaRecorder.onstop = () => {
      resolve(new Blob(chunks, { type: mediaRecorder.mimeType }));
    };
  });

  mediaRecorder.start();
  const interval = 1000 / fps;

  for (const frame of frames) {
    ctx.putImageData(frame, 0, 0);
    // Request a frame from the capture stream
    const track = stream.getVideoTracks()[0];
    if ("requestFrame" in track) {
      (track as unknown as { requestFrame: () => void }).requestFrame();
    }
    await sleep(interval);
  }

  mediaRecorder.stop();
  return done;
}

function getSupportedMimeType(): string {
  const types = [
    "video/webm;codecs=vp9",
    "video/webm;codecs=vp8",
    "video/webm",
    "video/mp4",
  ];
  for (const t of types) {
    if (MediaRecorder.isTypeSupported(t)) return t;
  }
  return "video/webm";
}

function sleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}
