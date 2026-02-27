"use client";

import { FileImage, FileVideo, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { VIDEO_FORMATS, TIFF_FORMATS } from "@/lib/constants";

interface ImagePreviewProps {
  file: File;
  onRemove: () => void;
}

export function ImagePreview({ file, onRemove }: ImagePreviewProps) {
  const ext = file.name.split(".").pop()?.toLowerCase() || "";
  const isVideo = VIDEO_FORMATS.includes(ext as typeof VIDEO_FORMATS[number]);
  const isTiff = TIFF_FORMATS.includes(ext as typeof TIFF_FORMATS[number]);
  const isImage = !isVideo && !isTiff;

  const sizeMB = (file.size / 1024 / 1024).toFixed(1);

  return (
    <div className="flex items-center gap-3 rounded-lg border bg-muted/30 p-3">
      <div className="rounded-md bg-muted p-2">
        {isVideo ? (
          <FileVideo className="h-5 w-5 text-blue-500" />
        ) : (
          <FileImage className="h-5 w-5 text-green-500" />
        )}
      </div>
      <div className="flex-1 min-w-0">
        <p className="text-sm font-medium truncate">{file.name}</p>
        <p className="text-xs text-muted-foreground">
          {sizeMB} MB
          {isVideo && " · Video"}
          {isTiff && " · Multi-page TIFF"}
          {isImage && " · Image"}
        </p>
      </div>
      <Button variant="ghost" size="icon" className="h-7 w-7" onClick={onRemove}>
        <X className="h-4 w-4" />
      </Button>
    </div>
  );
}
