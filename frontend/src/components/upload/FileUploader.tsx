"use client";

import { useCallback, useState, useRef } from "react";
import { Upload, FileImage, FileVideo } from "lucide-react";
import { ACCEPT_STRING, VIDEO_FORMATS, TIFF_FORMATS } from "@/lib/constants";

interface FileUploaderProps {
  onFile: (file: File) => void;
  disabled?: boolean;
}

export function FileUploader({ onFile, disabled }: FileUploaderProps) {
  const [isDragOver, setIsDragOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback(
    (file: File) => {
      if (!disabled) onFile(file);
    },
    [onFile, disabled],
  );

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragOver(false);
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile],
  );

  const ext = (name: string) => name.split(".").pop()?.toLowerCase() || "";
  const isVideo = (name: string) =>
    VIDEO_FORMATS.includes(ext(name) as typeof VIDEO_FORMATS[number]);
  const isTiff = (name: string) =>
    TIFF_FORMATS.includes(ext(name) as typeof TIFF_FORMATS[number]);

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        setIsDragOver(true);
      }}
      onDragLeave={() => setIsDragOver(false)}
      onDrop={onDrop}
      onClick={() => inputRef.current?.click()}
      className={`
        relative cursor-pointer rounded-xl border-2 border-dashed
        transition-all duration-200 ease-in-out
        flex flex-col items-center justify-center gap-3 py-12 px-6
        ${isDragOver
          ? "border-primary bg-primary/5 scale-[1.01]"
          : "border-muted-foreground/20 hover:border-primary/50 hover:bg-muted/30"
        }
        ${disabled ? "opacity-50 pointer-events-none" : ""}
      `}
    >
      <div className="rounded-full bg-muted p-3">
        <Upload className="h-6 w-6 text-muted-foreground" />
      </div>
      <div className="text-center">
        <p className="text-sm font-medium">
          Drop your file here or{" "}
          <span className="text-primary">click to browse</span>
        </p>
        <p className="text-xs text-muted-foreground mt-1.5 flex items-center justify-center gap-3">
          <span className="flex items-center gap-1">
            <FileImage className="h-3 w-3" /> Images
          </span>
          <span className="flex items-center gap-1">
            <FileVideo className="h-3 w-3" /> Videos
          </span>
          <span>TIF/TIFF</span>
        </p>
        <p className="text-[10px] text-muted-foreground/60 mt-1">
          PNG, JPG, BMP, GIF, TIF, TIFF, SVS, NDPI, SCN, CZI, LIF, MP4, AVI,
          MOV, MKV, WebM
        </p>
      </div>

      <input
        ref={inputRef}
        type="file"
        accept={ACCEPT_STRING}
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) handleFile(file);
          e.target.value = "";
        }}
      />
    </div>
  );
}
