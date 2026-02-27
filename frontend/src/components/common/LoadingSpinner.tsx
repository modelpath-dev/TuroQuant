"use client";

import { Loader2 } from "lucide-react";
import { Progress } from "@/components/ui/progress";

interface LoadingSpinnerProps {
  progress?: number;
  text?: string;
}

export function LoadingSpinner({ progress, text }: LoadingSpinnerProps) {
  return (
    <div className="flex flex-col items-center gap-4 py-12">
      <Loader2 className="h-10 w-10 animate-spin text-primary" />
      {text && <p className="text-sm text-muted-foreground">{text}</p>}
      {progress !== undefined && (
        <div className="w-full max-w-xs">
          <Progress value={progress} className="h-2" />
          <p className="text-xs text-muted-foreground text-center mt-1">
            {Math.round(progress)}%
          </p>
        </div>
      )}
    </div>
  );
}
