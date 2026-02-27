"use client";

import { Card, CardContent } from "@/components/ui/card";

interface OverlayImageProps {
  src: string;
}

export function OverlayImage({ src }: OverlayImageProps) {
  return (
    <Card>
      <CardContent className="p-3">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={src}
          alt="Cell overlay"
          className="w-full rounded-md"
        />
        <div className="flex items-center justify-center gap-4 mt-2 text-xs text-muted-foreground">
          <span className="flex items-center gap-1.5">
            <span className="h-2.5 w-2.5 rounded-full bg-red-500" />
            Positive cells
          </span>
          <span className="flex items-center gap-1.5">
            <span className="h-2.5 w-2.5 rounded-full bg-blue-500" />
            Negative cells
          </span>
        </div>
      </CardContent>
    </Card>
  );
}
