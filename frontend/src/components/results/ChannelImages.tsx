"use client";

import { useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import {
  Dialog,
  DialogContent,
  DialogTitle,
} from "@/components/ui/dialog";

interface ChannelImagesProps {
  images: Record<string, string>;
}

export function ChannelImages({ images }: ChannelImagesProps) {
  const [selected, setSelected] = useState<{ name: string; src: string } | null>(null);
  const entries = Object.entries(images);

  if (entries.length === 0) return null;

  return (
    <>
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-2">
        {entries.map(([name, base64]) => {
          const src = base64.startsWith("data:")
            ? base64
            : `data:image/png;base64,${base64}`;
          return (
            <Card
              key={name}
              className="cursor-pointer hover:ring-2 hover:ring-primary/50 transition-all"
              onClick={() => setSelected({ name, src })}
            >
              <CardContent className="p-2">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={src}
                  alt={name}
                  className="w-full rounded-sm"
                />
                <p className="text-xs text-center text-muted-foreground mt-1.5 font-medium">
                  {name}
                </p>
              </CardContent>
            </Card>
          );
        })}
      </div>

      <Dialog open={!!selected} onOpenChange={() => setSelected(null)}>
        <DialogContent className="max-w-3xl">
          <DialogTitle className="text-sm font-medium">
            {selected?.name}
          </DialogTitle>
          {selected && (
            /* eslint-disable-next-line @next/next/no-img-element */
            <img
              src={selected.src}
              alt={selected.name}
              className="w-full rounded-md"
            />
          )}
        </DialogContent>
      </Dialog>
    </>
  );
}
