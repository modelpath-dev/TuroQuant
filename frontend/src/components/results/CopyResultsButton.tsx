"use client";

import { Button } from "@/components/ui/button";
import { Check, Copy } from "lucide-react";
import { useClipboard } from "@/hooks/useClipboard";
import type { ScoringResult, DedupScoring } from "@/types";

interface CopyResultsButtonProps {
  scoring: ScoringResult | DedupScoring;
}

export function CopyResultsButton({ scoring }: CopyResultsButtonProps) {
  const { copied, copy } = useClipboard();

  const handleCopy = () => {
    // Exclude the cells array for a cleaner clipboard output
    const clean = Object.fromEntries(
      Object.entries(scoring).filter(([k]) => k !== "cells"),
    );
    copy(JSON.stringify(clean, null, 2));
  };

  return (
    <Button
      variant="outline"
      size="sm"
      className="text-xs"
      onClick={handleCopy}
    >
      {copied ? (
        <>
          <Check className="h-3.5 w-3.5 mr-1.5 text-green-500" />
          Copied!
        </>
      ) : (
        <>
          <Copy className="h-3.5 w-3.5 mr-1.5" />
          Copy Results
        </>
      )}
    </Button>
  );
}
