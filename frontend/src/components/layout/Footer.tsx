"use client";

import { AlertTriangle } from "lucide-react";

export function Footer() {
  return (
    <footer className="border-t bg-muted/30 mt-auto">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 py-4">
        <div className="flex items-start gap-2.5 text-xs text-muted-foreground">
          <AlertTriangle className="h-4 w-4 shrink-0 mt-0.5 text-amber-500" />
          <p>
            <strong>Disclaimer:</strong> TuroQuant is a research tool for
            investigational use only. It is not intended for clinical diagnosis,
            treatment decisions, or any other clinical use. Results should be
            validated by qualified pathologists before any clinical
            interpretation. This tool should not be used as a diagnostic
            instrument in any way or form.{" "}
            <a
              href="https://pmc.ncbi.nlm.nih.gov/articles/PMC9494834/"
              target="_blank"
              rel="noopener noreferrer"
              className="underline hover:text-foreground transition-colors"
            >
              DeepLIIF
            </a>
          </p>
        </div>
      </div>
    </footer>
  );
}
