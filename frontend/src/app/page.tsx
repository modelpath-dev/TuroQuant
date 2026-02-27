"use client";

import { useState, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Header } from "@/components/layout/Header";
import { Footer } from "@/components/layout/Footer";
import { SettingsPanel } from "@/components/settings/SettingsPanel";
import { FileUploader } from "@/components/upload/FileUploader";
import { ImagePreview } from "@/components/upload/ImagePreview";
import { LoadingSpinner } from "@/components/common/LoadingSpinner";
import {
  SingleResultView,
  VideoResultView,
} from "@/components/results/ResultsContainer";
import { CameraCapture } from "@/components/camera/CameraCapture";
import { useAnalysis } from "@/hooks/useAnalysis";
import { useServerHealth } from "@/hooks/useServerHealth";
import type { Settings, SourceMode, AnalysisState } from "@/types";
import { DEFAULT_SETTINGS } from "@/types";
import { VIDEO_FORMATS, TIFF_FORMATS } from "@/lib/constants";
import { Play, Menu } from "lucide-react";

export default function HomePage() {
  const [settings, setSettings] = useState<Settings>(DEFAULT_SETTINGS);
  const [sourceMode, setSourceMode] = useState<SourceMode>("file");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const { isHealthy, checking, check } = useServerHealth();
  const { state, analyzeImage, reset } = useAnalysis();

  const isVideoOrTiff = selectedFile
    ? (() => {
        const ext = selectedFile.name.split(".").pop()?.toLowerCase() || "";
        return (
          VIDEO_FORMATS.includes(ext as (typeof VIDEO_FORMATS)[number]) ||
          TIFF_FORMATS.includes(ext as (typeof TIFF_FORMATS)[number])
        );
      })()
    : false;

  const handleRun = useCallback(async () => {
    if (!selectedFile) return;
    const ok = await check();
    if (!ok) return;
    await analyzeImage(selectedFile, settings);
  }, [selectedFile, settings, check, analyzeImage]);

  const handleReset = useCallback(() => {
    reset();
    setSelectedFile(null);
  }, [reset]);

  const isProcessing =
    state.status !== "idle" &&
    state.status !== "done" &&
    state.status !== "error";

  return (
    <div className="min-h-screen flex flex-col bg-background">
      <Header isHealthy={isHealthy} checking={checking} />

      <div className="flex-1 flex flex-col lg:flex-row max-w-7xl mx-auto w-full">
        {/* Mobile sidebar toggle */}
        <div className="lg:hidden border-b px-4 py-2">
          <Button
            variant="ghost"
            size="sm"
            className="text-xs"
            onClick={() => setSidebarOpen(!sidebarOpen)}
          >
            <Menu className="h-4 w-4 mr-1.5" />
            {sidebarOpen ? "Hide Settings" : "Show Settings"}
          </Button>
        </div>

        {/* Sidebar */}
        <aside
          className={`w-full lg:w-72 xl:w-80 border-r bg-muted/10 p-4 sm:p-5 shrink-0 ${
            sidebarOpen ? "block" : "hidden lg:block"
          }`}
        >
          <SettingsPanel
            settings={settings}
            onChange={setSettings}
            sourceMode={sourceMode}
            onSourceModeChange={setSourceMode}
            onCheckServer={check}
            checkingServer={checking}
            showVideoSettings={isVideoOrTiff}
          />
        </aside>

        {/* Main content */}
        <main className="flex-1 p-4 sm:p-6 overflow-y-auto">
          {sourceMode === "file" ? (
            <FileUploadMode
              state={state}
              settings={settings}
              selectedFile={selectedFile}
              isHealthy={isHealthy}
              isProcessing={isProcessing}
              onSelectFile={setSelectedFile}
              onRun={handleRun}
              onReset={handleReset}
            />
          ) : (
            <div>
              <h2 className="text-xl font-semibold tracking-tight mb-1">
                TuroQuant — Camera ({settings.stain})
              </h2>
              <p className="text-xs text-muted-foreground mb-4">
                Use your browser camera to capture microscopy images. Click the
                snapshot or capture controls below.
              </p>
              <CameraCapture settings={settings} stain={settings.stain} />
            </div>
          )}
        </main>
      </div>

      <Footer />
    </div>
  );
}

// ─── File Upload sub-component ─────────────────────────────────────────────

interface FileUploadModeProps {
  state: AnalysisState;
  settings: Settings;
  selectedFile: File | null;
  isHealthy: boolean | null;
  isProcessing: boolean;
  onSelectFile: (f: File | null) => void;
  onRun: () => void;
  onReset: () => void;
}

function FileUploadMode({
  state,
  settings,
  selectedFile,
  isHealthy,
  isProcessing,
  onSelectFile,
  onRun,
  onReset,
}: FileUploadModeProps) {
  return (
    <div className="space-y-5">
      <div>
        <h2 className="text-xl font-semibold tracking-tight">
          TuroQuant Pipeline
        </h2>
        <p className="text-xs text-muted-foreground mt-0.5">
          IHC quantification via DeepLIIF — supports video, multi-page TIF, and
          standard images
        </p>
      </div>

      {state.status === "idle" && (
        <>
          {!selectedFile ? (
            <FileUploader onFile={onSelectFile} />
          ) : (
            <div className="space-y-3">
              <ImagePreview
                file={selectedFile}
                onRemove={() => onSelectFile(null)}
              />
              <Button
                onClick={onRun}
                disabled={isHealthy === false}
                className="w-full sm:w-auto"
              >
                <Play className="h-4 w-4 mr-2" />
                Run TuroQuant
              </Button>
              {isHealthy === false && (
                <p className="text-xs text-destructive">
                  Cannot reach server. Check your connection or try again later.
                </p>
              )}
            </div>
          )}
        </>
      )}

      {isProcessing && (
        <LoadingSpinner progress={state.progress} text={state.progressText} />
      )}

      {state.status === "error" && (
        <div className="rounded-lg border border-destructive/50 bg-destructive/5 p-4">
          <p className="text-sm text-destructive font-medium">
            Analysis failed
          </p>
          <p className="text-xs text-destructive/80 mt-1">{state.error}</p>
          <Button
            variant="outline"
            size="sm"
            className="mt-3 text-xs"
            onClick={onReset}
          >
            Try Again
          </Button>
        </div>
      )}

      {state.status === "done" && state.result && (
        <SingleResultView
          result={state.result}
          stain={settings.stain}
          onReset={onReset}
        />
      )}

      {state.status === "done" && state.videoResult && (
        <VideoResultView
          videoResult={state.videoResult}
          stain={settings.stain}
          onReset={onReset}
        />
      )}
    </div>
  );
}
