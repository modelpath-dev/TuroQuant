"use client";

import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Separator } from "@/components/ui/separator";
import { Slider } from "@/components/ui/slider";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import type { Settings, SourceMode, StainType, Resolution } from "@/types";
import { Activity, Settings2 } from "lucide-react";

interface SettingsPanelProps {
  settings: Settings;
  onChange: (s: Settings) => void;
  sourceMode: SourceMode;
  onSourceModeChange: (m: SourceMode) => void;
  onCheckServer: () => void;
  checkingServer: boolean;
  showVideoSettings: boolean;
}

export function SettingsPanel({
  settings,
  onChange,
  sourceMode,
  onSourceModeChange,
  onCheckServer,
  checkingServer,
  showVideoSettings,
}: SettingsPanelProps) {
  const update = <K extends keyof Settings>(key: K, value: Settings[K]) =>
    onChange({ ...settings, [key]: value });

  return (
    <div className="space-y-5">
      <div className="flex items-center gap-2">
        <Settings2 className="h-4 w-4 text-muted-foreground" />
        <h2 className="font-semibold text-sm">TuroQuant Options</h2>
      </div>

      {/* Source mode */}
      <div className="space-y-2">
        <label className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
          Source
        </label>
        <div className="flex gap-1 rounded-md bg-muted p-1">
          {(["file", "camera"] as const).map((mode) => (
            <button
              key={mode}
              onClick={() => onSourceModeChange(mode)}
              className={`flex-1 text-xs py-1.5 rounded-sm font-medium transition-colors ${
                sourceMode === mode
                  ? "bg-background text-foreground shadow-sm"
                  : "text-muted-foreground hover:text-foreground"
              }`}
            >
              {mode === "file" ? "File Upload" : "Camera"}
            </button>
          ))}
        </div>
      </div>

      {/* Stain */}
      <div className="space-y-2">
        <label className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
          Stain
        </label>
        <div className="flex gap-1 rounded-md bg-muted p-1">
          {(["KI67", "ER", "PR"] as StainType[]).map((s) => (
            <button
              key={s}
              onClick={() => update("stain", s)}
              className={`flex-1 text-xs py-1.5 rounded-sm font-medium transition-colors ${
                settings.stain === s
                  ? "bg-background text-foreground shadow-sm"
                  : "text-muted-foreground hover:text-foreground"
              }`}
            >
              {s}
            </button>
          ))}
        </div>
      </div>

      {sourceMode === "camera" && (
        <div className="space-y-2">
          <label className="text-xs font-medium text-muted-foreground">
            Dedup radius: {settings.dedupRadius}px
          </label>
          <Slider
            value={[settings.dedupRadius]}
            min={5}
            max={50}
            step={1}
            onValueChange={([v]) => update("dedupRadius", v)}
          />
        </div>
      )}

      <Separator />

      {/* Acquisition */}
      <div className="space-y-3">
        <label className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
          Acquisition
        </label>

        <div className="space-y-1.5">
          <label className="text-xs text-muted-foreground">Scan Resolution</label>
          <Select
            value={settings.resolution}
            onValueChange={(v) => update("resolution", v as Resolution)}
          >
            <SelectTrigger className="h-8 text-xs">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {(["40x", "20x", "10x"] as const).map((r) => (
                <SelectItem key={r} value={r} className="text-xs">
                  {r}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <div className="space-y-1.5">
          <label className="text-xs text-muted-foreground">
            Probability Threshold: {settings.probThresh.toFixed(2)}
          </label>
          <Slider
            value={[settings.probThresh]}
            min={0}
            max={1}
            step={0.05}
            onValueChange={([v]) => update("probThresh", v)}
          />
        </div>

        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <Checkbox
              id="slim"
              checked={settings.slim}
              onCheckedChange={(v) => update("slim", !!v)}
            />
            <label htmlFor="slim" className="text-xs">Slim mode</label>
          </div>
          <div className="flex items-center gap-2">
            <Checkbox
              id="nopost"
              checked={settings.nopost}
              onCheckedChange={(v) => update("nopost", !!v)}
            />
            <label htmlFor="nopost" className="text-xs">Skip postprocessing</label>
          </div>
          <div className="flex items-center gap-2">
            <Checkbox
              id="usePil"
              checked={settings.usePil}
              onCheckedChange={(v) => update("usePil", !!v)}
            />
            <label htmlFor="usePil" className="text-xs">Pillow loader (PNG/JPG only)</label>
          </div>
        </div>
      </div>

      {/* Video-specific settings */}
      {showVideoSettings && sourceMode === "file" && (
        <>
          <Separator />
          <div className="space-y-3">
            <label className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
              Video / Multi-page
            </label>
            <div className="space-y-1.5">
              <label className="text-xs text-muted-foreground">
                Extract every: {settings.everyNSec.toFixed(1)}s
              </label>
              <Slider
                value={[settings.everyNSec]}
                min={0.5}
                max={10}
                step={0.5}
                onValueChange={([v]) => update("everyNSec", v)}
              />
            </div>
            <div className="space-y-1.5">
              <label className="text-xs text-muted-foreground">
                Output FPS: {settings.outFps}
              </label>
              <Slider
                value={[settings.outFps]}
                min={1}
                max={30}
                step={1}
                onValueChange={([v]) => update("outFps", v)}
              />
            </div>
            <div className="space-y-1.5">
              <label className="text-xs text-muted-foreground">
                Interpolation steps: {settings.interpSteps}
              </label>
              <Slider
                value={[settings.interpSteps]}
                min={0}
                max={4}
                step={1}
                onValueChange={([v]) => update("interpSteps", v)}
              />
            </div>
            <div className="space-y-1.5">
              <label className="text-xs text-muted-foreground">
                Dedup radius: {settings.dedupRadius}px
              </label>
              <Slider
                value={[settings.dedupRadius]}
                min={5}
                max={50}
                step={1}
                onValueChange={([v]) => update("dedupRadius", v)}
              />
            </div>
          </div>
        </>
      )}

      <Separator />

      <Button
        variant="outline"
        size="sm"
        className="w-full text-xs"
        onClick={onCheckServer}
        disabled={checkingServer}
      >
        <Activity className="h-3.5 w-3.5 mr-1.5" />
        {checkingServer ? "Checking..." : "Check Server"}
      </Button>
    </div>
  );
}
