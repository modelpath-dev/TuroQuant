export interface LabelResult {
  labels: Int32Array;
  numLabels: number;
  width: number;
  height: number;
}

export interface RegionProp {
  label: number;
  area: number;
  centroid: [number, number]; // [y, x]
  maxIntensity?: number;
}
