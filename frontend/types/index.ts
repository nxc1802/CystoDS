export interface GroundTruth {
  binary: "ROI" | "Non-ROI";
  coarse: "Malignant" | "Non-malignant" | "Normal mucosa" | "Anatomical landmarks" | "Foreign bodies";
  fine?: string;
}

export interface SampleItem {
  id: string;
  name: string;
  filename: string;
  thumbnail_url: string;
  description: string;
  modality: string;
  patient_id?: string;
  stage?: string;
  urgency: "CRITICAL" | "HIGH" | "MEDIUM" | "NORMAL";
  ground_truth: GroundTruth;
  has_segmentation: boolean;
}

export interface BinaryResult {
  prediction: "ROI" | "Non-ROI";
  confidence: number;
  probabilities: {
    "ROI": number;
    "Non-ROI": number;
    [key: string]: number;
  };
}

export interface CoarseResult {
  prediction: string;
  confidence: number;
  probabilities: {
    "Malignant": number;
    "Non-malignant": number;
    "Normal mucosa": number;
    "Anatomical landmarks": number;
    "Foreign bodies": number;
    [key: string]: number;
  };
}

export interface FineTopItem {
  class: string;
  probability: number;
  parent: string;
}

export interface FineResult {
  prediction: string;
  confidence: number;
  top_3: FineTopItem[];
  top_5?: FineTopItem[];
  probabilities?: Record<string, number>;
}

export interface ConsistencyResult {
  is_consistent: boolean;
  parent_from_fine: string;
  direct_coarse_prediction: string;
}

export interface ClinicalAssessment {
  urgency: "CRITICAL" | "HIGH" | "MEDIUM" | "NORMAL" | "critical" | "high" | "medium" | "normal";
  badge_color: "rose" | "amber" | "yellow" | "blue" | "emerald" | string;
  title: string;
  action: string;
  recommendation?: string;
}

export interface PredictResponse {
  request_id: string;
  model_version: string;
  task_type: string;
  binary: BinaryResult;
  coarse: CoarseResult;
  fine: FineResult;
  urgency: string;
  consistency: ConsistencyResult;
  inference_time_ms: number;
  status: string;
  success: boolean;
  provenance?: {
    mode: string;
    checkpoint_hash: string;
    device: string;
    latency_ms: number;
    fps: number;
  };
  hierarchy_metrics?: ConsistencyResult;
  clinical_assessment?: ClinicalAssessment;
}

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  device: string;
  version: string;
  checkpoint_hash: string;
  service: string;
  model_parameters: number;
  memory_allocated_mb: number;
  uptime_seconds: number;
}

export interface ModelInfoResponse {
  name: string;
  version: string;
  task_type: string;
  architecture: string;
  parameters: number;
  taxonomy: {
    binary: string[];
    coarse: string[];
    fine_count: number;
    fine_by_parent?: Record<string, string[]>;
  };
  input_spec: {
    format: string[];
    color_space: string;
    max_size_bytes: number;
    dimension_bounds: number[];
    eval_resolution: number[];
    fov_crop_ratio: number;
    mean: number[];
    std: number[];
  };
  metrics_summary: {
    binary_auroc: number;
    coarse_macro_f1: number;
    fine_top1_accuracy: number;
    hierarchical_consistency_rate: number;
    split: string;
    holdout_type: string;
  };
  checkpoint_hash?: string;
  device?: string;
  checkpoint_loaded?: boolean;
}

export type InferenceStatus = "idle" | "ready" | "uploading" | "running" | "success" | "error";
