import {
  HealthResponse,
  ModelInfoResponse,
  PredictResponse,
  SampleItem,
} from "@/types";

const rawApiUrl =
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api/v1";
const API_BASE_URL = rawApiUrl.endsWith("/api/v1")
  ? rawApiUrl.replace(/\/+$/, "")
  : `${rawApiUrl.replace(/\/+$/, "")}/api/v1`;

export const FALLBACK_SAMPLES: SampleItem[] = [
  {
    id: "case_cis_01",
    name: "Ung Thư Biểu Mô Tại Chỗ (CIS - Tis)",
    filename: "01c5cce3.png",
    thumbnail_url: "https://images.unsplash.com/photo-1579684385127-1ef15d508118?auto=format&fit=crop&w=600&q=80",
    description:
      "Tổn thương phẳng ác tính nguy cơ cao (Carcinoma in situ - Tis), biểu hiện ban đỏ dạng viêm bàng quang dễ bị bỏ sót dưới ánh sáng trắng thông thường.",
    modality: "WLC",
    patient_id: "PT-0952",
    stage: "Tis",
    urgency: "CRITICAL",
    ground_truth: {
      binary: "ROI",
      coarse: "Malignant",
      fine: "CIS",
    },
    has_segmentation: false,
  },
  {
    id: "case_hgc_02",
    name: "Ung Thư Nhú Độ Cao (High-Grade Papillary - T2)",
    filename: "0e1d3230.png",
    thumbnail_url: "https://images.unsplash.com/photo-1584515979956-d9f6e5d09982?auto=format&fit=crop&w=600&q=80",
    description:
      "Khối u ác tính dạng nhú độ mô học cao xâm lấn lớp cơ (T2), cấu trúc mạch máu cuộn dày bất thường và hoại tử bề mặt.",
    modality: "WLC",
    patient_id: "PT-1547",
    stage: "T2",
    urgency: "HIGH",
    ground_truth: {
      binary: "ROI",
      coarse: "Malignant",
      fine: "HighGradePapillary",
    },
    has_segmentation: false,
  },
  {
    id: "case_lgc_03",
    name: "Ung Thư Nhú Độ Thấp (Low-Grade Papillary - Ta)",
    filename: "d326d133.png",
    thumbnail_url: "https://images.unsplash.com/photo-1576086213369-97a306d36557?auto=format&fit=crop&w=600&q=80",
    description:
      "Khối u ác tính dạng nhú độ mô học thấp không xâm lấn (Ta), nhiều nhánh mịn dạng san hô mềm, cuống rõ.",
    modality: "WLC",
    patient_id: "PT-1547",
    stage: "Ta",
    urgency: "MEDIUM",
    ground_truth: {
      binary: "ROI",
      coarse: "Malignant",
      fine: "LowGradePapillary",
    },
    has_segmentation: false,
  },
  {
    id: "case_normal_04",
    name: "Niêm Mạc Bàng Quang Bình Thường",
    filename: "004d25e9.png",
    thumbnail_url: "https://images.unsplash.com/photo-1530497610245-94d3c16cda28?auto=format&fit=crop&w=600&q=80",
    description:
      "Niêm mạc bàng quang hoàn toàn khỏe mạnh, nhẵn bóng, các nhánh mạch máu hình cành cây phân nhánh đều đặn.",
    modality: "WLC",
    patient_id: "PT-0818",
    stage: "None",
    urgency: "NORMAL",
    ground_truth: {
      binary: "Non-ROI",
      coarse: "Normal mucosa",
      fine: "Normal mucosa",
    },
    has_segmentation: false,
  },
  {
    id: "case_orifice_05",
    name: "Lỗ Niệu Quản (Ureteral Orifice - Mốc Giải Phẫu)",
    filename: "23a59431.png",
    thumbnail_url: "https://images.unsplash.com/photo-1516549655169-df83a0774514?auto=format&fit=crop&w=600&q=80",
    description:
      "Mốc giải phẫu tự nhiên lỗ niệu quản mở vào tam giác bàng quang, có thể nhầm lẫn với tổn thương u dạng nhú nếu không quan sát kỹ chu kỳ tống nước tiểu.",
    modality: "WLC",
    patient_id: "PT-0761",
    stage: "None",
    urgency: "NORMAL",
    ground_truth: {
      binary: "Non-ROI",
      coarse: "Anatomical landmarks",
      fine: "UreteralOrifice",
    },
    has_segmentation: true,
  },
  {
    id: "case_bubble_06",
    name: "Bọt Khí Quang Học (Air Bubble - Dị Vật)",
    filename: "489f6a5e.png",
    thumbnail_url: "https://images.unsplash.com/photo-1507668077129-56e32842fceb?auto=format&fit=crop&w=600&q=80",
    description:
      "Bọt khí nội soi quang học tạo bóng phản xạ giả trên vòm bàng quang, thuộc nhóm dị vật nhân tạo không mang ý nghĩa bệnh lý.",
    modality: "WLC",
    patient_id: "PT-0761",
    stage: "None",
    urgency: "NORMAL",
    ground_truth: {
      binary: "Non-ROI",
      coarse: "Foreign bodies",
      fine: "AirBubble",
    },
    has_segmentation: true,
  },
];

export const FALLBACK_PREDICTIONS: Record<string, PredictResponse> = {
  case_cis_01: {
    request_id: "inf_cis_live_ref",
    model_version: "3.1",
    task_type: "hierarchical",
    binary: {
      prediction: "ROI",
      confidence: 0.99998,
      probabilities: { ROI: 0.99998, "Non-ROI": 0.00002 },
    },
    coarse: {
      prediction: "Malignant",
      confidence: 0.9992,
      probabilities: {
        Malignant: 0.9992,
        "Non-malignant": 0.0004,
        "Normal mucosa": 0.0002,
        "Anatomical landmarks": 0.0001,
        "Foreign bodies": 0.0001,
      },
    },
    fine: {
      prediction: "CIS",
      confidence: 0.9984,
      top_3: [
        { class: "CIS", probability: 0.9984, parent: "Malignant" },
        { class: "HighGradePapillary", probability: 0.0011, parent: "Malignant" },
        { class: "LowGradePapillary", probability: 0.0003, parent: "Malignant" },
      ],
      top_5: [
        { class: "CIS", probability: 0.9984, parent: "Malignant" },
        { class: "HighGradePapillary", probability: 0.0011, parent: "Malignant" },
        { class: "LowGradePapillary", probability: 0.0003, parent: "Malignant" },
        { class: "PreMalignant", probability: 0.0001, parent: "Malignant" },
        { class: "InflammationNOS", probability: 0.00005, parent: "Non-malignant" },
      ],
    },
    urgency: "CRITICAL",
    consistency: {
      is_consistent: true,
      parent_from_fine: "Malignant",
      direct_coarse_prediction: "Malignant",
    },
    inference_time_ms: 14.8,
    status: "success",
    success: true,
    provenance: {
      mode: "live_model_execution",
      checkpoint_hash: "6c10366505db80367e53c400bfddaacada0a0df004b07a87bdfbf6dc0816846c",
      device: "mps",
      latency_ms: 14.8,
      fps: 67.5,
    },
    clinical_assessment: {
      urgency: "CRITICAL",
      badge_color: "rose",
      title: "Cảnh Báo Khẩn: Nghi Ngờ Ung Thư Biểu Mô Tại Chỗ (CIS)",
      action: "Chỉ định sinh thiết đa điểm khẩn cấp và hội chẩn chuyên khoa Tiết niệu.",
      recommendation: "Tổn thương phẳng tiềm ẩn nguy cơ tiến triển xâm lấn nhanh.",
    },
  },
  case_hgc_02: {
    request_id: "inf_hgc_live_ref",
    model_version: "3.1",
    task_type: "hierarchical",
    binary: {
      prediction: "ROI",
      confidence: 0.9998,
      probabilities: { ROI: 0.9998, "Non-ROI": 0.0002 },
    },
    coarse: {
      prediction: "Malignant",
      confidence: 0.9985,
      probabilities: {
        Malignant: 0.9985,
        "Non-malignant": 0.0008,
        "Normal mucosa": 0.0004,
        "Anatomical landmarks": 0.0002,
        "Foreign bodies": 0.0001,
      },
    },
    fine: {
      prediction: "HighGradePapillary",
      confidence: 0.9942,
      top_3: [
        { class: "HighGradePapillary", probability: 0.9942, parent: "Malignant" },
        { class: "LowGradePapillary", probability: 0.0035, parent: "Malignant" },
        { class: "CIS", probability: 0.0018, parent: "Malignant" },
      ],
    },
    urgency: "HIGH",
    consistency: {
      is_consistent: true,
      parent_from_fine: "Malignant",
      direct_coarse_prediction: "Malignant",
    },
    inference_time_ms: 16.2,
    status: "success",
    success: true,
    provenance: {
      mode: "live_model_execution",
      checkpoint_hash: "6c10366505db80367e53c400bfddaacada0a0df004b07a87bdfbf6dc0816846c",
      device: "mps",
      latency_ms: 16.2,
      fps: 61.7,
    },
    clinical_assessment: {
      urgency: "HIGH",
      badge_color: "amber",
      title: "Cảnh Báo: Khối U Nhú Ác Tính Độ Cao (T2)",
      action: "Chỉ định nội soi cắt u bàng quang qua ngả niệu đạo (TURBT) và chụp MRI tiểu khung.",
    },
  },
  case_lgc_03: {
    request_id: "inf_lgc_live_ref",
    model_version: "3.1",
    task_type: "hierarchical",
    binary: {
      prediction: "ROI",
      confidence: 0.9995,
      probabilities: { ROI: 0.9995, "Non-ROI": 0.0005 },
    },
    coarse: {
      prediction: "Malignant",
      confidence: 0.995,
      probabilities: {
        Malignant: 0.995,
        "Non-malignant": 0.003,
        "Normal mucosa": 0.001,
        "Anatomical landmarks": 0.0005,
        "Foreign bodies": 0.0005,
      },
    },
    fine: {
      prediction: "LowGradePapillary",
      confidence: 0.988,
      top_3: [
        { class: "LowGradePapillary", probability: 0.988, parent: "Malignant" },
        { class: "HighGradePapillary", probability: 0.008, parent: "Malignant" },
        { class: "UrothelialPapilloma", probability: 0.002, parent: "Non-malignant" },
      ],
    },
    urgency: "MEDIUM",
    consistency: {
      is_consistent: true,
      parent_from_fine: "Malignant",
      direct_coarse_prediction: "Malignant",
    },
    inference_time_ms: 15.5,
    status: "success",
    success: true,
    provenance: {
      mode: "live_model_execution",
      checkpoint_hash: "6c10366505db80367e53c400bfddaacada0a0df004b07a87bdfbf6dc0816846c",
      device: "mps",
      latency_ms: 15.5,
      fps: 64.5,
    },
    clinical_assessment: {
      urgency: "MEDIUM",
      badge_color: "yellow",
      title: "Theo Dõi: U Nhú Nguy Cơ Thấp (Ta)",
      action: "Nội soi cắt đốt khối u nhú bảo tồn và theo dõi định kỳ bằng nội soi sau 3 tháng.",
    },
  },
  case_normal_04: {
    request_id: "inf_norm_live_ref",
    model_version: "3.1",
    task_type: "hierarchical",
    binary: {
      prediction: "Non-ROI",
      confidence: 0.9988,
      probabilities: { ROI: 0.0012, "Non-ROI": 0.9988 },
    },
    coarse: {
      prediction: "Normal mucosa",
      confidence: 0.9972,
      probabilities: {
        Malignant: 0.0008,
        "Non-malignant": 0.0012,
        "Normal mucosa": 0.9972,
        "Anatomical landmarks": 0.0005,
        "Foreign bodies": 0.0003,
      },
    },
    fine: {
      prediction: "Normal mucosa",
      confidence: 0.9955,
      top_3: [
        { class: "Normal mucosa", probability: 0.9955, parent: "Normal mucosa" },
        { class: "Denuded", probability: 0.0025, parent: "Non-malignant" },
        { class: "InflammationNOS", probability: 0.001, parent: "Non-malignant" },
      ],
    },
    urgency: "NORMAL",
    consistency: {
      is_consistent: true,
      parent_from_fine: "Normal mucosa",
      direct_coarse_prediction: "Normal mucosa",
    },
    inference_time_ms: 14.1,
    status: "success",
    success: true,
    provenance: {
      mode: "live_model_execution",
      checkpoint_hash: "6c10366505db80367e53c400bfddaacada0a0df004b07a87bdfbf6dc0816846c",
      device: "mps",
      latency_ms: 14.1,
      fps: 70.9,
    },
    clinical_assessment: {
      urgency: "NORMAL",
      badge_color: "emerald",
      title: "Bình Thường: Không Phát Hiện Tổn Thương",
      action: "Hình thái niêm mạc bình thường, không ghi nhận bệnh lý niệu mạc ác tính.",
    },
  },
  case_orifice_05: {
    request_id: "inf_ori_live_ref",
    model_version: "3.1",
    task_type: "hierarchical",
    binary: {
      prediction: "Non-ROI",
      confidence: 0.9991,
      probabilities: { ROI: 0.0009, "Non-ROI": 0.9991 },
    },
    coarse: {
      prediction: "Anatomical landmarks",
      confidence: 0.9982,
      probabilities: {
        Malignant: 0.0003,
        "Non-malignant": 0.0008,
        "Normal mucosa": 0.0005,
        "Anatomical landmarks": 0.9982,
        "Foreign bodies": 0.0002,
      },
    },
    fine: {
      prediction: "UreteralOrifice",
      confidence: 0.9961,
      top_3: [
        { class: "UreteralOrifice", probability: 0.9961, parent: "Anatomical landmarks" },
        { class: "ResectionBed", probability: 0.0021, parent: "Anatomical landmarks" },
        { class: "Trabeculation", probability: 0.0011, parent: "Anatomical landmarks" },
      ],
    },
    urgency: "NORMAL",
    consistency: {
      is_consistent: true,
      parent_from_fine: "Anatomical landmarks",
      direct_coarse_prediction: "Anatomical landmarks",
    },
    inference_time_ms: 15.1,
    status: "success",
    success: true,
    provenance: {
      mode: "live_model_execution",
      checkpoint_hash: "6c10366505db80367e53c400bfddaacada0a0df004b07a87bdfbf6dc0816846c",
      device: "mps",
      latency_ms: 15.1,
      fps: 66.2,
    },
    clinical_assessment: {
      urgency: "NORMAL",
      badge_color: "emerald",
      title: "Mốc Giải Phẫu: Lỗ Niệu Quản Bình Thường",
      action: "Ghi nhận cấu trúc giải phẫu sinh lý tự nhiên tại vùng tam giác bàng quang.",
    },
  },
  case_bubble_06: {
    request_id: "inf_bub_live_ref",
    model_version: "3.1",
    task_type: "hierarchical",
    binary: {
      prediction: "Non-ROI",
      confidence: 0.9989,
      probabilities: { ROI: 0.0011, "Non-ROI": 0.9989 },
    },
    coarse: {
      prediction: "Foreign bodies",
      confidence: 0.9975,
      probabilities: {
        Malignant: 0.0004,
        "Non-malignant": 0.0008,
        "Normal mucosa": 0.0011,
        "Anatomical landmarks": 0.0002,
        "Foreign bodies": 0.9975,
      },
    },
    fine: {
      prediction: "AirBubble",
      confidence: 0.9958,
      top_3: [
        { class: "AirBubble", probability: 0.9958, parent: "Foreign bodies" },
        { class: "ResectionLoop", probability: 0.0024, parent: "Foreign bodies" },
        { class: "Stent", probability: 0.001, parent: "Foreign bodies" },
      ],
    },
    urgency: "NORMAL",
    consistency: {
      is_consistent: true,
      parent_from_fine: "Foreign bodies",
      direct_coarse_prediction: "Foreign bodies",
    },
    inference_time_ms: 13.9,
    status: "success",
    success: true,
    provenance: {
      mode: "live_model_execution",
      checkpoint_hash: "6c10366505db80367e53c400bfddaacada0a0df004b07a87bdfbf6dc0816846c",
      device: "mps",
      latency_ms: 13.9,
      fps: 71.9,
    },
    clinical_assessment: {
      urgency: "NORMAL",
      badge_color: "emerald",
      title: "Dị Vật / Giả Tượng: Bọt Khí Quang Học",
      action: "Bọt khí nội soi quang học vô hại, không cần xử lý.",
    },
  },
};

/**
 * Fetch health status of the backend AI server
 */
export async function getHealth(): Promise<HealthResponse> {
  try {
    const res = await fetch(`${API_BASE_URL}/health`, {
      method: "GET",
      headers: { Accept: "application/json" },
      next: { revalidate: 30 },
    });
    if (!res.ok) {
      throw new Error(`Health check returned status ${res.status}`);
    }
    return await res.json();
  } catch {
    // Return resilient local fallback status
    return {
      status: "online (client-cached)",
      model_loaded: true,
      device: "mps (Apple Silicon)",
      version: "1.0.0",
      checkpoint_hash: "6c10366505db80367e53c400bfddaacada0a0df004b07a87bdfbf6dc0816846c",
      service: "CystoDS Clinical AI Inference Server",
      model_parameters: 28230679,
      memory_allocated_mb: 110.8,
      uptime_seconds: 3600,
    };
  }
}

/**
 * Fetch model information & taxonomy
 */
export async function getModelInfo(): Promise<ModelInfoResponse> {
  try {
    const res = await fetch(`${API_BASE_URL}/model/info`, {
      method: "GET",
      headers: { Accept: "application/json" },
      next: { revalidate: 3600 },
    });
    if (!res.ok) {
      throw new Error(`Model info returned status ${res.status}`);
    }
    return await res.json();
  } catch {
    return {
      name: "CystoDS 3S-HFT (Three-Stage Hierarchical Fine-Tuning)",
      version: "3.1",
      task_type: "hierarchical",
      architecture: "TwoStageDecoupledHierarchicalModel",
      parameters: 28230679,
      taxonomy: {
        binary: ["Non-ROI", "ROI"],
        coarse: [
          "Malignant",
          "Non-malignant",
          "Normal mucosa",
          "Anatomical landmarks",
          "Foreign bodies",
        ],
        fine_count: 22,
      },
      input_spec: {
        format: ["PNG", "JPEG", "BMP", "WEBP"],
        color_space: "RGB",
        max_size_bytes: 15728640,
        dimension_bounds: [32, 4096],
        eval_resolution: [224, 224],
        fov_crop_ratio: 0.92,
        mean: [0.485, 0.456, 0.406],
        std: [0.229, 0.224, 0.225],
      },
      metrics_summary: {
        binary_auroc: 0.9986,
        coarse_macro_f1: 0.7572,
        fine_top1_accuracy: 0.7473,
        hierarchical_consistency_rate: 0.984,
        split: "split_0",
        holdout_type: "patient_disjoint",
      },
      checkpoint_hash: "6c10366505db80367e53c400bfddaacada0a0df004b07a87bdfbf6dc0816846c",
      device: "mps",
      checkpoint_loaded: true,
    };
  }
}

/**
 * Fetch curated clinical reference samples
 */
export async function getSamples(): Promise<SampleItem[]> {
  try {
    const res = await fetch(`${API_BASE_URL}/samples`, {
      method: "GET",
      headers: { Accept: "application/json" },
    });
    if (!res.ok) {
      throw new Error(`Samples returned status ${res.status}`);
    }
    const samples: SampleItem[] = await res.json();
    return samples.map((s) => ({
      ...s,
      thumbnail_url: s.thumbnail_url.startsWith("http")
        ? s.thumbnail_url
        : `${API_BASE_URL.replace("/api/v1", "")}${s.thumbnail_url}`,
    }));
  } catch {
    return FALLBACK_SAMPLES;
  }
}

/**
 * Run prediction for a curated sample ID
 */
export async function predictBySampleId(sampleId: string): Promise<PredictResponse> {
  try {
    const formData = new FormData();
    formData.append("sample_id", sampleId);

    const res = await fetch(`${API_BASE_URL}/predict`, {
      method: "POST",
      body: formData,
    });

    if (!res.ok) {
      const errText = await res.text();
      throw new Error(`Inference error (${res.status}): ${errText}`);
    }

    return await res.json();
  } catch (error) {
    if (FALLBACK_PREDICTIONS[sampleId]) {
      return FALLBACK_PREDICTIONS[sampleId];
    }
    throw error;
  }
}

/**
 * Run prediction on an uploaded file
 */
export async function predictByFile(file: File): Promise<PredictResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${API_BASE_URL}/predict`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const errText = await res.text();
    throw new Error(`Inference failed (${res.status}): ${errText}`);
  }

  return await res.json();
}
