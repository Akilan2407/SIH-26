const probabilities = { "No DR": 0.03, "Mild DR": 0.08, "Moderate DR": 0.81, "Severe DR": 0.06, "Proliferative DR": 0.02 };

export function mockAnalyze(image) {
  return Promise.resolve({ screeningId: "DR-0001", predictedClass: 2, predictedLabel: "Moderate DR", probabilities, confidence: 0.81, model: "EfficientNetB4", originalImage: image || "/fundus-placeholder.svg", createdAt: "19 Aug 2026" });
}