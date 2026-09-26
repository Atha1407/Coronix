export const analyzeSegment = async (file, pointA, pointB, catheterSize) => {
  // Mock API call to simulate backend analysis
  return new Promise((resolve) => {
    setTimeout(() => {
      resolve({
        segment_label: "LAD",
        lesion_detected: true,
        lesion_bbox: [
          Math.min(pointA.x, pointB.x) - 20,
          Math.min(pointA.y, pointB.y) - 20,
          Math.abs(pointA.x - pointB.x) + 40,
          Math.abs(pointA.y - pointB.y) + 40
        ],
        severity_category: "70-90%",
        confidence: 0.89,
        catheter_size: catheterSize ? Number(catheterSize) : null,
        overlay_image_base64: null // Not used in this frontend phase yet
      });
    }, 1000);
  });
};
