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

/**
 * Report generation frontend service interface.
 * Prepared for future backend integration (POST /report).
 *
 * Expected future backend contract:
 *   POST /report
 *   Body: { reportPayload }
 *   Response: application/pdf Blob
 *
 * Frontend phase: Simulates the network lifecycle, validates UI loading/success/error
 * states, and sets up download handling without creating a fake client-side PDF.
 */
export const generateReport = async (reportPayload) => {
  return new Promise((resolve) => {
    setTimeout(() => {
      // Future backend will return:
      // resolve({ success: true, blob: pdfBlob });
      resolve({
        success: true,
        blob: null
      });
    }, 1200);
  });
};
