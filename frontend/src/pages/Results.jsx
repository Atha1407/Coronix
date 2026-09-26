import React, { useState } from 'react';
import { Target, FileText, RotateCcw, AlertCircle, CheckCircle2, AlertTriangle, Loader2, PlugZap } from 'lucide-react';
import StepProgress from '../components/StepProgress';
import Disclaimer from '../components/Disclaimer';
import AngiogramViewer from '../components/AngiogramViewer';
import { generateReport } from '../services/api';

const ALLOWED_SEVERITY_CATEGORIES = ['<20%', '20-50%', '50-70%', '70-90%', '90-98%', '99%', '100%'];
const SOURCE_LABELS = {
  image:           'Uploaded Image',
  dicom_upload:    'Uploaded DICOM',
  dicom_connected: 'Connected DICOM',
};

export default function Results({ fileType, imageUrl, imageSource, analysisResult, pointA, pointB, catheterSize, onAnalyzeAnother, onReset }) {
  const displayCatheterSize = catheterSize || analysisResult?.catheter_size;
  const validSeverity = ALLOWED_SEVERITY_CATEGORIES.includes(analysisResult?.severity_category)
    ? analysisResult.severity_category
    : null;

  const [isGeneratingReport, setIsGeneratingReport] = useState(false);
  const [reportStatus, setReportStatus] = useState(null); // null | 'success' | 'error'
  const [reportErrorMessage, setReportErrorMessage] = useState('');

  const handleGenerateReport = async () => {
    if (isGeneratingReport) return;

    setIsGeneratingReport(true);
    setReportStatus(null);
    setReportErrorMessage('');

    // Conceptual report payload preserving all clinical parameters
    const reportPayload = {
      title: "CORONIX — Coronary Lesion Analysis Report",
      case_id: analysisResult?.case_id || null,
      segment_label: analysisResult?.segment_label || "LAD",
      pointA: pointA ? { x: pointA.x, y: pointA.y } : null,
      pointB: pointB ? { x: pointB.x, y: pointB.y } : null,
      catheterSize: displayCatheterSize ? Number(displayCatheterSize) : null,
      catheter_unit: "Fr",
      lesion_detected: analysisResult?.lesion_detected ?? null,
      lesion_bbox: analysisResult?.lesion_bbox || null,
      lesion_location: analysisResult?.lesion_location || null,
      severity_category: analysisResult?.severity_category || null,
      confidence: analysisResult?.confidence ?? null,
      overlay_image_base64: analysisResult?.overlay_image_base64 || null,
      short_finding: analysisResult?.short_finding || (analysisResult?.lesion_detected ? `Lesion detected in ${analysisResult?.segment_label || 'segment'} with ${analysisResult?.severity_category || 'stenosis'}.` : "No significant lesion detected."),
      disclaimer: "This tool is an AI-assisted research/hackathon prototype for demonstration purposes only. It is not a medical device, does not provide a diagnosis, and must not be used for clinical decision-making.",
      model_version: analysisResult?.model_version || null
    };

    try {
      const response = await generateReport(reportPayload);

      // Future backend PDF integration: triggers download when real Blob is returned
      if (response?.blob instanceof Blob) {
        const url = window.URL.createObjectURL(response.blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = 'coronix-analysis-report.pdf';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        window.URL.revokeObjectURL(url);
      }

      setReportStatus('success');
    } catch (error) {
      console.error("Report generation failed:", error);
      setReportStatus('error');
      setReportErrorMessage("Report could not be generated. Please try again.");
    } finally {
      setIsGeneratingReport(false);
    }
  };

  if (!analysisResult) {
    return (
      <div className="max-w-7xl mx-auto flex flex-col h-full items-center justify-center">
        <AlertCircle className="w-12 h-12 text-muted-teal mb-4" />
        <h2 className="text-xl font-medium text-charcoal-blue mb-2">No Analysis Results Found</h2>
        <p className="text-muted-teal mb-6">Please return to the analysis step and analyze a segment.</p>
        <button 
          onClick={onAnalyzeAnother}
          className="px-6 py-2 bg-teal text-white rounded-xl font-medium shadow-sm hover:bg-charcoal-blue transition-colors"
        >
          Return to Analysis
        </button>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto flex flex-col h-full">
      <div className="flex flex-wrap justify-between items-center mb-6 gap-4">
        <StepProgress currentStep={3} className="mb-0" />
        <div className="flex items-center gap-3 shrink-0">
          {imageSource && SOURCE_LABELS[imageSource] && (
            <span className="inline-flex items-center gap-1.5 text-xs font-semibold text-teal bg-teal/10 border border-teal/20 px-3.5 py-1.5 rounded-full whitespace-nowrap shrink-0 shadow-xs">
              <PlugZap className="w-3.5 h-3.5 shrink-0" />
              <span>{SOURCE_LABELS[imageSource]}</span>
            </span>
          )}
          <button 
            onClick={onReset}
            className="text-sm font-medium text-muted-teal hover:text-charcoal-blue transition-colors px-4 py-2 bg-white rounded-lg border border-border shadow-sm flex items-center gap-2 whitespace-nowrap shrink-0 hover:bg-slate-50"
          >
            <RotateCcw className="w-4 h-4 shrink-0" />
            <span>Start Over</span>
          </button>
        </div>
      </div>
      
      <div className="mb-6">
        <h1 className="text-3xl font-semibold text-charcoal-blue mb-2">Analysis Complete</h1>
        <p className="text-muted-teal text-lg">AI-assisted analysis of the selected coronary segment. Hover to inspect.</p>
      </div>
      
      <div className="flex flex-col lg:flex-row gap-6 flex-1 min-h-0">
        
        {/* Left: Main Viewer with Magnifier */}
        <AngiogramViewer
          fileType={fileType}
          imageUrl={imageUrl}
          pointA={pointA}
          pointB={pointB}
          analysisResult={analysisResult}
          isResultsMode={true}
        />
        
        {/* Right: Summary Panel (30%) */}
        <div className="flex-[3] flex flex-col min-w-0 space-y-4">
          <div className="glass-card rounded-2xl p-6 border border-white/50 flex-1 flex flex-col">
            <div className="flex items-center gap-2 mb-6 text-charcoal-blue">
              <Target className="w-5 h-5 text-teal" />
              <h2 className="text-xl font-semibold tracking-tight">Analysis Summary</h2>
            </div>
            
            <div className="space-y-4 mb-8">
              {/* Segment */}
              <div className="bg-white/60 p-4 rounded-xl border border-white/80 shadow-sm">
                <p className="text-xs font-semibold text-muted-teal uppercase tracking-wider mb-1">Segment</p>
                <p className="font-medium text-charcoal-blue text-lg">
                  {analysisResult.segment_label || 'Unknown Segment'}
                </p>
              </div>

              {/* Lesion & Severity */}
              <div className="bg-white/60 p-4 rounded-xl border border-white/80 shadow-sm flex items-center justify-between">
                <div>
                  <p className="text-xs font-semibold text-muted-teal uppercase tracking-wider mb-1">Lesion</p>
                  <p className="font-medium text-charcoal-blue text-lg">
                    {analysisResult.lesion_detected ? 'Detected' : 'Not Detected'}
                  </p>
                </div>
                {validSeverity && analysisResult.lesion_detected && (
                  <div className="bg-teal/10 border border-teal/20 px-3 py-1.5 rounded-lg text-teal font-semibold text-sm">
                    {validSeverity}
                  </div>
                )}
              </div>

              {/* Confidence */}
              {analysisResult.confidence != null && !isNaN(analysisResult.confidence) && (
                <div className="bg-white/60 p-4 rounded-xl border border-white/80 shadow-sm flex items-center justify-between">
                  <div>
                    <p className="text-xs font-semibold text-muted-teal uppercase tracking-wider mb-1">Confidence</p>
                    <p className="font-medium text-charcoal-blue text-lg">
                      {Math.round(analysisResult.confidence * 100)}%
                    </p>
                  </div>
                </div>
              )}

              {/* Catheter Size */}
              {displayCatheterSize != null && !isNaN(displayCatheterSize) && (
                <div className="bg-white/60 p-4 rounded-xl border border-white/80 shadow-sm flex items-center justify-between">
                  <div>
                    <p className="text-xs font-semibold text-muted-teal uppercase tracking-wider mb-1">Catheter Size</p>
                    <p className="font-medium text-charcoal-blue text-lg">
                      {displayCatheterSize} Fr
                    </p>
                  </div>
                  <div className="bg-teal/10 border border-teal/20 px-2.5 py-1 rounded-lg text-teal font-semibold text-xs">
                    Calibration Ref
                  </div>
                </div>
              )}
            </div>
            
            <div className="space-y-3 mb-6">
              <h3 className="text-sm font-semibold text-charcoal-blue mb-2">Coordinates</h3>
              {/* Point A Data */}
              <div className="flex items-center justify-between p-3 bg-white/60 rounded-xl border border-white/80 shadow-sm">
                <div className="flex items-center gap-3">
                  <div className="w-4 h-4 rounded-full bg-[#00E5FF] border-2 border-white shadow-xs shrink-0 ring-1 ring-[#00E5FF]/40"></div>
                  <span className="font-medium text-sm text-charcoal-blue">Point A</span>
                </div>
                {pointA && pointA.x != null && pointA.y != null && !isNaN(pointA.x) && !isNaN(pointA.y) ? (
                  <span className="font-mono text-xs text-muted-teal bg-black/5 px-2 py-1 rounded">
                    x: {pointA.x}, y: {pointA.y}
                  </span>
                ) : (
                  <span className="text-xs text-muted-teal italic">Unavailable</span>
                )}
              </div>
              
              {/* Point B Data */}
              <div className="flex items-center justify-between p-3 bg-white/60 rounded-xl border border-white/80 shadow-sm">
                <div className="flex items-center gap-3">
                  <div className="w-4 h-4 rounded-full bg-[#FF3366] border-2 border-white shadow-xs shrink-0 ring-1 ring-[#FF3366]/40"></div>
                  <span className="font-medium text-sm text-charcoal-blue">Point B</span>
                </div>
                {pointB && pointB.x != null && pointB.y != null && !isNaN(pointB.x) && !isNaN(pointB.y) ? (
                  <span className="font-mono text-xs text-muted-teal bg-black/5 px-2 py-1 rounded">
                    x: {pointB.x}, y: {pointB.y}
                  </span>
                ) : (
                  <span className="text-xs text-muted-teal italic">Unavailable</span>
                )}
              </div>
            </div>
            
            <div className="mt-auto space-y-3">
              {/* Success Notification */}
              {reportStatus === 'success' && (
                <div className="p-3 bg-emerald-500/10 border border-emerald-500/20 rounded-xl flex items-center gap-2.5 text-xs text-emerald-800 font-medium">
                  <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0" />
                  <span>Report generated successfully.</span>
                </div>
              )}

              {/* Error Notification with Retry */}
              {reportStatus === 'error' && (
                <div className="p-3 bg-red-500/10 border border-red-500/20 rounded-xl flex items-center justify-between gap-2 text-xs text-red-700 font-medium">
                  <div className="flex items-center gap-2">
                    <AlertTriangle className="w-4 h-4 text-red-500 shrink-0" />
                    <span>{reportErrorMessage || "Report could not be generated. Please try again."}</span>
                  </div>
                  <button
                    onClick={handleGenerateReport}
                    className="text-xs font-semibold underline text-red-800 hover:text-red-950 cursor-pointer shrink-0"
                  >
                    Retry
                  </button>
                </div>
              )}

              {/* Primary Action: Generate Report */}
              <button
                onClick={handleGenerateReport}
                disabled={isGeneratingReport}
                className={`w-full py-3.5 rounded-xl font-medium text-lg transition-all flex justify-center items-center gap-2 ${
                  isGeneratingReport
                    ? 'bg-teal/70 text-white cursor-not-allowed'
                    : 'bg-teal text-white shadow-md hover:bg-charcoal-blue hover:shadow-lg cursor-pointer'
                }`}
              >
                {isGeneratingReport ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin text-white" />
                    <span>Generating Report...</span>
                  </>
                ) : (
                  <>
                    <FileText className="w-5 h-5" />
                    <span>Generate Report</span>
                    <span>&rarr;</span>
                  </>
                )}
              </button>
              
              {/* Secondary Action: Analyze Another Segment */}
              <button 
                onClick={onAnalyzeAnother}
                className="w-full py-3 rounded-xl font-medium text-base transition-colors flex justify-center items-center gap-2 bg-white text-muted-teal border border-border shadow-sm hover:text-charcoal-blue hover:border-charcoal-blue/30 cursor-pointer"
              >
                Analyze Another Segment
              </button>
            </div>
          </div>
          
          <Disclaimer />
        </div>
      </div>
    </div>
  );
}
