import React from 'react';
import { Target, FileText, RotateCcw, AlertCircle } from 'lucide-react';
import StepProgress from '../components/StepProgress';
import Disclaimer from '../components/Disclaimer';
import AngiogramViewer from '../components/AngiogramViewer';

export default function Results({ fileType, imageUrl, analysisResult, pointA, pointB, catheterSize, onAnalyzeAnother, onReset }) {
  const displayCatheterSize = catheterSize || analysisResult?.catheter_size;

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
      <div className="flex justify-between items-center mb-6">
        <StepProgress currentStep={3} />
        <button 
          onClick={onReset}
          className="text-sm font-medium text-muted-teal hover:text-charcoal-blue transition-colors px-4 py-2 bg-white rounded-lg border border-border shadow-sm flex items-center gap-2"
        >
          <RotateCcw className="w-4 h-4" />
          Start Over
        </button>
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
                {analysisResult.severity_category && analysisResult.lesion_detected && (
                  <div className="bg-teal/10 border border-teal/20 px-3 py-1.5 rounded-lg text-teal font-semibold text-sm">
                    {analysisResult.severity_category}
                  </div>
                )}
              </div>

              {/* Confidence */}
              {analysisResult.confidence != null && (
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
              {displayCatheterSize && (
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
                  <div className="w-4 h-4 rounded-full bg-teal border-2 border-white shadow-sm"></div>
                  <span className="font-medium text-sm text-charcoal-blue">Point A</span>
                </div>
                {pointA ? (
                  <span className="font-mono text-xs text-muted-teal bg-black/5 px-2 py-1 rounded">
                    x: {pointA.x}, y: {pointA.y}
                  </span>
                ) : null}
              </div>
              
              {/* Point B Data */}
              <div className="flex items-center justify-between p-3 bg-white/60 rounded-xl border border-white/80 shadow-sm">
                <div className="flex items-center gap-3">
                  <div className="w-4 h-4 rounded-full bg-[#D9A6A0] border-2 border-white shadow-sm"></div>
                  <span className="font-medium text-sm text-charcoal-blue">Point B</span>
                </div>
                {pointB ? (
                  <span className="font-mono text-xs text-muted-teal bg-black/5 px-2 py-1 rounded">
                    x: {pointB.x}, y: {pointB.y}
                  </span>
                ) : null}
              </div>
            </div>
            
            <div className="mt-auto space-y-3">
              <button className="w-full py-3.5 rounded-xl font-medium text-lg transition-all flex justify-center items-center gap-2 bg-teal text-white shadow-md hover:bg-charcoal-blue hover:shadow-lg">
                <FileText className="w-5 h-5" />
                Generate Report
                <span>&rarr;</span>
              </button>
              
              <button 
                onClick={onAnalyzeAnother}
                className="w-full py-3 rounded-xl font-medium text-base transition-colors flex justify-center items-center gap-2 bg-white text-muted-teal border border-border shadow-sm hover:text-charcoal-blue hover:border-charcoal-blue/30"
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
