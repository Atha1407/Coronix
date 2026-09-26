import React, { useState, useMemo } from 'react';
import { Target, RotateCcw } from 'lucide-react';
import StepProgress from '../components/StepProgress';
import Disclaimer from '../components/Disclaimer';
import AngiogramViewer from '../components/AngiogramViewer';
import { analyzeSegment } from '../services/api';

export default function Analysis({ file, fileType, imageUrl, initialCatheterSize, onReset, onAnalysisComplete }) {
  const [pointA, setPointA] = useState(null);
  const [pointB, setPointB] = useState(null);
  const [catheterSize, setCatheterSize] = useState(initialCatheterSize ? initialCatheterSize.toString() : '');
  const [catheterTouched, setCatheterTouched] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  const isCatheterValid = useMemo(() => {
    if (catheterSize === '' || catheterSize === null || catheterSize === undefined) return false;
    const num = Number(catheterSize);
    return !isNaN(num) && num > 0 && isFinite(num);
  }, [catheterSize]);

  const handlePointClick = (point) => {
    // Basic vessel mask abstraction check
    if (!pointA) {
      setPointA(point);
    } else if (!pointB) {
      setPointB(point);
    }
  };

  const handleAnalyze = async () => {
    if (!pointA || !pointB || !isCatheterValid) {
      setCatheterTouched(true);
      return;
    }
    
    setIsAnalyzing(true);
    try {
      const parsedCatheter = Number(catheterSize);
      const result = await analyzeSegment(file, pointA, pointB, parsedCatheter);
      const enrichedResult = {
        ...result,
        catheter_size: parsedCatheter
      };
      if (onAnalysisComplete) {
        onAnalysisComplete(enrichedResult, pointA, pointB, parsedCatheter);
      }
    } catch (error) {
      console.error("Analysis failed", error);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleStartOver = () => {
    setPointA(null);
    setPointB(null);
    setCatheterSize('');
    setCatheterTouched(false);
    onReset();
  };

  let instructionText = "Select two points (A and B) defining the segment to analyze.";
  if (pointA && !pointB) {
    instructionText = "Point A selected. Now select ending point (B).";
  } else if (pointA && pointB) {
    if (!isCatheterValid) {
      instructionText = "Select or enter catheter size (Fr) to proceed.";
    } else {
      instructionText = "Parameters specified. The segment is ready for analysis.";
    }
  }

  return (
    <div className="max-w-7xl mx-auto flex flex-col h-full">
      <div className="flex justify-between items-center mb-6">
        <StepProgress currentStep={2} />
        <button 
          onClick={handleStartOver}
          className="text-sm font-medium text-muted-teal hover:text-charcoal-blue transition-colors px-4 py-2 bg-white rounded-lg border border-border shadow-sm flex items-center gap-2"
        >
          <RotateCcw className="w-4 h-4" />
          Start Over
        </button>
      </div>
      
      <div className="flex flex-col lg:flex-row gap-6 flex-1 min-h-0">
        
        {/* Left: Main Viewer component (handles pan/zoom internally, no magnifier during analysis) */}
        <AngiogramViewer
          fileType={fileType}
          imageUrl={imageUrl}
          pointA={pointA}
          pointB={pointB}
          isResultsMode={false}
          onPointClick={handlePointClick}
        />
        
        {/* Right: Control Panel (30%) */}
        <div className="flex-[3] flex flex-col min-w-0 space-y-4">
          <div className="glass-card rounded-2xl p-5 md:p-6 border border-white/50 flex-1 flex flex-col">
            {/* 1. Header */}
            <div className="flex items-center gap-2 mb-2 text-charcoal-blue">
              <Target className="w-5 h-5 text-teal shrink-0" />
              <h2 className="text-xl font-semibold tracking-tight">Select Analysis Segment</h2>
            </div>
            
            {/* 2. Short Instruction */}
            <p className="text-sm text-muted-teal mb-4 leading-relaxed">
              {instructionText}
            </p>

            {/* 3. Catheter Size Input (First-View Area) */}
            <div className="bg-white/70 p-4 rounded-xl border border-white/80 shadow-sm mb-4">
              <div className="flex items-center justify-between mb-1.5">
                <label htmlFor="catheter-size" className="text-sm font-semibold text-charcoal-blue">
                  Catheter Size
                </label>
                <span className="text-xs font-semibold px-2 py-0.5 rounded-md bg-teal/10 text-teal">
                  French (Fr)
                </span>
              </div>
              <p className="text-xs text-muted-teal mb-2.5">
                Enter the catheter size used for this study.
              </p>
              
              <div className="relative">
                <input
                  id="catheter-size"
                  type="number"
                  step="any"
                  min="1"
                  max="30"
                  placeholder="Select / Enter size"
                  value={catheterSize}
                  onChange={(e) => {
                    setCatheterSize(e.target.value);
                    if (!catheterTouched) setCatheterTouched(true);
                  }}
                  onBlur={() => setCatheterTouched(true)}
                  className={`w-full pl-3.5 pr-12 py-2 bg-white rounded-lg border text-sm font-medium text-charcoal-blue placeholder:text-muted-teal/50 focus:outline-none transition-all ${
                    catheterTouched && !isCatheterValid
                      ? 'border-red-400 ring-2 ring-red-200'
                      : 'border-border focus:border-teal focus:ring-2 focus:ring-teal/20'
                  }`}
                />
                <span className="absolute right-3.5 top-1/2 -translate-y-1/2 text-xs font-semibold text-muted-teal pointer-events-none">
                  Fr
                </span>
              </div>

              {/* Quick Select Options */}
              <div className="flex items-center gap-1.5 mt-2.5">
                <span className="text-xs text-muted-teal font-medium mr-1">Common:</span>
                {[5, 6, 7].map((size) => (
                  <button
                    key={size}
                    type="button"
                    onClick={() => {
                      setCatheterSize(size.toString());
                      setCatheterTouched(true);
                    }}
                    className={`px-2.5 py-0.5 text-xs rounded-md transition-all border font-medium ${
                      catheterSize === size.toString()
                        ? 'bg-teal text-white border-teal shadow-xs'
                        : 'bg-white/80 text-charcoal-blue border-border hover:border-teal/50 hover:bg-teal/5'
                    }`}
                  >
                    {size} Fr
                  </button>
                ))}
              </div>

              {/* Validation Feedback */}
              {catheterTouched && !isCatheterValid && (
                <p className="text-xs text-red-500 mt-2 font-medium">
                  Please enter a valid catheter size.
                </p>
              )}
            </div>

            {/* 4 & 5. Selected Points Status */}
            <div className="space-y-2.5 mb-5">
              <h3 className="text-xs font-semibold text-charcoal-blue uppercase tracking-wider px-0.5">
                Selected Points
              </h3>
              
              {/* Point A Status */}
              <div className="flex items-center justify-between p-3 bg-white/60 rounded-xl border border-white/80 shadow-sm">
                <div className="flex items-center gap-2.5">
                  <div className="w-3.5 h-3.5 rounded-full bg-teal border-2 border-white shadow-sm shrink-0"></div>
                  <span className="font-medium text-sm text-charcoal-blue">Point A</span>
                </div>
                <div className="flex items-center gap-2.5">
                  {pointA ? (
                    <span className="font-mono text-xs text-muted-teal bg-black/5 px-2 py-0.5 rounded">
                      x: {pointA.x}, y: {pointA.y}
                    </span>
                  ) : (
                    <span className="text-xs text-muted-teal italic">Not selected</span>
                  )}
                  {pointA && (
                    <button onClick={() => setPointA(null)} className="text-xs font-medium text-red-500 hover:text-red-700 uppercase tracking-wide">
                      Reset
                    </button>
                  )}
                </div>
              </div>
              
              {/* Point B Status */}
              <div className="flex items-center justify-between p-3 bg-white/60 rounded-xl border border-white/80 shadow-sm">
                <div className="flex items-center gap-2.5">
                  <div className="w-3.5 h-3.5 rounded-full bg-[#D9A6A0] border-2 border-white shadow-sm shrink-0"></div>
                  <span className="font-medium text-sm text-charcoal-blue">Point B</span>
                </div>
                <div className="flex items-center gap-2.5">
                  {pointB ? (
                    <span className="font-mono text-xs text-muted-teal bg-black/5 px-2 py-0.5 rounded">
                      x: {pointB.x}, y: {pointB.y}
                    </span>
                  ) : (
                    <span className="text-xs text-muted-teal italic">Not selected</span>
                  )}
                  {pointB && (
                    <button onClick={() => setPointB(null)} className="text-xs font-medium text-red-500 hover:text-red-700 uppercase tracking-wide">
                      Reset
                    </button>
                  )}
                </div>
              </div>
            </div>
            
            {/* 6. Analyze Segment Button */}
            <div className="mt-auto pt-2">
              <button
                onClick={handleAnalyze}
                disabled={!pointA || !pointB || !isCatheterValid || isAnalyzing || fileType === 'dicom'}
                className={`w-full py-3.5 rounded-xl font-medium text-base transition-all flex justify-center items-center gap-2 ${
                  pointA && pointB && isCatheterValid && !isAnalyzing
                    ? 'bg-teal text-white shadow-md hover:bg-charcoal-blue hover:shadow-lg cursor-pointer'
                    : 'bg-muted-teal/20 text-muted-teal cursor-not-allowed'
                }`}
              >
                {isAnalyzing ? (
                  <>
                    <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
                    Analyzing Segment...
                  </>
                ) : (
                  <>
                    Analyze Segment
                    <span>&rarr;</span>
                  </>
                )}
              </button>
              {pointA && pointB && !isCatheterValid && (
                <p className="text-center text-xs text-muted-teal mt-2">
                  Catheter size is required to begin analysis.
                </p>
              )}
            </div>
          </div>
          
          {/* 7. Disclaimer */}
          <Disclaimer />
        </div>
      </div>
    </div>
  );
}
