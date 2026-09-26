import React, { useState } from 'react';
import Navbar from './components/Navbar';
import Sidebar from './components/Sidebar';
import Upload from './pages/Upload';
import Analysis from './pages/Analysis';
import Results from './pages/Results';
import './index.css';

function App() {
  const [file, setFile] = useState(null);
  const [fileType, setFileType] = useState(null);
  const [imageUrl, setImageUrl] = useState(null);
  const [currentStep, setCurrentStep] = useState(1);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [points, setPoints] = useState({ pointA: null, pointB: null });
  const [catheterSize, setCatheterSize] = useState(null);

  const handleUploadComplete = (uploadedFile, type, url) => {
    setFile(uploadedFile);
    setFileType(type);
    setImageUrl(url);
    setCurrentStep(2);
  };

  const handleAnalysisComplete = (result, pointA, pointB, catSize) => {
    setAnalysisResult(result);
    setPoints({ pointA, pointB });
    setCatheterSize(catSize);
    setCurrentStep(3);
  };

  const handleAnalyzeAnother = () => {
    setAnalysisResult(null);
    setCurrentStep(2);
  };

  const resetUpload = () => {
    setFile(null);
    setFileType(null);
    if (imageUrl) URL.revokeObjectURL(imageUrl);
    setImageUrl(null);
    setAnalysisResult(null);
    setPoints({ pointA: null, pointB: null });
    setCatheterSize(null);
    setCurrentStep(1);
  };

  return (
    <div className="flex h-screen w-full bg-bright-snow overflow-hidden">
      <Sidebar />
      <div className="flex-1 flex flex-col min-w-0">
        <Navbar />
        <main className="flex-1 overflow-auto p-6 md:p-8">
          {currentStep === 1 && (
            <Upload onUploadComplete={handleUploadComplete} />
          )}
          {currentStep === 2 && (
            <Analysis 
              file={file} 
              fileType={fileType} 
              imageUrl={imageUrl} 
              onReset={resetUpload}
              onAnalysisComplete={handleAnalysisComplete}
            />
          )}
          {currentStep === 3 && (
            <Results
              fileType={fileType}
              imageUrl={imageUrl}
              analysisResult={analysisResult}
              pointA={points.pointA}
              pointB={points.pointB}
              catheterSize={catheterSize}
              onAnalyzeAnother={handleAnalyzeAnother}
              onReset={resetUpload}
            />
          )}
        </main>
      </div>
    </div>
  );
}

export default App;
