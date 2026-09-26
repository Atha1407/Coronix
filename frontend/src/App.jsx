import React, { useState } from 'react';
import Navbar from './components/Navbar';
import Sidebar from './components/Sidebar';
import Upload from './pages/Upload';
import Analysis from './pages/Analysis';
import Results from './pages/Results';
import HowItWorks from './pages/HowItWorks';
import About from './pages/About';
import './index.css';

function App() {
  const [file, setFile] = useState(null);
  const [fileType, setFileType] = useState(null);
  const [imageUrl, setImageUrl] = useState(null);
  const [currentStep, setCurrentStep] = useState(1);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [points, setPoints] = useState({ pointA: null, pointB: null });
  const [catheterSize, setCatheterSize] = useState(null);

  // 'analyze' | 'how-it-works' | 'about'
  const [activePage, setActivePage] = useState('analyze');

  const handleNavigate = (page) => {
    setActivePage(page);
  };

  const handleUploadComplete = (uploadedFile, type, url) => {
    setFile(uploadedFile);
    setFileType(type);
    setImageUrl(url);
    setCurrentStep(2);
    setActivePage('analyze');
  };

  const handleAnalysisComplete = (result, pointA, pointB, catSize) => {
    setAnalysisResult(result);
    setPoints({ pointA, pointB });
    setCatheterSize(catSize);
    setCurrentStep(3);
    setActivePage('analyze');
  };

  const handleAnalyzeAnother = () => {
    setAnalysisResult(null);
    setCurrentStep(2);
    setActivePage('analyze');
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
    setActivePage('analyze');
  };

  const renderMain = () => {
    if (activePage === 'how-it-works') {
      return <HowItWorks onNavigate={handleNavigate} />;
    }
    if (activePage === 'about') {
      return <About onNavigate={handleNavigate} />;
    }

    // activePage === 'analyze'
    if (currentStep === 1) {
      return <Upload onUploadComplete={handleUploadComplete} />;
    }
    if (currentStep === 2) {
      return (
        <Analysis
          file={file}
          fileType={fileType}
          imageUrl={imageUrl}
          initialCatheterSize={catheterSize}
          onReset={resetUpload}
          onAnalysisComplete={handleAnalysisComplete}
        />
      );
    }
    if (currentStep === 3) {
      return (
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
      );
    }
    return null;
  };

  return (
    <div className="flex h-screen w-full bg-bright-snow overflow-hidden">
      <Sidebar activePage={activePage} onNavigate={handleNavigate} />
      <div className="flex-1 flex flex-col min-w-0">
        <Navbar activePage={activePage} onNavigate={handleNavigate} />
        <main className="flex-1 overflow-auto p-6 md:p-8">
          {renderMain()}
        </main>
      </div>
    </div>
  );
}

export default App;
