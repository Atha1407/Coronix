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
  const [imageSource, setImageSource] = useState(null); // 'image' | 'dicom_upload' | 'dicom_connected'
  const [frameIndex, setFrameIndex] = useState(0);
  const [dicomMetadata, setDicomMetadata] = useState(null);
  const [currentStep, setCurrentStep] = useState(1);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [points, setPoints] = useState({ pointA: null, pointB: null });
  const [catheterSize, setCatheterSize] = useState(null);

  // 'analyze' | 'how-it-works' | 'about'
  const [activePage, setActivePage] = useState('analyze');

  const handleNavigate = (page) => {
    setActivePage(page);
  };

  const handleUploadComplete = (
    uploadedFile,
    type,
    url,
    sourceOrFrame = null,
    maybeFrameOrMeta = null,
    maybeMeta = null
  ) => {
    let source = type === 'dicom' ? 'dicom_upload' : 'image';
    let selectedFrame = 0;
    let meta = null;

    if (typeof sourceOrFrame === 'string') {
      source = sourceOrFrame;
      if (typeof maybeFrameOrMeta === 'number') {
        selectedFrame = maybeFrameOrMeta;
        meta = maybeMeta;
      } else if (typeof maybeFrameOrMeta === 'object') {
        meta = maybeFrameOrMeta;
      }
    } else if (typeof sourceOrFrame === 'number') {
      selectedFrame = sourceOrFrame;
      meta = maybeFrameOrMeta;
    } else if (typeof sourceOrFrame === 'object' && sourceOrFrame !== null) {
      source = sourceOrFrame.source || source;
      selectedFrame = sourceOrFrame.frameIndex || 0;
      meta = sourceOrFrame.metadata || null;
    }

    setFile(uploadedFile);
    setFileType(type);
    setImageUrl(url);
    setImageSource(source);
    setFrameIndex(selectedFrame);
    setDicomMetadata(meta);
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
    if (imageUrl && !imageUrl.startsWith('data:')) {
      URL.revokeObjectURL(imageUrl);
    }
    setImageUrl(null);
    setImageSource(null);
    setFrameIndex(0);
    setDicomMetadata(null);
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
          imageSource={imageSource}
          initialCatheterSize={catheterSize}
          frameIndex={frameIndex}
          dicomMetadata={dicomMetadata}
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
          imageSource={imageSource}
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
