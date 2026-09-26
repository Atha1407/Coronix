import React, { useState, useRef } from 'react';
import { UploadCloud, File, Image as ImageIcon, X } from 'lucide-react';
import StepProgress from '../components/StepProgress';

export default function Upload({ onUploadComplete }) {
  const [selectedFile, setSelectedFile] = useState(null);
  const [fileType, setFileType] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [error, setError] = useState('');
  const [isHovering, setIsHovering] = useState(false);
  const [isImageResolving, setIsImageResolving] = useState(false);
  
  const fileInputRef = useRef(null);

  const processFile = (file) => {
    setError('');
    
    if (!file) return;
    
    const isImage = file.type.startsWith('image/png') || file.type.startsWith('image/jpeg') || file.name.toLowerCase().endsWith('.jpg');
    const isDicom = file.name.toLowerCase().endsWith('.dcm') || file.type === 'application/dicom';
    
    if (!isImage && !isDicom) {
      setError('Unsupported file format. Please upload a PNG, JPG, or DICOM file.');
      return;
    }
    
    setSelectedFile(file);
    setFileType(isImage ? 'image' : 'dicom');
    
    if (isImage) {
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
      setIsImageResolving(true);
      // Let the CSS animation play out
      setTimeout(() => {
        setIsImageResolving(false);
      }, 1000);
    } else {
      setPreviewUrl(null);
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      processFile(e.target.files[0]);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsHovering(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      processFile(e.dataTransfer.files[0]);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsHovering(true);
  };

  const handleDragLeave = () => {
    setIsHovering(false);
  };

  const handleRemove = () => {
    setSelectedFile(null);
    setFileType(null);
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    setPreviewUrl(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleContinue = () => {
    if (selectedFile) {
      onUploadComplete(selectedFile, fileType, previewUrl);
    }
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <div className="max-w-4xl mx-auto flex flex-col h-full">
      <StepProgress currentStep={1} />
      
      <div className="mb-8">
        <h1 className="text-3xl font-semibold text-charcoal-blue mb-2">Upload Coronary Angiogram</h1>
        <p className="text-muted-teal text-lg">Upload a clear coronary angiography frame to begin segment analysis.</p>
      </div>
      
      {!selectedFile ? (
        <div 
          className={`glass-card rounded-2xl p-12 flex flex-col items-center justify-center text-center border-2 border-dashed transition-colors cursor-pointer ${
            isHovering ? 'border-teal bg-teal/5' : 'border-muted-teal/30 hover:border-teal/50 hover:bg-white/60'
          }`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
        >
          <div className="w-16 h-16 bg-white rounded-full shadow-sm flex items-center justify-center mb-6 text-teal">
            <UploadCloud className="w-8 h-8" />
          </div>
          <h3 className="text-xl font-medium text-charcoal-blue mb-2">Drop your angiogram here</h3>
          <p className="text-muted-teal mb-6">or browse from your computer</p>
          <div className="px-4 py-2 bg-white rounded-full border border-border shadow-sm text-sm text-charcoal-blue font-medium">
            PNG, JPG or DICOM • Single frame
          </div>
          
          <input 
            type="file" 
            ref={fileInputRef}
            onChange={handleFileChange}
            accept="image/png, image/jpeg, .jpg, .dcm, application/dicom"
            className="hidden"
          />
        </div>
      ) : (
        <div className="glass-card rounded-2xl p-8 mb-8 border border-white/50">
          <div className="flex justify-between items-start mb-6">
            <h3 className="text-lg font-medium text-charcoal-blue">File Selected</h3>
            <div className="flex gap-3">
              <button 
                onClick={() => fileInputRef.current?.click()}
                className="text-sm font-medium text-teal hover:text-charcoal-blue transition-colors px-3 py-1.5 bg-white rounded-lg border border-white/50 shadow-sm"
              >
                Change File
              </button>
              <button 
                onClick={handleRemove}
                className="text-sm font-medium text-red-500 hover:text-red-700 transition-colors px-3 py-1.5 bg-white rounded-lg border border-white/50 shadow-sm flex items-center gap-1"
              >
                <X className="w-4 h-4" />
                Remove
              </button>
            </div>
          </div>
          
          {fileType === 'image' && previewUrl ? (
            <div className="bg-charcoal-blue/5 rounded-xl border border-border p-4 flex gap-6">
              <div className="w-48 h-48 bg-black rounded-lg overflow-hidden shrink-0 flex items-center justify-center relative">
                <img 
                  src={previewUrl} 
                  alt="Angiogram preview" 
                  className={`w-full h-full object-contain ${isImageResolving ? 'reveal-animation' : ''}`}
                />
              </div>
              <div className="flex flex-col justify-center">
                <div className="flex items-center gap-2 mb-2">
                  <ImageIcon className="w-5 h-5 text-teal" />
                  <span className="font-semibold text-charcoal-blue text-lg truncate max-w-sm">{selectedFile.name}</span>
                </div>
                <div className="text-muted-teal text-sm space-y-1">
                  <p>Size: {formatFileSize(selectedFile.size)}</p>
                  <p>Type: {selectedFile.type || 'Image'}</p>
                </div>
                <div className="mt-4 inline-flex items-center gap-2 bg-teal/10 text-teal px-3 py-1.5 rounded-full text-sm font-medium">
                  <div className="w-2 h-2 rounded-full bg-teal"></div>
                  Ready for analysis
                </div>
              </div>
            </div>
          ) : (
            <div className="bg-charcoal-blue/5 rounded-xl border border-border p-8 flex items-center gap-6">
              <div className="w-20 h-24 bg-white border border-border rounded-lg shadow-sm flex flex-col items-center justify-center text-teal shrink-0">
                <File className="w-8 h-8 mb-2" />
                <span className="text-xs font-bold uppercase">DICOM</span>
              </div>
              <div>
                <h4 className="font-semibold text-charcoal-blue text-lg mb-1">{selectedFile.name}</h4>
                <p className="text-muted-teal text-sm mb-4">Size: {formatFileSize(selectedFile.size)}</p>
                <div className="inline-flex items-center gap-2 bg-teal/10 text-teal px-3 py-1.5 rounded-full text-sm font-medium">
                  <div className="w-2 h-2 rounded-full bg-teal animate-pulse"></div>
                  DICOM file ready for analysis
                </div>
              </div>
            </div>
          )}
        </div>
      )}
      
      {error && (
        <div className="mt-4 p-4 bg-red-50 text-red-600 rounded-xl border border-red-100 text-sm">
          {error}
        </div>
      )}
      
      <div className="mt-auto pt-8 flex justify-end">
        <button
          onClick={handleContinue}
          disabled={!selectedFile}
          className={`px-8 py-3 rounded-xl font-medium text-lg transition-all flex items-center gap-2 ${
            selectedFile 
              ? 'bg-teal text-white shadow-md hover:bg-charcoal-blue hover:shadow-lg' 
              : 'bg-muted-teal/20 text-muted-teal cursor-not-allowed'
          }`}
        >
          Continue to Point Selection
          <span>&rarr;</span>
        </button>
      </div>
    </div>
  );
}
