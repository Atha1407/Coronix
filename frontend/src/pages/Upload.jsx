import React, { useState, useRef } from 'react';
import {
  UploadCloud,
  File,
  Image as ImageIcon,
  X,
  Wifi,
  HardDriveDownload,
  ChevronLeft,
  ChevronRight,
  Layers,
} from 'lucide-react';
import StepProgress from '../components/StepProgress';
import ConnectedDicom from '../components/ConnectedDicom';
import { uploadDicom } from '../services/api';

// Source indicator — small pill shown once a file/study is ready
function SourceBadge({ source }) {
  const labels = {
    image:           'Uploaded Image',
    dicom_upload:    'Uploaded DICOM',
    dicom_connected: 'Connected DICOM',
  };
  return (
    <span className="inline-flex items-center gap-1.5 text-xs font-semibold text-teal bg-teal/10 border border-teal/20 px-3 py-1 rounded-full whitespace-nowrap shrink-0">
      <span className="w-1.5 h-1.5 rounded-full bg-teal" />
      Source: {labels[source] || source}
    </span>
  );
}

export default function Upload({ onUploadComplete }) {
  // 'choose' | 'upload_dicom' | 'upload_image' | 'connected'
  const [importMode, setImportMode] = useState('choose');

  // shared file state
  const [selectedFile, setSelectedFile] = useState(null);
  const [fileType, setFileType]         = useState(null);   // 'image' | 'dicom'
  const [fileSource, setFileSource]     = useState(null);   // 'image' | 'dicom_upload' | 'dicom_connected'
  const [previewUrl, setPreviewUrl]     = useState(null);
  const [isImageResolving, setIsImageResolving] = useState(false);
  const [isDicomLoading, setIsDicomLoading]     = useState(false);

  // DICOM multi-frame and metadata state
  const [dicomMetadata, setDicomMetadata]       = useState(null);
  const [totalFrames, setTotalFrames]           = useState(1);
  const [currentFrameIndex, setCurrentFrameIndex] = useState(0);

  const [error, setError]               = useState('');
  const [isHovering, setIsHovering]     = useState(false);

  // refs for hidden inputs
  const imageInputRef = useRef(null);
  const dicomInputRef = useRef(null);

  // ── helpers ──────────────────────────────────────────────────────────────
  const formatFileSize = (bytes) => {
    if (!bytes) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const clearFile = () => {
    setSelectedFile(null);
    setFileType(null);
    setFileSource(null);
    if (previewUrl && !previewUrl.startsWith('data:')) {
      URL.revokeObjectURL(previewUrl);
    }
    setPreviewUrl(null);
    setDicomMetadata(null);
    setTotalFrames(1);
    setCurrentFrameIndex(0);
    setError('');
    if (imageInputRef.current) imageInputRef.current.value = '';
    if (dicomInputRef.current) dicomInputRef.current.value = '';
  };

  // ── file processing ───────────────────────────────────────────────────────
  const processFile = async (file, sourceOverride) => {
    setError('');
    if (!file) return;

    const isImage = file.type.startsWith('image/png') ||
                    file.type.startsWith('image/jpeg') ||
                    file.name.toLowerCase().endsWith('.jpg');
    const isDicom = file.name.toLowerCase().endsWith('.dcm') ||
                    file.type === 'application/dicom';

    if (!isImage && !isDicom) {
      setError('Unsupported file format. Please upload a PNG, JPG, or DICOM (.dcm) file.');
      return;
    }

    const determinedSource = sourceOverride || (isImage ? 'image' : 'dicom_upload');
    setSelectedFile(file);
    setFileType(isImage ? 'image' : 'dicom');
    setFileSource(determinedSource);

    if (isImage) {
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
      setDicomMetadata(null);
      setTotalFrames(1);
      setCurrentFrameIndex(0);
      setIsImageResolving(true);
      setTimeout(() => setIsImageResolving(false), 800);
    } else {
      // Ingest DICOM via FastAPI backend
      setIsDicomLoading(true);
      try {
        const data = await uploadDicom(file, 0);
        setPreviewUrl(data.image_base64);
        setDicomMetadata(data.metadata);
        setTotalFrames(data.total_frames || 1);
        setCurrentFrameIndex(0);
      } catch (err) {
        console.error("DICOM ingestion error:", err);
        setError(`Failed to read DICOM file: ${err.message}`);
        setPreviewUrl(null);
      } finally {
        setIsDicomLoading(false);
      }
    }
  };

  const handleFrameChange = async (newIndex) => {
    if (!selectedFile || fileType !== 'dicom') return;
    if (newIndex < 0 || newIndex >= totalFrames) return;

    setIsDicomLoading(true);
    try {
      const data = await uploadDicom(selectedFile, newIndex);
      setPreviewUrl(data.image_base64);
      setCurrentFrameIndex(newIndex);
    } catch (err) {
      setError(`Failed to extract frame ${newIndex}: ${err.message}`);
    } finally {
      setIsDicomLoading(false);
    }
  };

  const handleImageChange = (e) => e.target.files?.[0] && processFile(e.target.files[0], 'image');
  const handleDicomChange = (e) => e.target.files?.[0] && processFile(e.target.files[0], 'dicom_upload');

  const handleDrop = (e) => {
    e.preventDefault();
    setIsHovering(false);
    if (e.dataTransfer.files?.[0]) {
      processFile(e.dataTransfer.files[0]);
    }
  };

  // ── connected DICOM study selected ────────────────────────────────────────
  const handleStudySelected = ({ study, series }) => {
    setFileType('dicom');
    setFileSource('dicom_connected');
    setSelectedFile(null);
    setPreviewUrl(null);
    setError(
      'Study selected. The image will be retrieved from the Coronix DICOM Gateway ' +
      'once the gateway backend is connected. You can still press Continue to proceed in demo mode.'
    );
  };

  // ── continue ─────────────────────────────────────────────────────────────
  const handleContinue = () => {
    if (selectedFile) {
      onUploadComplete(
        selectedFile,
        fileType,
        previewUrl,
        fileSource,
        currentFrameIndex,
        dicomMetadata
      );
    } else if (fileSource === 'dicom_connected') {
      onUploadComplete(
        null,
        'dicom',
        previewUrl,
        fileSource,
        currentFrameIndex,
        dicomMetadata
      );
    }
  };

  const canContinue = (selectedFile && previewUrl && !isDicomLoading) || (fileSource === 'dicom_connected');

  // ── drag helpers ──────────────────────────────────────────────────────────
  const handleDragOver  = (e) => { e.preventDefault(); setIsHovering(true); };
  const handleDragLeave = ()  => setIsHovering(false);

  // =========================================================================
  // RENDER
  // =========================================================================

  // ── Mode: choose ──────────────────────────────────────────────────────────
  const renderChoose = () => (
    <div className="space-y-4 animate-fade-in">
      <div className="grid sm:grid-cols-2 gap-4">
        {/* Upload DICOM */}
        <button
          onClick={() => setImportMode('upload_dicom')}
          className="group flex flex-col items-center gap-4 bg-white/60 backdrop-blur-sm border-2 border-white/60 hover:border-teal/40 hover:bg-teal/5 rounded-2xl p-8 text-center transition-all duration-200 shadow-soft hover:shadow-md"
        >
          <div className="w-14 h-14 rounded-2xl bg-teal/10 border border-teal/20 flex items-center justify-center group-hover:bg-teal/20 transition-colors">
            <HardDriveDownload className="w-7 h-7 text-teal" />
          </div>
          <div>
            <p className="font-semibold text-charcoal-blue text-base">Upload DICOM / Image</p>
            <p className="text-muted-teal text-sm mt-1">
              Import a .dcm angiography run or image from your computer.
            </p>
          </div>
          <span className="text-xs text-muted-teal border border-muted-teal/30 rounded-full px-3 py-1">
            .dcm · PNG · JPG · Single & Multi-frame
          </span>
        </button>

        {/* Connected DICOM */}
        <button
          onClick={() => setImportMode('connected')}
          className="group flex flex-col items-center gap-4 bg-white/60 backdrop-blur-sm border-2 border-white/60 hover:border-teal/40 hover:bg-teal/5 rounded-2xl p-8 text-center transition-all duration-200 shadow-soft hover:shadow-md"
        >
          <div className="w-14 h-14 rounded-2xl bg-teal/10 border border-teal/20 flex items-center justify-center group-hover:bg-teal/20 transition-colors">
            <Wifi className="w-7 h-7 text-teal" />
          </div>
          <div>
            <p className="font-semibold text-charcoal-blue text-base">Connected DICOM</p>
            <p className="text-muted-teal text-sm mt-1">
              Retrieve a study from a connected PACS or DICOM gateway.
            </p>
          </div>
          <span className="text-xs text-muted-teal border border-muted-teal/30 rounded-full px-3 py-1">
            DICOM Gateway
          </span>
        </button>
      </div>
    </div>
  );

  // ── Mode: upload_dicom ────────────────────────────────────────────────────
  const renderUploadDicom = () => (
    <div className="space-y-5 animate-fade-in">
      <button
        onClick={() => { clearFile(); setImportMode('choose'); }}
        className="text-sm text-muted-teal hover:text-charcoal-blue transition-colors flex items-center gap-1 cursor-pointer"
      >
        ← Back
      </button>

      {!selectedFile ? (
        <div
          className={`glass-card rounded-2xl p-12 flex flex-col items-center justify-center text-center border-2 border-dashed transition-colors cursor-pointer ${
            isHovering
              ? 'border-teal bg-teal/5'
              : 'border-muted-teal/30 hover:border-teal/50 hover:bg-white/60'
          }`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => dicomInputRef.current?.click()}
        >
          <div className="w-16 h-16 bg-white rounded-full shadow-sm flex items-center justify-center mb-6 text-teal">
            <UploadCloud className="w-8 h-8" />
          </div>
          <h3 className="text-xl font-medium text-charcoal-blue mb-2">Drop your angiogram here</h3>
          <p className="text-muted-teal mb-6">or browse from your computer</p>
          <div className="px-4 py-2 bg-white rounded-full border border-white/60 shadow-sm text-sm text-charcoal-blue font-medium">
            DICOM (.dcm) or PNG, JPG • Single & Multi-frame supported
          </div>
        </div>
      ) : (
        <div className="glass-card rounded-2xl p-8 border border-white/50">
          <div className="flex justify-between items-start mb-6">
            <div className="flex items-center gap-3">
              <h3 className="text-lg font-medium text-charcoal-blue">File Selected</h3>
              <SourceBadge source={fileSource} />
            </div>
            <div className="flex gap-3">
              <button
                onClick={() => dicomInputRef.current?.click()}
                className="text-sm font-medium text-teal hover:text-charcoal-blue transition-colors px-3 py-1.5 bg-white rounded-lg border border-white/50 shadow-sm cursor-pointer"
              >
                Change File
              </button>
              <button
                onClick={clearFile}
                className="text-sm font-medium text-red-500 hover:text-red-700 transition-colors px-3 py-1.5 bg-white rounded-lg border border-white/50 shadow-sm flex items-center gap-1 cursor-pointer"
              >
                <X className="w-4 h-4" />
                Remove
              </button>
            </div>
          </div>

          {/* Preview Card */}
          <div className="bg-charcoal-blue/5 rounded-xl border border-white/50 p-6 flex flex-col md:flex-row gap-6">
            {/* Image Preview Box */}
            <div className="w-56 h-56 bg-black rounded-lg overflow-hidden shrink-0 flex items-center justify-center relative border border-black/10">
              {isDicomLoading ? (
                <div className="flex flex-col items-center justify-center text-teal gap-2">
                  <div className="w-8 h-8 border-3 border-teal/30 border-t-teal rounded-full animate-spin" />
                  <span className="text-xs text-white/80">Decoding Frame...</span>
                </div>
              ) : previewUrl ? (
                <img
                  src={previewUrl}
                  alt="Angiogram preview"
                  className={`w-full h-full object-contain ${isImageResolving ? 'reveal-animation' : ''}`}
                />
              ) : (
                <div className="flex flex-col items-center justify-center text-muted-teal p-4 text-center">
                  <File className="w-10 h-10 text-muted-teal/60 mb-2" />
                  <span className="text-xs">No preview available</span>
                </div>
              )}
            </div>

            {/* Metadata & Controls */}
            <div className="flex flex-col justify-between flex-1 min-w-0">
              <div>
                <div className="flex items-center gap-2 mb-2">
                  {fileType === 'dicom' ? (
                    <File className="w-5 h-5 text-teal shrink-0" />
                  ) : (
                    <ImageIcon className="w-5 h-5 text-teal shrink-0" />
                  )}
                  <span className="font-semibold text-charcoal-blue text-lg truncate max-w-md">
                    {selectedFile.name}
                  </span>
                </div>

                <div className="text-muted-teal text-sm space-y-1 mb-4">
                  <p>File Size: {formatFileSize(selectedFile.size)}</p>
                  <p>Format: {fileType === 'dicom' ? 'DICOM Digital Angiography' : (selectedFile.type || 'Standard Image')}</p>
                  {dicomMetadata && (
                    <div className="mt-3 grid grid-cols-2 gap-2 text-xs bg-white/60 p-3 rounded-lg border border-border/60">
                      <div><strong className="text-charcoal-blue">Modality:</strong> {dicomMetadata.Modality || 'XA'}</div>
                      <div><strong className="text-charcoal-blue">Matrix:</strong> {dicomMetadata.Rows || '—'} × {dicomMetadata.Columns || '—'}</div>
                      <div><strong className="text-charcoal-blue">Total Frames:</strong> {totalFrames}</div>
                      <div><strong className="text-charcoal-blue">Study:</strong> {dicomMetadata.StudyDescription || 'Coronary Angiogram'}</div>
                    </div>
                  )}
                </div>
              </div>

              {/* Multi-frame DICOM Frame Selector */}
              {fileType === 'dicom' && totalFrames > 1 && (
                <div className="bg-white/80 p-3 rounded-xl border border-teal/20 shadow-xs mb-3 flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Layers className="w-4 h-4 text-teal" />
                    <span className="text-xs font-semibold text-charcoal-blue">
                      Frame {currentFrameIndex + 1} of {totalFrames}
                    </span>
                  </div>
                  <div className="flex items-center gap-2">
                    <button
                      type="button"
                      disabled={currentFrameIndex <= 0 || isDicomLoading}
                      onClick={() => handleFrameChange(currentFrameIndex - 1)}
                      className="p-1 rounded-md bg-white border border-border text-charcoal-blue hover:bg-teal/10 disabled:opacity-40 disabled:hover:bg-white cursor-pointer"
                      title="Previous Frame"
                    >
                      <ChevronLeft className="w-4 h-4" />
                    </button>
                    <span className="font-mono text-xs text-charcoal-blue px-2 font-medium">
                      idx: {currentFrameIndex}
                    </span>
                    <button
                      type="button"
                      disabled={currentFrameIndex >= totalFrames - 1 || isDicomLoading}
                      onClick={() => handleFrameChange(currentFrameIndex + 1)}
                      className="p-1 rounded-md bg-white border border-border text-charcoal-blue hover:bg-teal/10 disabled:opacity-40 disabled:hover:bg-white cursor-pointer"
                      title="Next Frame"
                    >
                      <ChevronRight className="w-4 h-4" />
                    </button>
                  </div>
                </div>
              )}

              <div className="inline-flex items-center gap-2 bg-teal/10 text-teal px-3 py-1.5 rounded-full text-sm font-medium w-fit">
                <div className="w-2 h-2 rounded-full bg-teal" />
                {previewUrl ? 'Ready for Point Selection' : 'Processing file...'}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );

  // ── Mode: connected ───────────────────────────────────────────────────────
  const renderConnected = () => (
    <div className="space-y-5 animate-fade-in">
      <button
        onClick={() => { clearFile(); setImportMode('choose'); }}
        className="text-sm text-muted-teal hover:text-charcoal-blue transition-colors flex items-center gap-1 cursor-pointer"
      >
        ← Back
      </button>
      <ConnectedDicom onStudySelected={handleStudySelected} />
    </div>
  );

  // ── layout ────────────────────────────────────────────────────────────────
  return (
    <div className="max-w-4xl mx-auto flex flex-col min-h-full pb-12">
      <StepProgress currentStep={1} />

      <div className="mb-8">
        <h1 className="text-3xl font-semibold text-charcoal-blue mb-2">Import Study</h1>
        <p className="text-muted-teal text-lg">
          Upload a DICOM study or retrieve one from a connected imaging source.
        </p>
      </div>

      {importMode === 'choose'       && renderChoose()}
      {importMode === 'upload_dicom' && renderUploadDicom()}
      {importMode === 'connected'    && renderConnected()}

      {error && (
        <div className="mt-4 p-4 rounded-xl border border-red-500/20 bg-red-500/10 text-red-700 text-sm">
          {error}
        </div>
      )}

      {/* Action footer */}
      {importMode !== 'choose' && (
        <div className="mt-auto pt-8 pb-8 flex justify-end">
          <button
            onClick={handleContinue}
            disabled={!canContinue}
            className={`px-8 py-3 rounded-xl font-medium text-lg transition-all flex items-center gap-2 ${
              canContinue
                ? 'bg-teal text-white hover:bg-charcoal-blue shadow-lg hover:shadow-xl cursor-pointer'
                : 'bg-charcoal-blue/20 text-charcoal-blue/40 cursor-not-allowed'
            }`}
          >
            Continue to Point Selection
            <span>&rarr;</span>
          </button>
        </div>
      )}

      {/* Hidden inputs */}
      <input
        type="file"
        ref={dicomInputRef}
        onChange={handleDicomChange}
        accept=".dcm, application/dicom, image/png, image/jpeg, .jpg"
        className="hidden"
      />
      <input
        type="file"
        ref={imageInputRef}
        onChange={handleImageChange}
        accept="image/png, image/jpeg, .jpg"
        className="hidden"
      />
    </div>
  );
}