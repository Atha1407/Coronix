import React, { useState, useRef } from 'react';
import {
  UploadCloud,
  File,
  Image as ImageIcon,
  X,
  Wifi,
  HardDriveDownload,
} from 'lucide-react';
import StepProgress from '../components/StepProgress';
import ConnectedDicom from '../components/ConnectedDicom';

// Source indicator — small pill shown once a file/study is ready
function SourceBadge({ source }) {
  const labels = {
    image:           'Uploaded Image',
    dicom_upload:    'Uploaded DICOM',
    dicom_connected: 'Connected DICOM',
  };
  return (
    <span className="inline-flex items-center gap-1.5 text-xs font-semibold text-teal bg-teal/10 border border-teal/20 px-3 py-1 rounded-full">
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

  const [error, setError]         = useState('');
  const [isHovering, setIsHovering] = useState(false);

  // refs for the two hidden inputs
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
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(null);
    setError('');
    if (imageInputRef.current) imageInputRef.current.value = '';
    if (dicomInputRef.current) dicomInputRef.current.value = '';
  };

  // ── file processing ───────────────────────────────────────────────────────
  const processFile = (file, sourceOverride) => {
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

    setSelectedFile(file);
    setFileType(isImage ? 'image' : 'dicom');
    setFileSource(sourceOverride || (isImage ? 'image' : 'dicom_upload'));

    if (isImage) {
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
      setIsImageResolving(true);
      setTimeout(() => setIsImageResolving(false), 1000);
    } else {
      setPreviewUrl(null);
    }
  };

  const handleImageChange  = (e) => e.target.files?.[0] && processFile(e.target.files[0]);
  const handleDicomChange  = (e) => e.target.files?.[0] && processFile(e.target.files[0]);

  const handleDrop = (e) => {
    e.preventDefault();
    setIsHovering(false);
    e.dataTransfer.files?.[0] && processFile(e.dataTransfer.files[0]);
  };

  // ── connected DICOM study selected ────────────────────────────────────────
  // Called when the user picks a study from the gateway panel.
  // The file object will remain null until the backend is connected and
  // getDicomRenderedFrame() is called to produce an object URL.
  const handleStudySelected = ({ study, series }) => {
    setFileType('dicom');
    setFileSource('dicom_connected');
    setSelectedFile(null);   // real file will come from the gateway later
    setPreviewUrl(null);
    setError('');
    // Store study ref so the gateway call can be made when Continue is pressed
    // (For now we surface a clear "backend not connected" message.)
    setError(
      'Study selected. The image will be retrieved from the Coronix DICOM Gateway ' +
      'once the backend is connected. You can still press Continue to proceed ' +
      'in demo mode.'
    );
  };

  // ── continue ─────────────────────────────────────────────────────────────
  const handleContinue = () => {
    if (selectedFile) {
      onUploadComplete(selectedFile, fileType, previewUrl, fileSource);
    }
    // Connected DICOM: when gateway is live, fetch frame and call:
    // onUploadComplete(null, 'dicom', renderedObjectUrl, 'dicom_connected');
  };

  const canContinue = !!selectedFile;

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
            <p className="font-semibold text-charcoal-blue text-base">Upload DICOM</p>
            <p className="text-muted-teal text-sm mt-1">
              Import a .dcm file or angiogram image from your computer.
            </p>
          </div>
          <span className="text-xs text-muted-teal border border-muted-teal/30 rounded-full px-3 py-1">
            .dcm · PNG · JPG
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

  // ── Mode: upload_dicom (covers both .dcm and image drag/drop) ─────────────
  const renderUploadDicom = () => (
    <div className="space-y-5 animate-fade-in">
      <button
        onClick={() => { clearFile(); setImportMode('choose'); }}
        className="text-sm text-muted-teal hover:text-charcoal-blue transition-colors flex items-center gap-1"
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
          <h3 className="text-xl font-medium text-charcoal-blue mb-2">Drop your file here</h3>
          <p className="text-muted-teal mb-6">or browse from your computer</p>
          <div className="px-4 py-2 bg-white rounded-full border border-white/60 shadow-sm text-sm text-charcoal-blue font-medium">
            PNG · JPG · DICOM (.dcm) · Single frame
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
                className="text-sm font-medium text-teal hover:text-charcoal-blue transition-colors px-3 py-1.5 bg-white rounded-lg border border-white/50 shadow-sm"
              >
                Change File
              </button>
              <button
                onClick={clearFile}
                className="text-sm font-medium text-powder-blush hover:text-charcoal-blue transition-colors px-3 py-1.5 bg-white rounded-lg border border-white/50 shadow-sm flex items-center gap-1"
              >
                <X className="w-4 h-4" />
                Remove
              </button>
            </div>
          </div>

          {fileType === 'image' && previewUrl ? (
            <div className="bg-charcoal-blue/5 rounded-xl border border-white/50 p-4 flex gap-6">
              <div className="w-48 h-48 bg-black rounded-lg overflow-hidden shrink-0 flex items-center justify-center">
                <img
                  src={previewUrl}
                  alt="Angiogram preview"
                  className="w-full h-full object-contain"
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
                  <div className="w-2 h-2 rounded-full bg-teal" />
                  Ready for analysis
                </div>
              </div>
            </div>
          ) : (
            <div className="bg-charcoal-blue/5 rounded-xl border border-white/50 p-8 flex items-center gap-6">
              <div className="w-20 h-24 bg-white border border-white/60 rounded-lg shadow-sm flex flex-col items-center justify-center text-teal shrink-0">
                <File className="w-8 h-8 mb-2" />
                <span className="text-xs font-bold uppercase">DICOM</span>
              </div>
              <div>
                <h4 className="font-semibold text-charcoal-blue text-lg mb-1">{selectedFile.name}</h4>
                <p className="text-muted-teal text-sm mb-4">Size: {formatFileSize(selectedFile.size)}</p>
                <div className="inline-flex items-center gap-2 bg-teal/10 text-teal px-3 py-1.5 rounded-full text-sm font-medium">
                  <div className="w-2 h-2 rounded-full bg-teal animate-pulse" />
                  DICOM file ready for analysis
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );

  // ── Mode: connected ───────────────────────────────────────────────────────
  const renderConnected = () => (
    <div className="space-y-5 animate-fade-in">
      <button
        onClick={() => { clearFile(); setImportMode('choose'); }}
        className="text-sm text-muted-teal hover:text-charcoal-blue transition-colors flex items-center gap-1"
      >
        ← Back
      </button>
      <ConnectedDicom onStudySelected={handleStudySelected} />
    </div>
  );

  // ── layout ────────────────────────────────────────────────────────────────
  return (
    <div className="max-w-4xl mx-auto flex flex-col h-full">
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
        <div className="mt-auto pt-8 flex justify-end">
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
        accept="image/png, image/jpeg, .jpg, .dcm, application/dicom"
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