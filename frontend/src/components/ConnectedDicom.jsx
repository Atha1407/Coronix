import React, { useState, useCallback } from 'react';
import {
  Wifi,
  WifiOff,
  RefreshCw,
  FolderOpen,
  AlertCircle,
  Layers,
  ChevronRight,
  ChevronDown,
  Loader2,
  ServerCrash,
  PlugZap,
} from 'lucide-react';
import {
  getDicomConnectionStatus,
  getDicomStudies,
  getDicomSeries,
} from '../services/dicomApi';

/**
 * Connection status values:
 *   'not_connected' | 'connecting' | 'connected' | 'error'
 */

function StatusDot({ status }) {
  const map = {
    not_connected: 'bg-muted-teal/50',
    connecting:    'bg-amber-400 animate-pulse',
    connected:     'bg-emerald-500',
    error:         'bg-powder-blush',
  };
  return (
    <span className={`inline-block w-2.5 h-2.5 rounded-full shrink-0 ${map[status] || 'bg-muted-teal/50'}`} />
  );
}

function StatusLabel({ status }) {
  const labels = {
    not_connected: 'Not Connected',
    connecting:    'Connecting…',
    connected:     'Connected',
    error:         'Connection Error',
  };
  return <span>{labels[status] || 'Not Connected'}</span>;
}

// ---------------------------------------------------------------------------
// Study card
// ---------------------------------------------------------------------------
function StudyCard({ study, onOpen }) {
  const [expanded, setExpanded] = useState(false);
  const [series, setSeries] = useState(null);
  const [loadingSeries, setLoadingSeries] = useState(false);
  const [seriesError, setSeriesError] = useState(null);

  const handleExpand = async () => {
    if (expanded) { setExpanded(false); return; }
    setExpanded(true);
    if (series !== null) return;
    setLoadingSeries(true);
    setSeriesError(null);
    try {
      const data = await getDicomSeries(study.studyId);
      setSeries(data);
    } catch (e) {
      setSeriesError('Unable to load series.');
    } finally {
      setLoadingSeries(false);
    }
  };

  return (
    <div className="bg-white/70 border border-white/60 rounded-2xl overflow-hidden shadow-soft hover:shadow-md transition-shadow duration-200">
      {/* Study header */}
      <div className="p-5 flex items-start justify-between gap-4">
        <div className="flex items-start gap-3 min-w-0">
          <div className="w-10 h-10 rounded-xl bg-teal/10 border border-teal/20 flex items-center justify-center shrink-0 mt-0.5">
            <Layers className="w-5 h-5 text-teal" />
          </div>
          <div className="min-w-0">
            <p className="font-semibold text-charcoal-blue truncate">
              Study
            </p>
            <div className="text-xs text-muted-teal space-y-0.5 mt-1">
              {study.studyId  && <p>ID: <span className="font-mono">{study.studyId}</span></p>}
              {study.studyDate && <p>Date: {study.studyDate}</p>}
              {study.description && <p>{study.description}</p>}
            </div>
          </div>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          <button
            onClick={() => onOpen(study)}
            className="inline-flex items-center gap-1.5 bg-teal text-white text-xs font-semibold px-4 py-2 rounded-xl hover:bg-charcoal-blue transition-colors"
          >
            Open Study
            <ChevronRight className="w-3.5 h-3.5" />
          </button>
          <button
            onClick={handleExpand}
            className="p-2 rounded-xl hover:bg-teal/10 text-muted-teal hover:text-teal transition-colors"
            title="View series"
          >
            <ChevronDown className={`w-4 h-4 transition-transform duration-200 ${expanded ? 'rotate-180' : ''}`} />
          </button>
        </div>
      </div>

      {/* Series list */}
      {expanded && (
        <div className="border-t border-white/60 bg-bright-snow/40 px-5 py-4 space-y-2">
          {loadingSeries && (
            <div className="flex items-center gap-2 text-muted-teal text-sm">
              <Loader2 className="w-4 h-4 animate-spin" />
              Loading series…
            </div>
          )}
          {seriesError && (
            <p className="text-sm text-powder-blush">{seriesError}</p>
          )}
          {series && series.length === 0 && (
            <p className="text-sm text-muted-teal">No series in this study.</p>
          )}
          {series && series.map((s, i) => (
            <div
              key={s.seriesId || i}
              className="flex items-center justify-between bg-white/60 rounded-xl px-4 py-3 border border-white/50"
            >
              <div className="text-xs text-charcoal-blue space-y-0.5">
                {s.description && <p className="font-medium">{s.description}</p>}
                {s.seriesId && <p className="text-muted-teal font-mono">ID: {s.seriesId}</p>}
                {s.instanceCount != null && (
                  <p className="text-muted-teal">{s.instanceCount} instance{s.instanceCount !== 1 ? 's' : ''}</p>
                )}
              </div>
              <button
                onClick={() => onOpen(study, s)}
                className="text-xs text-teal font-semibold hover:text-charcoal-blue transition-colors px-3 py-1.5 bg-teal/10 rounded-lg"
              >
                Open
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------
export default function ConnectedDicom({ onStudySelected }) {
  const [connectionStatus, setConnectionStatus] = useState('not_connected');
  const [studies, setStudies] = useState(null);          // null = not yet fetched
  const [loadingStudies, setLoadingStudies] = useState(false);
  const [studiesError, setStudiesError] = useState(null);

  const handleRefresh = useCallback(async () => {
    setConnectionStatus('connecting');
    setStudies(null);
    setStudiesError(null);
    setLoadingStudies(false);

    try {
      await getDicomConnectionStatus();
      setConnectionStatus('connected');
      // Fetch studies once connected
      setLoadingStudies(true);
      try {
        const data = await getDicomStudies();
        setStudies(data);
      } catch (se) {
        setStudiesError('Unable to load studies.');
        setStudies([]);
      } finally {
        setLoadingStudies(false);
      }
    } catch (e) {
      // Expected: DICOM_NOT_CONNECTED — show not_connected, not a crash
      if (e.code === 'DICOM_NOT_CONNECTED') {
        setConnectionStatus('not_connected');
      } else {
        setConnectionStatus('error');
      }
    }
  }, []);

  const handleOpenStudy = (study, series) => {
    // Prepares the study/series for the viewer once the backend provides
    // the rendered image. The caller (Upload / App) will receive these and
    // can fetch the rendered frame via getDicomRenderedFrame().
    if (onStudySelected) {
      onStudySelected({ study, series: series || null });
    }
  };

  // ── render ───────────────────────────────────────────────────────────────
  return (
    <div className="space-y-5 animate-fade-in">
      {/* Source / status card */}
      <div className="bg-white/60 backdrop-blur-sm border border-white/60 rounded-2xl p-6 shadow-soft">
        <div className="flex items-start justify-between gap-4 flex-wrap">
          <div className="space-y-3">
            <div>
              <p className="text-xs text-muted-teal font-semibold uppercase tracking-widest mb-1">
                Source
              </p>
              <p className="text-charcoal-blue font-semibold flex items-center gap-2">
                <PlugZap className="w-4 h-4 text-teal" />
                Coronix DICOM Gateway
              </p>
            </div>
            <div>
              <p className="text-xs text-muted-teal font-semibold uppercase tracking-widest mb-1">
                Status
              </p>
              <p className="flex items-center gap-2 text-sm font-medium text-charcoal-blue">
                <StatusDot status={connectionStatus} />
                <StatusLabel status={connectionStatus} />
              </p>
            </div>
          </div>

          <button
            onClick={handleRefresh}
            disabled={connectionStatus === 'connecting'}
            className="inline-flex items-center gap-2 border border-teal/30 text-teal text-sm font-semibold px-5 py-2.5 rounded-xl hover:bg-teal/10 active:scale-95 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {connectionStatus === 'connecting'
              ? <Loader2 className="w-4 h-4 animate-spin" />
              : <RefreshCw className="w-4 h-4" />
            }
            Refresh Connection
          </button>
        </div>
      </div>

      {/* Studies area */}
      <div className="space-y-3">
        <p className="text-sm font-semibold text-charcoal-blue px-1">Available Studies</p>

        {/* Not yet tried */}
        {connectionStatus === 'not_connected' && studies === null && (
          <div className="bg-white/50 border border-white/60 rounded-2xl p-8 text-center space-y-3">
            <WifiOff className="w-10 h-10 text-muted-teal/60 mx-auto" />
            <p className="text-charcoal-blue font-medium">No studies available yet.</p>
            <p className="text-muted-teal text-sm max-w-xs mx-auto leading-relaxed">
              Connect a DICOM source through the Coronix backend to retrieve studies.
            </p>
          </div>
        )}

        {/* Connection error */}
        {connectionStatus === 'error' && (
          <div className="bg-powder-blush/10 border border-powder-blush/30 rounded-2xl p-6 flex gap-4">
            <ServerCrash className="w-6 h-6 text-powder-blush shrink-0 mt-0.5" />
            <div>
              <p className="text-charcoal-blue font-semibold text-sm mb-1">Unable to connect to the DICOM source.</p>
              <p className="text-muted-teal text-sm">
                Ensure the Coronix DICOM Gateway is running and try again.
              </p>
            </div>
          </div>
        )}

        {/* Connecting */}
        {connectionStatus === 'connecting' && (
          <div className="bg-white/50 border border-white/60 rounded-2xl p-8 text-center space-y-3">
            <Loader2 className="w-10 h-10 text-teal mx-auto animate-spin" />
            <p className="text-muted-teal text-sm">Connecting to DICOM gateway…</p>
          </div>
        )}

        {/* Loading studies */}
        {connectionStatus === 'connected' && loadingStudies && (
          <div className="bg-white/50 border border-white/60 rounded-2xl p-8 text-center space-y-3">
            <Loader2 className="w-10 h-10 text-teal mx-auto animate-spin" />
            <p className="text-muted-teal text-sm">Loading studies…</p>
          </div>
        )}

        {/* Studies error */}
        {studiesError && !loadingStudies && (
          <div className="bg-white/50 border border-white/60 rounded-2xl p-6 flex gap-3 items-start">
            <AlertCircle className="w-5 h-5 text-powder-blush shrink-0 mt-0.5" />
            <p className="text-sm text-muted-teal">{studiesError}</p>
          </div>
        )}

        {/* Empty studies list */}
        {connectionStatus === 'connected' && !loadingStudies && studies && studies.length === 0 && !studiesError && (
          <div className="bg-white/50 border border-white/60 rounded-2xl p-8 text-center space-y-3">
            <FolderOpen className="w-10 h-10 text-muted-teal/60 mx-auto" />
            <p className="text-charcoal-blue font-medium">No studies available.</p>
            <p className="text-muted-teal text-sm">The connected gateway returned no studies.</p>
          </div>
        )}

        {/* Study list */}
        {studies && studies.length > 0 && studies.map((study, i) => (
          <StudyCard
            key={study.studyId || i}
            study={study}
            onOpen={handleOpenStudy}
          />
        ))}
      </div>

      {/* Integration readiness note */}
      <div className="bg-teal/5 border border-teal/15 rounded-2xl p-5 flex gap-3">
        <Wifi className="w-5 h-5 text-teal shrink-0 mt-0.5" />
        <p className="text-muted-teal text-xs leading-relaxed">
          <span className="text-charcoal-blue font-semibold">Integration ready.</span>{' '}
          This frontend is prepared to retrieve studies from the Coronix DICOM Gateway
          once the backend is connected. No patient data is stored or transmitted by
          the frontend.
        </p>
      </div>
    </div>
  );
}