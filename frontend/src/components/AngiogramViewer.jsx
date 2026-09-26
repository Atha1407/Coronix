import React, { useState, useRef, useEffect } from 'react';
import { Move, ZoomIn, Sliders, Maximize, Activity } from 'lucide-react';
import AnalysisOverlay from './AnalysisOverlay';

const distanceToSegment = (p, v, w) => {
  const l2 = (v.x - w.x)**2 + (v.y - w.y)**2;
  if (l2 === 0) return Math.hypot(p.x - v.x, p.y - v.y);
  let t = ((p.x - v.x) * (w.x - v.x) + (p.y - v.y) * (w.y - v.y)) / l2;
  t = Math.max(0, Math.min(1, t));
  const proj = { x: v.x + t * (w.x - v.x), y: v.y + t * (w.y - v.y) };
  return Math.hypot(p.x - proj.x, p.y - proj.y);
};

export default function AngiogramViewer({
  fileType,
  imageUrl,
  pointA,
  pointB,
  analysisResult = null,
  isResultsMode = false,
  onPointClick = () => {}
}) {
  const defaultTool = isResultsMode ? 'inspect' : 'select';
  const [activeTool, setActiveTool] = useState(defaultTool);
  
  const [naturalDimensions, setNaturalDimensions] = useState(null);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [zoom, setZoom] = useState(1);
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  
  const [showAdjustments, setShowAdjustments] = useState(false);
  const [adjustments, setAdjustments] = useState({ brightness: 100, contrast: 100 });
  
  const [lensState, setLensState] = useState({ active: false, screenX: 0, screenY: 0, origX: 0, origY: 0 });

  const svgRef = useRef(null);
  const viewerContainerRef = useRef(null);
  const wrapperRef = useRef(null); 
  const activeToolRef = useRef(activeTool);

  useEffect(() => {
    activeToolRef.current = activeTool;
  }, [activeTool]);

  useEffect(() => {
    const container = viewerContainerRef.current;
    if (!container) return;

    const handleNativeWheel = (e) => {
      if (activeToolRef.current === 'zoom') {
        e.preventDefault(); 
        
        const rect = container.getBoundingClientRect();
        const cx = rect.width / 2;
        const cy = rect.height / 2;
        
        const dx = (e.clientX - rect.left) - cx;
        const dy = (e.clientY - rect.top) - cy;

        setZoom(prevZoom => {
          const zoomDelta = e.deltaY < 0 ? 0.15 : -0.15;
          const newZoom = Math.min(Math.max(1, prevZoom + zoomDelta), 4);
          
          if (newZoom !== prevZoom) {
            setPan(prevPan => {
              const newPanX = prevPan.x + ((dx - prevPan.x) / prevZoom) * (prevZoom - newZoom);
              const newPanY = prevPan.y + ((dy - prevPan.y) / prevZoom) * (prevZoom - newZoom);
              return { x: newPanX, y: newPanY };
            });
          }
          return newZoom;
        });
      }
    };

    container.addEventListener('wheel', handleNativeWheel, { passive: false });
    return () => container.removeEventListener('wheel', handleNativeWheel);
  }, []);

  const handleImageLoad = (e) => {
    setNaturalDimensions({
      width: e.target.naturalWidth,
      height: e.target.naturalHeight
    });
  };

  const getOriginalCoordsFromScreen = (clientX, clientY) => {
    if (!svgRef.current) return null;
    const svg = svgRef.current;
    const pt = svg.createSVGPoint();
    pt.x = clientX;
    pt.y = clientY;
    return pt.matrixTransform(svg.getScreenCTM().inverse());
  };

  const handleSvgClick = (e) => {
    if (activeTool !== 'select') return;
    if (!naturalDimensions) return;

    const svgP = getOriginalCoordsFromScreen(e.clientX, e.clientY);
    if (!svgP) return;
    
    const x = Math.round(svgP.x);
    const y = Math.round(svgP.y);
    
    if (x >= 0 && x <= naturalDimensions.width && y >= 0 && y <= naturalDimensions.height) {
      onPointClick({ x, y });
    }
  };

  const handlePointerDown = (e) => {
    if (activeTool === 'pan') {
      setIsDragging(true);
      setDragStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
      e.currentTarget.setPointerCapture(e.pointerId); 
    }
  };

  const handlePointerMove = (e) => {
    if (isDragging && activeTool === 'pan') {
      setPan({
        x: e.clientX - dragStart.x,
        y: e.clientY - dragStart.y
      });
      if (lensState.active) setLensState(prev => ({ ...prev, active: false }));
      return;
    }

    // Magnifier Logic (ONLY active in Results mode and inspect tool)
    if (isResultsMode && activeTool === 'inspect' && pointA && pointB && naturalDimensions && viewerContainerRef.current) {
      const svgP = getOriginalCoordsFromScreen(e.clientX, e.clientY);
      if (svgP) {
        let dist = Infinity;
        if (analysisResult && analysisResult.lesion_bbox) {
          const [bx, by, bw, bh] = analysisResult.lesion_bbox;
          // Distance to rect is 0 if inside
          if (svgP.x >= bx && svgP.x <= bx + bw && svgP.y >= by && svgP.y <= by + bh) {
            dist = 0;
          } else {
            // roughly check if within 40px of bbox
            dist = Math.max(0, Math.min(Math.abs(svgP.x - bx), Math.abs(svgP.x - (bx + bw)))) + 
                   Math.max(0, Math.min(Math.abs(svgP.y - by), Math.abs(svgP.y - (by + bh))));
          }
        } else {
          dist = distanceToSegment(svgP, pointA, pointB);
        }

        if (dist <= 40) {
          const rect = viewerContainerRef.current.getBoundingClientRect();
          setLensState({
            active: true,
            screenX: e.clientX - rect.left,
            screenY: e.clientY - rect.top,
            origX: svgP.x,
            origY: svgP.y
          });
          return;
        }
      }
    }
    
    if (lensState.active) setLensState(prev => ({ ...prev, active: false }));
  };

  const handlePointerUp = (e) => {
    if (isDragging) {
      setIsDragging(false);
      if (e.currentTarget.hasPointerCapture(e.pointerId)) {
        e.currentTarget.releasePointerCapture(e.pointerId);
      }
    }
  };

  const handlePointerLeave = () => {
    if (lensState.active) setLensState(prev => ({ ...prev, active: false }));
    handlePointerUp();
  };

  const toggleFullscreen = () => {
    if (!document.fullscreenElement) {
      wrapperRef.current?.requestFullscreen().catch(err => {
        console.error(`Error attempting to enable fullscreen: ${err.message}`);
      });
    } else {
      document.exitFullscreen();
    }
  };

  const getMagnifierStyle = () => {
    if (!viewerContainerRef.current) return {};
    const rect = viewerContainerRef.current.getBoundingClientRect();
    
    const LENS_SIZE = 180;
    const OFFSET = 20;
    
    let left = lensState.screenX + OFFSET;
    let top = lensState.screenY + OFFSET;
    
    if (left + LENS_SIZE > rect.width) left = lensState.screenX - OFFSET - LENS_SIZE;
    if (top + LENS_SIZE > rect.height) top = lensState.screenY - OFFSET - LENS_SIZE;
    
    if (left < 0) left = 0;
    if (top < 0) top = 0;

    return {
      left,
      top,
      width: LENS_SIZE,
      height: LENS_SIZE,
    };
  };

  const getCursor = () => {
    if (activeTool === 'pan') return isDragging ? 'cursor-grabbing' : 'cursor-grab';
    if (activeTool === 'zoom') return 'cursor-zoom-in';
    if (activeTool === 'select') return 'cursor-crosshair';
    return 'cursor-default'; // inspect mode
  };

  const lensSize = 180;
  const lensRadius = lensSize / 2;
  const magScale = 2.5;

  const displayImage = isResultsMode && analysisResult?.overlay_image_base64 
    ? `data:image/png;base64,${analysisResult.overlay_image_base64}` 
    : imageUrl;

  return (
    <div className="flex-[7] flex flex-col min-w-0 min-h-[500px]" ref={wrapperRef}>
      <div className="glass-card rounded-2xl p-2 flex flex-col h-full border border-white/50 bg-bright-snow">
        {/* Toolbar */}
        <div className="flex items-center gap-2 px-4 py-2 border-b border-border/50 mb-2 bg-white/50 rounded-t-xl shrink-0">
          <button 
            onClick={() => setActiveTool(activeTool === 'pan' ? defaultTool : 'pan')}
            className={`p-2 rounded-lg transition-colors ${activeTool === 'pan' ? 'text-teal bg-teal/10' : 'text-muted-teal hover:text-charcoal-blue hover:bg-black/5'}`} 
            title="Pan"
          >
            <Move className="w-5 h-5" />
          </button>
          <button 
            onClick={() => setActiveTool(activeTool === 'zoom' ? defaultTool : 'zoom')}
            className={`p-2 rounded-lg transition-colors ${activeTool === 'zoom' ? 'text-teal bg-teal/10' : 'text-muted-teal hover:text-charcoal-blue hover:bg-black/5'}`}
            title="Zoom"
          >
            <ZoomIn className="w-5 h-5" />
          </button>
          <button 
            onClick={() => setShowAdjustments(!showAdjustments)}
            className={`p-2 rounded-lg transition-colors ${showAdjustments ? 'text-teal bg-teal/10' : 'text-muted-teal hover:text-charcoal-blue hover:bg-black/5'}`}
            title="Image Adjustments"
          >
            <Sliders className="w-5 h-5" />
          </button>
          <div className="flex-1"></div>
          <button 
            onClick={toggleFullscreen}
            className="p-2 text-muted-teal hover:text-charcoal-blue hover:bg-black/5 rounded-lg transition-colors" 
            title="Fullscreen"
          >
            <Maximize className="w-5 h-5" />
          </button>
        </div>
        
        {/* Image Area */}
        <div 
          className={`flex-1 bg-black/5 rounded-xl overflow-hidden relative border border-border flex items-center justify-center select-none ${getCursor()}`}
          ref={viewerContainerRef}
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={handlePointerUp}
          onPointerCancel={handlePointerLeave}
          onPointerLeave={handlePointerLeave}
          style={{
            touchAction: (activeTool === 'pan' || activeTool === 'zoom') ? 'none' : 'auto',
            WebkitUserSelect: 'none',
            userSelect: 'none'
          }}
        >
          {showAdjustments && (
            <div 
              className="absolute top-4 left-4 z-20 glass-card p-4 rounded-xl border border-white/50 shadow-lg w-64 bg-white/90 backdrop-blur-md cursor-default"
              onPointerDown={e => e.stopPropagation()}
              onWheel={e => e.stopPropagation()}
            >
              <h4 className="text-sm font-semibold text-charcoal-blue mb-3">Display Adjustments</h4>
              <div className="space-y-4">
                <div>
                  <label className="text-xs font-medium text-muted-teal flex justify-between mb-1">
                    <span>Brightness</span>
                    <span>{adjustments.brightness}%</span>
                  </label>
                  <input 
                    type="range" min="0" max="200" 
                    value={adjustments.brightness} 
                    onChange={(e) => setAdjustments({...adjustments, brightness: Number(e.target.value)})}
                    className="w-full accent-teal"
                  />
                </div>
                <div>
                  <label className="text-xs font-medium text-muted-teal flex justify-between mb-1">
                    <span>Contrast</span>
                    <span>{adjustments.contrast}%</span>
                  </label>
                  <input 
                    type="range" min="0" max="200" 
                    value={adjustments.contrast} 
                    onChange={(e) => setAdjustments({...adjustments, contrast: Number(e.target.value)})}
                    className="w-full accent-teal"
                  />
                </div>
              </div>
              <button 
                onClick={() => setAdjustments({ brightness: 100, contrast: 100 })}
                className="mt-5 w-full py-2 text-xs font-medium text-teal border border-teal/30 rounded-lg hover:bg-teal/5 transition-colors"
              >
                Reset Adjustments
              </button>
            </div>
          )}

          {/* Magnifying Lens Overlay (ONLY IN RESULTS) */}
          {isResultsMode && lensState.active && !isDragging && activeTool === 'inspect' && naturalDimensions && (
            <svg 
              className="absolute pointer-events-none z-30 drop-shadow-xl"
              style={getMagnifierStyle()}
            >
              <defs>
                <clipPath id="lens-clip">
                  <circle cx={lensRadius} cy={lensRadius} r={lensRadius - 2} />
                </clipPath>
              </defs>
              
              <circle cx={lensRadius} cy={lensRadius} r={lensRadius - 2} fill="#111" stroke="var(--color-teal)" strokeWidth="3" />
              
              <g clipPath="url(#lens-clip)">
                <g transform={`translate(${lensRadius}, ${lensRadius}) scale(${magScale}) translate(${-lensState.origX}, ${-lensState.origY})`}>
                  <image 
                    href={displayImage} 
                    width={naturalDimensions.width} 
                    height={naturalDimensions.height} 
                    style={{ filter: `brightness(${adjustments.brightness}%) contrast(${adjustments.contrast}%)` }}
                  />
                  <AnalysisOverlay 
                    pointA={pointA} 
                    pointB={pointB} 
                    isMagnifier={true}
                    analysisResult={analysisResult} 
                  />
                </g>
              </g>
              
              <circle cx={lensRadius} cy={lensRadius} r={lensRadius - 4} fill="none" stroke="white" strokeWidth="1" opacity="0.3"/>
            </svg>
          )}

          {fileType === 'dicom' && !displayImage ? (
            <div className="text-center p-8">
              <Activity className="w-12 h-12 text-teal mx-auto mb-4" />
              <h3 className="text-xl font-medium text-charcoal-blue mb-2">DICOM File Accepted</h3>
              <p className="text-muted-teal max-w-md">
                Backend processing is required to render DICOM frames. 
                In this phase, point selection is only available for direct image uploads (PNG/JPG).
              </p>
            </div>
          ) : displayImage ? (
            <div 
              className="relative w-full h-full flex items-center justify-center transition-transform"
              style={{ 
                transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
                transitionDuration: isDragging || activeTool === 'zoom' ? '0ms' : '200ms'
              }}
            >
              <img 
                src={displayImage} 
                alt="Angiogram for analysis" 
                className="absolute inset-0 w-full h-full object-contain pointer-events-none"
                draggable={false}
                style={{ 
                  filter: `brightness(${adjustments.brightness}%) contrast(${adjustments.contrast}%)`,
                  WebkitUserDrag: 'none'
                }}
                onLoad={handleImageLoad}
              />
              {naturalDimensions && (
                <svg 
                  ref={svgRef}
                  className="absolute inset-0 w-full h-full z-10"
                  viewBox={`0 0 ${naturalDimensions.width} ${naturalDimensions.height}`}
                  preserveAspectRatio="xMidYMid meet"
                  onClick={handleSvgClick}
                >
                  <AnalysisOverlay 
                    pointA={pointA} 
                    pointB={pointB} 
                    isMagnifier={false}
                    analysisResult={analysisResult} 
                  />
                </svg>
              )}
            </div>
          ) : null}
        </div>
      </div>
    </div>
  );
}
