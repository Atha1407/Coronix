import React from 'react';

export default function AnalysisOverlay({ pointA, pointB, isMagnifier = false, analysisResult = {} }) {
  // Safely extract potential future OpenCV fields
  const { vesselMask, vesselCenterline, roiPolygon, lesionOverlay, lesion_bbox } = analysisResult || {};

  return (
    <>
      {/* 
        FRONTEND FALLBACK VISUALIZATION:
        When OpenCV backend returns real geometry, it will take priority here.
      */}
      {!vesselMask && !roiPolygon && pointA && pointB && (
        <line 
          x1={pointA.x} 
          y1={pointA.y} 
          x2={pointB.x} 
          y2={pointB.y} 
          stroke="var(--color-teal)" 
          strokeWidth="40" 
          strokeLinecap="round"
          opacity="0.15" 
        />
      )}

      {/* Backend provided lesion bounding box */}
      {lesion_bbox && (
        <rect
          x={lesion_bbox[0]}
          y={lesion_bbox[1]}
          width={lesion_bbox[2]}
          height={lesion_bbox[3]}
          fill="rgba(217, 166, 160, 0.15)"
          stroke="var(--color-powder-blush)"
          strokeWidth={isMagnifier ? "1.5" : "3"}
          strokeDasharray="6 4"
        />
      )}

      {/* Fallback centerline */}
      {!vesselCenterline && pointA && pointB && (
        <line 
          x1={pointA.x} 
          y1={pointA.y} 
          x2={pointB.x} 
          y2={pointB.y} 
          stroke="var(--color-teal)" 
          strokeWidth={isMagnifier ? "1" : "2"} 
          opacity="0.9"
        />
      )}
      
      {/* Point A */}
      {pointA && (
        <g transform={`translate(${pointA.x}, ${pointA.y})`}>
          <circle 
            r={isMagnifier ? "4" : "6"} 
            fill="var(--color-teal)" 
            stroke="white" 
            strokeWidth={isMagnifier ? "1" : "2"} 
            filter={!isMagnifier ? "drop-shadow(0 2px 4px rgba(0,0,0,0.3))" : ""} 
          />
          {!isMagnifier && (
            <text x="12" y="4" fill="white" fontSize="14" fontWeight="bold" filter="drop-shadow(0 1px 2px rgba(0,0,0,0.8))">A</text>
          )}
        </g>
      )}
      
      {/* Point B */}
      {pointB && (
        <g transform={`translate(${pointB.x}, ${pointB.y})`}>
          <circle 
            r={isMagnifier ? "4" : "6"} 
            fill="var(--color-powder-blush)" 
            stroke="white" 
            strokeWidth={isMagnifier ? "1" : "2"} 
            filter={!isMagnifier ? "drop-shadow(0 2px 4px rgba(0,0,0,0.3))" : ""} 
          />
          {!isMagnifier && (
            <text x="12" y="4" fill="white" fontSize="14" fontWeight="bold" filter="drop-shadow(0 1px 2px rgba(0,0,0,0.8))">B</text>
          )}
        </g>
      )}
    </>
  );
}
