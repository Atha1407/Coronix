import React from 'react';

export default function AnalysisOverlay({ pointA, pointB, isMagnifier = false, analysisResult = {} }) {
  // Safely extract potential future OpenCV fields
  const { vesselMask, vesselCenterline, roiPolygon, lesionOverlay, lesion_bbox } = analysisResult || {};

  return (
    <>
      {/* Backend provided lesion bounding box - Clear dashed line, completely transparent fill */}
      {lesion_bbox && (
        <rect
          x={lesion_bbox[0]}
          y={lesion_bbox[1]}
          width={lesion_bbox[2]}
          height={lesion_bbox[3]}
          fill="none"
          stroke="#FFB300"
          strokeWidth={isMagnifier ? "2" : "2.5"}
          strokeDasharray="6 4"
          rx="4"
        />
      )}

      {/* Centerline - Thin, clean line with no glow obstructing vessel lumen */}
      {!vesselCenterline && pointA && pointB && (
        <line 
          x1={pointA.x} 
          y1={pointA.y} 
          x2={pointB.x} 
          y2={pointB.y} 
          stroke="#00E5FF" 
          strokeWidth={isMagnifier ? "1" : "1.75"} 
          strokeLinecap="round"
          opacity="0.9"
        />
      )}
      
      {/* Point A - Crisp circle, no colored glow halo */}
      {pointA && (
        <g transform={`translate(${pointA.x}, ${pointA.y})`}>
          <circle 
            r={isMagnifier ? "4" : "6"} 
            fill="#00E5FF" 
            stroke="#FFFFFF" 
            strokeWidth={isMagnifier ? "1.5" : "2"} 
            filter={!isMagnifier ? "drop-shadow(0 1px 2px rgba(0,0,0,0.7))" : ""} 
          />
          {!isMagnifier && (
            <text 
              x="12" 
              y="5" 
              fill="#FFFFFF" 
              stroke="#0f172a" 
              strokeWidth="2.5" 
              paintOrder="stroke fill" 
              fontSize="13" 
              fontWeight="800"
            >
              A
            </text>
          )}
        </g>
      )}
      
      {/* Point B - Crisp circle, no colored glow halo */}
      {pointB && (
        <g transform={`translate(${pointB.x}, ${pointB.y})`}>
          <circle 
            r={isMagnifier ? "4" : "6"} 
            fill="#FF3366" 
            stroke="#FFFFFF" 
            strokeWidth={isMagnifier ? "1.5" : "2"} 
            filter={!isMagnifier ? "drop-shadow(0 1px 2px rgba(0,0,0,0.7))" : ""} 
          />
          {!isMagnifier && (
            <text 
              x="12" 
              y="5" 
              fill="#FFFFFF" 
              stroke="#0f172a" 
              strokeWidth="2.5" 
              paintOrder="stroke fill" 
              fontSize="13" 
              fontWeight="800"
            >
              B
            </text>
          )}
        </g>
      )}
    </>
  );
}
