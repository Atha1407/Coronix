/**
 * Coronix DICOM API Service — Frontend Integration Layer
 *
 * CURRENT STATE  : Integration placeholders only. No real requests are made.
 *                  All functions reject with code 'DICOM_NOT_CONNECTED' so
 *                  callers can render the "Not Connected" UI state cleanly.
 *
 * FUTURE STATE   : Replace placeholder bodies with real fetch/axios calls
 *                  once the backend team provides endpoint URLs + shapes.
 *
 * Expected future data model (names TBC by backend team):
 *   Study    { studyId, studyDate, series: Series[] }
 *   Series   { seriesId, description, instanceCount }
 *   Instance { instanceId, sopClassUid, frameCount }
 */

// ---------------------------------------------------------------------------
// Integration constant — set this when backend provides the gateway URL
// ---------------------------------------------------------------------------
const DICOM_GATEWAY_BASE_URL = null; // e.g. 'http://localhost:8042'

// ---------------------------------------------------------------------------
// Internal helper
// ---------------------------------------------------------------------------
function notConnected(fnName) {
  const err = new Error(
    '[dicomApi] ' + fnName + ': DICOM gateway not connected. ' +
    'Update DICOM_GATEWAY_BASE_URL when the backend is available.'
  );
  err.code = 'DICOM_NOT_CONNECTED';
  return Promise.reject(err);
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Check whether the DICOM gateway is reachable.
 *
 * Future contract:
 *   GET /dicom/status
 *   Response: { connected: boolean, source: string, version?: string }
 */
export async function getDicomConnectionStatus() {
  // TODO: replace with real fetch
  // const res = await fetch(DICOM_GATEWAY_BASE_URL + '/dicom/status');
  // return res.json();
  return notConnected('getDicomConnectionStatus');
}

/**
 * Retrieve the list of available studies from the gateway.
 *
 * Future contract:
 *   GET /dicom/studies
 *   Response: Study[]
 */
export async function getDicomStudies() {
  // TODO: replace with real fetch
  // const res = await fetch(DICOM_GATEWAY_BASE_URL + '/dicom/studies');
  // return res.json();
  return notConnected('getDicomStudies');
}

/**
 * Retrieve the series list for a given study.
 *
 * Future contract:
 *   GET /dicom/studies/:studyId/series
 *   Response: Series[]
 *
 * @param {string} studyId
 */
export async function getDicomSeries(studyId) {
  // TODO: replace with real fetch
  // const res = await fetch(DICOM_GATEWAY_BASE_URL + '/dicom/studies/' + studyId + '/series');
  // return res.json();
  return notConnected('getDicomSeries');
}

/**
 * Retrieve metadata for a specific instance.
 *
 * Future contract:
 *   GET /dicom/instances/:instanceId
 *   Response: Instance metadata
 *
 * @param {string} instanceId
 */
export async function getDicomInstance(instanceId) {
  return notConnected('getDicomInstance');
}

/**
 * Request a rendered frame from a DICOM instance.
 * The returned object URL is passed directly to the existing Coronix viewer.
 *
 * Future contract:
 *   GET /dicom/instances/:instanceId/rendered?frame=0
 *   Response: image/jpeg or image/png
 *
 * @param {string} instanceId
 * @param {number} frame
 * @returns {Promise<string>} Object URL for AngiogramViewer
 */
export async function getDicomRenderedFrame(instanceId, frame) {
  return notConnected('getDicomRenderedFrame');
}