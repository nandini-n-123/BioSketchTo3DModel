import { useCallback, useEffect, useRef, useState } from "react";
import Webcam from "react-webcam";
import ReactCrop from "react-image-crop";
import "react-image-crop/dist/ReactCrop.css";
import ModelViewer from "./ModelViewer";

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

const DEFAULT_CROP = {
  unit: "%",
  x: 5,
  y: 5,
  width: 90,
  height: 90,
};

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function percentToPixelCrop(percentCrop, image) {
  const displayedWidth = image.width || image.naturalWidth;
  const displayedHeight = image.height || image.naturalHeight;

  return {
    unit: "px",
    x: Math.round((percentCrop.x / 100) * displayedWidth),
    y: Math.round((percentCrop.y / 100) * displayedHeight),
    width: Math.round((percentCrop.width / 100) * displayedWidth),
    height: Math.round((percentCrop.height / 100) * displayedHeight),
  };
}

function detectSketchCropPercent(image) {
  const naturalWidth = image.naturalWidth;
  const naturalHeight = image.naturalHeight;

  if (!naturalWidth || !naturalHeight) {
    return DEFAULT_CROP;
  }

  const maxCanvasSide = 520;
  const scale = Math.min(
    1,
    maxCanvasSide / Math.max(naturalWidth, naturalHeight),
  );

  const width = Math.max(1, Math.round(naturalWidth * scale));
  const height = Math.max(1, Math.round(naturalHeight * scale));

  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;

  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) {
    return DEFAULT_CROP;
  }

  ctx.drawImage(image, 0, 0, width, height);
  const { data } = ctx.getImageData(0, 0, width, height);

  let minX = width;
  let minY = height;
  let maxX = 0;
  let maxY = 0;
  let inkPixels = 0;

  const step = Math.max(1, Math.floor(Math.min(width, height) / 360));

  for (let y = 0; y < height; y += step) {
    for (let x = 0; x < width; x += step) {
      const idx = (y * width + x) * 4;
      const r = data[idx];
      const g = data[idx + 1];
      const b = data[idx + 2];
      const alpha = data[idx + 3];

      if (alpha < 10) continue;

      const luminance = 0.299 * r + 0.587 * g + 0.114 * b;
      const channelSpread = Math.max(r, g, b) - Math.min(r, g, b);

      const looksLikeInk =
        luminance < 215 && (channelSpread < 95 || luminance < 170);

      if (looksLikeInk) {
        inkPixels += 1;
        minX = Math.min(minX, x);
        minY = Math.min(minY, y);
        maxX = Math.max(maxX, x);
        maxY = Math.max(maxY, y);
      }
    }
  }

  if (inkPixels < 40 || minX >= maxX || minY >= maxY) {
    return DEFAULT_CROP;
  }

  const boxWidth = maxX - minX;
  const boxHeight = maxY - minY;
  const paddingX = Math.max(18, boxWidth * 0.22);
  const paddingY = Math.max(18, boxHeight * 0.22);

  const x1 = clamp(minX - paddingX, 0, width);
  const y1 = clamp(minY - paddingY, 0, height);
  const x2 = clamp(maxX + paddingX, 0, width);
  const y2 = clamp(maxY + paddingY, 0, height);

  const crop = {
    unit: "%",
    x: clamp((x1 / width) * 100, 0, 99),
    y: clamp((y1 / height) * 100, 0, 99),
    width: clamp(((x2 - x1) / width) * 100, 8, 100),
    height: clamp(((y2 - y1) / height) * 100, 8, 100),
  };

  if (crop.x + crop.width > 100) {
    crop.width = 100 - crop.x;
  }

  if (crop.y + crop.height > 100) {
    crop.height = 100 - crop.y;
  }

  return crop;
}

function formatMs(value) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "-";
  }

  return `${value} ms`;
}

function formatConfidence(value) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "-";
  }

  return `${value}%`;
}

function buildBackendErrorMessage(data) {
  const detail = data?.detail;

  if (typeof detail === "string") {
    return detail;
  }

  if (detail && typeof detail === "object") {
    return detail.message || "Backend failed to process the sketch.";
  }

  return data?.message || "Backend failed to process the sketch.";
}

function UploadSection() {
  const [file, setFile] = useState(null);
  const [imageSrc, setImageSrc] = useState(null);
  const [prediction, setPrediction] = useState("");
  const [model, setModel] = useState("");
  const [metrics, setMetrics] = useState(null);
  const [debugImages, setDebugImages] = useState(null);
  const [fullscreen, setFullscreen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [useCamera, setUseCamera] = useState(false);
  const [selectedOrgan, setSelectedOrgan] = useState("auto");
  const [crop, setCrop] = useState(DEFAULT_CROP);
  const [completedCrop, setCompletedCrop] = useState(null);
  const [statusMessage, setStatusMessage] = useState("");
  const [errorMessage, setErrorMessage] = useState("");
  const [unsupportedInfo, setUnsupportedInfo] = useState(null);
  const [timing, setTiming] = useState(null);

  const webcamRef = useRef(null);
  const imgRef = useRef(null);
  const canvasRef = useRef(null);
  const viewerBoxRef = useRef(null);
  const submitStartRef = useRef(null);
  const responseReceivedRef = useRef(null);

  useEffect(() => {
    const handleFullscreenChange = () => {
      setFullscreen(document.fullscreenElement === viewerBoxRef.current);
    };

    document.addEventListener("fullscreenchange", handleFullscreenChange);

    return () => {
      document.removeEventListener("fullscreenchange", handleFullscreenChange);
    };
  }, []);

  const toggleFullscreen = async () => {
    const viewerElement = viewerBoxRef.current;

    if (!viewerElement) return;

    try {
      if (document.fullscreenElement) {
        await document.exitFullscreen();
        setFullscreen(false);
        return;
      }

      await viewerElement.requestFullscreen();
      setFullscreen(true);
    } catch (error) {
      console.error("Fullscreen failed:", error);
      setFullscreen((current) => !current);
    }
  };

  const resetResult = useCallback(() => {
    setPrediction("");
    setModel("");
    setMetrics(null);
    setDebugImages(null);
    setErrorMessage("");
    setUnsupportedInfo(null);
    setTiming(null);
  }, []);

  const handleFile = useCallback(
    (selectedFile) => {
      if (!selectedFile) return;

      if (!selectedFile.type.startsWith("image/")) {
        setErrorMessage("Please choose an image file.");
        return;
      }

      setFile(selectedFile);
      setCompletedCrop(null);
      setCrop(DEFAULT_CROP);
      resetResult();
      setStatusMessage("Image loaded. Auto-detecting sketch area...");

      const reader = new FileReader();

      reader.onload = () => {
        setImageSrc(reader.result);
      };

      reader.onerror = () => {
        setErrorMessage("Could not read the selected image.");
      };

      reader.readAsDataURL(selectedFile);
    },
    [resetResult],
  );

  const handleFileChange = (event) => {
    handleFile(event.target.files?.[0]);
  };

  const handleDrop = (event) => {
    event.preventDefault();
    handleFile(event.dataTransfer.files?.[0]);
  };

  const handleDragOver = (event) => {
    event.preventDefault();
  };

  const removeFile = () => {
    setFile(null);
    setImageSrc(null);
    setCrop(DEFAULT_CROP);
    setCompletedCrop(null);
    setStatusMessage("");
    resetResult();
  };

  const capture = useCallback(async () => {
    const screenshot = webcamRef.current?.getScreenshot();

    if (!screenshot) {
      setErrorMessage(
        "Camera capture failed. Please allow webcam access and try again.",
      );
      return;
    }

    const blob = await fetch(screenshot).then((res) => res.blob());

    const capturedFile = new File([blob], `webcam-sketch-${Date.now()}.jpg`, {
      type: "image/jpeg",
    });

    handleFile(capturedFile);
  }, [handleFile]);

  const handleImageLoad = (event) => {
    const image = event.currentTarget;
    imgRef.current = image;

    const autoCrop = detectSketchCropPercent(image);
    setCrop(autoCrop);
    setCompletedCrop(percentToPixelCrop(autoCrop, image));
    setStatusMessage(
      "Auto crop is ready. Adjust the crop box if needed, then submit.",
    );
  };

  const getCroppedBlob = () => {
    return new Promise((resolve, reject) => {
      const canvas = canvasRef.current;
      const image = imgRef.current;

      const activeCrop =
        completedCrop ||
        (image ? percentToPixelCrop(DEFAULT_CROP, image) : null);

      if (!canvas || !image || !activeCrop?.width || !activeCrop?.height) {
        reject(new Error("Please crop the image before submitting."));
        return;
      }

      const scaleX = image.naturalWidth / image.width;
      const scaleY = image.naturalHeight / image.height;

      const outputWidth = Math.max(1, Math.round(activeCrop.width * scaleX));
      const outputHeight = Math.max(1, Math.round(activeCrop.height * scaleY));

      canvas.width = outputWidth;
      canvas.height = outputHeight;

      const ctx = canvas.getContext("2d");

      if (!ctx) {
        reject(new Error("Could not prepare the cropped image."));
        return;
      }

      ctx.drawImage(
        image,
        activeCrop.x * scaleX,
        activeCrop.y * scaleY,
        activeCrop.width * scaleX,
        activeCrop.height * scaleY,
        0,
        0,
        outputWidth,
        outputHeight,
      );

      canvas.toBlob(
        (blob) => {
          if (!blob) {
            reject(new Error("Could not create the cropped image."));
            return;
          }

          resolve(blob);
        },
        "image/jpeg",
        0.92,
      );
    });
  };

  const handleUpload = async () => {
    if (!imageSrc) {
      setErrorMessage("Please capture or upload a sketch first.");
      return;
    }

    submitStartRef.current = performance.now();
    responseReceivedRef.current = null;

    setTiming(null);
    setUnsupportedInfo(null);
    setLoading(true);
    setErrorMessage("");
    setStatusMessage("Submitting cropped sketch to backend...");

    try {
      const blob = await getCroppedBlob();

      const formData = new FormData();
      formData.append("file", blob, "cropped-sketch.jpg");
      formData.append("save_debug", "true");

      if (selectedOrgan !== "auto") {
        formData.append("organ", selectedOrgan);
      }

      const response = await fetch(`${API_BASE_URL}/api/deform-sketch`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json().catch(() => ({}));
      responseReceivedRef.current = performance.now();

      if (!response.ok) {
        const detail = data?.detail;

        if (
          detail &&
          typeof detail === "object" &&
          detail.code === "UNSUPPORTED_ORGAN"
        ) {
          setUnsupportedInfo(detail);
          setStatusMessage("");
          setErrorMessage("");
          return;
        }

        throw new Error(buildBackendErrorMessage(data));
      }

      const detectedOrgan =
        data.organ || data.prediction?.organ || selectedOrgan;

      const returnedModelUrl =
        data.model_url ||
        data.output_url ||
        data.glb_url ||
        data.url ||
        data.model ||
        "";

      if (!returnedModelUrl) {
        console.log("Backend response:", data);
        throw new Error("Backend did not return a GLB model URL.");
      }

      const finalModelUrl = returnedModelUrl.startsWith("http")
        ? returnedModelUrl
        : `${API_BASE_URL}${returnedModelUrl}`;

      const backendTiming = data.timing || {};
      const submitToResponseMs = Math.round(
        responseReceivedRef.current - submitStartRef.current,
      );

      setTiming({
        submitToResponseMs,
        backendProcessingMs:
          data.processing_time_ms ??
          backendTiming.processing_time_ms ??
          backendTiming.backend_total_time_ms ??
          null,
        backendUploadSaveMs: backendTiming.upload_save_time_ms ?? null,
        backendPredictionMs: backendTiming.prediction_time_ms ?? null,
        backendDeformationMs: backendTiming.deformation_time_ms ?? null,
        modelRenderMs: null,
        totalToDisplayMs: null,
      });

      setPrediction(detectedOrgan);
      setModel(finalModelUrl);
      setMetrics(data.metrics || null);
      setDebugImages(data.debug || null);
      setStatusMessage("Sketch processed successfully.");
    } catch (error) {
      setErrorMessage(error.message || "Backend connection error.");
      setStatusMessage("");
    } finally {
      setLoading(false);
    }
  };

  const analyzeAnother = () => {
    if (document.fullscreenElement) {
      document.exitFullscreen().catch(() => {});
    }

    setFile(null);
    setImageSrc(null);
    setCrop(DEFAULT_CROP);
    setCompletedCrop(null);
    setFullscreen(false);
    setStatusMessage("");
    resetResult();
  };

  return (
    <div className="upload-container">
      <div className="upload-box">
        {!prediction && (
          <>
            <h2>Upload Biological Diagram</h2>

            <p className="upload-subtitle">
              Capture from your laptop webcam or upload a sketch image. The app
              currently supports Brain, Heart, and Lungs. If another organ is
              detected, the backend will stop before deformation instead of
              showing a wrong 3D model.
            </p>

            <div className="toggle-buttons">
              <button
                type="button"
                className={!useCamera ? "active-toggle" : ""}
                onClick={() => setUseCamera(false)}
              >
                Upload File
              </button>

              <button
                type="button"
                className={useCamera ? "active-toggle" : ""}
                onClick={() => setUseCamera(true)}
              >
                Use Camera
              </button>
            </div>

            <div className="manual-organ-row">
              <label htmlFor="organ-select">Organ mode</label>

              <select
                id="organ-select"
                value={selectedOrgan}
                onChange={(event) => setSelectedOrgan(event.target.value)}
              >
                <option value="auto">Auto detect using CNN</option>
                <option value="brain">Brain</option>
                <option value="heart">Heart</option>
                <option value="lungs">Lungs</option>
              </select>
            </div>

            {!useCamera ? (
              <label
                className="upload-area"
                onDrop={handleDrop}
                onDragOver={handleDragOver}
              >
                <input
                  hidden
                  type="file"
                  accept="image/*"
                  onChange={handleFileChange}
                />

                {!file ? (
                  <div>
                    📤 Drag & Drop Image
                    <br />
                    Click to Upload
                  </div>
                ) : (
                  <div>📄 {file.name}</div>
                )}
              </label>
            ) : (
              <div className="camera-section">
                <Webcam
                  audio={false}
                  ref={webcamRef}
                  screenshotFormat="image/jpeg"
                  screenshotQuality={1}
                  forceScreenshotSourceSize
                  mirrored={false}
                  videoConstraints={{
                    width: { ideal: 1920, min: 1280 },
                    height: { ideal: 1080, min: 720 },
                    facingMode: "user",
                  }}
                  className="webcam"
                  onUserMedia={(stream) => {
                    const track = stream.getVideoTracks()[0];
                    const settings = track.getSettings();

                    setStatusMessage(
                      `Camera ready: ${settings.width || "auto"} x ${
                        settings.height || "auto"
                      }. Use bright light and a dark pen for best results.`,
                    );
                  }}
                  onUserMediaError={() =>
                    setErrorMessage(
                      "Could not access webcam. Please allow camera permission in the browser.",
                    )
                  }
                />

                <div className="camera-tips">
                  <strong>For best sketch detection:</strong>
                  <span>Use a dark pen or marker instead of light pencil.</span>
                  <span>Keep the paper flat and fill most of the frame.</span>
                  <span>Use bright lighting and avoid shadows.</span>
                </div>

                <button type="button" className="capture-btn" onClick={capture}>
                  Capture Sketch
                </button>
              </div>
            )}

            {imageSrc && (
              <div className="crop-section">
                <h3>Auto Crop + Manual Crop</h3>

                <p className="crop-help">
                  The crop box is auto-positioned around the visible sketch.
                  Drag or resize it before submitting if you want a tighter
                  crop.
                </p>

                <ReactCrop
                  crop={crop}
                  onChange={(_, percentCrop) => setCrop(percentCrop)}
                  onComplete={(pixelCrop) => setCompletedCrop(pixelCrop)}
                  keepSelection
                >
                  <img
                    ref={imgRef}
                    src={imageSrc}
                    alt="Selected sketch for cropping"
                    className="crop-image"
                    onLoad={handleImageLoad}
                  />
                </ReactCrop>
              </div>
            )}

            <canvas ref={canvasRef} style={{ display: "none" }} />

            {statusMessage && <p className="status-message">{statusMessage}</p>}

            {unsupportedInfo && (
              <div className="error-message">
                <strong>Unsupported sketch detected.</strong>
                <br />
                {unsupportedInfo.message}
                <br />
                <br />
                Predicted as:{" "}
                <strong>{unsupportedInfo.predicted_organ || "unknown"}</strong>
                <br />
                Confidence:{" "}
                <strong>
                  {formatConfidence(unsupportedInfo.confidence_percent)}
                </strong>
                <br />
                Minimum required confidence:{" "}
                <strong>
                  {formatConfidence(unsupportedInfo.minimum_required_percent)}
                </strong>
              </div>
            )}

            {errorMessage && <p className="error-message">{errorMessage}</p>}

            <div className="upload-actions">
              {file && (
                <button
                  type="button"
                  className="remove-btn"
                  onClick={removeFile}
                  disabled={loading}
                >
                  Remove Image
                </button>
              )}

              <button
                type="button"
                className="analyze-btn"
                onClick={handleUpload}
                disabled={loading || !imageSrc}
              >
                {loading ? "Processing Sketch..." : "Submit Cropped Sketch"}
              </button>
            </div>
          </>
        )}

        {prediction && (
          <div className="result-layout">
            <div className="organ-info">
              <span className="badge">Detected Organ</span>

              <h2>{prediction}</h2>

              <p>
                Your sketch is not used to hallucinate a new organ from scratch.
                The backend detects the organ, extracts the sketch contour, and
                deforms a validated base GLB model so it feels like your drawing
                is shaping the final 3D anatomy.
              </p>

              <div className="deformation-note">
                <strong>Sketch-based deformation:</strong> the model keeps its
                anatomical structure, while its outer proportions bend toward
                your drawing. Rotate, zoom, and inspect the result manually in
                the viewer.
              </div>

              {timing && (
                <div className="timing-box">
                  <h3>Processing Time</h3>

                  <div className="metrics-grid">
                    <span>Submit to backend response</span>
                    <strong>{formatMs(timing.submitToResponseMs)}</strong>

                    <span>Backend total pipeline</span>
                    <strong>{formatMs(timing.backendProcessingMs)}</strong>

                    <span>Backend upload save</span>
                    <strong>{formatMs(timing.backendUploadSaveMs)}</strong>

                    <span>CNN/manual detection</span>
                    <strong>{formatMs(timing.backendPredictionMs)}</strong>

                    <span>Deformation pipeline</span>
                    <strong>{formatMs(timing.backendDeformationMs)}</strong>

                    <span>GLB load/render</span>
                    <strong>{formatMs(timing.modelRenderMs)}</strong>

                    <span>Total time to display</span>
                    <strong>
                      {timing.totalToDisplayMs !== null
                        ? formatMs(timing.totalToDisplayMs)
                        : "Loading model..."}
                    </strong>
                  </div>
                </div>
              )}

              {metrics && (
                <div className="metrics-box">
                  <h3>Deformation Metrics</h3>

                  <div className="metrics-grid">
                    <span>RMS before</span>
                    <strong>{metrics.rms_before ?? "-"}</strong>

                    <span>RMS after</span>
                    <strong>{metrics.rms_after ?? "-"}</strong>

                    <span>Similarity after</span>
                    <strong>{metrics.similarity_after_percent ?? "-"}%</strong>

                    <span>Mean deformation</span>
                    <strong>{metrics.mean_deformation_percent ?? "-"}%</strong>
                  </div>
                </div>
              )}

              {debugImages && (
                <div className="debug-links">
                  {debugImages.correspondence_url && (
                    <a
                      href={debugImages.correspondence_url}
                      target="_blank"
                      rel="noreferrer"
                    >
                      View correspondence debug
                    </a>
                  )}

                  {debugImages.sketch_border_url && (
                    <a
                      href={debugImages.sketch_border_url}
                      target="_blank"
                      rel="noreferrer"
                    >
                      View sketch border debug
                    </a>
                  )}
                </div>
              )}

              <button
                type="button"
                className="remove-btn"
                onClick={analyzeAnother}
              >
                Analyze Another Sketch
              </button>
            </div>

            <div
              ref={viewerBoxRef}
              className={`viewer-box ${fullscreen ? "fullscreen" : ""}`}
            >
              <button
                type="button"
                className="fullscreen-btn"
                onClick={toggleFullscreen}
              >
                {fullscreen ? "Exit" : "Expand"}
              </button>

              <ModelViewer
                modelUrl={model}
                onModelDisplayed={() => {
                  if (!submitStartRef.current) return;

                  const displayedAt = performance.now();

                  setTiming((previousTiming) => ({
                    ...previousTiming,
                    modelRenderMs: responseReceivedRef.current
                      ? Math.round(displayedAt - responseReceivedRef.current)
                      : null,
                    totalToDisplayMs: Math.round(
                      displayedAt - submitStartRef.current,
                    ),
                  }));
                }}
              />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default UploadSection;
