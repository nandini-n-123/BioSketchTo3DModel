import { useCallback, useEffect, useRef, useState } from "react";
import { QRCodeSVG } from "qrcode.react";
import Webcam from "react-webcam";
import ReactCrop from "react-image-crop";
import "react-image-crop/dist/ReactCrop.css";
import ModelViewer from "./ModelViewer";
import "./mobile-scan.css";

function getApiBaseUrl() {
  const apiUrl = new URL(window.location.href);
  apiUrl.port = "8000";
  apiUrl.pathname = "";
  apiUrl.search = "";
  apiUrl.hash = "";
  return apiUrl.origin;
}

const API_BASE_URL = getApiBaseUrl();

function isLoopbackHostname(hostname) {
  return ["localhost", "127.0.0.1", "::1"].includes(hostname);
}

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

  if (!naturalWidth || !naturalHeight) return DEFAULT_CROP;

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
  if (!ctx) return DEFAULT_CROP;

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

  if (inkPixels < 40 || minX >= maxX || minY >= maxY) return DEFAULT_CROP;

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

  if (crop.x + crop.width > 100) crop.width = 100 - crop.x;
  if (crop.y + crop.height > 100) crop.height = 100 - crop.y;

  return crop;
}

function formatMs(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  return `${value} ms`;
}

function formatConfidence(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  return `${value}%`;
}

function buildBackendErrorMessage(data) {
  const detail = data?.detail;
  if (typeof detail === "string") return detail;
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
  const [inputMode, setInputMode] = useState("upload");
  const [selectedOrgan, setSelectedOrgan] = useState("auto");
  const [crop, setCrop] = useState(DEFAULT_CROP);
  const [completedCrop, setCompletedCrop] = useState(null);
  const [statusMessage, setStatusMessage] = useState("");
  const [errorMessage, setErrorMessage] = useState("");
  const [unsupportedInfo, setUnsupportedInfo] = useState(null);
  const [timing, setTiming] = useState(null);
  const [mobileSessionId, setMobileSessionId] = useState("");
  const [mobileScanUrl, setMobileScanUrl] = useState("");
  const [mobileScanStatus, setMobileScanStatus] = useState("idle");
  const [mobileScanError, setMobileScanError] = useState("");
  const [mobileSessionExpires, setMobileSessionExpires] = useState(null);

  const webcamRef = useRef(null);
  const imgRef = useRef(null);
  const canvasRef = useRef(null);
  const fileInputRef = useRef(null);
  const viewerBoxRef = useRef(null);
  const submitStartRef = useRef(null);
  const responseReceivedRef = useRef(null);
  const mobilePollBusyRef = useRef(false);

  useEffect(() => {
    const handleFullscreenChange = () => {
      setFullscreen(document.fullscreenElement === viewerBoxRef.current);
    };

    document.addEventListener("fullscreenchange", handleFullscreenChange);
    return () =>
      document.removeEventListener("fullscreenchange", handleFullscreenChange);
  }, []);

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
      setImageSrc(null);
      setCompletedCrop(null);
      setCrop(DEFAULT_CROP);
      resetResult();
      setStatusMessage("Image loaded. Auto-detecting sketch area...");

      const reader = new FileReader();
      reader.onload = () => setImageSrc(reader.result);
      reader.onerror = () =>
        setErrorMessage("Could not read the selected image.");
      reader.readAsDataURL(selectedFile);
    },
    [resetResult],
  );

  useEffect(() => {
    if (!mobileSessionId || mobileScanStatus !== "waiting") return undefined;

    let cancelled = false;

    const pollMobileScan = async () => {
      if (mobilePollBusyRef.current || cancelled) return;
      mobilePollBusyRef.current = true;

      try {
        const response = await fetch(
          `${API_BASE_URL}/api/mobile-scan/${mobileSessionId}`,
        );
        const data = await response.json().catch(() => ({}));

        if (!response.ok) {
          throw new Error(
            data.detail || "Phone scan session could not be checked.",
          );
        }

        if (data.status === "received" && data.image_url) {
          const imageResponse = await fetch(data.image_url);
          if (!imageResponse.ok)
            throw new Error("Laptop could not fetch the phone image.");

          const blob = await imageResponse.blob();
          const receivedFile = new File(
            [blob],
            `phone-sketch-${Date.now()}.jpg`,
            { type: blob.type || "image/jpeg" },
          );

          if (cancelled) return;

          handleFile(receivedFile);
          setMobileScanStatus("received");
          setStatusMessage(
            "Phone-camera sketch received. Adjust the crop box if needed, then submit it.",
          );
        }
      } catch (error) {
        if (!cancelled) {
          setMobileScanError(
            error.message || "Could not receive the phone sketch.",
          );
          setMobileScanStatus("error");
        }
      } finally {
        mobilePollBusyRef.current = false;
      }
    };

    pollMobileScan();
    const intervalId = window.setInterval(pollMobileScan, 1000);

    return () => {
      cancelled = true;
      window.clearInterval(intervalId);
    };
  }, [mobileSessionId, mobileScanStatus, handleFile]);

  const toggleFullscreen = async () => {
    const viewerElement = viewerBoxRef.current;
    if (!viewerElement) return;

    try {
      if (document.fullscreenElement) {
        await document.exitFullscreen();
      } else {
        await viewerElement.requestFullscreen();
      }
    } catch (error) {
      console.error("Fullscreen failed:", error);
      setFullscreen((current) => !current);
    }
  };

  const startMobileScanSession = async () => {
    setMobileScanError("");
    setMobileScanStatus("creating");

    try {
      const response = await fetch(`${API_BASE_URL}/api/mobile-scan/session`, {
        method: "POST",
      });
      const data = await response.json().catch(() => ({}));

      if (!response.ok || !data.session_id) {
        throw new Error(
          data.detail || "Could not create a phone scan session.",
        );
      }

      const mobileCaptureOrigin = window.location.origin;

      let mobileHostname = "";
      try {
        mobileHostname = new URL(mobileCaptureOrigin).hostname;
      } catch {
        throw new Error(
          "Invalid frontend origin. Open the application using the Vite Network URL.",
        );
      }

      if (isLoopbackHostname(mobileHostname)) {
        throw new Error(
          "The QR code cannot use localhost. Open this laptop page using the Vite Network URL printed by npm run dev -- --host 0.0.0.0, such as http://192.168.x.x:5173.",
        );
      }

      setMobileSessionId(data.session_id);
      setMobileScanUrl(`${mobileCaptureOrigin}/mobile-scan/${data.session_id}`);
      setMobileSessionExpires(data.expires_in_minutes ?? 10);
      setMobileScanStatus("waiting");
      setStatusMessage(
        "QR code ready. Scan it using a phone connected to the same Wi-Fi network or hotspot.",
      );
    } catch (error) {
      setMobileScanError(
        error.message || "Could not create a phone scan session.",
      );
      setMobileScanStatus("error");
    }
  };

  const cancelMobileScan = () => {
    setMobileSessionId("");
    setMobileScanUrl("");
    setMobileScanStatus("idle");
    setMobileScanError("");
    setMobileSessionExpires(null);
  };

  const openFilePicker = () => {
    setErrorMessage("");
    fileInputRef.current?.click();
  };

  const handleFileChange = (event) => {
    const selectedFile = event.target.files?.[0];
    handleFile(selectedFile);

    // Allow the same file to be selected again after Retake / Choose Again.
    event.target.value = "";
  };

  const handleUploadAreaKeyDown = (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      openFilePicker();
    }
  };

  const handleDrop = (event) => {
    event.preventDefault();
    event.stopPropagation();
    handleFile(event.dataTransfer.files?.[0]);
  };

  const handleDragOver = (event) => {
    event.preventDefault();
    event.stopPropagation();
  };

  const removeFile = () => {
    setFile(null);
    setImageSrc(null);
    setCrop(DEFAULT_CROP);
    setCompletedCrop(null);
    setStatusMessage("");

    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }

    resetResult();
  };

  const capture = useCallback(async () => {
    const screenshot = webcamRef.current?.getScreenshot();

    if (!screenshot) {
      setErrorMessage(
        "Camera capture failed. Allow webcam access and try again.",
      );
      return;
    }

    const blob = await fetch(screenshot).then((response) => response.blob());
    handleFile(
      new File([blob], `webcam-sketch-${Date.now()}.jpg`, {
        type: "image/jpeg",
      }),
    );
  }, [handleFile]);

  const handleImageLoad = (event) => {
    const image = event.currentTarget;
    imgRef.current = image;

    const autoCrop = detectSketchCropPercent(image);
    setCrop(autoCrop);
    setCompletedCrop(percentToPixelCrop(autoCrop, image));
    setStatusMessage(
      "Auto crop is ready. Adjust the crop box if needed, then submit the cropped sketch.",
    );
  };

  const renderCroppedImage = () => {
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
        0.95,
      );
    });
  };

  const handleUpload = async () => {
    if (!imageSrc) {
      setErrorMessage("Capture or upload a sketch first.");
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
      const blob = await renderCroppedImage();
      const formData = new FormData();
      formData.append("file", blob, "cropped-sketch.jpg");
      formData.append("save_debug", "true");

      if (selectedOrgan !== "auto") formData.append("organ", selectedOrgan);

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

      if (!returnedModelUrl)
        throw new Error("Backend did not return a GLB model URL.");

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
          data.processing_time_ms ?? backendTiming.processing_time_ms ?? null,
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
    if (document.fullscreenElement) document.exitFullscreen().catch(() => {});
    removeFile();
    setFullscreen(false);
  };

  return (
    <div className="upload-container">
      <div className="upload-box">
        {!prediction && (
          <>
            <h2>Upload Biological Diagram</h2>

            <p className="upload-subtitle">
              Capture from your laptop webcam, upload a sketch image, or scan a
              QR code to use your phone camera over the same Wi-Fi network. For
              now, use a plain unruled sheet so notebook lines do not interfere
              with contour extraction. The app currently supports Brain, Heart,
              and Lungs. If another organ is detected, the backend stops before
              deformation instead of showing a wrong 3D model.
            </p>

            <div className="toggle-buttons">
              <button
                type="button"
                className={inputMode === "upload" ? "active-toggle" : ""}
                onClick={() => setInputMode("upload")}
              >
                Upload File
              </button>
              <button
                type="button"
                className={inputMode === "webcam" ? "active-toggle" : ""}
                onClick={() => setInputMode("webcam")}
              >
                Laptop Camera
              </button>
              <button
                type="button"
                className={inputMode === "phone" ? "active-toggle" : ""}
                onClick={() => setInputMode("phone")}
              >
                Phone Camera QR
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

            {inputMode === "upload" && (
              <div
                className="upload-area"
                role="button"
                tabIndex={0}
                aria-label="Choose a sketch image from this device"
                onClick={openFilePicker}
                onKeyDown={handleUploadAreaKeyDown}
                onDrop={handleDrop}
                onDragOver={handleDragOver}
              >
                <input
                  ref={fileInputRef}
                  className="file-input-hidden"
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
              </div>
            )}

            {inputMode === "webcam" && (
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
                    const settings =
                      stream.getVideoTracks()[0]?.getSettings() || {};
                    setStatusMessage(
                      `Laptop camera ready: ${settings.width || "auto"} x ${settings.height || "auto"}.`,
                    );
                  }}
                  onUserMediaError={(error) => {
                    console.error("Webcam access error:", error);

                    if (!window.isSecureContext) {
                      setErrorMessage(
                        "Laptop webcam access requires a secure browser page. Open this app on the laptop using http://localhost:5173 instead of the Wi-Fi Network URL. Use the Network URL only for the Phone Camera QR option.",
                      );
                      return;
                    }

                    setErrorMessage(
                      "Could not access the laptop webcam. Allow camera permission in the browser, check Windows camera privacy settings, and close other apps that may be using the camera.",
                    );
                  }}
                />

                <div className="camera-tips">
                  <strong>For best sketch detection:</strong>
                  <span>Use a plain unruled sheet.</span>
                  <span>Use a dark pen or marker instead of faint pencil.</span>
                  <span>
                    Keep the paper flat, fill most of the frame, and avoid
                    shadows.
                  </span>
                </div>

                <button type="button" className="capture-btn" onClick={capture}>
                  Capture Sketch
                </button>
              </div>
            )}

            {inputMode === "phone" && (
              <div className="phone-qr-panel">
                <h3>Use Phone Camera on the Same Wi-Fi Network</h3>
                <p>
                  Connect the laptop and phone to the same Wi-Fi network or
                  phone hotspot. Open this laptop page using the Vite Network
                  URL, generate a QR code, and scan it using the phone camera.
                </p>

                <div className="phone-qr-actions">
                  <button
                    type="button"
                    className="secondary-btn"
                    onClick={startMobileScanSession}
                  >
                    {mobileScanStatus === "creating"
                      ? "Creating QR..."
                      : "Generate Phone QR Code"}
                  </button>
                  {mobileScanUrl && (
                    <button
                      type="button"
                      className="remove-btn"
                      onClick={cancelMobileScan}
                    >
                      Clear QR Session
                    </button>
                  )}
                </div>

                {mobileScanUrl && (
                  <div className="phone-qr-content">
                    <div className="phone-qr-code">
                      <QRCodeSVG
                        value={mobileScanUrl}
                        size={224}
                        bgColor="#ffffff"
                        fgColor="#020617"
                        level="M"
                        includeMargin
                      />
                    </div>
                    <div className="phone-qr-instructions">
                      <strong>Scan this QR code on the phone.</strong>
                      <span>
                        QR session expires in about {mobileSessionExpires ?? 10}{" "}
                        minutes.
                      </span>
                      <span>
                        The phone page will let you capture, preview, and send
                        the photo.
                      </span>
                      <small>{mobileScanUrl}</small>
                    </div>
                  </div>
                )}

                {mobileScanStatus === "waiting" && (
                  <p className="status-message">
                    Waiting for the phone-camera sketch...
                  </p>
                )}
                {mobileScanStatus === "received" && (
                  <p className="status-message">
                    Phone-camera sketch received successfully.
                  </p>
                )}
                {mobileScanError && (
                  <p className="error-message">{mobileScanError}</p>
                )}
              </div>
            )}

            {imageSrc && (
              <div className="crop-section">
                <h3>Auto Crop + Manual Crop</h3>
                <p className="crop-help">
                  Drag or resize the crop box if needed. The frontend sends the
                  cropped original image, and the backend performs the final
                  OpenCV cleanup before classification and deformation.
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
                  Retake / Choose Again
                </button>
              )}
              <button
                type="button"
                className="analyze-btn"
                onClick={handleUpload}
                disabled={loading || !imageSrc}
              >
                {loading ? "Processing Sketch..." : "Submit Sketch"}
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
                Your sketch is not used to hallucinate anatomy from scratch. The
                backend detects the organ, extracts the contour, and deforms a
                validated base GLB model so your drawing shapes the final
                interactive result.
              </p>

              <div className="deformation-note">
                <strong>Sketch-based deformation:</strong> the model keeps its
                recognizable anatomical structure while its proportions bend
                toward your drawing. Rotate, zoom, and inspect the result
                manually.
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
