import { useState, useRef, useCallback } from "react";
import Webcam from "react-webcam";
import ReactCrop from "react-image-crop";
import "react-image-crop/dist/ReactCrop.css";

function UploadSection() {
  const [file, setFile] = useState(null);
  const [imageSrc, setImageSrc] = useState(null);

  const [prediction, setPrediction] = useState("");
  const [model, setModel] = useState("");

  const [fullscreen, setFullscreen] = useState(false);
  const [loading, setLoading] = useState(false);

  const [useCamera, setUseCamera] = useState(false);

  const [crop, setCrop] = useState({
    unit: "%",
    x: 20,
    y: 20,
    width: 60,
    height: 60,
  });

  const [completedCrop, setCompletedCrop] = useState(null);

  const webcamRef = useRef(null);
  const imgRef = useRef(null);
  const canvasRef = useRef(null);

  // -----------------------------
  // HANDLE FILE
  // -----------------------------
  const handleFile = (selectedFile) => {
    if (!selectedFile) return;

    setFile(selectedFile);

    const reader = new FileReader();

    reader.onload = () => {
      setImageSrc(reader.result);
    };

    reader.readAsDataURL(selectedFile);

    setPrediction("");
    setModel("");
  };

  const handleFileChange = (e) => {
    handleFile(e.target.files[0]);
  };

  // -----------------------------
  // DRAG DROP
  // -----------------------------
  const handleDrop = (e) => {
    e.preventDefault();

    if (e.dataTransfer.files.length > 0) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  // -----------------------------
  // REMOVE FILE
  // -----------------------------
  const removeFile = () => {
    setFile(null);
    setImageSrc(null);
    setPrediction("");
    setModel("");
  };

  // -----------------------------
  // CAMERA CAPTURE
  // -----------------------------
  const capture = useCallback(() => {
    const screenshot = webcamRef.current.getScreenshot();

    fetch(screenshot)
      .then((res) => res.blob())
      .then((blob) => {
        const capturedFile = new File(
          [blob],
          "capture.jpg",
          {
            type: "image/jpeg",
          }
        );

        handleFile(capturedFile);
      });
  }, [webcamRef]);

  // -----------------------------
  // CROPPED IMAGE
  // -----------------------------
  const getCroppedBlob = () => {
    return new Promise((resolve) => {
      const canvas = canvasRef.current;
      const image = imgRef.current;
      const crop = completedCrop;

      if (!crop || !canvas || !image) return;

      const scaleX =
        image.naturalWidth / image.width;

      const scaleY =
        image.naturalHeight / image.height;

      const ctx = canvas.getContext("2d");

      canvas.width = crop.width;
      canvas.height = crop.height;

      ctx.drawImage(
        image,
        crop.x * scaleX,
        crop.y * scaleY,
        crop.width * scaleX,
        crop.height * scaleY,
        0,
        0,
        crop.width,
        crop.height
      );

      canvas.toBlob(
        (blob) => {
          resolve(blob);
        },
        "image/jpeg",
        0.8
      );
    });
  };

  // -----------------------------
  // ANALYZE
  // -----------------------------
  const handleUpload = async () => {
    if (!completedCrop) {
      alert("Please crop the image first");
      return;
    }

    setLoading(true);

    const blob = await getCroppedBlob();

    const formData = new FormData();

    formData.append(
      "file",
      blob,
      "cropped.jpg"
    );

    try {
      const res = await fetch(
        "http://127.0.0.1:8000/api/classify-sketch",
        {
          method: "POST",
          body: formData,
        }
      );

      const data = await res.json();

      setPrediction(data.prediction);

      // -----------------------------
      // MODEL MAPPING
      // -----------------------------
      if (data.prediction === "brain") {
        setModel("/models/brain.glb");
      } else if (data.prediction === "lungs") {
        setModel("/models/lungs.glb");
      } else if (data.prediction === "heart") {
        setModel("/models/heart.glb");
      } else {
        setModel("/models/brain.glb");
      }
    } catch (err) {
      console.log(err);
      alert("Backend connection error");
    }

    setLoading(false);
  };

  // -----------------------------
  // RESET
  // -----------------------------
  const analyzeAnother = () => {
    setPrediction("");
    setModel("");
    setFile(null);
    setImageSrc(null);
    setFullscreen(false);
  };

  return (
    <div className="upload-container">

      <div className="upload-box">

        {!prediction && (
          <>
            <h2>Upload Biological Diagram</h2>

            <p className="upload-subtitle">
              Capture or upload your sketch
            </p>

            {/* TOGGLE */}
            <div className="toggle-buttons">

              <button
                className={
                  !useCamera
                    ? "active-toggle"
                    : ""
                }
                onClick={() =>
                  setUseCamera(false)
                }
              >
                Upload File
              </button>

              <button
                className={
                  useCamera
                    ? "active-toggle"
                    : ""
                }
                onClick={() =>
                  setUseCamera(true)
                }
              >
                Use Camera
              </button>

            </div>

            {/* FILE MODE */}
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
                  <div>
                    📄 {file.name}
                  </div>
                )}

              </label>
            ) : (
              // CAMERA MODE
              <div className="camera-section">

                <Webcam
                  audio={false}
                  ref={webcamRef}
                  screenshotFormat="image/jpeg"
                  videoConstraints={{
                    facingMode: "environment",
                  }}
                  className="webcam"
                />

                <button
                  className="capture-btn"
                  onClick={capture}
                >
                  Capture Sketch
                </button>

              </div>
            )}

            {/* CROP */}
            {imageSrc && (
              <div className="crop-section">

                <h3>Crop Your Diagram</h3>

                <ReactCrop
                  crop={crop}
                  onChange={(c) => setCrop(c)}
                  onComplete={(c) =>
                    setCompletedCrop(c)
                  }
                >

                  <img
                    ref={imgRef}
                    src={imageSrc}
                    alt="crop"
                    className="crop-image"
                  />

                </ReactCrop>

              </div>
            )}

            <canvas
              ref={canvasRef}
              style={{ display: "none" }}
            />

            {/* ACTIONS */}
            <div className="upload-actions">

              {file && (
                <button
                  className="remove-btn"
                  onClick={removeFile}
                >
                  Remove File
                </button>
              )}

            </div>

            {/* ANALYZE */}
            <button
              className="analyze-btn"
              onClick={handleUpload}
            >

              {loading
                ? "Analyzing Sketch..."
                : "Analyze Sketch"}

            </button>
          </>
        )}

        {/* RESULT SCREEN */}
        {prediction && (
          <div className="result-layout">

            {/* LEFT INFO */}
            <div className="organ-info">

              <span className="badge">
                Detected Organ
              </span>

              <h2>{prediction}</h2>

              <p>
                The uploaded biological
                diagram has been analyzed
                using computer vision and
                deep learning techniques.

                The corresponding interactive
                3D anatomical structure is
                rendered for educational
                exploration and visualization.
              </p>

              {/* MISSING PARTS */}
              <div className="missing-box">

                <h3>Detected / Missing Parts</h3>

                <ul>
                  <li>Frontal Lobe</li>
                  <li>Cerebellum</li>
                  <li>Brain Stem</li>
                </ul>

              </div>

              <button
                className="remove-btn"
                onClick={analyzeAnother}
              >
                Analyze Another Sketch
              </button>

            </div>

            {/* 3D VIEWER */}
            {model && (
              <div
                className={`viewer-box ${
                  fullscreen
                    ? "fullscreen"
                    : ""
                }`}
              >

                <button
                  className="fullscreen-btn"
                  onClick={() =>
                    setFullscreen(
                      !fullscreen
                    )
                  }
                >

                  {fullscreen
                    ? "Exit"
                    : "Expand"}

                </button>

                <model-viewer
                  src={model}
                  auto-rotate
                  camera-controls
                  style={{
                    width: "100%",
                    height: "100%",
                  }}
                />

              </div>
            )}

          </div>
        )}

      </div>

    </div>
  );
}

export default UploadSection;