import { useState, useRef } from "react";
import "./index.css";

// 3D
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, useGLTF } from "@react-three/drei";

// =======================
// 🧠 Brain Model
// =======================
function BrainModel() {
  const { scene } = useGLTF("/models/brain.glb");
  const ref = useRef();

  useFrame(() => {
    if (ref.current) ref.current.rotation.y += 0.01;
  });

  const handleClick = () => {
    if (ref.current) ref.current.scale.set(0.8, 0.8, 0.8); // zoom in
  };

  return (
    <primitive
      ref={ref}
      object={scene}
      scale={0.6}
      position={[0, 0, 0]}
      onClick={handleClick}
    />
  );
}

// =======================
// 🌺 Hibiscus Model
// =======================
function HibiscusModel() {
  const { scene } = useGLTF("/models/hibiscus.glb");
  const ref = useRef();

  useFrame(() => {
    if (ref.current) ref.current.rotation.y += 0.01;
  });

  const handleClick = () => {
    if (ref.current) ref.current.scale.set(0.8, 0.8, 0.8);
  };

  return (
    <primitive
      ref={ref}
      object={scene}
      scale={0.6}
      position={[0, 0, 0]}
      onClick={handleClick}
    />
  );
}

function App() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [prediction, setPrediction] = useState("");
  const [loading, setLoading] = useState(false);

  // File select
  const handleFileChange = (e) => {
    const file = e.target.files[0];
    setSelectedImage(file);
    setPreview(URL.createObjectURL(file));
  };

  // API call
  const handleUpload = async () => {
    if (!selectedImage) return;

    setLoading(true);

    const formData = new FormData();
    formData.append("file", selectedImage);

    try {
      const response = await fetch(
        "http://127.0.0.1:8000/api/classify-sketch",
        {
          method: "POST",
          body: formData,
        }
      );

      const data = await response.json();
      setPrediction(data.prediction);
    } catch (err) {
      console.error(err);
      alert("Error connecting to backend");
    }

    setLoading(false);
  };

  return (
    <div className="app">
      {/* LEFT PANEL */}
      <div className="panel left">
        <h1>Sketch2Anatomy</h1>

        <input type="file" onChange={handleFileChange} />

        {preview && (
          <img src={preview} alt="preview" className="preview" />
        )}

        <button onClick={handleUpload} disabled={!selectedImage}>
          {loading ? "Processing..." : "Identify & Visualize"}
        </button>

        {prediction && (
          <p className="result">Detected: {prediction}</p>
        )}
      </div>

      {/* RIGHT PANEL */}
      <div className="panel right">
        <h2>3D Viewer</h2>

        <Canvas camera={{ position: [0, 0, 5] }} style={{ height: "400px" }}>
          <ambientLight intensity={0.7} />
          <directionalLight position={[2, 2, 2]} intensity={1} />

          {/* SAFE MODEL SWITCHING */}
          {prediction === "brain" && <BrainModel />}
          {prediction === "hibiscus" && <HibiscusModel />}

          <OrbitControls />
        </Canvas>
      </div>
    </div>
  );
}

export default App;