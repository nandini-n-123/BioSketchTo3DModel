import UploadSection from "./UploadSection";
import MobileScanPage from "./MobileScanPage";
import "./index.css";
import "./mobile-scan.css";

function App() {
  const mobileScanMatch = window.location.pathname.match(
    /^\/mobile-scan\/([a-zA-Z0-9_-]+)\/?$/,
  );

  if (mobileScanMatch) {
    return <MobileScanPage sessionId={mobileScanMatch[1]} />;
  }

  const scrollToUpload = () => {
    document.querySelector(".upload-wrapper")?.scrollIntoView({
      behavior: "smooth",
    });
  };

  return (
    <div className="main">
      <div className="bg-glow"></div>
      <div className="bg-grid"></div>

      <nav className="navbar">
        <h2>BioSketch3D</h2>
        <div className="nav-links">
          <a href="#home">Home</a>
          <a href="#how">How it Works</a>
          <a href="#why">About</a>
        </div>
      </nav>

      <section id="home" className="hero">
        <h1>
          Convert your <span>sketches</span> into 3D
        </h1>

        <p>
          Capture or upload a hand-drawn biological diagram and turn it into an
          interactive 3D organ model using computer vision, AI classification,
          and deterministic sketch-based deformation.
        </p>

        <button className="primary-btn" onClick={scrollToUpload}>
          Get Started
        </button>
      </section>

      <section id="how" className="section">
        <h2>How it works</h2>

        <div className="grid">
          <div className="card">
            <h3>Capture</h3>
            <p>
              Upload an image, use the laptop webcam, or scan a QR code to use a
              phone camera over the same Wi-Fi network.
            </p>
          </div>

          <div className="card">
            <h3>Crop & Process</h3>
            <p>
              Adjust the crop box if needed. The backend then cleans the sketch
              using OpenCV before classification and deformation.
            </p>
          </div>

          <div className="card">
            <h3>AI Detection</h3>
            <p>
              A CNN accepts supported Brain, Heart, or Lungs sketches and
              rejects low-confidence inputs.
            </p>
          </div>

          <div className="card">
            <h3>3D Deformation</h3>
            <p>
              A validated GLB model bends toward the sketch rather than
              generating risky anatomy from scratch.
            </p>
          </div>
        </div>
      </section>

      <section id="why" className="section">
        <h2>Why this matters</h2>

        <div className="grid">
          <div className="card">
            <h3>Create Your Own Model</h3>
            <p>
              Your drawing shapes the proportions of the final interactive organ
              model.
            </p>
          </div>

          <div className="card">
            <h3>Better Learning</h3>
            <p>
              Move from a flat diagram to a model that you can rotate, zoom, and
              inspect.
            </p>
          </div>

          <div className="card">
            <h3>Controlled Output</h3>
            <p>
              The system uses trusted baseline models and stops unsupported
              sketches before deformation.
            </p>
          </div>
        </div>
      </section>

      <section className="upload-wrapper">
        <UploadSection />
      </section>
    </div>
  );
}

export default App;
