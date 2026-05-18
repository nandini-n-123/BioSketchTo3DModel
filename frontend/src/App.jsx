import UploadSection from "./UploadSection";
import "./index.css";

function App() {
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
          Capture or upload hand-drawn biological diagrams and turn them into
          interactive 3D anatomy using computer vision, organ classification,
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
            <h3>Upload</h3>
            <p>Use your laptop webcam or upload a hand-drawn sketch.</p>
          </div>

          <div className="card">
            <h3>Crop</h3>
            <p>
              The page auto-detects the sketch area, then lets you refine it.
            </p>
          </div>

          <div className="card">
            <h3>AI Detection</h3>
            <p>
              A CNN identifies whether the sketch is a brain, heart, or lungs.
            </p>
          </div>

          <div className="card">
            <h3>Deformation</h3>
            <p>
              The backend bends a validated GLB model toward your sketch instead
              of generating risky anatomy from scratch.
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
              Students can feel like their drawing shaped the 3D output while
              the base anatomy remains protected.
            </p>
          </div>

          <div className="card">
            <h3>Better Learning</h3>
            <p>
              Moving from flat notes to an inspectable model makes biological
              structure easier to understand.
            </p>
          </div>

          <div className="card">
            <h3>Fast & Safe</h3>
            <p>
              The system deforms existing models, avoiding slow generative
              pipelines and reducing anatomical hallucinations.
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
