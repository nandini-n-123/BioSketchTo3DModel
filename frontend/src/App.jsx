import UploadSection from "./UploadSection";
import "./index.css";

function App() {
  const scrollToUpload = () => {
    document.querySelector(".upload-wrapper").scrollIntoView({
      behavior: "smooth",
    });
  };

  return (
    <div className="main">

      {/* BACKGROUND */}
      <div className="bg-glow"></div>
      <div className="bg-grid"></div>

      {/* NAVBAR */}
      <nav className="navbar">
        <h2>BioSketch3D</h2>
        <div className="nav-links">
          <a href="#">Home</a>
          <a href="#how">How it Works</a>
          <a href="#why">About</a>
        </div>
      </nav>

      {/* HERO */}
      <section className="hero">
        <h1>
          Convert your <span>sketches</span> into 3D
        </h1>

        <p>
          Upload hand-drawn biological diagrams and convert them into
          interactive 3D models using AI + Computer Vision.
        </p>

        <button className="primary-btn" onClick={scrollToUpload}>
          Get Started
        </button>
      </section>

      {/* HOW */}
      <section id="how" className="section">
        <h2>How it works</h2>

        <div className="grid">
          <div className="card">
            <h3>Upload</h3>
            <p>Upload your hand-drawn sketch</p>
          </div>

          <div className="card">
            <h3>Processing</h3>
            <p>OpenCV cleans and extracts structure</p>
          </div>

          <div className="card">
            <h3>AI Detection</h3>
            <p>CNN identifies the diagram</p>
          </div>

          <div className="card">
            <h3>3D Model</h3>
            <p>Interactive model is rendered</p>
          </div>
        </div>
      </section>

      {/* WHY */}
      <section id="why" className="section">
        <h2>Why this matters</h2>

        <div className="grid">
          <div className="card">
            <h3>Better Learning</h3>
            <p>Visualize complex biological structures in 3D</p>
          </div>

          <div className="card">
            <h3>Student Friendly</h3>
            <p>Turn notes into interactive understanding</p>
          </div>

          <div className="card">
            <h3>AI Powered</h3>
            <p>Combines computer vision + deep learning</p>
          </div>
        </div>
      </section>

      {/* UPLOAD */}
      <section className="upload-wrapper">
        <UploadSection />
      </section>

    </div>
  );
}

export default App;