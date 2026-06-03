import { useEffect, useState } from "react";
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

function MobileScanPage({ sessionId }) {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");
  const [status, setStatus] = useState("");
  const [sending, setSending] = useState(false);

  useEffect(() => {
    return () => {
      if (preview) URL.revokeObjectURL(preview);
    };
  }, [preview]);

  const handleFileChange = (event) => {
    const selectedFile = event.target.files?.[0];

    if (!selectedFile) return;

    if (!selectedFile.type.startsWith("image/")) {
      setStatus("Please capture or select an image file.");
      return;
    }

    if (preview) URL.revokeObjectURL(preview);

    setFile(selectedFile);
    setPreview(URL.createObjectURL(selectedFile));
    setStatus("Photo captured. Check the preview and press Send to Laptop.");
  };

  const sendToLaptop = async () => {
    if (!sessionId) {
      setStatus(
        "This QR link is invalid. Generate a new QR code on the laptop.",
      );
      return;
    }

    if (!file) {
      setStatus("Capture a sketch first.");
      return;
    }

    setSending(true);
    setStatus("Sending sketch to laptop...");

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(
        `${API_BASE_URL}/api/mobile-scan/${sessionId}/upload`,
        {
          method: "POST",
          body: formData,
        },
      );

      const data = await response.json().catch(() => ({}));

      if (!response.ok) {
        throw new Error(data.detail || "Could not send the sketch.");
      }

      setStatus("Sketch sent successfully. Return to the laptop screen.");
    } catch (error) {
      setStatus(error.message || "Could not send the sketch.");
    } finally {
      setSending(false);
    }
  };

  return (
    <main className="mobile-scan-page">
      <section className="mobile-scan-card">
        <div className="mobile-scan-badge">BioSketch3D Mobile Capture</div>

        <h1>Scan your sketch using the phone camera</h1>

        <p>
          Capture the biological sketch using your phone camera. The image will
          be sent to the laptop over the same Wi-Fi network or hotspot. On the
          laptop, adjust the crop box if needed and submit it for deformation.
        </p>

        <div className="mobile-paper-note">
          <strong>Important:</strong> for now, use a plain unruled sheet. Draw
          using a dark pen or marker, keep the paper flat, and avoid shadows.
        </div>

        <label className="mobile-capture-button">
          Capture Sketch
          <input
            hidden
            type="file"
            accept="image/*"
            capture="environment"
            onChange={handleFileChange}
          />
        </label>

        {preview && (
          <img
            src={preview}
            alt="Captured biological sketch preview"
            className="mobile-scan-preview"
          />
        )}

        <button
          type="button"
          className="mobile-send-button"
          onClick={sendToLaptop}
          disabled={!file || sending}
        >
          {sending ? "Sending..." : "Send to Laptop"}
        </button>

        {status && <p className="mobile-scan-status">{status}</p>}
      </section>
    </main>
  );
}

export default MobileScanPage;
