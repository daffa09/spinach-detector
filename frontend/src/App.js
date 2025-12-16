import { useEffect, useRef, useState } from "react";
import axios from "axios";
import "./index.css";

export default function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const overlayCanvasRef = useRef(null);
  const intervalRef = useRef(null);

  const [model, setModel] = useState("yolo11");
  const [status, setStatus] = useState("Idle");
  const [confidence, setConfidence] = useState(null);
  const [running, setRunning] = useState(false);
  const [detections, setDetections] = useState([]);

  useEffect(() => {
    navigator.mediaDevices
      .getUserMedia({ video: { width: 1280, height: 720 } })
      .then((stream) => {
        videoRef.current.srcObject = stream;
        // Wait for video to load metadata
        videoRef.current.onloadedmetadata = () => {
          // Set canvas sizes to match video
          const video = videoRef.current;
          canvasRef.current.width = video.videoWidth;
          canvasRef.current.height = video.videoHeight;
          overlayCanvasRef.current.width = video.videoWidth;
          overlayCanvasRef.current.height = video.videoHeight;
        };
      })
      .catch(() => alert("Camera access denied"));
  }, []);

  // Draw bounding boxes on overlay canvas - YOLO style
  const drawBoundingBoxes = (detectionData) => {
    const canvas = overlayCanvasRef.current;
    const ctx = canvas.getContext("2d");
    
    // Clear previous drawings
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    if (!detectionData || detectionData.length === 0) return;
    
    detectionData.forEach((detection) => {
      const x = detection.x * canvas.width;
      const y = detection.y * canvas.height;
      const width = detection.width * canvas.width;
      const height = detection.height * canvas.height;
      
      // Box color - bright green
      const boxColor = "#10b981";
      
      // Draw main bounding box with thick border
      ctx.strokeStyle = boxColor;
      ctx.lineWidth = 4;
      ctx.shadowBlur = 0;
      ctx.strokeRect(x, y, width, height);
      
      // Prepare label text
      const label = `bayam`;
      const confidenceText = `${detection.confidence.toFixed(1)}%`;
      
      // Configure text style for measuring
      ctx.font = "bold 18px Inter, sans-serif";
      const labelWidth = ctx.measureText(label).width;
      ctx.font = "600 14px Inter, sans-serif";
      const confWidth = ctx.measureText(confidenceText).width;
      
      const textPadding = 8;
      const labelHeight = 28;
      const boxWidth = Math.max(labelWidth, confWidth) + textPadding * 2 + 4;
      
      // Draw label background (filled rectangle above the box)
      ctx.fillStyle = boxColor;
      ctx.fillRect(
        x - 2, 
        y - labelHeight - 8, 
        boxWidth, 
        labelHeight + 4
      );
      
      // Draw label text (class name)
      ctx.fillStyle = "#FFFFFF";
      ctx.font = "bold 18px Inter, sans-serif";
      ctx.fillText(label, x + textPadding, y - labelHeight + 12);
      
      // Draw confidence text below label
      ctx.font = "600 14px Inter, sans-serif";
      ctx.fillText(confidenceText, x + textPadding, y - 8);
      
      // Draw corner brackets for professional YOLO look
      const cornerLength = 20;
      ctx.strokeStyle = boxColor;
      ctx.lineWidth = 5;
      ctx.lineCap = "round";
      
      // Top-left corner
      ctx.beginPath();
      ctx.moveTo(x, y + cornerLength);
      ctx.lineTo(x, y);
      ctx.lineTo(x + cornerLength, y);
      ctx.stroke();
      
      // Top-right corner
      ctx.beginPath();
      ctx.moveTo(x + width - cornerLength, y);
      ctx.lineTo(x + width, y);
      ctx.lineTo(x + width, y + cornerLength);
      ctx.stroke();
      
      // Bottom-left corner
      ctx.beginPath();
      ctx.moveTo(x, y + height - cornerLength);
      ctx.lineTo(x, y + height);
      ctx.lineTo(x + cornerLength, y + height);
      ctx.stroke();
      
      // Bottom-right corner
      ctx.beginPath();
      ctx.moveTo(x + width - cornerLength, y + height);
      ctx.lineTo(x + width, y + height);
      ctx.lineTo(x + width, y + height - cornerLength);
      ctx.stroke();
    });
  };

  const captureAndDetect = async () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");

    ctx.drawImage(video, 0, 0);

    const blob = await new Promise((resolve) =>
      canvas.toBlob(resolve, "image/jpeg")
    );

    const formData = new FormData();
    formData.append("image", blob);
    formData.append("model", model);

    try {
      const res = await axios.post("http://localhost:5000/predict", formData);

      setStatus(res.data.is_bayam ? "✅ Bayam Detected" : "🔍 Scanning...");
      setConfidence(res.data.confidence);
      setDetections(res.data.detections || []);
      
      // Draw bounding boxes
      drawBoundingBoxes(res.data.detections || []);
    } catch (error) {
      console.error("Detection error:", error);
      setStatus("❌ Error");
      setDetections([]);
      drawBoundingBoxes([]);
    }
  };

  const startDetection = () => {
    setRunning(true);
    setStatus("🔍 Starting...");
    intervalRef.current = setInterval(captureAndDetect, 500);
  };

  const stopDetection = () => {
    clearInterval(intervalRef.current);
    setRunning(false);
    setStatus("⏸️ Stopped");
    setDetections([]);
    // Clear bounding boxes
    const ctx = overlayCanvasRef.current.getContext("2d");
    ctx.clearRect(0, 0, overlayCanvasRef.current.width, overlayCanvasRef.current.height);
  };

  return (
    <div style={styles.container}>
      <div style={styles.header} className="fade-in">
        <h1 style={styles.title}>
          <span style={styles.icon}>🥬</span>
          Spinach Detector AI
        </h1>
        <p style={styles.subtitle}>Real-time Object Detection with YOLO</p>
      </div>

      <div style={styles.mainCard} className="glass fade-in">
        <div style={styles.videoContainer}>
          <video 
            ref={videoRef} 
            autoPlay 
            playsInline 
            style={styles.video}
            muted
          />
          <canvas 
            ref={overlayCanvasRef} 
            style={styles.overlayCanvas}
          />
          <canvas ref={canvasRef} style={{ display: "none" }} />
          
          {/* Status Badge Overlay */}
          <div style={{
            ...styles.statusBadge,
            background: detections.length > 0 
              ? 'linear-gradient(135deg, #10b981 0%, #059669 100%)'
              : 'linear-gradient(135deg, #64748b 0%, #475569 100%)'
          }}>
            <span style={styles.statusText}>{status}</span>
            {detections.length > 0 && (
              <span style={styles.detectionCount}>
                {detections.length} object{detections.length > 1 ? 's' : ''}
              </span>
            )}
          </div>
        </div>

        <div style={styles.controls}>
          <div style={styles.controlRow}>
            <label style={styles.label}>
              <span style={styles.labelText}>Model:</span>
              <select 
                value={model} 
                onChange={(e) => setModel(e.target.value)}
                style={styles.select}
                disabled={running}
              >
                <option value="yolo11">YOLO 11</option>
                <option value="yolo9">YOLO 9</option>
              </select>
            </label>

            {!running ? (
              <button onClick={startDetection} style={styles.buttonStart}>
                <span style={styles.buttonIcon}>▶</span>
                Start Detection
              </button>
            ) : (
              <button onClick={stopDetection} style={styles.buttonStop}>
                <span style={styles.buttonIcon}>⏸</span>
                Stop
              </button>
            )}
          </div>

          {confidence !== null && (
            <div style={styles.statsCard} className="glass">
              <div style={styles.statItem}>
                <span style={styles.statLabel}>Max Confidence</span>
                <span style={styles.statValue}>{confidence}%</span>
              </div>
              <div style={styles.statItem}>
                <span style={styles.statLabel}>Detections</span>
                <span style={styles.statValue}>{detections.length}</span>
              </div>
              <div style={styles.statItem}>
                <span style={styles.statLabel}>Model</span>
                <span style={styles.statValue}>{model.toUpperCase()}</span>
              </div>
            </div>
          )}
        </div>
      </div>

      <div style={styles.footer}>
        <p style={styles.footerText}>
          Powered by YOLO AI • Real-time Object Detection
        </p>
      </div>
    </div>
  );
}

const styles = {
  container: {
    minHeight: "100vh",
    padding: "2rem 1rem",
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    gap: "2rem",
    position: "relative",
    zIndex: 1,
  },
  header: {
    textAlign: "center",
    marginBottom: "1rem",
  },
  title: {
    fontSize: "3rem",
    fontWeight: "800",
    background: "linear-gradient(135deg, #10b981 0%, #34d399 100%)",
    WebkitBackgroundClip: "text",
    WebkitTextFillColor: "transparent",
    backgroundClip: "text",
    marginBottom: "0.5rem",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    gap: "1rem",
  },
  icon: {
    fontSize: "3.5rem",
    filter: "drop-shadow(0 0 20px rgba(16, 185, 129, 0.5))",
  },
  subtitle: {
    fontSize: "1.125rem",
    color: "#94a3b8",
    fontWeight: "400",
  },
  mainCard: {
    width: "100%",
    maxWidth: "900px",
    borderRadius: "24px",
    padding: "2rem",
    display: "flex",
    flexDirection: "column",
    gap: "2rem",
  },
  videoContainer: {
    position: "relative",
    borderRadius: "16px",
    overflow: "hidden",
    boxShadow: "0 20px 60px rgba(0, 0, 0, 0.4)",
  },
  video: {
    width: "100%",
    height: "auto",
    display: "block",
    borderRadius: "16px",
  },
  overlayCanvas: {
    position: "absolute",
    top: 0,
    left: 0,
    width: "100%",
    height: "100%",
    pointerEvents: "none",
  },
  statusBadge: {
    position: "absolute",
    top: "1rem",
    left: "1rem",
    padding: "0.75rem 1.25rem",
    borderRadius: "12px",
    backdropFilter: "blur(10px)",
    display: "flex",
    flexDirection: "column",
    gap: "0.25rem",
    boxShadow: "0 4px 20px rgba(0, 0, 0, 0.3)",
  },
  statusText: {
    fontSize: "1rem",
    fontWeight: "700",
    color: "#ffffff",
  },
  detectionCount: {
    fontSize: "0.75rem",
    fontWeight: "500",
    color: "rgba(255, 255, 255, 0.8)",
  },
  controls: {
    display: "flex",
    flexDirection: "column",
    gap: "1.5rem",
  },
  controlRow: {
    display: "flex",
    gap: "1rem",
    alignItems: "center",
    flexWrap: "wrap",
  },
  label: {
    display: "flex",
    alignItems: "center",
    gap: "0.75rem",
    flex: 1,
    minWidth: "200px",
  },
  labelText: {
    fontSize: "0.95rem",
    fontWeight: "600",
    color: "black",
  },
  select: {
    flex: 1,
    padding: "0.75rem 1rem",
    borderRadius: "12px",
    border: "2px solid rgba(255, 255, 255, 0.1)",
    background: "rgba(255, 255, 255, 0.05)",
    color: "black",
    fontSize: "0.95rem",
    fontWeight: "500",
    cursor: "pointer",
    transition: "all 0.3s ease",
    outline: "none",
  },
  buttonStart: {
    padding: "0.875rem 2rem",
    borderRadius: "12px",
    border: "none",
    background: "linear-gradient(135deg, #10b981 0%, #059669 100%)",
    color: "#ffffff",
    fontSize: "1rem",
    fontWeight: "700",
    cursor: "pointer",
    transition: "all 0.3s ease",
    display: "flex",
    alignItems: "center",
    gap: "0.5rem",
    boxShadow: "0 4px 20px rgba(16, 185, 129, 0.4)",
  },
  buttonStop: {
    padding: "0.875rem 2rem",
    borderRadius: "12px",
    border: "none",
    background: "linear-gradient(135deg, #ef4444 0%, #dc2626 100%)",
    color: "#ffffff",
    fontSize: "1rem",
    fontWeight: "700",
    cursor: "pointer",
    transition: "all 0.3s ease",
    display: "flex",
    alignItems: "center",
    gap: "0.5rem",
    boxShadow: "0 4px 20px rgba(239, 68, 68, 0.4)",
  },
  buttonIcon: {
    fontSize: "1.125rem",
  },
  statsCard: {
    padding: "1.5rem",
    borderRadius: "16px",
    display: "flex",
    justifyContent: "space-around",
    gap: "2rem",
  },
  statItem: {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    gap: "0.5rem",
  },
  statLabel: {
    fontSize: "0.875rem",
    color: "#94a3b8",
    fontWeight: "500",
  },
  statValue: {
    fontSize: "1.5rem",
    fontWeight: "700",
    background: "linear-gradient(135deg, #10b981 0%, #34d399 100%)",
    WebkitBackgroundClip: "text",
    WebkitTextFillColor: "transparent",
    backgroundClip: "text",
  },
  footer: {
    marginTop: "auto",
    textAlign: "center",
  },
  footerText: {
    fontSize: "0.875rem",
    color: "#64748b",
    fontWeight: "400",
  },
};
