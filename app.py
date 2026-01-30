from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cv2
import numpy as np
from insightface.app import FaceAnalysis
import base64
import uvicorn
from threading import Lock
import os

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- GLOBAL STATE --------------------
class EnrollmentState:
    def __init__(self):
        self.face_app = None
        self.POSE_BINS = 8
        self.SIMILARITY_THRESHOLD = 0.6
        self.ALPHA = 0.3
        self.pose_filled = [False] * self.POSE_BINS
        self.bin_attempts = [0] * self.POSE_BINS
        self.identity_embedding = None
        self.count = 0
        self.is_enrolling = False
        self.last_message = ""
        self.current_bin = -1
        self.lock = Lock()
        
    def reset(self):
        with self.lock:
            self.pose_filled = [False] * self.POSE_BINS
            self.bin_attempts = [0] * self.POSE_BINS
            self.identity_embedding = None
            self.count = 0
            self.is_enrolling = True
            self.last_message = "Starting enrollment..."
            self.current_bin = -1

state = EnrollmentState()

# -------------------- PYDANTIC MODELS --------------------
class FrameData(BaseModel):
    image: str

# -------------------- HELPERS --------------------
def l2_normalize(x):
    return x / np.linalg.norm(x)

def cosine_distance(a, b):
    return 1 - np.dot(a, b)

def init_face_model():
    """Initialize the face analysis model"""
    if state.face_app is None:
        print("Loading InsightFace model...")
        state.face_app = FaceAnalysis(name="buffalo_l")
        state.face_app.prepare(ctx_id=-1, det_size=(640, 640))  # -1 for CPU
        print("Model loaded successfully!")

# -------------------- PROCESSING --------------------
def process_enrollment_frame(frame):
    """Process a single frame for enrollment"""
    with state.lock:
        if state.face_app is None:
            return {'status': 'error', 'message': 'Model not initialized'}
        
        # Detect faces
        faces = state.face_app.get(frame)
        
        # Handle no face
        if len(faces) == 0:
            state.last_message = "No face detected - move closer"
            return {
                'status': 'no_face',
                'message': state.last_message,
                'progress': (sum(state.pose_filled) / state.POSE_BINS) * 100,
                'pose_filled': state.pose_filled,
                'current_bin': -1
            }
        
        # Handle multiple faces
        if len(faces) > 1:
            state.last_message = "Multiple faces - show only one person"
            return {
                'status': 'multiple_faces',
                'message': state.last_message,
                'progress': (sum(state.pose_filled) / state.POSE_BINS) * 100,
                'pose_filled': state.pose_filled,
                'current_bin': -1
            }
        
        face = faces[0]
        
        # Check pose availability
        if not hasattr(face, 'pose') or face.pose is None:
            state.last_message = "Pose estimation unavailable"
            return {
                'status': 'no_pose',
                'message': state.last_message,
                'progress': (sum(state.pose_filled) / state.POSE_BINS) * 100,
                'pose_filled': state.pose_filled,
                'current_bin': -1
            }
        
        # Get pose bin
        yaw = face.pose[1]
        bin_index = int(((yaw + 90) / 180) * state.POSE_BINS)
        bin_index = max(0, min(state.POSE_BINS - 1, bin_index))
        state.current_bin = bin_index
        
        # Normalize embedding
        emb = l2_normalize(face.embedding)
        
        # Verify same person
        if state.identity_embedding is not None:
            similarity = 1 - cosine_distance(emb, state.identity_embedding)
            if similarity < state.SIMILARITY_THRESHOLD:
                state.last_message = "Different person detected!"
                return {
                    'status': 'different_person',
                    'message': state.last_message,
                    'progress': (sum(state.pose_filled) / state.POSE_BINS) * 100,
                    'pose_filled': state.pose_filled,
                    'current_bin': bin_index
                }
        
        # Update embedding for this bin
        if not state.pose_filled[bin_index]:
            state.pose_filled[bin_index] = True
            
            if state.identity_embedding is None:
                state.identity_embedding = emb
                state.count = 1
                state.last_message = f"Bin {bin_index + 1}/{state.POSE_BINS} captured"
            else:
                # Exponential moving average
                state.identity_embedding = state.ALPHA * emb + (1 - state.ALPHA) * state.identity_embedding
                state.identity_embedding = l2_normalize(state.identity_embedding)
                state.count += 1
                state.last_message = f"Bin {bin_index + 1}/{state.POSE_BINS} captured"
        
        # Timeout handling
        state.bin_attempts[bin_index] += 1
        if state.bin_attempts[bin_index] > 150 and not state.pose_filled[bin_index]:
            state.bin_attempts[bin_index] = 0
        
        # Check if complete
        filled_count = sum(state.pose_filled)
        is_complete = filled_count == state.POSE_BINS
        
        if is_complete:
            state.last_message = "🎉 Enrollment complete!"
            state.is_enrolling = False
        
        return {
            'status': 'success',
            'message': state.last_message,
            'progress': (filled_count / state.POSE_BINS) * 100,
            'pose_filled': state.pose_filled,
            'current_bin': bin_index,
            'is_complete': is_complete,
            'bbox': face.bbox.tolist()
        }

# -------------------- ROUTES --------------------
@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the main page"""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Face Enrollment System</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #FFEB99 0%, #FFD966 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }

        .container {
            background: white;
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            max-width: 900px;
            width: 100%;
        }

        h1 {
            text-align: center;
            color: #333;
            margin-bottom: 30px;
            font-size: 2.5em;
        }

        .video-container {
            position: relative;
            width: 100%;
            max-width: 640px;
            margin: 0 auto 30px;
            border-radius: 15px;
            overflow: hidden;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }

        #video {
            width: 100%;
            height: auto;
            display: block;
            transform: scaleX(-1);
        }

        #canvas {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            transform: scaleX(-1);
        }

        .progress-ring {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: 240px;
            height: 240px;
        }

        .controls {
            display: flex;
            gap: 15px;
            justify-content: center;
            margin-bottom: 30px;
        }

        button {
            padding: 15px 30px;
            font-size: 16px;
            border: none;
            border-radius: 10px;
            cursor: pointer;
            transition: all 0.3s;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        #startBtn {
            background: #4CAF50;
            color: white;
        }

        #startBtn:hover {
            background: #45a049;
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(76, 175, 80, 0.4);
        }

        #saveBtn {
            background: #2196F3;
            color: white;
        }

        #saveBtn:hover {
            background: #0b7dda;
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(33, 150, 243, 0.4);
        }

        #saveBtn:disabled {
            background: #ccc;
            cursor: not-allowed;
            transform: none;
        }

        .status {
            background: #f5f5f5;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }

        .status-item {
            display: flex;
            justify-content: space-between;
            padding: 10px 0;
            border-bottom: 1px solid #ddd;
        }

        .status-item:last-child {
            border-bottom: none;
        }

        .status-label {
            font-weight: 600;
            color: #555;
        }

        .status-value {
            color: #667eea;
            font-weight: 600;
        }

        #message {
            text-align: center;
            font-size: 18px;
            color: #333;
            min-height: 30px;
            font-weight: 500;
        }

        .progress-bar {
            width: 100%;
            height: 30px;
            background: #e0e0e0;
            border-radius: 15px;
            overflow: hidden;
            margin-top: 15px;
        }

        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            transition: width 0.3s;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: 600;
        }

        .instruction {
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin-bottom: 20px;
            border-radius: 5px;
        }

        .instruction p {
            color: #856404;
            margin: 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎭 Face Enrollment System</h1>
        
        <div class="instruction">
            <p><strong>Instructions:</strong> Click "Start Enrollment" and slowly turn your head left and right. The system will capture different angles of your face.</p>
        </div>

        <div class="video-container">
            <video id="video" autoplay playsinline></video>
            <canvas id="canvas"></canvas>
        </div>

        <div class="controls">
            <button id="startBtn">Start Enrollment</button>
            <button id="saveBtn" disabled>Save Embedding</button>
        </div>

        <div class="status">
            <div class="status-item">
                <span class="status-label">Status:</span>
                <span class="status-value" id="statusText">Ready</span>
            </div>
            <div class="status-item">
                <span class="status-label">Progress:</span>
                <span class="status-value" id="progressText">0%</span>
            </div>
            <div class="progress-bar">
                <div class="progress-fill" id="progressFill" style="width: 0%;">0%</div>
            </div>
        </div>

        <div id="message"></div>
    </div>

    <script>
        const video = document.getElementById('video');
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        const startBtn = document.getElementById('startBtn');
        const saveBtn = document.getElementById('saveBtn');
        const message = document.getElementById('message');
        const statusText = document.getElementById('statusText');
        const progressText = document.getElementById('progressText');
        const progressFill = document.getElementById('progressFill');

        let isEnrolling = false;
        let processingInterval = null;

        // Initialize camera
        async function initCamera() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ 
                    video: { width: 640, height: 480 } 
                });
                video.srcObject = stream;
                video.onloadedmetadata = () => {
                    canvas.width = video.videoWidth;
                    canvas.height = video.videoHeight;
                };
            } catch (err) {
                console.error('Camera error:', err);
                message.textContent = 'Error accessing camera: ' + err.message;
            }
        }

        // Start enrollment
        startBtn.addEventListener('click', async () => {
            try {
                const response = await fetch('/start_enrollment', { method: 'POST' });
                const data = await response.json();
                
                if (data.status === 'success') {
                    isEnrolling = true;
                    startBtn.disabled = true;
                    saveBtn.disabled = true;
                    statusText.textContent = 'Enrolling...';
                    message.textContent = 'Turn your head slowly left and right';
                    
                    // Start processing frames
                    processingInterval = setInterval(processFrame, 100);
                }
            } catch (err) {
                console.error('Start enrollment error:', err);
                message.textContent = 'Error starting enrollment: ' + err.message;
            }
        });

        // Save embedding
        saveBtn.addEventListener('click', async () => {
            try {
                const response = await fetch('/save_embedding', { method: 'POST' });
                const data = await response.json();
                
                if (data.status === 'success') {
                    message.textContent = `✓ Embedding saved! Samples: ${data.samples}`;
                    statusText.textContent = 'Saved';
                } else {
                    message.textContent = 'Error: ' + data.message;
                }
            } catch (err) {
                console.error('Save error:', err);
                message.textContent = 'Error saving: ' + err.message;
            }
        });

        // Process single frame
        async function processFrame() {
            if (!isEnrolling) return;

            try {
                // Draw video to canvas
                ctx.save();
                ctx.scale(-1, 1);
                ctx.drawImage(video, -canvas.width, 0, canvas.width, canvas.height);
                ctx.restore();

                // Get image data
                const imageData = canvas.toDataURL('image/jpeg', 0.8);

                // Send to server
                const response = await fetch('/process_frame', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ image: imageData })
                });

                const data = await response.json();

                // Update UI
                if (data.progress !== undefined) {
                    const progress = Math.round(data.progress);
                    progressText.textContent = progress + '%';
                    progressFill.style.width = progress + '%';
                    progressFill.textContent = progress + '%';
                }

                if (data.message) {
                    message.textContent = data.message;
                }

                // Draw visualization
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                
                if (data.bbox) {
                    // Draw face bounding box
                    ctx.save();
                    ctx.scale(-1, 1);
                    ctx.strokeStyle = '#00ff00';
                    ctx.lineWidth = 3;
                    ctx.strokeRect(-data.bbox[2], data.bbox[1], 
                                  -(data.bbox[2] - data.bbox[0]), 
                                  data.bbox[3] - data.bbox[1]);
                    ctx.restore();
                }

                // Draw ring
                if (data.pose_filled) {
                    drawRing(data.pose_filled, data.current_bin);
                }

                // Check if complete
                if (data.is_complete) {
                    isEnrolling = false;
                    clearInterval(processingInterval);
                    startBtn.disabled = false;
                    saveBtn.disabled = false;
                    statusText.textContent = 'Complete!';
                    statusText.style.color = '#4CAF50';
                }

            } catch (err) {
                console.error('Processing error:', err);
            }
        }

        // Draw progress ring
        function drawRing(poseFilled, currentBin) {
            const centerX = canvas.width / 2;
            const centerY = canvas.height / 2;
            const radius = 120;
            const binCount = poseFilled.length;

            for (let i = 0; i < binCount; i++) {
                const startAngle = (360 / binCount) * i;
                const endAngle = startAngle + (360 / binCount);

                let color;
                if (i === currentBin && !poseFilled[i]) {
                    color = '#FFFF00'; // Yellow for current
                } else if (poseFilled[i]) {
                    color = '#00FF00'; // Green for filled
                } else {
                    color = '#505050'; // Gray for unfilled
                }

                ctx.beginPath();
                ctx.arc(centerX, centerY, radius, 
                       (startAngle - 90) * Math.PI / 180, 
                       (endAngle - 90) * Math.PI / 180);
                ctx.strokeStyle = color;
                ctx.lineWidth = 12;
                ctx.stroke();
            }
        }

        // Initialize on load
        initCamera();
    </script>
</body>
</html>
    """

@app.post("/start_enrollment")
async def start_enrollment():
    """Start a new enrollment session"""
    try:
        init_face_model()
        state.reset()
        return {"status": "success", "message": "Enrollment started"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process_frame")
async def process_frame(frame_data: FrameData):
    """Process a single frame from the client"""
    try:
        if not state.is_enrolling:
            return {"status": "error", "message": "Not enrolling"}
        
        # Decode base64 image
        image_data = frame_data.image.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Process frame
        result = process_enrollment_frame(frame)
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get_status")
async def get_status():
    """Get current enrollment status"""
    with state.lock:
        filled_count = sum(state.pose_filled)
        progress = (filled_count / state.POSE_BINS) * 100
        
        return {
            "status": "success",
            "progress": progress,
            "pose_filled": state.pose_filled,
            "is_complete": filled_count == state.POSE_BINS,
            "message": state.last_message
        }

@app.post("/save_embedding")
async def save_embedding():
    """Save the final embedding"""
    try:
        if state.identity_embedding is not None:
            # Save to file
            np.save("identity_embedding.npy", state.identity_embedding)
            
            return {
                "status": "success",
                "message": "Embedding saved successfully",
                "shape": list(state.identity_embedding.shape),
                "samples": state.count
            }
        else:
            return {"status": "error", "message": "No embedding to save"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -------------------- STARTUP --------------------
if __name__ == "__main__":
    # Get port from environment variable (Render sets this)
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
