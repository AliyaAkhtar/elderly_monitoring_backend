from fastapi import FastAPI, HTTPException, WebSocket, UploadFile, File
from inference import predict_action
import cv2
import asyncio
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from collections import deque, Counter
import time
from fastapi import UploadFile, File
import tempfile
from fastapi.middleware.cors import CORSMiddleware
import os
import subprocess
from collections import deque

app = FastAPI(title="Real-Time Human Action Detection API")

from fastapi.staticfiles import StaticFiles

# Serve a "processed_videos" folder
PROCESSED_VIDEO_DIR = "processed_videos"
os.makedirs(PROCESSED_VIDEO_DIR, exist_ok=True)

app.mount("/videos", StaticFiles(directory=PROCESSED_VIDEO_DIR), name="videos")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define serious actions
SERIOUS_ACTIONS = [
    "staggering", 
    "falling", 
    "nausea/vomiting", 
    "touch chest (stomachache/heart pain)", 
    "touch head (headache)"
]

# Email configuration
EMAIL_SENDER = "aliya10akhtar3a@gmail.com"
EMAIL_PASSWORD = "zkbk foko nncf lsfr"  
EMAIL_RECEIVER = "aminah30akhtar3a@gmail.com"
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

# ==================== CONFIGURATION ====================
PREDICTION_HISTORY_SIZE = 7  # Number of predictions to smooth over
CONFIDENCE_THRESHOLD = 0.35  # Minimum confidence to accept prediction
STRIDE = 4  # Process every N frames (smaller = smoother but slower)
TARGET_FPS = 30  # Target webcam FPS
ALERT_COOLDOWN = 10  # Seconds between alerts for same action
# ======================================================


class PredictionSmoother:
    """
    Handles temporal smoothing of predictions using multiple strategies
    """
    def __init__(self, history_size=PREDICTION_HISTORY_SIZE, confidence_threshold=CONFIDENCE_THRESHOLD):
        self.history = deque(maxlen=history_size)
        self.confidence_threshold = confidence_threshold
        self.last_stable_action = None
    
    def add_prediction(self, action, confidence):
        """Add new prediction with confidence"""
        self.history.append((action, confidence))
    
    def get_smoothed_prediction(self):
        """
        Get smoothed prediction using weighted voting
        Higher confidence predictions have more weight
        """
        if not self.history:
            return self.last_stable_action, 0.0
        
        # Filter by confidence threshold
        valid_predictions = [
            (action, conf) for action, conf in self.history 
            if conf >= self.confidence_threshold
        ]
        
        if not valid_predictions:
            # No confident predictions, return last stable
            return self.last_stable_action, 0.0
        
        # Weighted voting based on confidence
        action_weights = {}
        for action, confidence in valid_predictions:
            if action not in action_weights:
                action_weights[action] = 0
            action_weights[action] += confidence
        
        # Get action with highest total weight
        best_action = max(action_weights.items(), key=lambda x: x[1])
        smoothed_action = best_action[0]
        avg_confidence = best_action[1] / len(valid_predictions)
        
        # Update last stable action if we have strong consensus
        if avg_confidence >= self.confidence_threshold:
            self.last_stable_action = smoothed_action
        
        return smoothed_action, avg_confidence
    
    def reset(self):
        """Reset the smoother"""
        self.history.clear()
        self.last_stable_action = None


def send_email_alert(action):
    """Send email alert for serious action"""
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_SENDER
        msg['To'] = EMAIL_RECEIVER
        msg['Subject'] = f"⚠️ Alert: Serious Action Detected - {action}"

        body = f"""
        ⚠️ ALERT: Serious Action Detected
        
        Action: {action}
        Time: {time.strftime('%Y-%m-%d %H:%M:%S')}
        
        Please check the live feed for more details.
        
        This is an automated alert from your action detection system.
        """
        msg.attach(MIMEText(body, 'plain'))

        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.send_message(msg)
        server.quit()

        print(f"📧 Alert email sent for action: {action}")
        return True
    except Exception as e:
        print(f"❌ Failed to send email: {e}")
        return False


@app.websocket("/ws/stream")
async def stream_actions(websocket: WebSocket):
    await websocket.accept()
    print("📶 Client connected to /ws/stream")

    # Initialize webcam with specific settings
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        await websocket.send_text("ERROR: Webcam not accessible")
        await websocket.close()
        return
    
    # Set webcam properties for consistent FPS
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
    
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"🎥 Webcam initialized at {actual_fps} FPS")

    # Initialize tracking variables
    frames = deque(maxlen=16)
    smoother = PredictionSmoother()
    frame_count = 0
    last_alert_time = {}  # Track last alert time for each action
    
    # Statistics
    total_predictions = 0
    start_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                await asyncio.sleep(0.01)
                continue

            frames.append(frame)
            frame_count += 1

            # Process every STRIDE frames once we have 16 frames
            if len(frames) == 16 and frame_count % STRIDE == 0:
                try:
                    # Get prediction with confidence
                    action, confidence = predict_action(list(frames), return_confidence=True)
                    
                    # Add to smoother
                    smoother.add_prediction(action, confidence)
                    
                    # Get smoothed prediction
                    smoothed_action, avg_confidence = smoother.get_smoothed_prediction()
                    
                    total_predictions += 1
                    
                    # Log predictions
                    print(f"🔎 Raw: {action} ({confidence:.3f}) | "
                          f"Smoothed: {smoothed_action} ({avg_confidence:.3f})")
                    
                    # Send prediction to client
                    message = f"{smoothed_action}|{avg_confidence:.3f}"
                    await websocket.send_text(message)
                    
                    # Handle serious action alerts
                    if smoothed_action in SERIOUS_ACTIONS and avg_confidence >= CONFIDENCE_THRESHOLD:
                        current_time = time.time()
                        last_alert = last_alert_time.get(smoothed_action, 0)
                        
                        # Send alert if cooldown period has passed
                        if current_time - last_alert >= ALERT_COOLDOWN:
                            print(f"🚨 SERIOUS ACTION DETECTED: {smoothed_action}")
                            if send_email_alert(smoothed_action):
                                last_alert_time[smoothed_action] = current_time
                    
                except Exception as e:
                    error_msg = f"ERROR: {str(e)}"
                    print(f"❌ {error_msg}")
                    await websocket.send_text(error_msg)

            await asyncio.sleep(0.01)

    except Exception as e:
        print(f"⚠️ WebSocket loop ended: {e}")
    finally:
        # Cleanup
        cap.release()
        await websocket.close()
        
        # Print statistics
        elapsed_time = time.time() - start_time
        if total_predictions > 0:
            avg_prediction_rate = total_predictions / elapsed_time
            print(f"\n📊 Session Statistics:")
            print(f"   Total predictions: {total_predictions}")
            print(f"   Session duration: {elapsed_time:.1f}s")
            print(f"   Avg prediction rate: {avg_prediction_rate:.2f} predictions/sec")
        
        print("👋 Websocket closed, webcam released.")


@app.get("/")
async def root():
    return {
        "message": "Real-Time Action Detection API",
        "websocket_endpoint": "/ws/stream",
        "configuration": {
            "prediction_history_size": PREDICTION_HISTORY_SIZE,
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "stride": STRIDE,
            "target_fps": TARGET_FPS,
            "alert_cooldown_seconds": ALERT_COOLDOWN
        }
    }

# @app.post("/api/detect-video")
# async def detect_video(file: UploadFile = File(...)):
#     """Process uploaded video, overlay actions, convert to browser-friendly MP4, return URL."""

#     # Save uploaded video to temp file
#     suffix = os.path.splitext(file.filename)[1].lower()
#     with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_video:
#         temp_video.write(await file.read())
#         uploaded_path = temp_video.name

#     # Convert non-MP4 videos to MP4 first
#     converted_path = uploaded_path
#     if suffix != ".mp4":
#         converted_path = os.path.join(PROCESSED_VIDEO_DIR, f"{time.time_ns()}_converted.mp4")
#         subprocess.run([
#             "ffmpeg", "-y", "-i", uploaded_path,
#             "-c:v", "libx264", "-preset", "fast",
#             "-crf", "22", "-c:a", "aac", "-b:a", "128k",
#             converted_path
#         ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

#     # Open video
#     cap = cv2.VideoCapture(converted_path)
#     if not cap.isOpened():
#         os.remove(uploaded_path)
#         if converted_path != uploaded_path:
#             os.remove(converted_path)
#         raise HTTPException(status_code=400, detail="Invalid video file")

#     # Prepare OpenCV output (temporary)
#     fps = cap.get(cv2.CAP_PROP_FPS) or 25
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     temp_output_path = os.path.join(PROCESSED_VIDEO_DIR, f"{time.time_ns()}_temp.mp4")
#     fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#     out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))

#     frames = deque(maxlen=16)
#     smoother = PredictionSmoother()
#     total_frames = 0

#     try:
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             frames.append(frame)
#             total_frames += 1

#             if len(frames) == 16 and total_frames % STRIDE == 0:
#                 action, conf = predict_action(list(frames), return_confidence=True)
#                 smoother.add_prediction(action, conf)
#                 smoothed_action, avg_conf = smoother.get_smoothed_prediction()
#             else:
#                 smoothed_action, avg_conf = smoother.last_stable_action or "Analyzing...", 0.0

#             # Overlay text
#             display_text = f"{smoothed_action} ({avg_conf*100:.1f}%)"
#             cv2.putText(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
#             out.write(frame)

#     finally:
#         cap.release()
#         out.release()
#         os.remove(uploaded_path)
#         if converted_path != uploaded_path:
#             os.remove(converted_path)

#     # Convert OpenCV output to browser-friendly H.264 + AAC
#     final_output_path = os.path.join(PROCESSED_VIDEO_DIR, f"{int(time.time())}_final.mp4")
#     subprocess.run([
#         "ffmpeg", "-y", "-i", temp_output_path,
#         "-c:v", "libx264", "-preset", "fast",
#         "-crf", "22", "-c:a", "aac", "-b:a", "128k",
#         final_output_path
#     ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

#     os.remove(temp_output_path)  # Remove temp OpenCV output

#     # Return final URL
#     final_action, final_conf = smoother.get_smoothed_prediction()
#     return {
#         "status": "success",
#         "detected_action": final_action,
#         "confidence": float(f"{final_conf:.3f}"),
#         "frames_processed": total_frames,
#         "video_url": f"http://localhost:8000/videos/{os.path.basename(final_output_path)}"
#     }


@app.post("/api/detect-video")
async def detect_video(file: UploadFile = File(...)):
    """Process uploaded video, overlay actions, convert to MP4, send alerts if needed."""

    # Save uploaded video to temp file
    suffix = os.path.splitext(file.filename)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_video:
        temp_video.write(await file.read())
        uploaded_path = temp_video.name

    # Convert to MP4 if necessary
    converted_path = uploaded_path
    if suffix != ".mp4":
        converted_path = os.path.join(PROCESSED_VIDEO_DIR, f"{time.time_ns()}_converted.mp4")
        subprocess.run([
            "ffmpeg", "-y", "-i", uploaded_path,
            "-c:v", "libx264", "-preset", "fast",
            "-crf", "22",
            "-c:a", "aac", "-b:a", "128k",  
            converted_path
        ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Open video file
    cap = cv2.VideoCapture(converted_path)
    if not cap.isOpened():
        os.remove(uploaded_path)
        if converted_path != uploaded_path:
            os.remove(converted_path)
        raise HTTPException(status_code=400, detail="Invalid video file")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    temp_output_path = os.path.join(PROCESSED_VIDEO_DIR, f"{time.time_ns()}_temp.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))

    frames = deque(maxlen=16)
    smoother = PredictionSmoother()
    total_frames = 0

    # NEW 👉 Track last alert times for each serious action
    last_alert_time = {}

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frames.append(frame)
            total_frames += 1

            # Process every STRIDE frames
            if len(frames) == 16 and total_frames % STRIDE == 0:
                action, conf = predict_action(list(frames), return_confidence=True)
                smoother.add_prediction(action, conf)
                smoothed_action, avg_conf = smoother.get_smoothed_prediction()
            else:
                smoothed_action, avg_conf = smoother.last_stable_action or "Analyzing...", 0.0

            # ========== EMAIL ALERT LOGIC ADDED HERE ==========
            if (
                smoothed_action in SERIOUS_ACTIONS and
                avg_conf >= CONFIDENCE_THRESHOLD
            ):
                current = time.time()
                last_sent = last_alert_time.get(smoothed_action, 0)

                if current - last_sent > ALERT_COOLDOWN:
                    print(f"🚨 SERIOUS ACTION IN VIDEO: {smoothed_action}")
                    send_email_alert(smoothed_action)
                    last_alert_time[smoothed_action] = current
            # ==================================================

            # Overlay text on the output frame
            display_text = f"{smoothed_action} ({avg_conf*100:.1f}%)"
            cv2.putText(frame, display_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            out.write(frame)

    finally:
        cap.release()
        out.release()

        # cleanup uploaded temporary files
        os.remove(uploaded_path)
        if converted_path != uploaded_path:
            os.remove(converted_path)

    # Convert output to proper MP4 format
    final_output_path = os.path.join(PROCESSED_VIDEO_DIR, f"{int(time.time())}_final.mp4")
    subprocess.run([
        "ffmpeg", "-y", "-i", temp_output_path,
        "-c:v", "libx264", "-preset", "fast",
        "-crf", "22",
        "-c:a", "aac", "-b:a", "128k",   
        final_output_path
    ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Remove temporary cv2 output
    os.remove(temp_output_path)

    # Final smoothed result
    final_action, final_conf = smoother.get_smoothed_prediction()

    return {
        "status": "success",
        "detected_action": final_action,
        "confidence": float(f"{final_conf:.3f}"),
        "frames_processed": total_frames,
        "video_url": f"http://localhost:8000/videos/{os.path.basename(final_output_path)}"
    }



@app.get("/health")
async def health():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

