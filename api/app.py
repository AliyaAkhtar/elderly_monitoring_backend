from fastapi import FastAPI, WebSocket
from inference import predict_action
import cv2
import asyncio
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

app = FastAPI(title="Real-Time Human Action Detection API")

# Define serious actions
SERIOUS_ACTIONS = ["staggering", "falling", "nausea/vomiting", "touch chest (stomachache/heart pain)", "touch head (headache)"]

# Email configuration
EMAIL_SENDER = "aliya10akhtar3a@gmail.com"
EMAIL_PASSWORD = "zkbk foko nncf lsfr"  
EMAIL_RECEIVER = "aminah30akhtar3a@gmail.com"
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

def send_email_alert(action):
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_SENDER
        msg['To'] = EMAIL_RECEIVER
        msg['Subject'] = f"⚠️ Alert: Serious Action Detected - {action}"

        body = f"A serious action '{action}' has been detected in real-time."
        msg.attach(MIMEText(body, 'plain'))

        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.send_message(msg)
        server.quit()

        print(f"📧 Alert email sent for action: {action}")
    except Exception as e:
        print(f"❌ Failed to send email: {e}")


@app.websocket("/ws/stream")
async def stream_actions(websocket: WebSocket):
    await websocket.accept()
    print("📶 Client connected to /ws/stream")

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        await websocket.send_text("ERROR: Webcam not accessible")
        await websocket.close()
        return

    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                await asyncio.sleep(0.01)
                continue

            frames.append(frame)

            if len(frames) >= 16:
                buffer = frames[:16]
                frames = frames[16:]

                try:
                    action = predict_action(buffer)
                except Exception as e:
                    action = f"ERROR: {e}"
                    print(action)

                print("🔎 Predicted action:", action)

                # Send email if serious
                if action in SERIOUS_ACTIONS:
                    send_email_alert(action)

                await websocket.send_text(action)

            await asyncio.sleep(0.01)
    except Exception as e:
        print("WebSocket loop ended:", e)
    finally:
        cap.release()
        await websocket.close()
        print("👋 Websocket closed, webcam released.")
