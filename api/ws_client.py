# ws_client.py
import asyncio
import websockets

async def run():
    uri = "ws://127.0.0.1:8000/ws/stream"
    async with websockets.connect(uri) as ws:
        print("Connected to server. Waiting for predictions...")
        try:
            while True:
                msg = await ws.recv()
                print("Received:", msg)
        except Exception as e:
            print("Connection closed:", e)

if __name__ == "__main__":
    asyncio.run(run())
