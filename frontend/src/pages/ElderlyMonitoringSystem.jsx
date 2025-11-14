import React, { useState, useEffect, useRef } from 'react';
import { AlertCircle, Activity, Bell, Video, Clock, Heart, User, Settings, Phone, Mail, X } from 'lucide-react';

const ElderlyMonitoringSystem = () => {
  const [isConnected, setIsConnected] = useState(false);
  const [currentAction, setCurrentAction] = useState('Initializing...');
  const [confidence, setConfidence] = useState(0);
  const [alerts, setAlerts] = useState([]);
  const [stats, setStats] = useState({ totalDetections: 0, alertsCount: 0, sessionTime: 0 });
  const [showSettings, setShowSettings] = useState(false);
  const [alertThreshold, setAlertThreshold] = useState(0.35);
  const [cameraEnabled, setCameraEnabled] = useState(true);
  const [mode, setMode] = useState("upload"); // "realtime" or "upload"
  const [uploadedVideoResult, setUploadedVideoResult] = useState(null);
  const [uploadedVideoURL, setUploadedVideoURL] = useState(null);

  const videoRef = useRef(null);
  const wsRef = useRef(null);
  const startTimeRef = useRef(null);

  const SERIOUS_ACTIONS = [
    "staggering",
    "falling",
    "nausea/vomiting",
    "touch chest (stomachache/heart pain)",
    "touch head (headache)"
  ];

  // Start session timer
  useEffect(() => {
    startTimeRef.current = Date.now();
    const timer = setInterval(() => {
      if (startTimeRef.current) {
        const elapsed = Math.floor((Date.now() - startTimeRef.current) / 1000);
        setStats(prev => ({ ...prev, sessionTime: elapsed }));
      }
    }, 1000);

    return () => clearInterval(timer);
  }, []);

  // Handle mode switching (webcam vs uploaded video)
  useEffect(() => {
    if (mode === "realtime") {
      startCamera();
    } else if (mode === "upload" && uploadedVideoURL) {
      // Stop webcam stream if running
      if (videoRef.current?.srcObject) {
        videoRef.current.srcObject.getTracks().forEach(track => track.stop());
        videoRef.current.srcObject = null;
      }
      videoRef.current.src = uploadedVideoURL;
      videoRef.current.play();
      // Close WebSocket if open
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
        setIsConnected(false);
      }
    }
  }, [mode, uploadedVideoURL]);

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      if (videoRef.current) videoRef.current.srcObject = stream;

      wsRef.current = new WebSocket('ws://localhost:8000/ws/stream');
      wsRef.current.onopen = () => {
        console.log('Connected to monitoring system');
        setIsConnected(true);
        setCurrentAction('Monitoring...');
      };
      wsRef.current.onmessage = (event) => {
        const data = event.data.split('|');
        const action = data[0];
        const conf = parseFloat(data[1]);
        setCurrentAction(action);
        setConfidence(conf);
        setStats(prev => ({ ...prev, totalDetections: prev.totalDetections + 1 }));

        if (SERIOUS_ACTIONS.includes(action) && conf >= alertThreshold) {
          addAlert(action, conf);
        }
      };
      wsRef.current.onerror = (err) => {
        console.error('WebSocket error:', err);
        setCurrentAction('Connection error');
        setIsConnected(false);
      };
      wsRef.current.onclose = () => {
        console.log('WebSocket closed');
        setIsConnected(false);
        setCurrentAction('Disconnected');
        // Auto reconnect in realtime mode
        if (mode === "realtime") setTimeout(() => startCamera(), 3000);
      };
    } catch (err) {
      console.error("Error accessing camera:", err);
    }
  };

  const stopCamera = () => {
    if (videoRef.current?.srcObject) {
      videoRef.current.srcObject.getTracks().forEach(track => track.stop());
      videoRef.current.srcObject = null;
    }
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
      setIsConnected(false);
    }
  };

  const addAlert = (action, conf) => {
    const newAlert = {
      id: Date.now(),
      action,
      confidence: conf,
      timestamp: new Date().toLocaleTimeString(),
      date: new Date().toLocaleDateString()
    };
    setAlerts(prev => [newAlert, ...prev].slice(0, 10));
    setStats(prev => ({ ...prev, alertsCount: prev.alertsCount + 1 }));
    playAlertSound();
  };

  const playAlertSound = () => {
    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const oscillator = audioContext.createOscillator();
    const gainNode = audioContext.createGain();
    oscillator.connect(gainNode);
    gainNode.connect(audioContext.destination);
    oscillator.frequency.value = 800;
    oscillator.type = 'sine';
    gainNode.gain.setValueAtTime(0.3, audioContext.currentTime);
    gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.5);
    oscillator.start(audioContext.currentTime);
    oscillator.stop(audioContext.currentTime + 0.5);
  };

  const formatTime = (seconds) => {
    const hrs = Math.floor(seconds / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    return `${hrs.toString().padStart(2, '0')}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  const getActionColor = (action) => SERIOUS_ACTIONS.includes(action) ? 'text-red-600 bg-red-50 border-red-200' : 'text-green-600 bg-green-50 border-green-200';
  const getConfidenceColor = (conf) => conf >= 0.7 ? 'bg-green-500' : conf >= 0.5 ? 'bg-yellow-500' : 'bg-orange-500';

  const handleVideoUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("http://localhost:8000/api/detect-video", {
        method: "POST",
        body: formData
      });

      const result = await response.json();
      console.log("Processed Result:", result);

      setUploadedVideoResult(result);
      setCurrentAction(result.detected_action);
      setConfidence(result.confidence);

      // ✅ Use backend processed MP4 only
      setUploadedVideoURL(result.video_url);

    } catch (err) {
      console.error("Upload failed:", err);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50">
      {/* Header */}
      <header className="bg-white shadow-md border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-6 py-4 flex justify-between items-center">
          <div className="flex items-center space-x-3">
            <div className="bg-blue-600 p-2 rounded-lg"><Heart className="w-6 h-6 text-white" /></div>
            <div>
              <h1 className="text-2xl font-bold text-gray-900">Elderly Care Monitor</h1>
              <p className="text-sm text-gray-600">Real-time Activity Detection System</p>
            </div>
          </div>
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <div className={`w-3 h-3 rounded-full ${isConnected ? 'bg-green-500 animate-pulse' : 'bg-red-500'}`}></div>
              <span className={`text-sm font-medium ${isConnected ? 'text-green-600' : 'text-red-600'}`}>
                {isConnected ? 'Connected' : 'Disconnected'}
              </span>
            </div>
            <button onClick={() => setShowSettings(true)} className="p-2 hover:bg-gray-100 rounded-lg"><Settings className="w-5 h-5 text-gray-600" /></button>
          </div>
        </div>
      </header>

      {/* Settings Modal */}
      {showSettings && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-white rounded-xl w-96 p-6 relative shadow-lg">
            <button className="absolute top-3 right-3" onClick={() => setShowSettings(false)}><X className="w-5 h-5 text-gray-600" /></button>
            <h2 className="text-xl font-bold mb-4">Settings</h2>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Alert Threshold</label>
                <input type="number" step="0.05" min="0" max="1" value={alertThreshold} onChange={e => setAlertThreshold(parseFloat(e.target.value))} className="w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500" />
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium text-gray-700">Enable Camera</span>
                <input type="checkbox" checked={cameraEnabled} onChange={e => setCameraEnabled(e.target.checked)} className="h-5 w-5 text-blue-600 border-gray-300 rounded" />
              </div>
            </div>
            <button onClick={() => setShowSettings(false)} className="mt-6 w-full bg-blue-600 text-white py-2 rounded-lg hover:bg-blue-700 transition-colors">Save Settings</button>
          </div>
        </div>
      )}

      <div className="max-w-7xl mx-auto px-6 py-6">
        {/* Stats */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
          <StatsCard title="Session Time" value={formatTime(stats.sessionTime)} Icon={Clock} />
          <StatsCard title="Total Detections" value={stats.totalDetections} Icon={Activity} />
          <StatsCard title="Alerts Triggered" value={stats.alertsCount} Icon={Bell} isRed />
          <StatsCard title="Confidence" value={`${(confidence * 100).toFixed(1)}%`} Icon={Video} />
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Live Detection Panel */}
          <div className="lg:col-span-2">
            <div className="bg-white rounded-xl shadow-lg border border-gray-200 overflow-hidden">
              <div className="bg-gradient-to-r from-blue-600 to-purple-600 px-6 py-4">
                <h2 className="text-xl font-bold text-white flex items-center"><Activity className="w-5 h-5 mr-2" />Live Activity Detection</h2>
              </div>

              <div className="p-6">
                {/* Mode Selector */}
                <div className="flex space-x-3 mt-3">
                  <button className={`px-4 py-1 rounded-lg text-sm font-semibold ${mode === "realtime" ? "bg-white text-blue-600" : "bg-blue-500/40 text-white"}`} onClick={() => { setMode("realtime"); setUploadedVideoResult(null); }}>Real-Time Mode</button>
                  <button className={`px-4 py-1 rounded-lg text-sm font-semibold ${mode === "upload" ? "bg-white text-blue-600" : "bg-blue-500/40 text-white"}`} onClick={() => { setMode("upload"); stopCamera(); }}>Upload Video</button>
                </div>

                {/* Video Display */}
                <div className="bg-gray-900 rounded-lg aspect-video mb-4 flex items-center justify-center relative overflow-hidden mt-4">
                  {mode === "realtime" && (
                    <div className="absolute top-4 left-4 flex items-center space-x-2 bg-red-600 px-3 py-1 rounded-full">
                      <div className="w-2 h-2 bg-white rounded-full animate-pulse"></div>
                      <span className="text-white text-xs font-semibold">LIVE</span>
                    </div>
                  )}
                  <video
                    ref={videoRef}
                    src={mode === "upload" ? uploadedVideoURL : undefined}
                    autoPlay={mode === "realtime"}
                    muted={mode === "realtime"}
                    controls={mode === "upload"}
                    playsInline
                    className="w-full h-full object-cover"
                  />

                  {mode === "upload" && !uploadedVideoURL && <p className="absolute text-gray-400">Upload a video to preview</p>}
                  {uploadedVideoResult && mode === "upload" && (
                    <div className="absolute bottom-4 left-4 p-3 bg-gray-50 rounded-lg border">
                      <p className="text-sm font-bold">Detected Action: {uploadedVideoResult.detected_action}</p>
                      <p className="text-xs text-gray-700">Confidence: {(uploadedVideoResult.confidence * 100).toFixed(1)}%</p>
                    </div>
                  )}

                  {mode === "upload" && <input type="file" accept="video/*" onChange={handleVideoUpload} className="absolute inset-0 opacity-0 w-full h-full cursor-pointer" />}
                </div>

                {/* Current Action Display */}
                <div className={`border-2 rounded-lg p-6 ${getActionColor(currentAction)}`}>
                  <div className="flex items-center justify-between mb-2">
                    <p className="text-sm font-semibold uppercase tracking-wide opacity-70">Current Action</p>
                    {SERIOUS_ACTIONS.includes(currentAction) && <AlertCircle className="w-5 h-5 animate-pulse" />}
                  </div>
                  <p className="text-2xl font-bold mb-3">{currentAction}</p>

                  <div className="space-y-2">
                    <div className="flex items-center justify-between text-sm">
                      <span className="font-medium">Confidence Level</span>
                      <span className="font-bold">{(confidence * 100).toFixed(1)}%</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-3 overflow-hidden">
                      <div className={`h-full transition-all duration-300 rounded-full ${getConfidenceColor(confidence)}`} style={{ width: `${confidence * 100}%` }}></div>
                    </div>
                  </div>
                </div>

                {/* Serious Actions Legend */}
                <div className="mt-4 p-4 bg-gray-50 rounded-lg">
                  <p className="text-xs font-semibold text-gray-700 mb-2">Monitored Serious Actions:</p>
                  <div className="flex flex-wrap gap-2">
                    {SERIOUS_ACTIONS.map(action => (
                      <span key={action} className="text-xs bg-red-100 text-red-700 px-2 py-1 rounded-full">{action}</span>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Alerts Panel */}
          <div className="lg:col-span-1">
            <div className="bg-white rounded-xl shadow-lg border border-gray-200 overflow-hidden">
              <div className="bg-gradient-to-r from-red-600 to-orange-600 px-6 py-4">
                <h2 className="text-xl font-bold text-white flex items-center"><Bell className="w-5 h-5 mr-2" />Alert History</h2>
              </div>
              <div className="p-4 max-h-[600px] overflow-y-auto">
                {alerts.length === 0 ? (
                  <div className="text-center py-12">
                    <Bell className="w-12 h-12 text-gray-300 mx-auto mb-3" />
                    <p className="text-gray-500 text-sm">No alerts yet</p>
                    <p className="text-gray-400 text-xs mt-1">System is monitoring...</p>
                  </div>
                ) : (
                  <div className="space-y-3">
                    {alerts.map((alert, index) => (
                      <div key={alert.id} className={`border-l-4 border-red-500 bg-red-50 p-3 rounded-r-lg transform transition-all duration-300 ${index === 0 ? 'scale-100' : 'scale-95'}`}>
                        <div className="flex items-start justify-between mb-2">
                          <div className="flex items-center space-x-2">
                            <AlertCircle className="w-4 h-4 text-red-600 flex-shrink-0 mt-0.5" />
                            <p className="font-semibold text-red-900 text-sm leading-tight">{alert.action}</p>
                          </div>
                          {index === 0 && <span className="bg-red-600 text-white text-xs px-2 py-0.5 rounded-full font-semibold">NEW</span>}
                        </div>
                        <div className="ml-6 space-y-1">
                          <p className="text-xs text-gray-600"><strong>Confidence:</strong> {(alert.confidence * 100).toFixed(1)}%</p>
                          <p className="text-xs text-gray-500">{alert.date} at {alert.timestamp}</p>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {/* Emergency Contacts */}
              <div className="mt-6 bg-white rounded-xl shadow-lg border border-gray-200 overflow-hidden">
                <div className="bg-gradient-to-r from-purple-600 to-blue-600 px-6 py-4">
                  <h2 className="text-lg font-bold text-white flex items-center"><User className="w-5 h-5 mr-2" />Emergency Contacts</h2>
                </div>
                <div className="p-4 space-y-3">
                  <ContactCard name="Guardian" value="+1 234 567 8900" Icon={Phone} color="blue" />
                  <ContactCard name="Email Alert" value="guardian@email.com" Icon={Mail} color="green" />
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

// Helper components
const StatsCard = ({ title, value, Icon, isRed }) => (
  <div className={`bg-white rounded-xl shadow-md p-4 border border-gray-200`}>
    <div className="flex items-center justify-between">
      <div>
        <p className="text-sm text-gray-600 mb-1">{title}</p>
        <p className={`text-2xl font-bold ${isRed ? 'text-red-600' : 'text-gray-900'}`}>{value}</p>
      </div>
      <Icon className={`w-10 h-10 opacity-20 ${isRed ? 'text-red-500' : 'text-blue-500'}`} />
    </div>
  </div>
);

const ContactCard = ({ name, value, Icon, color }) => (
  <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors cursor-pointer">
    <div className="flex items-center space-x-3">
      <div className={`bg-${color}-100 p-2 rounded-full`}><Icon className={`w-4 h-4 text-${color}-600`} /></div>
      <div>
        <p className="text-sm font-semibold text-gray-900">{name}</p>
        <p className="text-xs text-gray-600">{value}</p>
      </div>
    </div>
  </div>
);

export default ElderlyMonitoringSystem;
