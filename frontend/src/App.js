import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import SettingsPage from './pages/SettingsPage';
import ElderlyCareHomePage from './pages/ElderlyCareHomePage';
import ElderlyMonitoringSystem from './pages/ElderlyMonitoringSystem';
import './index.css'
// import AuthPages from './pages/Authpages';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<ElderlyCareHomePage />} />
        {/* <Route path="/auth" element={<AuthPages />} /> */}
        <Route path="/monitor" element={<ElderlyMonitoringSystem />} />
        <Route path="/settings" element={<SettingsPage />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;