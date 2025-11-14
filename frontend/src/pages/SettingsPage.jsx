import React, { useState, useEffect } from 'react';
import { Settings, Bell, Users, Phone, Mail, Clock, Shield, Save, Plus, Trash2, Edit2, AlertCircle, Check, Video, Volume2, Moon } from 'lucide-react';

const SettingsPage = () => {
  const [activeTab, setActiveTab] = useState('notifications');
  const [saveStatus, setSaveStatus] = useState(null);
  
  // Notification Settings State
  const [notificationSettings, setNotificationSettings] = useState({
    emailEnabled: true,
    smsEnabled: false,
    pushEnabled: true,
    quietHoursStart: '22:00',
    quietHoursEnd: '07:00'
  });

  // System Settings State
  const [systemSettings, setSystemSettings] = useState({
    confidenceThreshold: 0.35,
    alertCooldown: 10,
    recordingEnabled: true,
    recordingDuration: 30
  });

  // Guardians State
  const [guardians, setGuardians] = useState([
    {
      id: 1,
      name: "Sarah Johnson",
      email: "sarah.johnson@email.com",
      phone: "+1 234 567 8900",
      relationship: "Daughter",
      isPrimary: true
    },
    {
      id: 2,
      name: "Michael Johnson",
      email: "michael.j@email.com",
      phone: "+1 234 567 8901",
      relationship: "Son",
      isPrimary: false
    }
  ]);

  const [isAddingGuardian, setIsAddingGuardian] = useState(false);
  const [newGuardian, setNewGuardian] = useState({
    name: '',
    email: '',
    phone: '',
    relationship: ''
  });

  // Mock API calls - replace with actual API calls
  const saveNotificationSettings = async () => {
    setSaveStatus('saving');
    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 1000));
    setSaveStatus('saved');
    setTimeout(() => setSaveStatus(null), 3000);
  };

  const saveSystemSettings = async () => {
    setSaveStatus('saving');
    await new Promise(resolve => setTimeout(resolve, 1000));
    setSaveStatus('saved');
    setTimeout(() => setSaveStatus(null), 3000);
  };

  const addGuardian = () => {
    if (newGuardian.name && newGuardian.email) {
      setGuardians([...guardians, { ...newGuardian, id: Date.now(), isPrimary: false }]);
      setNewGuardian({ name: '', email: '', phone: '', relationship: '' });
      setIsAddingGuardian(false);
      setSaveStatus('saved');
      setTimeout(() => setSaveStatus(null), 3000);
    }
  };

  const removeGuardian = (id) => {
    setGuardians(guardians.filter(g => g.id !== id));
    setSaveStatus('saved');
    setTimeout(() => setSaveStatus(null), 3000);
  };

  const setPrimaryGuardian = (id) => {
    setGuardians(guardians.map(g => ({ ...g, isPrimary: g.id === id })));
    setSaveStatus('saved');
    setTimeout(() => setSaveStatus(null), 3000);
  };

  const tabs = [
    { id: 'notifications', label: 'Notifications', icon: Bell },
    { id: 'guardians', label: 'Guardians', icon: Users },
    { id: 'system', label: 'System', icon: Settings },
    { id: 'security', label: 'Security', icon: Shield }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50">
      {/* Header */}
      <header className="bg-white shadow-md border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="bg-blue-600 p-2 rounded-lg">
                <Settings className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-gray-900">Settings</h1>
                <p className="text-sm text-gray-600">Configure your ElderWatch system</p>
              </div>
            </div>
            
            {/* Save Status */}
            {saveStatus && (
              <div className={`flex items-center space-x-2 px-4 py-2 rounded-lg ${
                saveStatus === 'saved' ? 'bg-green-100 text-green-700' : 'bg-blue-100 text-blue-700'
              }`}>
                {saveStatus === 'saved' ? (
                  <>
                    <Check className="w-4 h-4" />
                    <span className="text-sm font-medium">Saved</span>
                  </>
                ) : (
                  <>
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-700"></div>
                    <span className="text-sm font-medium">Saving...</span>
                  </>
                )}
              </div>
            )}
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-6 py-6">
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Sidebar Navigation */}
          <div className="lg:col-span-1">
            <div className="bg-white rounded-xl shadow-lg border border-gray-200 overflow-hidden sticky top-6">
              <div className="p-4">
                <h3 className="text-sm font-semibold text-gray-500 uppercase tracking-wide mb-3">
                  Settings Menu
                </h3>
                <nav className="space-y-1">
                  {tabs.map(tab => {
                    const Icon = tab.icon;
                    return (
                      <button
                        key={tab.id}
                        onClick={() => setActiveTab(tab.id)}
                        className={`w-full flex items-center space-x-3 px-4 py-3 rounded-lg transition-all ${
                          activeTab === tab.id
                            ? 'bg-blue-600 text-white shadow-md'
                            : 'text-gray-700 hover:bg-gray-100'
                        }`}
                      >
                        <Icon className="w-5 h-5" />
                        <span className="font-medium">{tab.label}</span>
                      </button>
                    );
                  })}
                </nav>
              </div>
            </div>
          </div>

          {/* Main Content */}
          <div className="lg:col-span-3">
            {/* Notifications Tab */}
            {activeTab === 'notifications' && (
              <div className="space-y-6">
                <div className="bg-white rounded-xl shadow-lg border border-gray-200 overflow-hidden">
                  <div className="bg-gradient-to-r from-blue-600 to-purple-600 px-6 py-4">
                    <h2 className="text-xl font-bold text-white flex items-center">
                      <Bell className="w-5 h-5 mr-2" />
                      Notification Preferences
                    </h2>
                  </div>

                  <div className="p-6 space-y-6">
                    {/* Email Notifications */}
                    <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                      <div className="flex items-center space-x-3">
                        <div className="bg-blue-100 p-2 rounded-lg">
                          <Mail className="w-5 h-5 text-blue-600" />
                        </div>
                        <div>
                          <h3 className="font-semibold text-gray-900">Email Notifications</h3>
                          <p className="text-sm text-gray-600">Receive alerts via email</p>
                        </div>
                      </div>
                      <label className="relative inline-flex items-center cursor-pointer">
                        <input
                          type="checkbox"
                          checked={notificationSettings.emailEnabled}
                          onChange={(e) => setNotificationSettings({
                            ...notificationSettings,
                            emailEnabled: e.target.checked
                          })}
                          className="sr-only peer"
                        />
                        <div className="w-11 h-6 bg-gray-300 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
                      </label>
                    </div>

                    {/* SMS Notifications */}
                    <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                      <div className="flex items-center space-x-3">
                        <div className="bg-green-100 p-2 rounded-lg">
                          <Phone className="w-5 h-5 text-green-600" />
                        </div>
                        <div>
                          <h3 className="font-semibold text-gray-900">SMS Notifications</h3>
                          <p className="text-sm text-gray-600">Receive urgent alerts via SMS</p>
                        </div>
                      </div>
                      <label className="relative inline-flex items-center cursor-pointer">
                        <input
                          type="checkbox"
                          checked={notificationSettings.smsEnabled}
                          onChange={(e) => setNotificationSettings({
                            ...notificationSettings,
                            smsEnabled: e.target.checked
                          })}
                          className="sr-only peer"
                        />
                        <div className="w-11 h-6 bg-gray-300 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
                      </label>
                    </div>

                    {/* Push Notifications */}
                    <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                      <div className="flex items-center space-x-3">
                        <div className="bg-purple-100 p-2 rounded-lg">
                          <Volume2 className="w-5 h-5 text-purple-600" />
                        </div>
                        <div>
                          <h3 className="font-semibold text-gray-900">Push Notifications</h3>
                          <p className="text-sm text-gray-600">Browser push notifications</p>
                        </div>
                      </div>
                      <label className="relative inline-flex items-center cursor-pointer">
                        <input
                          type="checkbox"
                          checked={notificationSettings.pushEnabled}
                          onChange={(e) => setNotificationSettings({
                            ...notificationSettings,
                            pushEnabled: e.target.checked
                          })}
                          className="sr-only peer"
                        />
                        <div className="w-11 h-6 bg-gray-300 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
                      </label>
                    </div>

                    {/* Quiet Hours */}
                    <div className="p-4 bg-gray-50 rounded-lg">
                      <div className="flex items-center space-x-3 mb-4">
                        <div className="bg-indigo-100 p-2 rounded-lg">
                          <Moon className="w-5 h-5 text-indigo-600" />
                        </div>
                        <div>
                          <h3 className="font-semibold text-gray-900">Quiet Hours</h3>
                          <p className="text-sm text-gray-600">Mute non-critical notifications</p>
                        </div>
                      </div>
                      <div className="grid grid-cols-2 gap-4">
                        <div>
                          <label className="block text-sm font-medium text-gray-700 mb-2">
                            Start Time
                          </label>
                          <input
                            type="time"
                            value={notificationSettings.quietHoursStart}
                            onChange={(e) => setNotificationSettings({
                              ...notificationSettings,
                              quietHoursStart: e.target.value
                            })}
                            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                          />
                        </div>
                        <div>
                          <label className="block text-sm font-medium text-gray-700 mb-2">
                            End Time
                          </label>
                          <input
                            type="time"
                            value={notificationSettings.quietHoursEnd}
                            onChange={(e) => setNotificationSettings({
                              ...notificationSettings,
                              quietHoursEnd: e.target.value
                            })}
                            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                          />
                        </div>
                      </div>
                    </div>

                    <div className="pt-4">
                      <button
                        onClick={saveNotificationSettings}
                        className="w-full px-6 py-3 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-lg hover:shadow-lg transition-all font-semibold flex items-center justify-center space-x-2"
                      >
                        <Save className="w-5 h-5" />
                        <span>Save Notification Settings</span>
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Guardians Tab */}
            {activeTab === 'guardians' && (
              <div className="space-y-6">
                <div className="bg-white rounded-xl shadow-lg border border-gray-200 overflow-hidden">
                  <div className="bg-gradient-to-r from-blue-600 to-purple-600 px-6 py-4">
                    <div className="flex items-center justify-between">
                      <h2 className="text-xl font-bold text-white flex items-center">
                        <Users className="w-5 h-5 mr-2" />
                        Emergency Contacts
                      </h2>
                      <button
                        onClick={() => setIsAddingGuardian(true)}
                        className="px-4 py-2 bg-white text-blue-600 rounded-lg hover:bg-blue-50 transition-colors font-semibold flex items-center space-x-2"
                      >
                        <Plus className="w-4 h-4" />
                        <span>Add Guardian</span>
                      </button>
                    </div>
                  </div>

                  <div className="p-6 space-y-4">
                    {/* Add Guardian Form */}
                    {isAddingGuardian && (
                      <div className="p-4 bg-blue-50 border-2 border-blue-200 rounded-lg space-y-4">
                        <h3 className="font-semibold text-gray-900">Add New Guardian</h3>
                        <div className="grid grid-cols-2 gap-4">
                          <input
                            type="text"
                            placeholder="Full Name"
                            value={newGuardian.name}
                            onChange={(e) => setNewGuardian({...newGuardian, name: e.target.value})}
                            className="px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                          />
                          <input
                            type="email"
                            placeholder="Email Address"
                            value={newGuardian.email}
                            onChange={(e) => setNewGuardian({...newGuardian, email: e.target.value})}
                            className="px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                          />
                          <input
                            type="tel"
                            placeholder="Phone Number"
                            value={newGuardian.phone}
                            onChange={(e) => setNewGuardian({...newGuardian, phone: e.target.value})}
                            className="px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                          />
                          <input
                            type="text"
                            placeholder="Relationship"
                            value={newGuardian.relationship}
                            onChange={(e) => setNewGuardian({...newGuardian, relationship: e.target.value})}
                            className="px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                          />
                        </div>
                        <div className="flex space-x-3">
                          <button
                            onClick={addGuardian}
                            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors font-semibold"
                          >
                            Add Guardian
                          </button>
                          <button
                            onClick={() => {
                              setIsAddingGuardian(false);
                              setNewGuardian({ name: '', email: '', phone: '', relationship: '' });
                            }}
                            className="px-4 py-2 bg-gray-300 text-gray-700 rounded-lg hover:bg-gray-400 transition-colors font-semibold"
                          >
                            Cancel
                          </button>
                        </div>
                      </div>
                    )}

                    {/* Guardian List */}
                    {guardians.map(guardian => (
                      <div 
                        key={guardian.id}
                        className={`p-4 rounded-lg border-2 ${
                          guardian.isPrimary 
                            ? 'bg-blue-50 border-blue-300' 
                            : 'bg-gray-50 border-gray-200'
                        }`}
                      >
                        <div className="flex items-center justify-between">
                          <div className="flex items-center space-x-4">
                            <div className={`w-12 h-12 rounded-full flex items-center justify-center font-bold text-white ${
                              guardian.isPrimary ? 'bg-blue-600' : 'bg-gray-400'
                            }`}>
                              {guardian.name.charAt(0)}
                            </div>
                            <div>
                              <div className="flex items-center space-x-2">
                                <h3 className="font-bold text-gray-900">{guardian.name}</h3>
                                {guardian.isPrimary && (
                                  <span className="px-2 py-0.5 bg-blue-600 text-white text-xs font-semibold rounded-full">
                                    PRIMARY
                                  </span>
                                )}
                              </div>
                              <p className="text-sm text-gray-600">{guardian.relationship}</p>
                              <div className="flex items-center space-x-4 mt-1">
                                <span className="text-sm text-gray-700 flex items-center space-x-1">
                                  <Mail className="w-3 h-3" />
                                  <span>{guardian.email}</span>
                                </span>
                                <span className="text-sm text-gray-700 flex items-center space-x-1">
                                  <Phone className="w-3 h-3" />
                                  <span>{guardian.phone}</span>
                                </span>
                              </div>
                            </div>
                          </div>
                          <div className="flex items-center space-x-2">
                            {!guardian.isPrimary && (
                              <button
                                onClick={() => setPrimaryGuardian(guardian.id)}
                                className="px-3 py-1.5 bg-blue-100 text-blue-700 rounded-lg hover:bg-blue-200 transition-colors text-sm font-semibold"
                              >
                                Set Primary
                              </button>
                            )}
                            <button
                              onClick={() => removeGuardian(guardian.id)}
                              className="p-2 bg-red-100 text-red-600 rounded-lg hover:bg-red-200 transition-colors"
                            >
                              <Trash2 className="w-4 h-4" />
                            </button>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {/* System Tab */}
            {activeTab === 'system' && (
              <div className="space-y-6">
                <div className="bg-white rounded-xl shadow-lg border border-gray-200 overflow-hidden">
                  <div className="bg-gradient-to-r from-blue-600 to-purple-600 px-6 py-4">
                    <h2 className="text-xl font-bold text-white flex items-center">
                      <Settings className="w-5 h-5 mr-2" />
                      System Configuration
                    </h2>
                  </div>

                  <div className="p-6 space-y-6">
                    {/* Confidence Threshold */}
                    <div className="p-4 bg-gray-50 rounded-lg">
                      <div className="flex items-center justify-between mb-3">
                        <div>
                          <h3 className="font-semibold text-gray-900">Confidence Threshold</h3>
                          <p className="text-sm text-gray-600">Minimum confidence to trigger alerts</p>
                        </div>
                        <span className="text-2xl font-bold text-blue-600">
                          {(systemSettings.confidenceThreshold * 100).toFixed(0)}%
                        </span>
                      </div>
                      <input
                        type="range"
                        min="0.2"
                        max="0.8"
                        step="0.05"
                        value={systemSettings.confidenceThreshold}
                        onChange={(e) => setSystemSettings({
                          ...systemSettings,
                          confidenceThreshold: parseFloat(e.target.value)
                        })}
                        className="w-full h-2 bg-gray-300 rounded-lg appearance-none cursor-pointer"
                      />
                      <div className="flex justify-between text-xs text-gray-500 mt-1">
                        <span>Less Sensitive (20%)</span>
                        <span>More Sensitive (80%)</span>
                      </div>
                    </div>

                    {/* Alert Cooldown */}
                    <div className="p-4 bg-gray-50 rounded-lg">
                      <div className="flex items-center justify-between mb-3">
                        <div>
                          <h3 className="font-semibold text-gray-900">Alert Cooldown</h3>
                          <p className="text-sm text-gray-600">Seconds between repeated alerts</p>
                        </div>
                        <span className="text-2xl font-bold text-blue-600">
                          {systemSettings.alertCooldown}s
                        </span>
                      </div>
                      <input
                        type="range"
                        min="5"
                        max="60"
                        step="5"
                        value={systemSettings.alertCooldown}
                        onChange={(e) => setSystemSettings({
                          ...systemSettings,
                          alertCooldown: parseInt(e.target.value)
                        })}
                        className="w-full h-2 bg-gray-300 rounded-lg appearance-none cursor-pointer"
                      />
                      <div className="flex justify-between text-xs text-gray-500 mt-1">
                        <span>5 seconds</span>
                        <span>60 seconds</span>
                      </div>
                    </div>

                    {/* Video Recording */}
                    <div className="p-4 bg-gray-50 rounded-lg">
                      <div className="flex items-center justify-between mb-4">
                        <div className="flex items-center space-x-3">
                          <div className="bg-red-100 p-2 rounded-lg">
                            <Video className="w-5 h-5 text-red-600" />
                          </div>
                          <div>
                            <h3 className="font-semibold text-gray-900">Automatic Recording</h3>
                            <p className="text-sm text-gray-600">Record video when alert triggered</p>
                          </div>
                        </div>
                        <label className="relative inline-flex items-center cursor-pointer">
                          <input
                            type="checkbox"
                            checked={systemSettings.recordingEnabled}
                            onChange={(e) => setSystemSettings({
                              ...systemSettings,
                              recordingEnabled: e.target.checked
                            })}
                            className="sr-only peer"
                          />
                          <div className="w-11 h-6 bg-gray-300 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
                        </label>
                      </div>
                      
                      {systemSettings.recordingEnabled && (
                        <div>
                          <label className="block text-sm font-medium text-gray-700 mb-2">
                            Recording Duration (seconds)
                          </label>
                          <input
                            type="number"
                            min="10"
                            max="60"
                            value={systemSettings.recordingDuration}
                            onChange={(e) => setSystemSettings({
                              ...systemSettings,
                              recordingDuration: parseInt(e.target.value)
                            })}
                            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                          />
                        </div>
                      )}
                    </div>

                    <div className="pt-4">
                      <button
                        onClick={saveSystemSettings}
                        className="w-full px-6 py-3 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-lg hover:shadow-lg transition-all font-semibold flex items-center justify-center space-x-2"
                      >
                        <Save className="w-5 h-5" />
                        <span>Save System Settings</span>
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Security Tab */}
            {activeTab === 'security' && (
              <div className="space-y-6">
                <div className="bg-white rounded-xl shadow-lg border border-gray-200 overflow-hidden">
                  <div className="bg-gradient-to-r from-blue-600 to-purple-600 px-6 py-4">
                    <h2 className="text-xl font-bold text-white flex items-center">
                      <Shield className="w-5 h-5 mr-2" />
                      Security & Privacy
                    </h2>
                  </div>

                  <div className="p-6 space-y-6">
                    {/* Change Password */}
                    <div className="p-4 bg-gray-50 rounded-lg">
                      <h3 className="font-semibold text-gray-900 mb-4">Change Password</h3>
                      <div className="space-y-3">
                        <input
                          type="password"
                          placeholder="Current Password"
                          className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                        />
                        <input
                          type="password"
                          placeholder="New Password"
                          className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                        />
                        <input
                          type="password"
                          placeholder="Confirm New Password"
                          className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                        />
                        <button className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors font-semibold">
                          Update Password
                        </button>
                      </div>
                    </div>

                    {/* Data Privacy */}
                    <div className="p-4 bg-gray-50 rounded-lg">
                      <h3 className="font-semibold text-gray-900 mb-3">Data & Privacy</h3>
                      <div className="space-y-3">
                        <button className="w-full px-4 py-3 bg-white border-2 border-gray-300 rounded-lg hover:border-blue-500 transition-colors text-left">
                          <div className="flex items-center justify-between">
                            <span className="font-medium text-gray-900">Download My Data</span>
                            <span className="text-sm text-gray-600">Export all recordings & logs</span>
                          </div>
                        </button>
                        <button className="w-full px-4 py-3 bg-white border-2 border-gray-300 rounded-lg hover:border-blue-500 transition-colors text-left">
                          <div className="flex items-center justify-between">
                            <span className="font-medium text-gray-900">Clear Alert History</span>
                            <span className="text-sm text-gray-600">Delete old alerts</span>
                          </div>
                        </button>
                        <button className="w-full px-4 py-3 bg-red-50 border-2 border-red-300 rounded-lg hover:border-red-500 transition-colors text-left">
                          <div className="flex items-center justify-between">
                            <span className="font-medium text-red-700">Delete Account</span>
                            <span className="text-sm text-red-600">Permanently remove data</span>
                          </div>
                        </button>
                      </div>
                    </div>

                    {/* Two-Factor Authentication */}
                    <div className="p-4 bg-gray-50 rounded-lg">
                      <div className="flex items-center justify-between mb-3">
                        <div>
                          <h3 className="font-semibold text-gray-900">Two-Factor Authentication</h3>
                          <p className="text-sm text-gray-600">Add extra security to your account</p>
                        </div>
                        <label className="relative inline-flex items-center cursor-pointer">
                          <input
                            type="checkbox"
                            className="sr-only peer"
                          />
                          <div className="w-11 h-6 bg-gray-300 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
                        </label>
                      </div>
                    </div>

                    {/* Session Management */}
                    <div className="p-4 bg-gray-50 rounded-lg">
                      <h3 className="font-semibold text-gray-900 mb-3">Active Sessions</h3>
                      <div className="space-y-2">
                        <div className="flex items-center justify-between p-3 bg-white rounded-lg border border-gray-200">
                          <div>
                            <p className="font-medium text-gray-900">Current Device</p>
                            <p className="text-xs text-gray-600">Chrome on Windows • Active now</p>
                          </div>
                          <span className="text-xs bg-green-100 text-green-700 px-2 py-1 rounded-full font-semibold">
                            Active
                          </span>
                        </div>
                        <button className="w-full px-4 py-2 bg-red-100 text-red-700 rounded-lg hover:bg-red-200 transition-colors font-semibold text-sm">
                          Sign Out All Other Sessions
                        </button>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Security Info Box */}
                <div className="bg-blue-50 border-2 border-blue-200 rounded-xl p-4">
                  <div className="flex items-start space-x-3">
                    <AlertCircle className="w-5 h-5 text-blue-600 flex-shrink-0 mt-0.5" />
                    <div>
                      <h3 className="font-semibold text-blue-900 mb-1">Security Best Practices</h3>
                      <ul className="text-sm text-blue-800 space-y-1">
                        <li>• Use a strong, unique password for your account</li>
                        <li>• Enable two-factor authentication for extra security</li>
                        <li>• Review active sessions regularly</li>
                        <li>• Never share your login credentials</li>
                      </ul>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default SettingsPage;