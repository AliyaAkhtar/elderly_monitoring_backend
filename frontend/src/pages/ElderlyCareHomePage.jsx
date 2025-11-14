import React, { useState, useEffect } from 'react';
import { Heart, Shield, Bell, Activity, Video, Phone, Clock, Users, ChevronRight, CheckCircle, AlertTriangle, Camera, Mail, Settings, BarChart3, Menu, X } from 'lucide-react';
import { Link } from "react-router-dom";

const ElderlyCareHomePage = () => {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);
  const [visibleSections, setVisibleSections] = useState({});

  // Handle scroll for navbar
  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 50);
    };
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  // Smooth scroll to section
  const scrollToSection = (sectionId) => {
    const element = document.getElementById(sectionId);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth', block: 'start' });
      setMobileMenuOpen(false);
    }
  };

  // Intersection Observer for animations
  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            setVisibleSections((prev) => ({
              ...prev,
              [entry.target.id]: true,
            }));
          }
        });
      },
      { threshold: 0.1 }
    );

    const sections = document.querySelectorAll('[data-animate]');
    sections.forEach((section) => observer.observe(section));

    return () => observer.disconnect();
  }, []);

  // Statistics (would come from backend)
  const stats = {
    activeUsers: 12,
    alertsToday: 3,
    uptime: "99.9%",
    responseTime: "< 2s"
  };

  const features = [
    {
      icon: Video,
      title: "Real-Time Monitoring",
      description: "24/7 video monitoring with AI-powered action detection for immediate awareness of activities",
      color: "blue"
    },
    {
      icon: AlertTriangle,
      title: "Instant Alerts",
      description: "Immediate notifications via email, SMS, and push notifications when serious actions are detected",
      color: "red"
    },
    {
      icon: Activity,
      title: "Health Tracking",
      description: "Track daily activities, patterns, and health indicators to ensure wellbeing",
      color: "green"
    },
    {
      icon: Phone,
      title: "Emergency Contacts",
      description: "Quick access to emergency contacts and automatic alert routing to guardians",
      color: "purple"
    },
    {
      icon: BarChart3,
      title: "Analytics Dashboard",
      description: "Comprehensive reports and insights on activity patterns and health trends",
      color: "orange"
    },
    {
      icon: Shield,
      title: "Privacy & Security",
      description: "End-to-end encryption and secure data storage to protect sensitive information",
      color: "indigo"
    }
  ];

  const criticalActions = [
    { name: "Falling", severity: "critical", count: 0 },
    { name: "Staggering", severity: "high", count: 1 },
    { name: "Chest Pain", severity: "critical", count: 0 },
    { name: "Nausea/Vomiting", severity: "high", count: 2 },
    { name: "Headache", severity: "medium", count: 0 }
  ];

  const testimonials = [
    {
      name: "Sarah Johnson",
      role: "Daughter & Guardian",
      text: "This system gives me peace of mind knowing my mother is safe even when I can't be there. The instant alerts have been life-saving.",
      rating: 5
    },
    {
      name: "Dr. Michael Chen",
      role: "Geriatric Specialist",
      text: "As a healthcare professional, I highly recommend this monitoring system. It's non-intrusive yet highly effective.",
      rating: 5
    },
    {
      name: "Robert Martinez",
      role: "Son & Caregiver",
      text: "The activity tracking helps me understand my father's daily routines and any changes in his behavior patterns.",
      rating: 5
    }
  ];

  const getColorClasses = (color) => {
    const colors = {
      blue: 'bg-blue-100 text-blue-600 border-blue-200',
      red: 'bg-red-100 text-red-600 border-red-200',
      green: 'bg-green-100 text-green-600 border-green-200',
      purple: 'bg-purple-100 text-purple-600 border-purple-200',
      orange: 'bg-orange-100 text-orange-600 border-orange-200',
      indigo: 'bg-indigo-100 text-indigo-600 border-indigo-200'
    };
    return colors[color] || colors.blue;
  };

  const getSeverityColor = (severity) => {
    const colors = {
      critical: 'bg-red-100 text-red-700 border-red-300',
      high: 'bg-orange-100 text-orange-700 border-orange-300',
      medium: 'bg-yellow-100 text-yellow-700 border-yellow-300'
    };
    return colors[severity];
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50">
      {/* Navigation Bar */}
      <nav className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${scrolled ? 'bg-white shadow-lg' : 'bg-white/80 backdrop-blur-md shadow-md'
        } border-b border-gray-200`}>
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="bg-gradient-to-br from-blue-600 to-purple-600 p-2 rounded-xl transform hover:scale-110 transition-transform">
                <Heart className="w-7 h-7 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                  ElderWatch
                </h1>
                <p className="text-xs text-gray-600">Care Beyond Distance</p>
              </div>
            </div>

            {/* Desktop Menu */}
            <div className="hidden md:flex items-center space-x-6">
              <button
                onClick={() => scrollToSection('features')}
                className="text-gray-700 hover:text-blue-600 font-medium transition-colors"
              >
                Features
              </button>
              <button
                onClick={() => scrollToSection('how-it-works')}
                className="text-gray-700 hover:text-blue-600 font-medium transition-colors"
              >
                How It Works
              </button>
              <button
                onClick={() => scrollToSection('pricing')}
                className="text-gray-700 hover:text-blue-600 font-medium transition-colors"
              >
                Pricing
              </button>
              <Link to="/settings" className="py-2">
                <button className="w-full text-left text-gray-700 hover:text-blue-600 font-medium transition-colors">
                  Settings
                </button>
              </Link>
            </div>

            <div className="flex items-center space-x-3">
              {/* <a href="/auth">
                <button className="hidden md:block px-4 py-2 text-gray-700 hover:bg-gray-100 rounded-lg transition-colors font-medium">
                  Sign In
                </button>
              </a>
              <a href="/auth">
                <button className="hidden md:block px-6 py-2 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-lg hover:shadow-lg transition-all font-medium">
                  Get Started
                </button>
              </a> */}

              {/* Mobile Menu Button */}
              <button
                onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
                className="md:hidden p-2 hover:bg-gray-100 rounded-lg transition-colors"
              >
                {mobileMenuOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
              </button>
            </div>
          </div>

          {/* Mobile Menu */}
          {mobileMenuOpen && (
            <div className="md:hidden mt-4 pb-4 border-t border-gray-200 pt-4 animate-fadeIn">
              <div className="flex flex-col space-y-3">
                <button
                  onClick={() => scrollToSection('features')}
                  className="text-left text-gray-700 hover:text-blue-600 font-medium transition-colors py-2"
                >
                  Features
                </button>
                <button
                  onClick={() => scrollToSection('how-it-works')}
                  className="text-left text-gray-700 hover:text-blue-600 font-medium transition-colors py-2"
                >
                  How It Works
                </button>
                <button
                  onClick={() => scrollToSection('pricing')}
                  className="text-left text-gray-700 hover:text-blue-600 font-medium transition-colors py-2"
                >
                  Pricing
                </button>
                <button
                  onClick={() => scrollToSection('contact')}
                  className="text-left text-gray-700 hover:text-blue-600 font-medium transition-colors py-2"
                >
                  Contact
                </button>
                {/* <Link to="/settings" className="py-2">
                  <button className="w-full text-left text-gray-700 hover:text-blue-600 font-medium transition-colors">
                    Sign In
                  </button>
                </Link>
                <Link to="/monitor">
                  <button className="w-full px-6 py-2 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-lg hover:shadow-lg transition-all font-medium">
                    Get Started
                  </button>
                </Link> */}
              </div>
            </div>
          )}
        </div>
      </nav>

      {/* Add padding to account for fixed navbar */}
      <div className="h-20"></div>

      {/* Hero Section */}
      <section className="max-w-7xl mx-auto px-6 py-16" data-animate id="hero">
        <div className={`grid md:grid-cols-2 gap-12 items-center transition-all duration-1000 ${visibleSections['hero'] ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-10'
          }`}>
          <div className="space-y-6">
            <div className="inline-flex items-center space-x-2 bg-blue-100 text-blue-700 px-4 py-2 rounded-full text-sm font-semibold animate-bounce-slow">
              <Shield className="w-4 h-4" />
              <span>Trusted by 500+ Families</span>
            </div>

            <h1 className="text-5xl md:text-6xl font-bold text-gray-900 leading-tight">
              Peace of Mind for
              <span className="bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent"> Your Loved Ones</span>
            </h1>

            <p className="text-xl text-gray-600 leading-relaxed">
              AI-powered monitoring system that keeps elderly individuals safe with real-time activity detection, instant alerts, and 24/7 care support.
            </p>

            <div className="flex flex-wrap gap-4 pt-4">
              <Link to="/monitor">
                <button className="px-8 py-4 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-xl hover:shadow-xl hover:scale-105 transition-all font-semibold text-lg flex items-center space-x-2">
                  <span>Start Monitoring</span>
                  <ChevronRight className="w-5 h-5" />
                </button>
              </Link>
              <button
                onClick={() => scrollToSection('how-it-works')}
                className="px-8 py-4 bg-white text-gray-700 border-2 border-gray-300 rounded-xl hover:border-blue-600 hover:text-blue-600 hover:scale-105 transition-all font-semibold text-lg"
              >
                Watch Demo
              </button>
            </div>

            <div className="flex items-center space-x-8 pt-6">
              <div className="transform hover:scale-110 transition-transform">
                <p className="text-3xl font-bold text-gray-900">99.9%</p>
                <p className="text-sm text-gray-600">Uptime</p>
              </div>
              <div className="w-px h-12 bg-gray-300"></div>
              <div className="transform hover:scale-110 transition-transform">
                <p className="text-3xl font-bold text-gray-900">&lt;2s</p>
                <p className="text-sm text-gray-600">Response Time</p>
              </div>
              <div className="w-px h-12 bg-gray-300"></div>
              <div className="transform hover:scale-110 transition-transform">
                <p className="text-3xl font-bold text-gray-900">24/7</p>
                <p className="text-sm text-gray-600">Monitoring</p>
              </div>
            </div>
          </div>

          <div className="relative">
            <div className="bg-gradient-to-br from-blue-600 to-purple-600 rounded-3xl p-8 shadow-2xl transform hover:scale-105 transition-transform duration-300">
              <div className="bg-white rounded-2xl p-6 space-y-4">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="font-bold text-gray-900">Live Dashboard</h3>
                  <div className="flex items-center space-x-2">
                    <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                    <span className="text-sm text-green-600 font-semibold">ACTIVE</span>
                  </div>
                </div>

                {/* Mock Dashboard */}
                <div className="grid grid-cols-2 gap-3">
                  <div className="bg-gradient-to-br from-blue-50 to-blue-100 p-4 rounded-xl border border-blue-200 transform hover:scale-105 transition-transform">
                    <Users className="w-6 h-6 text-blue-600 mb-2" />
                    <p className="text-2xl font-bold text-blue-900">{stats.activeUsers}</p>
                    <p className="text-xs text-blue-700">Active Users</p>
                  </div>
                  <div className="bg-gradient-to-br from-red-50 to-red-100 p-4 rounded-xl border border-red-200 transform hover:scale-105 transition-transform">
                    <Bell className="w-6 h-6 text-red-600 mb-2" />
                    <p className="text-2xl font-bold text-red-900">{stats.alertsToday}</p>
                    <p className="text-xs text-red-700">Alerts Today</p>
                  </div>
                </div>

                <div className="bg-gray-50 rounded-xl p-4 space-y-2">
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-gray-600">Current Status</span>
                    <span className="text-green-600 font-semibold">All Clear</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <Activity className="w-4 h-4 text-green-600" />
                    <span className="text-sm text-gray-700">Normal activity detected</span>
                  </div>
                </div>

                <div className="flex items-center justify-between text-xs text-gray-500 pt-2">
                  <span>Last updated: Just now</span>
                  <Clock className="w-4 h-4" />
                </div>
              </div>
            </div>

            {/* Floating Stats */}
            <div className="absolute -bottom-6 -left-6 bg-white rounded-2xl shadow-xl p-4 border border-gray-200 animate-float">
              <div className="flex items-center space-x-3">
                <div className="bg-green-100 p-2 rounded-lg">
                  <CheckCircle className="w-6 h-6 text-green-600" />
                </div>
                <div>
                  <p className="text-sm font-semibold text-gray-900">No Falls Detected</p>
                  <p className="text-xs text-gray-600">Last 30 days</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="max-w-7xl mx-auto px-6 py-16" data-animate id="features">
        <div className={`text-center mb-12 transition-all duration-1000 ${visibleSections['features'] ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-10'
          }`}>
          <h2 className="text-4xl font-bold text-gray-900 mb-4">
            Comprehensive Care Features
          </h2>
          <p className="text-xl text-gray-600 max-w-2xl mx-auto">
            Everything you need to ensure the safety and wellbeing of your loved ones
          </p>
        </div>

        <div className={`grid md:grid-cols-2 lg:grid-cols-3 gap-6 transition-all duration-1000 delay-200 ${visibleSections['features'] ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-10'
          }`}>
          {features.map((feature, index) => {
            const Icon = feature.icon;
            return (
              <div
                key={index}
                className="bg-white rounded-2xl p-6 shadow-lg hover:shadow-2xl transition-all border border-gray-200 hover:border-blue-300 group transform hover:-translate-y-2 duration-300"
                style={{ transitionDelay: `${index * 100}ms` }}
              >
                <div className={`inline-flex p-3 rounded-xl border-2 mb-4 ${getColorClasses(feature.color)} group-hover:scale-110 transition-transform`}>
                  <Icon className="w-6 h-6" />
                </div>
                <h3 className="text-xl font-bold text-gray-900 mb-2 group-hover:text-blue-600 transition-colors">
                  {feature.title}
                </h3>
                <p className="text-gray-600 leading-relaxed">
                  {feature.description}
                </p>
              </div>
            );
          })}
        </div>
      </section>

      {/* Monitored Actions */}
      <section className="max-w-7xl mx-auto px-6 py-16" data-animate id="monitored-actions">
        <div className={`bg-white rounded-3xl shadow-xl border border-gray-200 overflow-hidden transition-all duration-1000 ${visibleSections['monitored-actions'] ? 'opacity-100 scale-100' : 'opacity-0 scale-95'
          }`}>
          <div className="bg-gradient-to-r from-red-600 to-orange-600 px-8 py-6">
            <h2 className="text-3xl font-bold text-white mb-2">Critical Actions Monitored</h2>
            <p className="text-red-100">Real-time detection of serious health events</p>
          </div>

          <div className="p-8">
            <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
              {criticalActions.map((action, index) => (
                <div
                  key={index}
                  className={`border-2 rounded-xl p-4 ${getSeverityColor(action.severity)} transform hover:scale-105 transition-all duration-300`}
                  style={{ transitionDelay: `${index * 50}ms` }}
                >
                  <div className="flex items-center justify-between mb-2">
                    <h3 className="font-bold text-lg">{action.name}</h3>
                    <AlertTriangle className="w-5 h-5" />
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm uppercase tracking-wide font-semibold">
                      {action.severity}
                    </span>
                    <span className="text-sm">
                      Today: <strong>{action.count}</strong>
                    </span>
                  </div>
                </div>
              ))}
            </div>

            <div className="mt-6 p-4 bg-blue-50 rounded-xl border border-blue-200">
              <p className="text-sm text-blue-900">
                <strong>Note:</strong> All critical actions trigger instant notifications to designated guardians via email, SMS, and push notifications.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* How It Works */}
      <section className="max-w-7xl mx-auto px-6 py-16" data-animate id="how-it-works">
        <div className={`text-center mb-12 transition-all duration-1000 ${visibleSections['how-it-works'] ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-10'
          }`}>
          <h2 className="text-4xl font-bold text-gray-900 mb-4">
            How It Works
          </h2>
          <p className="text-xl text-gray-600">
            Simple setup, powerful protection in 3 easy steps
          </p>
        </div>

        <div className={`grid md:grid-cols-3 gap-8 transition-all duration-1000 delay-200 ${visibleSections['how-it-works'] ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-10'
          }`}>
          <div className="text-center transform hover:scale-105 transition-transform duration-300">
            <div className="bg-gradient-to-br from-blue-600 to-purple-600 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4 shadow-lg hover:shadow-2xl transition-shadow">
              <Camera className="w-8 h-8 text-white" />
            </div>
            <div className="bg-white rounded-2xl p-6 shadow-lg border border-gray-200 hover:shadow-xl transition-shadow">
              <h3 className="text-xl font-bold text-gray-900 mb-3">1. Install Camera</h3>
              <p className="text-gray-600">
                Set up the camera in common areas. Our team guides you through the entire process.
              </p>
            </div>
          </div>

          <div className="text-center transform hover:scale-105 transition-transform duration-300 md:delay-100">
            <div className="bg-gradient-to-br from-blue-600 to-purple-600 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4 shadow-lg hover:shadow-2xl transition-shadow">
              <Settings className="w-8 h-8 text-white" />
            </div>
            <div className="bg-white rounded-2xl p-6 shadow-lg border border-gray-200 hover:shadow-xl transition-shadow">
              <h3 className="text-xl font-bold text-gray-900 mb-3">2. Configure Settings</h3>
              <p className="text-gray-600">
                Add guardian contacts, set alert preferences, and customize monitoring parameters.
              </p>
            </div>
          </div>

          <div className="text-center transform hover:scale-105 transition-transform duration-300 md:delay-200">
            <div className="bg-gradient-to-br from-blue-600 to-purple-600 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4 shadow-lg hover:shadow-2xl transition-shadow">
              <Bell className="w-8 h-8 text-white" />
            </div>
            <div className="bg-white rounded-2xl p-6 shadow-lg border border-gray-200 hover:shadow-xl transition-shadow">
              <h3 className="text-xl font-bold text-gray-900 mb-3">3. Stay Connected</h3>
              <p className="text-gray-600">
                Receive instant alerts and monitor activity 24/7 from your phone or computer.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Testimonials */}
      <section className="max-w-7xl mx-auto px-6 py-16" data-animate id="testimonials">
        <div className={`text-center mb-12 transition-all duration-1000 ${visibleSections['testimonials'] ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-10'
          }`}>
          <h2 className="text-4xl font-bold text-gray-900 mb-4">
            Trusted by Families Worldwide
          </h2>
          <p className="text-xl text-gray-600">
            Hear from guardians who use ElderWatch every day
          </p>
        </div>

        <div className={`grid md:grid-cols-3 gap-6 transition-all duration-1000 delay-200 ${visibleSections['testimonials'] ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-10'
          }`}>
          {testimonials.map((testimonial, index) => (
            <div
              key={index}
              className="bg-white rounded-2xl p-6 shadow-lg border border-gray-200 hover:shadow-xl transform hover:-translate-y-2 transition-all duration-300"
              style={{ transitionDelay: `${index * 100}ms` }}
            >
              <div className="flex items-center space-x-1 mb-4">
                {[...Array(testimonial.rating)].map((_, i) => (
                  <svg key={i} className="w-5 h-5 text-yellow-400 fill-current" viewBox="0 0 20 20">
                    <path d="M10 15l-5.878 3.09 1.123-6.545L.489 6.91l6.572-.955L10 0l2.939 5.955 6.572.955-4.756 4.635 1.123 6.545z" />
                  </svg>
                ))}
              </div>
              <p className="text-gray-700 mb-4 leading-relaxed italic">
                "{testimonial.text}"
              </p>
              <div className="pt-4 border-t border-gray-200">
                <p className="font-bold text-gray-900">{testimonial.name}</p>
                <p className="text-sm text-gray-600">{testimonial.role}</p>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Pricing Section */}
      <section className="max-w-7xl mx-auto px-6 py-16" data-animate id="pricing">
        <div className={`bg-gradient-to-r from-blue-600 to-purple-600 rounded-3xl p-12 text-center shadow-2xl transition-all duration-1000 ${visibleSections['pricing'] ? 'opacity-100 scale-100' : 'opacity-0 scale-95'
          }`}>
          <h2 className="text-4xl font-bold text-white mb-4">
            Ready to Ensure Their Safety?
          </h2>
          <p className="text-xl text-blue-100 mb-8 max-w-2xl mx-auto">
            Join hundreds of families who trust ElderWatch to keep their loved ones safe
          </p>
          <div className="flex flex-wrap justify-center gap-4">
            <Link to="/monitor">
              <button className="px-8 py-4 bg-white text-blue-600 rounded-xl hover:shadow-2xl hover:scale-105 transition-all font-semibold text-lg">
                Start Free Trial
              </button>
            </Link>
            <button
              onClick={() => scrollToSection('contact')}
              className="px-8 py-4 bg-transparent text-white border-2 border-white rounded-xl hover:bg-white hover:text-blue-600 hover:scale-105 transition-all font-semibold text-lg"
            >
              Schedule Demo
            </button>
          </div>
          <p className="text-blue-100 text-sm mt-6">
            No credit card required • 30-day money-back guarantee
          </p>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-gray-900 text-white" id="contact">
        <div className="max-w-7xl mx-auto px-6 py-12">
          <div className="grid md:grid-cols-4 gap-8">
            <div>
              <div className="flex items-center space-x-2 mb-4">
                <div className="bg-gradient-to-br from-blue-600 to-purple-600 p-2 rounded-lg">
                  <Heart className="w-5 h-5 text-white" />
                </div>
                <span className="text-xl font-bold">ElderWatch</span>
              </div>
              <p className="text-gray-400 text-sm">
                Providing peace of mind through intelligent elderly care monitoring.
              </p>
            </div>

            <div>
              <h3 className="font-bold mb-4">Product</h3>
              <ul className="space-y-2 text-gray-400 text-sm">
                <li><button onClick={() => scrollToSection('features')} className="hover:text-white transition-colors">Features</button></li>
                <li><button onClick={() => scrollToSection('pricing')} className="hover:text-white transition-colors">Pricing</button></li>
                <li><button onClick={() => scrollToSection('how-it-works')} className="hover:text-white transition-colors">Demo</button></li>
                <li><a href="#" className="hover:text-white transition-colors">Documentation</a></li>
              </ul>
            </div>

            <div>
              <h3 className="font-bold mb-4">Company</h3>
              <ul className="space-y-2 text-gray-400 text-sm">
                <li><a href="#" className="hover:text-white transition-colors">About Us</a></li>
                <li><a href="#" className="hover:text-white transition-colors">Blog</a></li>
                <li><a href="#" className="hover:text-white transition-colors">Careers</a></li>
                <li><button onClick={() => scrollToSection('contact')} className="hover:text-white transition-colors">Contact</button></li>
              </ul>
            </div>

            <div>
              <h3 className="font-bold mb-4">Support</h3>
              <ul className="space-y-2 text-gray-400 text-sm">
                <li><a href="#" className="hover:text-white transition-colors">Help Center</a></li>
                <li><a href="#" className="hover:text-white transition-colors">Privacy Policy</a></li>
                <li><a href="#" className="hover:text-white transition-colors">Terms of Service</a></li>
                <li><a href="#" className="hover:text-white transition-colors">Security</a></li>
              </ul>
            </div>
          </div>

          <div className="border-t border-gray-800 mt-8 pt-8 text-center text-gray-400 text-sm">
            <p>© 2024 ElderWatch. All rights reserved.</p>
          </div>
        </div>
      </footer>

      {/* Add custom animations in style tag */}
      <style jsx>{`
        @keyframes fadeIn {
          from {
            opacity: 0;
          }
          to {
            opacity: 1;
          }
        }

        @keyframes bounce-slow {
          0%, 100% {
            transform: translateY(0);
          }
          50% {
            transform: translateY(-5px);
          }
        }

        @keyframes float {
          0%, 100% {
            transform: translateY(0px);
          }
          50% {
            transform: translateY(-10px);
          }
        }

        .animate-fadeIn {
          animation: fadeIn 0.3s ease-in-out;
        }

        .animate-bounce-slow {
          animation: bounce-slow 2s ease-in-out infinite;
        }

        .animate-float {
          animation: float 3s ease-in-out infinite;
        }

        /* Smooth scroll behavior */
        html {
          scroll-behavior: smooth;
        }

        /* Custom scrollbar */
        ::-webkit-scrollbar {
          width: 10px;
        }

        ::-webkit-scrollbar-track {
          background: #f1f1f1;
        }

        ::-webkit-scrollbar-thumb {
          background: linear-gradient(to bottom, #2563eb, #9333ea);
          border-radius: 5px;
        }

        ::-webkit-scrollbar-thumb:hover {
          background: linear-gradient(to bottom, #1d4ed8, #7e22ce);
        }
      `}</style>
    </div>
  );
};

export default ElderlyCareHomePage;