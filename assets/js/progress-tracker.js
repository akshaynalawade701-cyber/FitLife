console.log('=== PROGRESS TRACKER LOADING ===');

// Progress Tracking and Photo Comparison System
class ProgressTracker {
  constructor() {
    this.scans = this.loadScans();
    this.photoComparison = new PhotoComparison();
  }

  // Load scans from localStorage
  loadScans() {
    const saved = localStorage.getItem('fitlife_scans');
    return saved ? JSON.parse(saved) : [];
  }

  // Save scans to localStorage
  saveScans() {
    localStorage.setItem('fitlife_scans', JSON.stringify(this.scans));
  }

  // Add new scan with photo
  async addScan(scanData, photoFile) {
    const scan = {
      id: Date.now(),
      date: new Date().toISOString(),
      ...scanData,
      photo: null,
      measurements: await this.extractMeasurements(photoFile)
    };

    // Store photo as base64
    if (photoFile) {
      scan.photo = await this.fileToBase64(photoFile);
    }

    this.scans.push(scan);
    this.saveScans();
    
    return scan;
  }

  // Convert file to base64
  fileToBase64(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.readAsDataURL(file);
      reader.onload = () => resolve(reader.result);
      reader.onerror = error => reject(error);
    });
  }

  // Extract basic measurements from photo (simplified)
  async extractMeasurements(photoFile) {
    if (!photoFile) return null;

    return new Promise((resolve) => {
      const img = new Image();
      img.onload = () => {
        // Simplified measurement extraction
        const measurements = {
          width: img.width,
          height: img.height,
          aspectRatio: img.width / img.height,
          timestamp: Date.now()
        };
        resolve(measurements);
      };
      img.src = URL.createObjectURL(photoFile);
    });
  }

  // Compare two scans
  compareScans(scanId1, scanId2) {
    const scan1 = this.scans.find(s => s.id === scanId1);
    const scan2 = this.scans.find(s => s.id === scanId2);

    if (!scan1 || !scan2) return null;

    const comparison = {
      timeSpan: Math.round((new Date(scan2.date) - new Date(scan1.date)) / (1000 * 60 * 60 * 24)),
      weightChange: scan2.weight - scan1.weight,
      bodyFatChange: scan2.bodyFatPercent - scan1.bodyFatPercent,
      muscleChange: scan2.muscleMass - scan1.muscleMass,
      visceralFatChange: scan2.visceralFat - scan1.visceralFat,
      healthScoreChange: scan2.healthScore - scan1.healthScore,
      progress: this.analyzeProgress(scan1, scan2),
      recommendations: []
    };

    // Generate recommendations based on progress
    comparison.recommendations = this.generateProgressRecommendations(comparison);

    return comparison;
  }

  // Analyze progress trends
  analyzeProgress(scan1, scan2) {
    const progress = {
      weight: this.getProgressStatus(scan2.weight - scan1.weight, 'weight'),
      bodyFat: this.getProgressStatus(scan1.bodyFatPercent - scan2.bodyFatPercent, 'bodyFat'),
      muscle: this.getProgressStatus(scan2.muscleMass - scan1.muscleMass, 'muscle'),
      overall: 'maintaining'
    };

    // Determine overall progress
    const positiveChanges = [progress.weight, progress.bodyFat, progress.muscle].filter(p => p === 'improving').length;
    const negativeChanges = [progress.weight, progress.bodyFat, progress.muscle].filter(p => p === 'declining').length;

    if (positiveChanges > negativeChanges) {
      progress.overall = 'improving';
    } else if (negativeChanges > positiveChanges) {
      progress.overall = 'declining';
    }

    return progress;
  }

  // Get progress status for a metric
  getProgressStatus(change, metric) {
    const thresholds = {
      weight: { good: -0.5, bad: 0.5 },
      bodyFat: { good: -1, bad: 1 },
      muscle: { good: 0.5, bad: -0.5 }
    };

    const threshold = thresholds[metric];
    if (metric === 'muscle') {
      if (change >= threshold.good) return 'improving';
      if (change <= threshold.bad) return 'declining';
      return 'maintaining';
    }

    if (change <= threshold.good) return 'improving';
    if (change >= threshold.bad) return 'declining';
    return 'maintaining';
  }

  // Generate progress-based recommendations
  generateProgressRecommendations(comparison) {
    const recommendations = [];

    // Weight recommendations
    if (comparison.weightChange > 1) {
      recommendations.push({
        type: 'weight',
        priority: 'high',
        message: 'Weight increased significantly. Consider reducing caloric intake by 200-300 calories daily.',
        action: 'Reduce portion sizes and increase cardio frequency'
      });
    } else if (comparison.weightChange < -1) {
      recommendations.push({
        type: 'weight',
        priority: 'medium',
        message: 'Weight decreased significantly. Monitor to ensure it\'s not muscle loss.',
        action: 'Increase protein intake and maintain strength training'
      });
    }

    // Body fat recommendations
    if (comparison.bodyFatChange > 2) {
      recommendations.push({
        type: 'bodyFat',
        priority: 'high',
        message: 'Body fat increased. Focus on fat loss through diet and cardio.',
        action: 'Increase cardio to 4-5 times per week and reduce refined carbs'
      });
    } else if (comparison.bodyFatChange < -1) {
      recommendations.push({
        type: 'bodyFat',
        priority: 'low',
        message: 'Great fat loss progress! Continue current approach.',
        action: 'Maintain current diet and exercise routine'
      });
    }

    // Muscle recommendations
    if (comparison.muscleChange < -1) {
      recommendations.push({
        type: 'muscle',
        priority: 'high',
        message: 'Muscle mass decreased. Increase protein and strength training.',
        action: 'Increase protein to 2.2g per kg body weight and strength train 3-4 times per week'
      });
    } else if (comparison.muscleChange > 1) {
      recommendations.push({
        type: 'muscle',
        priority: 'low',
        message: 'Excellent muscle gain! Keep up the great work.',
        action: 'Continue current strength training program'
      });
    }

    // General recommendations
    if (comparison.progress.overall === 'declining') {
      recommendations.push({
        type: 'general',
        priority: 'high',
        message: 'Overall progress is declining. Consider adjusting your approach.',
        action: 'Review diet and exercise routine, consider consulting a fitness professional'
      });
    }

    return recommendations;
  }

  // Get latest scan
  getLatestScan() {
    return this.scans.length > 0 ? this.scans[this.scans.length - 1] : null;
  }

  // Get scan history
  getScanHistory() {
    return this.scans.sort((a, b) => new Date(b.date) - new Date(a.date));
  }

  // Generate progress report
  generateProgressReport() {
    if (this.scans.length < 2) {
      return {
        message: 'Need at least 2 scans to generate progress report',
        scans: this.scans.length
      };
    }

    const latest = this.getLatestScan();
    const previous = this.scans[this.scans.length - 2];
    const comparison = this.compareScans(previous.id, latest.id);

    return {
      currentScan: latest,
      previousScan: previous,
      comparison,
      trends: this.analyzeTrends(),
      nextSteps: this.generateNextSteps(comparison)
    };
  }

  // Analyze long-term trends
  analyzeTrends() {
    if (this.scans.length < 3) return null;

    const recentScans = this.scans.slice(-3);
    const trends = {
      weight: this.calculateTrend(recentScans.map(s => s.weight)),
      bodyFat: this.calculateTrend(recentScans.map(s => s.bodyFatPercent)),
      muscle: this.calculateTrend(recentScans.map(s => s.muscleMass))
    };

    return trends;
  }

  // Calculate trend direction
  calculateTrend(values) {
    if (values.length < 2) return 'insufficient_data';
    
    const first = values[0];
    const last = values[values.length - 1];
    const change = last - first;
    
    if (Math.abs(change) < 0.5) return 'stable';
    return change > 0 ? 'increasing' : 'decreasing';
  }

  // Generate next steps based on progress
  generateNextSteps(comparison) {
    const nextSteps = [];

    if (comparison.progress.overall === 'improving') {
      nextSteps.push('Continue current program');
      nextSteps.push('Consider increasing workout intensity');
      nextSteps.push('Maintain consistent tracking');
    } else if (comparison.progress.overall === 'declining') {
      nextSteps.push('Review and adjust diet plan');
      nextSteps.push('Increase workout frequency');
      nextSteps.push('Consider professional guidance');
    } else {
      nextSteps.push('Maintain current approach');
      nextSteps.push('Focus on consistency');
      nextSteps.push('Track more frequently');
    }

    return nextSteps;
  }
}

// Photo Comparison System
class PhotoComparison {
  constructor() {
    this.canvas = null;
    this.ctx = null;
  }

  // Initialize canvas for photo comparison
  initializeCanvas() {
    this.canvas = document.createElement('canvas');
    this.ctx = this.canvas.getContext('2d');
    return this.canvas;
  }

  // Compare two photos side by side
  async comparePhotos(photo1, photo2, container) {
    const canvas = this.initializeCanvas();
    canvas.width = 800;
    canvas.height = 600;
    
    // Load both images
    const img1 = await this.loadImage(photo1);
    const img2 = await this.loadImage(photo2);
    
    // Draw side by side comparison
    this.ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw photo 1 (left side)
    this.ctx.drawImage(img1, 0, 0, canvas.width / 2, canvas.height);
    
    // Draw photo 2 (right side)
    this.ctx.drawImage(img2, canvas.width / 2, 0, canvas.width / 2, canvas.height);
    
    // Add labels
    this.ctx.fillStyle = 'white';
    this.ctx.font = 'bold 20px Arial';
    this.ctx.fillText('Before', 20, 30);
    this.ctx.fillText('After', canvas.width / 2 + 20, 30);
    
    // Add to container
    container.innerHTML = '';
    container.appendChild(canvas);
    
    return canvas;
  }

  // Load image from base64 or file
  loadImage(photo) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = reject;
      img.src = photo;
    });
  }

  // Analyze visual changes between photos
  analyzeVisualChanges(photo1, photo2) {
    // Simplified visual analysis
    // In a real implementation, this would use computer vision
    return {
      bodyShape: 'maintained',
      muscleDefinition: 'improved',
      posture: 'maintained',
      confidence: 'high'
    };
  }
}

// Initialize global progress tracker
window.progressTracker = new ProgressTracker();

console.log('=== PROGRESS TRACKER LOADED ===');

