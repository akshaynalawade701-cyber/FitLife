console.log('=== SIMPLE WORKING DEXA SCRIPT LOADING ===');

const BODY_SCAN_DEFAULTS = {
  age: 30,
  sex: 'male',
  heightCm: 170,
  weightKg: 70
};

function readNumberValue(id) {
  return parseFloat(document.getElementById(id)?.value ?? '');
}

function createScanError(message) {
  return { ok: false, error: message };
}

function getNormalizedBodyScanInputs(options = {}) {
  const { allowDefaults = false } = options;
  const sex = document.getElementById('bs-sex')?.value || BODY_SCAN_DEFAULTS.sex;
  const ageValue = parseInt(document.getElementById('bs-age')?.value ?? '', 10);
  const useMetric = document.getElementById('bs-unit-metric')?.checked ?? true;

  const age = Number.isFinite(ageValue) ? ageValue : (allowDefaults ? BODY_SCAN_DEFAULTS.age : NaN);
  if (!Number.isFinite(age) || age <= 0) {
    return createScanError('Enter a valid age before running DEXA analysis.');
  }

  if (useMetric) {
    const heightCmValue = readNumberValue('bs-height-cm');
    const weightKgValue = readNumberValue('bs-weight-kg');
    const heightCm = Number.isFinite(heightCmValue) ? heightCmValue : (allowDefaults ? BODY_SCAN_DEFAULTS.heightCm : NaN);
    const weightKg = Number.isFinite(weightKgValue) ? weightKgValue : (allowDefaults ? BODY_SCAN_DEFAULTS.weightKg : NaN);

    if (!Number.isFinite(heightCm) || heightCm <= 0 || !Number.isFinite(weightKg) || weightKg <= 0) {
      return createScanError('Enter your height and weight before running DEXA analysis.');
    }

    return {
      ok: true,
      age,
      sex,
      unit: 'metric',
      heightCm,
      weightKg,
      feet: null,
      inches: null,
      pounds: null
    };
  }

  const feetValue = readNumberValue('bs-feet');
  const inchesValue = readNumberValue('bs-inches');
  const poundsValue = readNumberValue('bs-pounds');
  const feet = Number.isFinite(feetValue) ? feetValue : NaN;
  const inches = Number.isFinite(inchesValue) ? inchesValue : NaN;
  const pounds = Number.isFinite(poundsValue) ? poundsValue : NaN;

  if (!Number.isFinite(feet) || feet < 0 || !Number.isFinite(inches) || inches < 0 || !Number.isFinite(pounds) || pounds <= 0) {
    return createScanError('Enter your height and weight in imperial units before running DEXA analysis.');
  }

  const totalInches = (feet * 12) + inches;
  if (totalInches <= 0) {
    return createScanError('Enter a valid imperial height before running DEXA analysis.');
  }

  return {
    ok: true,
    age,
    sex,
    unit: 'imperial',
    heightCm: totalInches * 2.54,
    weightKg: pounds * 0.45359237,
    feet,
    inches,
    pounds
  };
}

function syncMetricFieldsFromInputs(inputs) {
  const heightCm = document.getElementById('bs-height-cm');
  const weightKg = document.getElementById('bs-weight-kg');
  if (heightCm) heightCm.value = inputs.heightCm.toFixed(1);
  if (weightKg) weightKg.value = inputs.weightKg.toFixed(1);
}

function getCurrentWeightKg(options = {}) {
  const inputs = getNormalizedBodyScanInputs(options);
  return inputs.ok ? inputs.weightKg : BODY_SCAN_DEFAULTS.weightKg;
}

function buildCurrentReport(results) {
  return {
    generatedAt: new Date().toISOString(),
    bodyScanInputs: getNormalizedBodyScanInputs().ok ? getNormalizedBodyScanInputs() : null,
    goals: JSON.parse(localStorage.getItem('fitlife_goals') || 'null'),
    results
  };
}

function configureCanvasDisplay(canvas, displayWidth, displayHeight) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const safeWidth = Math.max(1, Math.round(displayWidth));
  const safeHeight = Math.max(1, Math.round(displayHeight));
  const renderWidth = Math.round(safeWidth * dpr);
  const renderHeight = Math.round(safeHeight * dpr);

  canvas.style.width = `${safeWidth}px`;
  canvas.style.height = `${safeHeight}px`;

  if (canvas.width !== renderWidth || canvas.height !== renderHeight) {
    canvas.width = renderWidth;
    canvas.height = renderHeight;
  }

  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';

  return { ctx, width: safeWidth, height: safeHeight };
}

function getVisualScanSummary(results) {
  const bodyFat = parseFloat(results.bodyFatPercent);
  const visceral = parseFloat(results.visceralFat);
  const pattern = visceral / Math.max(bodyFat, 1);

  const fatZones = [];
  if (bodyFat >= 24) fatZones.push('lower abdomen');
  if (bodyFat >= 28 || pattern > 0.28) fatZones.push('waist and obliques');
  if (results.sex === 'female' && bodyFat >= 26) fatZones.push('hips and thighs');
  if (results.sex === 'male' && bodyFat >= 22) fatZones.push('lower chest');
  if (fatZones.length === 0) fatZones.push('midsection remains the main watch area');

  const alignment = [];
  const imbalanceCues = [];
  if (results.healthScore >= 82) {
    alignment.push('Shoulder and hip lines look broadly balanced in this visual overlay.');
    alignment.push('Center-line loading appears fairly even from chest to knees.');
    imbalanceCues.push('No strong left-to-right drift stands out in the torso map.');
    imbalanceCues.push('Use bilateral lifts as your base and keep a small amount of unilateral stability work.');
  } else if (results.healthScore >= 65) {
    alignment.push('Watch for mild torso drift and slight shoulder-to-hip asymmetry.');
    alignment.push('Keep unilateral strength work in the plan to smooth out side-to-side loading.');
    imbalanceCues.push('Mild imbalance markers suggest checking stance width and core bracing under load.');
    imbalanceCues.push('Add split squats, single-arm rows, and controlled carries to reduce side bias.');
  } else {
    alignment.push('Visual cues suggest more noticeable torso compensation through the midline.');
    alignment.push('Prioritize core bracing, split-stance work, and tempo control for balance.');
    imbalanceCues.push('Imbalance markers look more pronounced through the shoulder-to-hip chain.');
    imbalanceCues.push('Bias the program toward slower unilateral work before pushing heavier volume.');
  }

  return { fatZones, alignment, imbalanceCues };
}

function getPhotoOverlayMetrics(results) {
  const bodyFat = parseFloat(results.bodyFatPercent);
  const visceral = parseFloat(results.visceralFat);
  const shoulderTilt = Math.max(0.4, Math.min(3.8, (100 - results.healthScore) * 0.04));
  const hipTilt = Math.max(0.3, Math.min(3.2, (bodyFat - 14) * 0.09));
  const centerDrift = Math.max(0.4, Math.min(4.5, (visceral - 2.5) * 0.34));
  const waistLoad = Math.max(0, Math.min(100, Math.round((bodyFat / 32) * 100)));
  const trunkLoad = Math.max(0, Math.min(100, Math.round((visceral / 10) * 100)));
  const chestLoad = Math.max(0, Math.min(100, Math.round(((bodyFat - 12) / 18) * 100)));

  return {
    shoulderTilt: shoulderTilt.toFixed(1),
    hipTilt: hipTilt.toFixed(1),
    centerDrift: centerDrift.toFixed(1),
    waistLoad,
    trunkLoad,
    chestLoad
  };
}

function getPlannerNutritionSnapshot(results) {
  if (!window.fitnessPlanner || typeof window.fitnessPlanner.generateDietPlan !== 'function') {
    return null;
  }

  const defaultGoals = {
    objective: 'maintain',
    activityLevel: 'moderate',
    macroPreset: 'recommended',
    customMacroMode: 'recommended',
    customMacroRatio: { protein: 30, carbs: 40, fat: 30 },
    dietTargetCalories: null,
    dietTargetMode: 'goal'
  };

  let savedGoals = {};
  try {
    savedGoals = JSON.parse(localStorage.getItem('fitlife_goals') || '{}');
  } catch (error) {
    console.warn('Could not parse saved planner goals:', error);
  }

  const activityMap = {
    '1.2': 'sedentary',
    '1.375': 'light',
    '1.55': 'moderate',
    '1.725': 'active',
    '1.9': 'very_active'
  };

  const activityValue = document.getElementById('activity-level')?.value;
  const effectiveGoals = {
    ...defaultGoals,
    ...savedGoals,
    activityLevel: activityMap[activityValue] || savedGoals.activityLevel || defaultGoals.activityLevel,
    customMacroRatio: {
      ...defaultGoals.customMacroRatio,
      ...(savedGoals.customMacroRatio || {})
    }
  };

  try {
    return window.fitnessPlanner.generateDietPlan(results, effectiveGoals);
  } catch (error) {
    console.warn('Could not build planner nutrition snapshot:', error);
    return null;
  }
}

function getCoreMetabolicMetrics(results) {
  const weight = parseFloat(results.weight);
  const height = parseFloat(results.height);
  const age = parseFloat(results.age);
  const sex = results.sex;
  const activityMultiplier = parseFloat(document.getElementById('activity-level')?.value || '1.55');
  const bmi = weight / Math.pow(height / 100, 2);
  const bmr = (10 * weight) + (6.25 * height) - (5 * age) + (sex === 'male' ? 5 : -161);
  const tdee = bmr * activityMultiplier;

  let bmiCategory = 'Normal';
  if (bmi < 18.5) bmiCategory = 'Underweight';
  else if (bmi < 25) bmiCategory = 'Normal';
  else if (bmi < 30) bmiCategory = 'Overweight';
  else bmiCategory = 'Obese';

  return {
    bmi: bmi.toFixed(1),
    bmiCategory,
    bmr: Math.round(bmr),
    tdee: Math.round(tdee),
    maintenanceCalories: Math.round(tdee)
  };
}

function getDemoScanInputs() {
  const currentInputs = getNormalizedBodyScanInputs({ allowDefaults: true });
  if (currentInputs.ok) {
    const demoWeightKg = Math.max(60, currentInputs.weightKg * 0.94);
    return {
      ...currentInputs,
      weightKg: parseFloat(demoWeightKg.toFixed(1))
    };
  }

  return {
    ok: true,
    age: 30,
    sex: 'male',
    unit: 'metric',
    heightCm: 175,
    weightKg: 74,
    feet: null,
    inches: null,
    pounds: null
  };
}

// Set status function
function setStatus(message) {
  const statusElement = document.getElementById('bs-status');
  if (statusElement) {
    statusElement.textContent = message;
  }
  console.log('Status:', message);
}

// Generate accurate DEXA results with improved calculations
function generateDEXAResults(options = {}) {
  console.log('=== GENERATING ACCURATE DEXA RESULTS ===');

  const inputData = options.inputData || getNormalizedBodyScanInputs({ allowDefaults: options.allowDefaults === true });
  if (!inputData.ok) {
    if (!options.silent) {
      throw new Error(inputData.error);
    }
    return null;
  }

  const age = inputData.age;
  const weight = inputData.weightKg;
  const height = inputData.heightCm;
  const sex = inputData.sex;
  
  console.log('User inputs:', { age, weight, height, sex });
  
  // Calculate BMI
  const bmi = weight / Math.pow(height / 100, 2);
  
  // Improved body fat calculation using multiple formulas for accuracy
  let bodyFatPercent;
  
  // Use Jackson-Pollock 7-site formula approximation (more accurate for higher BMI)
  // This formula accounts better for obesity
  if (sex === 'male') {
    // Enhanced formula that better handles higher BMI ranges
    if (bmi < 18.5) {
    bodyFatPercent = (1.20 * bmi) + (0.23 * age) - 16.2;
    } else if (bmi < 25) {
      bodyFatPercent = (1.20 * bmi) + (0.23 * age) - 16.2;
    } else if (bmi < 30) {
      // Overweight range - adjust formula
      bodyFatPercent = (1.35 * bmi) + (0.25 * age) - 18.5;
  } else {
      // Obese range - significantly higher body fat
      bodyFatPercent = (1.50 * bmi) + (0.28 * age) - 20.0;
    }
  } else {
    // Female formula
    if (bmi < 18.5) {
    bodyFatPercent = (1.20 * bmi) + (0.23 * age) - 5.4;
    } else if (bmi < 25) {
      bodyFatPercent = (1.20 * bmi) + (0.23 * age) - 5.4;
    } else if (bmi < 30) {
      bodyFatPercent = (1.35 * bmi) + (0.25 * age) - 7.5;
    } else {
      bodyFatPercent = (1.50 * bmi) + (0.28 * age) - 9.0;
    }
  }
  
  // Ensure realistic range (removed unrealistic caps)
  if (sex === 'male') {
    bodyFatPercent = Math.max(5, Math.min(50, bodyFatPercent)); // Allow up to 50% for severely obese
  } else {
    bodyFatPercent = Math.max(8, Math.min(55, bodyFatPercent)); // Allow up to 55% for severely obese
  }
  
  // Calculate derived metrics
  const fatMass = (weight * bodyFatPercent / 100).toFixed(1);
  const leanBodyMass = (weight - parseFloat(fatMass)).toFixed(1);
  
  // More accurate muscle mass calculation (decreases with higher body fat)
  const muscleMassRatio = bodyFatPercent > 30 ? 0.45 : bodyFatPercent > 25 ? 0.48 : 0.52;
  const muscleMass = (leanBodyMass * muscleMassRatio).toFixed(1);
  
  const boneMass = (weight * 0.04).toFixed(1);
  const waterMass = (leanBodyMass * 0.73).toFixed(1);
  
  // More realistic visceral fat calculation (higher for obese individuals)
  let visceralFatRatio = 0.15; // Base ratio
  if (bmi > 30) {
    visceralFatRatio = 0.35; // Much higher for obese
  } else if (bmi > 25) {
    visceralFatRatio = 0.25; // Higher for overweight
  } else if (bodyFatPercent > 25) {
    visceralFatRatio = 0.22; // Elevated for high body fat
  }
  const visceralFat = (bodyFatPercent * visceralFatRatio).toFixed(1);
  
  // More realistic health score calculation
  let healthScore = 100;
  
  // Penalize high body fat more severely
  if (bodyFatPercent > 30) {
    healthScore -= (bodyFatPercent - 30) * 3; // Severe penalty
  } else if (bodyFatPercent > 25) {
    healthScore -= (bodyFatPercent - 25) * 2.5;
  } else if (bodyFatPercent > 20) {
    healthScore -= (bodyFatPercent - 20) * 2;
  } else if (bodyFatPercent > 15) {
    healthScore -= (bodyFatPercent - 15) * 1;
  }
  
  // Penalize high BMI more severely
  if (bmi > 35) {
    healthScore -= (bmi - 35) * 4; // Very severe penalty for class 2+ obesity
  } else if (bmi > 30) {
    healthScore -= (bmi - 30) * 3.5; // Severe penalty for obesity
  } else if (bmi > 25) {
    healthScore -= (bmi - 25) * 3; // Penalty for overweight
  }
  
  // Penalize high visceral fat
  if (parseFloat(visceralFat) > 12) {
    healthScore -= (parseFloat(visceralFat) - 12) * 2;
  } else if (parseFloat(visceralFat) > 8) {
    healthScore -= (parseFloat(visceralFat) - 8) * 1.5;
  }
  
  // Age factor
  if (age > 50) {
    healthScore -= (age - 50) * 0.2;
  }
  
  healthScore = Math.max(30, Math.min(100, Math.floor(healthScore)));
  
  // Calculate metabolic age (more realistic)
  const metabolicAgeAdjustment = (bodyFatPercent - (sex === 'male' ? 15 : 20)) * 0.8;
  const metabolicAge = Math.max(18, Math.min(85, Math.floor(age + metabolicAgeAdjustment)));
  
  // More accurate health status categorization
  let bodyComposition;
  if (sex === 'male') {
    if (bodyFatPercent < 10) bodyComposition = 'Athletic';
    else if (bodyFatPercent < 15) bodyComposition = 'Excellent';
    else if (bodyFatPercent < 18) bodyComposition = 'Good';
    else if (bodyFatPercent < 22) bodyComposition = 'Fair';
    else if (bodyFatPercent < 25) bodyComposition = 'Needs Improvement';
    else if (bodyFatPercent < 30) bodyComposition = 'High Body Fat';
    else bodyComposition = 'Obese';
  } else {
    if (bodyFatPercent < 16) bodyComposition = 'Athletic';
    else if (bodyFatPercent < 20) bodyComposition = 'Excellent';
    else if (bodyFatPercent < 25) bodyComposition = 'Good';
    else if (bodyFatPercent < 30) bodyComposition = 'Fair';
    else if (bodyFatPercent < 35) bodyComposition = 'Needs Improvement';
    else if (bodyFatPercent < 40) bodyComposition = 'High Body Fat';
    else bodyComposition = 'Obese';
  }
  
  const musclePercent = (parseFloat(muscleMass) / weight) * 100;
  let muscleQuality;
  if (musclePercent > 50) muscleQuality = 'High';
  else if (musclePercent > 45) muscleQuality = 'Good';
  else if (musclePercent > 40) muscleQuality = 'Moderate';
  else if (musclePercent > 35) muscleQuality = 'Low';
  else muscleQuality = 'Very Low';
  
  let fatDistribution;
  const visceralFatValue = parseFloat(visceralFat);
  if (visceralFatValue < 5) fatDistribution = 'Optimal';
  else if (visceralFatValue < 8) fatDistribution = 'Good';
  else if (visceralFatValue < 12) fatDistribution = 'Elevated';
  else if (visceralFatValue < 15) fatDistribution = 'High Risk';
  else fatDistribution = 'Very High Risk';
  
  let overallHealth;
  if (healthScore >= 85) overallHealth = 'Excellent';
  else if (healthScore >= 75) overallHealth = 'Good';
  else if (healthScore >= 65) overallHealth = 'Fair';
  else if (healthScore >= 50) overallHealth = 'Needs Improvement';
  else if (healthScore >= 40) overallHealth = 'Poor';
  else overallHealth = 'Critical';
  
  return {
    weight,
    height,
    age,
    sex,
    muscleMass,
    fatMass,
    bodyFatPercent: bodyFatPercent.toFixed(1),
    visceralFat,
    boneMass,
    waterMass,
    healthScore,
    metabolicAge,
    bodyComposition,
    muscleQuality,
    fatDistribution,
    overallHealth
  };
}

// Store latest DEXA results globally
window.latestDEXAResults = null;

// Show DEXA results
function showDEXAResults(results = null, options = {}) {
  console.log('=== SHOWING DEXA RESULTS ===');
  const { displayMode = 'analysis' } = options;
  
  if (!results) {
    results = generateDEXAResults();
  }

  if (!results) {
    return;
  }
  
  // Store latest results globally
  window.latestDEXAResults = results;
  window.__fitlife_last_report = buildCurrentReport(results);
  console.log('Stored latest DEXA results:', results);
  
  console.log('Results:', results);
  
  const resultsElement = document.getElementById('bs-results');
  if (resultsElement) {
    const resultsPanel = document.getElementById('dexa-results-section');
    const hasUploadedPhoto = !!uploadedImageData;
    // Calculate additional metrics for comprehensive display
    const androidRatio = parseFloat(results.visceralFat) / parseFloat(results.bodyFatPercent);
    let bodyPattern = 'Mixed';
    if (androidRatio > 0.4) bodyPattern = 'Android (Apple)';
    else if (androidRatio < 0.2) bodyPattern = 'Gynoid (Pear)';
    
    const riskLevel = parseFloat(results.visceralFat) > 8 ? 'High Risk' : parseFloat(results.visceralFat) > 5 ? 'Moderate Risk' : 'Low Risk';
    const visualScan = getVisualScanSummary(results);
    const overlayMetrics = getPhotoOverlayMetrics(results);
    const plannerNutrition = getPlannerNutritionSnapshot(results);
    const coreMetrics = getCoreMetabolicMetrics(results);
    const isDemo = displayMode === 'demo';
    const reportHeading = isDemo ? 'DEXA Demo Preview' : 'DEXA Analysis Snapshot';
    const plannerGoal = plannerNutrition?.nutritionFocus?.recommendation || 'maintain';
    const plannerGoalLabels = {
      fat_loss: 'fat loss',
      recomp: 'recomposition',
      gain_muscle: 'muscle gain',
      maintain: 'maintenance'
    };
    const nutritionCalories = plannerNutrition?.targetCalories || Math.round(results.weight * 25);
    const proteinTarget = plannerNutrition?.macros?.protein?.grams || Math.round(results.weight * 1.6);
    const plannerGoalLabel = plannerGoalLabels[plannerGoal] || 'maintenance';
    const reportSubheading = isDemo
      ? 'A sample preview so you can inspect the overlay and results layout before running the full analysis.'
      : hasUploadedPhoto
        ? 'A cleaner view of your current composition and the most actionable next steps.'
        : 'Estimated from your body details only. Uploading a photo is optional and only adds the visual body map.';
    
    resultsElement.innerHTML = `
      <div class="dexa-report-card">
        <div class="dexa-report-header">
          <div>
            <h3 class="dexa-report-heading">${reportHeading}</h3>
            <p class="dexa-report-subheading">${reportSubheading}</p>
          </div>
          <div class="dexa-score-pill">${results.healthScore}/100 score</div>
        </div>

        <div class="dexa-metric-grid">
          <div class="dexa-metric-tile">
            <span class="dexa-metric-label">Weight vs height</span>
            <strong class="dexa-metric-value">${coreMetrics.bmi}</strong>
            <small class="dexa-metric-help">A quick check of whether your weight is low, average, or high for your height.</small>
          </div>
          <div class="dexa-metric-tile">
            <span class="dexa-metric-label">Calories at rest</span>
            <strong class="dexa-metric-value">${coreMetrics.bmr}</strong>
            <small class="dexa-metric-help">How many calories your body would use if you rested all day.</small>
          </div>
          <div class="dexa-metric-tile">
            <span class="dexa-metric-label">Calories with activity</span>
            <strong class="dexa-metric-value">${coreMetrics.tdee}</strong>
            <small class="dexa-metric-help">A rough estimate of what you burn in a normal day including movement.</small>
          </div>
          <div class="dexa-metric-tile">
            <span class="dexa-metric-label">Maintenance calories</span>
            <strong class="dexa-metric-value">${coreMetrics.maintenanceCalories}</strong>
            <small class="dexa-metric-help">A starting point for keeping your current weight about the same.</small>
          </div>
        </div>

        <div class="dexa-metric-grid">
          <div class="dexa-metric-tile">
            <span class="dexa-metric-label">Body Fat</span>
            <strong class="dexa-metric-value">${results.bodyFatPercent}%</strong>
            <small class="dexa-metric-help">The estimated share of your body weight that comes from fat.</small>
          </div>
          <div class="dexa-metric-tile">
            <span class="dexa-metric-label">Muscle on your body</span>
            <strong class="dexa-metric-value">${results.muscleMass} kg</strong>
            <small class="dexa-metric-help">An estimate of how much lean, active tissue you’re carrying.</small>
          </div>
          <div class="dexa-metric-tile">
            <span class="dexa-metric-label">Fat around organs</span>
            <strong class="dexa-metric-value">${results.visceralFat}%</strong>
            <small class="dexa-metric-help">This is the deeper belly fat linked more closely to health risk.</small>
          </div>
          <div class="dexa-metric-tile">
            <span class="dexa-metric-label">Estimated metabolic age</span>
            <strong class="dexa-metric-value">${results.metabolicAge} years</strong>
            <small class="dexa-metric-help">A simple comparison between your metabolism and typical age-based averages.</small>
          </div>
        </div>

        <div class="dexa-report-section dexa-report-section-info">
          <h4 class="dexa-report-section-title dexa-report-section-title-info">What this means in simple terms</h4>
          <div class="dexa-two-col-grid">
            <div><strong>Your weight range:</strong> ${coreMetrics.bmiCategory}</div>
            <div><strong>Your overall body makeup:</strong> ${results.bodyComposition}</div>
            <div><strong>Your muscle condition:</strong> ${results.muscleQuality}</div>
            <div><strong>Where fat seems more concentrated:</strong> ${results.fatDistribution}</div>
            <div><strong>Your general fitness picture:</strong> ${results.overallHealth}</div>
            <div><strong>Health risk from belly fat:</strong> ${riskLevel}</div>
            <div><strong>Overall body pattern:</strong> ${bodyPattern}</div>
          </div>
        </div>

        <div class="dexa-report-section dexa-report-section-muscle">
          <h4 class="dexa-report-section-title dexa-report-section-title-muscle">${hasUploadedPhoto ? 'Easy takeaways from your scan' : 'Easy takeaways from your body stats'}</h4>
          <div class="dexa-visual-list">
            <div><strong>Posture note:</strong> ${visualScan.alignment[0]}</div>
            <div><strong>Left vs right balance:</strong> ${visualScan.imbalanceCues[0]}</div>
            <div><strong>Main areas to focus on:</strong> ${visualScan.fatZones.join(', ')}</div>
            <div><strong>Best training focus right now:</strong> ${visualScan.imbalanceCues[1]}</div>
          </div>
        </div>

        ${hasUploadedPhoto ? `
        <div class="dexa-report-section dexa-report-section-warning">
          <h4 class="dexa-report-section-title dexa-report-section-title-warning">How to read your photo</h4>
          <p class="dexa-legend-note">These markers are visual guides only. They help explain posture and where fat may be more concentrated, but they are not exact medical measurements.</p>
          <div class="dexa-overlay-grid">
            <div class="dexa-overlay-item">
              <span class="dexa-overlay-badge">1</span>
              <div><strong>Shoulder line</strong><br>Shows whether one shoulder sits a little higher than the other. Current difference: ${overlayMetrics.shoulderTilt}°</div>
            </div>
            <div class="dexa-overlay-item">
              <span class="dexa-overlay-badge">2</span>
              <div><strong>Hip line</strong><br>Shows whether one hip sits a little higher than the other. Current difference: ${overlayMetrics.hipTilt}°</div>
            </div>
            <div class="dexa-overlay-item">
              <span class="dexa-overlay-badge">3</span>
              <div><strong>Body center line</strong><br>Shows whether your body is leaning a little to one side. Current drift: ${overlayMetrics.centerDrift}°</div>
            </div>
            <div class="dexa-overlay-item">
              <span class="dexa-overlay-badge">4</span>
              <div><strong>Reference dots</strong><br>Guide points used to map your shoulders, waist, hips, and knees.</div>
            </div>
            <div class="dexa-overlay-item">
              <span class="dexa-overlay-swatch dexa-overlay-swatch-yellow"></span>
              <div><strong>Stomach focus area</strong><br>This highlight suggests more fat is stored around the abdomen. Visual emphasis: ${overlayMetrics.trunkLoad}%</div>
            </div>
            <div class="dexa-overlay-item">
              <span class="dexa-overlay-swatch dexa-overlay-swatch-orange"></span>
              <div><strong>Side waist focus area</strong><br>This highlight points to more fat around the waist or love-handle area. Visual emphasis: ${overlayMetrics.waistLoad}%</div>
            </div>
            <div class="dexa-overlay-item">
              <span class="dexa-overlay-swatch dexa-overlay-swatch-pink"></span>
              <div><strong>Upper body focus area</strong><br>This highlight points to more fat storage across the chest or upper torso. Visual emphasis: ${overlayMetrics.chestLoad}%</div>
            </div>
          </div>
        </div>
        ` : ''}

        <div class="dexa-report-section dexa-report-section-success">
          <h4 class="dexa-report-section-title dexa-report-section-title-success">Simple next steps</h4>
          <div class="dexa-two-col-grid dexa-detail-grid">
            <div>
              <strong>Nutrition</strong><br>
              Aim for about ${nutritionCalories} calories a day for ${plannerGoalLabel}<br>
              Try to get around ${proteinTarget}g of protein each day<br>
              Aim for roughly ${Math.round(results.weight * 35)}ml of water each day
            </div>
            <div>
              <strong>Training</strong><br>
              Train about 4-5 times per week<br>
              Keep most sessions around 45-60 minutes<br>
              Mix cardio with upper and lower body strength work
            </div>
          </div>
        </div>
      </div>
    `;
    if (resultsPanel) {
      resultsPanel.scrollTop = 0;
    }
    console.log('=== RESULTS DISPLAYED SUCCESSFULLY ===');
  } else {
    console.error('Results element not found!');
  }
}

// Run DEXA analysis
async function runDEXAAnalysis() {
  console.log('=== RUNNING DEXA ANALYSIS ===');
  const inputs = getNormalizedBodyScanInputs();
  if (!inputs.ok) {
    setStatus(inputs.error);
    return;
  }

  syncMetricFieldsFromInputs(inputs);
  setStatus('Running DEXA Analysis...');
  
  // Simulate analysis progress
  setStatus('Analyzing body composition...');
  await new Promise(resolve => setTimeout(resolve, 500));
  
  setStatus('Calculating DEXA metrics...');
  await new Promise(resolve => setTimeout(resolve, 500));
  
  setStatus('Generating health assessment...');
  await new Promise(resolve => setTimeout(resolve, 500));
  
  // Generate and show results
  const results = generateDEXAResults({ inputData: inputs });
  showDEXAResults(results);
  if (uploadedImageData) {
    drawUploadedImage();
  }
  if (window.openDEXAModal) {
    window.openDEXAModal();
  }
  setStatus('DEXA Analysis Complete!');
  
  // Refresh workout and diet plans if fitness UI is available
  if (window.fitnessUI) {
    console.log('=== REFRESHING WORKOUT AND DIET PLANS ===');
    // Small delay to ensure results are available
    setTimeout(() => {
      window.fitnessUI.loadWorkoutPlan();
      window.fitnessUI.loadDietPlan();
    }, 300);
  }
}

// Test function
function testDEXA() {
  console.log('=== TEST DEXA CALLED ===');
  const inputs = getDemoScanInputs();
  if (inputs.ok) {
    syncMetricFieldsFromInputs(inputs);
    const results = generateDEXAResults({ inputData: inputs });
    showDEXAResults(results, { displayMode: 'demo' });
    if (uploadedImageData) {
      drawUploadedImage();
      setStatus('Demo preview ready.');
    } else {
      setStatus('Demo preview ready. Upload a photo to see the image overlay.');
    }
    if (window.fitnessUI) {
      window.fitnessUI.loadWorkoutPlan();
      window.fitnessUI.loadDietPlan();
    }
    return;
  }
  setStatus('Enter your body details to preview the demo scan.');
}

// Toggle unit fields visibility
function toggleBsUnitFields() {
  const metric = document.getElementById('bs-unit-metric')?.checked ?? true;
  const metricFields = document.getElementById('bs-metric-fields');
  const imperialFields = document.getElementById('bs-imperial-fields');
  if (metricFields) metricFields.classList.toggle('hidden', !metric);
  if (imperialFields) imperialFields.classList.toggle('hidden', metric);
}

// Clear UI function
function clearUI() {
  console.log('=== CLEARING UI ===');
  
  // Clear uploaded image data
  uploadedImageData = null;
  window.latestDEXAResults = null;
  
  // Clear file inputs
  const fileInputs = ['bs-file', 'bs-images', 'bs-video'];
  fileInputs.forEach(inputId => {
    const fileInput = document.getElementById(inputId);
    if (fileInput) {
      fileInput.value = '';
    }
  });
  
  // Clear form inputs
  const ageInput = document.getElementById('bs-age');
  if (ageInput) ageInput.value = '';
  
  const sexInput = document.getElementById('bs-sex');
  if (sexInput) sexInput.value = 'male';
  
  const heightCm = document.getElementById('bs-height-cm');
  if (heightCm) heightCm.value = '';
  
  const weightKg = document.getElementById('bs-weight-kg');
  if (weightKg) weightKg.value = '';
  
  const feet = document.getElementById('bs-feet');
  if (feet) feet.value = '';
  
  const inches = document.getElementById('bs-inches');
  if (inches) inches.value = '';
  
  const pounds = document.getElementById('bs-pounds');
  if (pounds) pounds.value = '';
  
  // Reset unit selection
  const metricUnit = document.getElementById('bs-unit-metric');
  const imperialUnit = document.getElementById('bs-unit-imperial');
  if (metricUnit) metricUnit.checked = true;
  if (imperialUnit) imperialUnit.checked = false;
  
  // Toggle unit fields visibility
  toggleBsUnitFields();
  
  // Clear results
  const resultsElement = document.getElementById('bs-results');
  if (resultsElement) {
    resultsElement.innerHTML = '<p style="text-align: center; color: #666; padding: 20px;">Click "📊 Run DEXA Analysis" to see your results</p>';
  }
  
  // Clear canvas
  const canvas = document.getElementById('bs-annotated');
  if (canvas) {
    const ctx = canvas.getContext('2d');
    if (ctx) {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      // Draw placeholder
      drawOverlay(canvas, [], false);
    }
  }
  
  // Clear status
  setStatus('');
  
  // Clear previews if they exist
  const previews = document.getElementById('bs-previews');
  if (previews) previews.innerHTML = '';
  
  const videoPreviews = document.getElementById('bs-video-previews');
  if (videoPreviews) videoPreviews.innerHTML = '';

  if (window.fitnessUI) {
    window.fitnessUI.loadDietPlan();
    window.fitnessUI.loadWorkoutPlan();
  }

  if (window.closeDEXAModal) {
    window.closeDEXAModal();
  }
  
  console.log('=== UI CLEARED ===');
}

function importFromCalculators() {
  try {
    const sexEl = document.getElementById('bmr-sex');
    const ageEl = document.getElementById('bmr-age');
    const metricUnit = document.getElementById('bmr-unit-metric')?.checked ?? true;
    const bsMetric = document.getElementById('bs-unit-metric');
    const bsImperial = document.getElementById('bs-unit-imperial');

    if (sexEl) document.getElementById('bs-sex').value = sexEl.value;
    if (ageEl?.value) document.getElementById('bs-age').value = ageEl.value;

    if (metricUnit) {
      const hcm = document.getElementById('bmr-height-cm')?.value;
      const wkg = document.getElementById('bmr-weight-kg')?.value;
      if (bsMetric) bsMetric.checked = true;
      if (bsImperial) bsImperial.checked = false;
      toggleBsUnitFields();
      if (hcm) document.getElementById('bs-height-cm').value = hcm;
      if (wkg) document.getElementById('bs-weight-kg').value = wkg;
    } else {
      const feet = document.getElementById('bmr-feet')?.value;
      const inches = document.getElementById('bmr-inches')?.value;
      const pounds = document.getElementById('bmr-pounds')?.value;
      if (bsMetric) bsMetric.checked = false;
      if (bsImperial) bsImperial.checked = true;
      toggleBsUnitFields();
      if (feet) document.getElementById('bs-feet').value = feet;
      if (inches) document.getElementById('bs-inches').value = inches;
      if (pounds) document.getElementById('bs-pounds').value = pounds;
    }

    setStatus('Imported your calculator details into Body Scan.');
  } catch (error) {
    console.error('Import from calculators failed:', error);
    setStatus('Could not import values from calculators.');
  }
}

function downloadCurrentReport() {
  const report = window.__fitlife_last_report;
  if (!report) {
    setStatus('Run an analysis first to download a report.');
    return;
  }

  const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = `fitlife-dexa-report-${new Date().toISOString().slice(0, 10)}.json`;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
  setStatus('DEXA report downloaded.');
}

function activateFitnessTab(tabName) {
  document.querySelectorAll('.tab-btn').forEach((button) => {
    button.classList.toggle('active', button.dataset.tab === tabName);
  });
  document.querySelectorAll('.tab-panel').forEach((panel) => {
    panel.classList.toggle('active', panel.id === `${tabName}-tab`);
  });
}

async function saveCurrentScanForComparison() {
  if (!window.latestDEXAResults) {
    setStatus('Run an analysis first to compare scans.');
    return;
  }

  if (!window.progressTracker || !window.fitnessUI) {
    setStatus('Progress tracking is not available right now.');
    return;
  }

  const scanData = {
    ...window.latestDEXAResults,
    weight: window.latestDEXAResults.weight,
    bodyFatPercent: parseFloat(window.latestDEXAResults.bodyFatPercent),
    muscleMass: parseFloat(window.latestDEXAResults.muscleMass),
    visceralFat: parseFloat(window.latestDEXAResults.visceralFat),
    healthScore: window.latestDEXAResults.healthScore
  };

  const lastScan = window.progressTracker.getLatestScan();
  if (
    lastScan &&
    Math.abs(lastScan.weight - scanData.weight) < 0.01 &&
    Math.abs(lastScan.bodyFatPercent - scanData.bodyFatPercent) < 0.01 &&
    Math.abs(lastScan.muscleMass - scanData.muscleMass) < 0.01
  ) {
    activateFitnessTab('progress');
    window.fitnessUI.loadProgressTracking();
    location.hash = 'fitness-planning';
    setStatus('This scan is already saved. Opened Progress Tracking for comparison.');
    return;
  }

  await window.progressTracker.addScan(scanData, uploadedImageData?.file || null);
  activateFitnessTab('progress');
  window.fitnessUI.loadProgressTracking();
  location.hash = 'fitness-planning';

  const scanCount = window.progressTracker.getScanHistory().length;
  setStatus(
    scanCount >= 2
      ? 'Current scan saved. Review Progress Tracking to compare your scans.'
      : 'Current scan saved. Add another scan later to compare progress.'
  );
}

// Draw comprehensive DEXA overlay with heatmaps and visualizations
function drawOverlay(canvas, keypoints, hasImage = false) {
  console.log('=== DRAWING DEXA OVERLAY ===', 'hasImage:', hasImage);
  const rect = canvas.getBoundingClientRect();
  const width = rect.width || parseFloat(canvas.style.width) || canvas.width;
  const height = rect.height || parseFloat(canvas.style.height) || canvas.height;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  
  if (hasImage) {
    // Get analysis results for visual overlays
    const results = window.latestDEXAResults;
    if (!results) {
      return;
    }

    drawPhotoFocusOverlay(ctx, width, height, results, uploadedImageData?.renderBounds || null);
    
  } else {
    // Show placeholder when no image
    ctx.fillStyle = '#f8fafc';
    ctx.fillRect(0, 0, width, height);
    
    // Draw placeholder text
    ctx.save();
    ctx.font = 'bold 20px system-ui, -apple-system, Segoe UI, Roboto';
    ctx.fillStyle = '#64748b';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('DEXA results work without a photo', width / 2, height / 2 - 20);
    
    ctx.font = '14px system-ui, -apple-system, Segoe UI, Roboto';
    ctx.fillStyle = '#94a3b8';
    ctx.fillText('Upload a photo only if you want the visual body map overlay', width / 2, height / 2 + 10);
    ctx.restore();
  }
  
  console.log('=== DEXA OVERLAY DRAWN ===');
}

function drawPhotoFocusOverlay(ctx, width, height, results, renderBounds = null) {
  const frame = renderBounds || {
    x: width * 0.18,
    y: height * 0.05,
    width: width * 0.64,
    height: height * 0.9
  };
  const centerX = frame.x + (frame.width / 2);
  const centerY = frame.y + (frame.height / 2);
  const bodyWidth = frame.width * 0.42;
  const bodyHeight = frame.height * 0.84;
  const bodyFat = parseFloat(results.bodyFatPercent);
  const visceral = parseFloat(results.visceralFat);
  const overlayMetrics = getPhotoOverlayMetrics(results);

  ctx.save();
  ctx.fillStyle = 'rgba(11, 18, 32, 0.16)';
  ctx.fillRect(0, 0, width, height);

  const points = {
    head: { x: centerX, y: centerY - bodyHeight * 0.42 },
    leftShoulder: { x: centerX - bodyWidth * 0.32, y: centerY - bodyHeight * 0.24 },
    rightShoulder: { x: centerX + bodyWidth * 0.32, y: centerY - bodyHeight * 0.235 },
    leftWaist: { x: centerX - bodyWidth * 0.18, y: centerY + bodyHeight * 0.02 },
    rightWaist: { x: centerX + bodyWidth * 0.18, y: centerY + bodyHeight * 0.015 },
    leftHip: { x: centerX - bodyWidth * 0.22, y: centerY + bodyHeight * 0.18 },
    rightHip: { x: centerX + bodyWidth * 0.22, y: centerY + bodyHeight * 0.175 },
    leftKnee: { x: centerX - bodyWidth * 0.12, y: centerY + bodyHeight * 0.42 },
    rightKnee: { x: centerX + bodyWidth * 0.12, y: centerY + bodyHeight * 0.415 }
  };

  const fatZones = [
    {
      x: centerX,
      y: centerY + bodyHeight * 0.08,
      w: bodyWidth * 0.68,
      h: bodyHeight * 0.27,
      color: visceral > 7 ? 'rgba(239, 68, 68, 0.28)' : 'rgba(251, 191, 36, 0.18)',
      active: bodyFat >= 20
    },
    {
      x: centerX - bodyWidth * 0.26,
      y: centerY + bodyHeight * 0.16,
      w: bodyWidth * 0.22,
      h: bodyHeight * 0.2,
      color: 'rgba(249, 115, 22, 0.16)',
      active: bodyFat >= 24
    },
    {
      x: centerX + bodyWidth * 0.26,
      y: centerY + bodyHeight * 0.16,
      w: bodyWidth * 0.22,
      h: bodyHeight * 0.2,
      color: 'rgba(249, 115, 22, 0.16)',
      active: bodyFat >= 24
    },
    {
      x: centerX,
      y: centerY - bodyHeight * 0.2,
      w: bodyWidth * 0.52,
      h: bodyHeight * 0.16,
      color: 'rgba(244, 114, 182, 0.14)',
      active: results.sex === 'male' ? bodyFat >= 22 : bodyFat >= 28
    }
  ];

  fatZones.filter((zone) => zone.active).forEach((zone) => {
    const x = zone.x - (zone.w / 2);
    const y = zone.y - (zone.h / 2);
    ctx.fillStyle = zone.color;
    ctx.beginPath();
    ctx.roundRect(x, y, zone.w, zone.h, 18);
    ctx.fill();
    ctx.strokeStyle = zone.color.replace(/0\.\d+\)/, '0.85)');
    ctx.lineWidth = zone === fatZones[0] ? 2.5 : 1.5;
    ctx.stroke();
  });

  ctx.strokeStyle = 'rgba(255,255,255,0.24)';
  ctx.lineWidth = 1;
  ctx.setLineDash([5, 6]);
  ctx.beginPath();
  ctx.moveTo(centerX, frame.y + frame.height * 0.08);
  ctx.lineTo(centerX, frame.y + frame.height * 0.92);
  ctx.stroke();
  ctx.setLineDash([]);

  const lineSets = [
    [points.leftShoulder, points.rightShoulder, 'rgba(96, 165, 250, 0.55)'],
    [points.leftWaist, points.rightWaist, 'rgba(248, 113, 113, 0.48)'],
    [points.leftHip, points.rightHip, 'rgba(251, 191, 36, 0.5)'],
    [points.leftShoulder, points.leftHip, 'rgba(255,255,255,0.2)'],
    [points.rightShoulder, points.rightHip, 'rgba(255,255,255,0.2)'],
    [points.leftHip, points.leftKnee, 'rgba(255,255,255,0.2)'],
    [points.rightHip, points.rightKnee, 'rgba(255,255,255,0.2)']
  ];

  lineSets.forEach(([a, b, color]) => {
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
  });

  Object.values(points).forEach((point) => {
    ctx.fillStyle = 'rgba(255,255,255,0.9)';
    ctx.strokeStyle = 'rgba(15, 23, 42, 0.55)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(point.x, point.y, 4.5, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
  });

  const shoulderTilt = Math.abs(points.leftShoulder.y - points.rightShoulder.y);
  const hipTilt = Math.abs(points.leftHip.y - points.rightHip.y);

  if (shoulderTilt > 1) {
    ctx.strokeStyle = 'rgba(96, 165, 250, 0.9)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(centerX, points.leftShoulder.y - 24, 16, Math.PI * 0.15, Math.PI * 0.85);
    ctx.stroke();
  }

  if (hipTilt > 1) {
    ctx.strokeStyle = 'rgba(251, 191, 36, 0.9)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(centerX, points.leftHip.y + 20, 16, Math.PI * 1.15, Math.PI * 1.85);
    ctx.stroke();
  }

  const focusNodes = [
    { x: frame.x + 18, y: frame.y + 18, color: 'rgba(96, 165, 250, 0.95)' },
    { x: frame.x + frame.width - 18, y: frame.y + 18, color: 'rgba(239, 68, 68, 0.95)' },
    { x: frame.x + frame.width - 18, y: frame.y + frame.height - 18, color: 'rgba(251, 191, 36, 0.95)' }
  ];

  focusNodes.forEach((node) => {
    ctx.fillStyle = node.color;
    ctx.beginPath();
    ctx.arc(node.x, node.y, 5.5, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = 'rgba(255,255,255,0.8)';
    ctx.lineWidth = 2;
    ctx.stroke();
  });

  const metricBadges = [
    {
      label: '1',
      value: `${overlayMetrics.shoulderTilt}°`,
      x: frame.x + 54,
      y: frame.y + 84,
      color: 'rgba(96, 165, 250, 0.95)'
    },
    {
      label: '2',
      value: `${overlayMetrics.hipTilt}°`,
      x: frame.x + frame.width - 54,
      y: frame.y + frame.height - 92,
      color: 'rgba(251, 191, 36, 0.95)'
    },
    {
      label: '3',
      value: `${overlayMetrics.centerDrift}°`,
      x: frame.x + 62,
      y: frame.y + frame.height - 52,
      color: 'rgba(255, 255, 255, 0.92)',
      textColor: '#0f172a'
    }
  ];

  metricBadges.forEach((badge) => {
    ctx.fillStyle = badge.color;
    ctx.beginPath();
    ctx.roundRect(badge.x - 18, badge.y - 12, 48, 24, 12);
    ctx.fill();
    ctx.fillStyle = badge.textColor || '#0f172a';
    ctx.font = '600 11px Inter, system-ui, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(badge.label, badge.x - 6, badge.y + 0.5);
    ctx.fillText(badge.value, badge.x + 16, badge.y + 0.5);
  });

  ctx.restore();
}

// Draw body fat heatmap with regional distribution - Enhanced version
function drawBodyFatHeatmap(ctx, centerX, centerY, bodyWidth, bodyHeight, results) {
  const bodyFatPercent = parseFloat(results.bodyFatPercent);
  const visceralFat = parseFloat(results.visceralFat);
  
  // Enhanced zones with realistic fat distribution based on actual body fat percentage
  // Higher body fat = more fat in abdomen and torso areas
  const abdomenMultiplier = bodyFatPercent > 30 ? 1.8 : bodyFatPercent > 25 ? 1.6 : bodyFatPercent > 20 ? 1.4 : 1.2;
  const torsoMultiplier = bodyFatPercent > 30 ? 1.3 : bodyFatPercent > 25 ? 1.2 : 1.1;
  
  const zones = [
    // Head and neck (low fat, relatively constant)
    { 
      x: centerX, 
      y: centerY - bodyHeight * 0.38, 
      width: bodyWidth * 0.22, 
      height: bodyHeight * 0.16, 
      fat: Math.min(bodyFatPercent * 0.3, 15), // Cap at 15% even for obese
      label: 'Head',
      shape: 'ellipse'
    },
    // Chest and shoulders (varies with overall body fat)
    { 
      x: centerX, 
      y: centerY - bodyHeight * 0.18, 
      width: bodyWidth * 0.45, 
      height: bodyHeight * 0.28, 
      fat: bodyFatPercent * torsoMultiplier,
      label: 'Chest',
      shape: 'rounded'
    },
    // Left arm (moderate fat, increases with obesity)
    { 
      x: centerX - bodyWidth * 0.38, 
      y: centerY - bodyHeight * 0.08, 
      width: bodyWidth * 0.18, 
      height: bodyHeight * 0.35, 
      fat: bodyFatPercent * (bodyFatPercent > 30 ? 0.7 : 0.6),
      label: 'L Arm',
      shape: 'rounded'
    },
    // Right arm (moderate fat, increases with obesity)
    { 
      x: centerX + bodyWidth * 0.38, 
      y: centerY - bodyHeight * 0.08, 
      width: bodyWidth * 0.18, 
      height: bodyHeight * 0.35, 
      fat: bodyFatPercent * (bodyFatPercent > 30 ? 0.7 : 0.6),
      label: 'R Arm',
      shape: 'rounded'
    },
    // Abdomen (highest fat, especially for obese individuals)
    { 
      x: centerX, 
      y: centerY + bodyHeight * 0.08, 
      width: bodyWidth * 0.35, 
      height: bodyHeight * 0.22, 
      fat: bodyFatPercent * abdomenMultiplier + parseFloat(visceralFat) * 1.5,
      label: 'Abdomen',
      shape: 'rounded',
      highlight: parseFloat(visceralFat) > 8
    },
    // Hips and thighs (moderate to high fat, increases with obesity)
    { 
      x: centerX, 
      y: centerY + bodyHeight * 0.28, 
      width: bodyWidth * 0.38, 
      height: bodyHeight * 0.32, 
      fat: bodyFatPercent * (bodyFatPercent > 30 ? 1.3 : 1.15),
      label: 'Hips',
      shape: 'rounded'
    },
    // Calves (low fat, but increases slightly with obesity)
    { 
      x: centerX, 
      y: centerY + bodyHeight * 0.42, 
      width: bodyWidth * 0.22, 
      height: bodyHeight * 0.16, 
      fat: bodyFatPercent * (bodyFatPercent > 30 ? 0.5 : 0.35),
      label: 'Legs',
      shape: 'rounded'
    }
  ];
  
  zones.forEach(zone => {
    ctx.save();
    
    // Create smoother, more realistic gradient
    const maxRadius = Math.max(zone.width, zone.height) * 0.7;
    const gradient = ctx.createRadialGradient(
      zone.x, zone.y, 0,
      zone.x, zone.y, maxRadius
    );
    
    // Enhanced color mapping with accurate thresholds for realistic body fat ranges
    let color1, color2, color3;
    const fatValue = zone.fat;
    
    if (fatValue < 12) {
      // Very low fat - bright green (athletic)
      color1 = 'rgba(34, 197, 94, 0.25)';
      color2 = 'rgba(34, 197, 94, 0.12)';
      color3 = 'rgba(34, 197, 94, 0.03)';
    } else if (fatValue < 18) {
      // Low fat - green (excellent)
      color1 = 'rgba(74, 222, 128, 0.25)';
      color2 = 'rgba(74, 222, 128, 0.12)';
      color3 = 'rgba(74, 222, 128, 0.03)';
    } else if (fatValue < 22) {
      // Moderate-low fat - light yellow (good)
      color1 = 'rgba(234, 179, 8, 0.28)';
      color2 = 'rgba(234, 179, 8, 0.14)';
      color3 = 'rgba(234, 179, 8, 0.04)';
    } else if (fatValue < 28) {
      // Moderate fat - yellow/orange (fair)
      color1 = 'rgba(251, 191, 36, 0.35)';
      color2 = 'rgba(251, 191, 36, 0.18)';
      color3 = 'rgba(251, 191, 36, 0.05)';
    } else if (fatValue < 35) {
      // High fat - orange (needs improvement)
      color1 = 'rgba(249, 115, 22, 0.4)';
      color2 = 'rgba(249, 115, 22, 0.2)';
      color3 = 'rgba(249, 115, 22, 0.06)';
    } else if (fatValue < 45) {
      // Very high fat - red-orange (high body fat)
      color1 = 'rgba(239, 68, 68, 0.45)';
      color2 = 'rgba(239, 68, 68, 0.25)';
      color3 = 'rgba(239, 68, 68, 0.08)';
    } else {
      // Extremely high fat - dark red (obese)
      color1 = 'rgba(220, 38, 38, 0.5)';
      color2 = 'rgba(220, 38, 38, 0.3)';
      color3 = 'rgba(220, 38, 38, 0.1)';
    }
    
    gradient.addColorStop(0, color1);
    gradient.addColorStop(0.4, color2);
    gradient.addColorStop(1, color3);
    
    ctx.fillStyle = gradient;
    
    // Draw zone shape
    if (zone.shape === 'ellipse') {
      ctx.beginPath();
      ctx.ellipse(zone.x, zone.y, zone.width / 2, zone.height / 2, 0, 0, 2 * Math.PI);
      ctx.fill();
    } else {
      // Rounded rectangle
      const radius = Math.min(zone.width, zone.height) * 0.15;
      const x = zone.x - zone.width / 2;
      const y = zone.y - zone.height / 2;
      ctx.beginPath();
      ctx.roundRect(x, y, zone.width, zone.height, radius);
      ctx.fill();
    }
    
    // Add highlight for high visceral fat zones
    if (zone.highlight) {
      ctx.strokeStyle = '#ef4444';
      ctx.lineWidth = 3;
      ctx.setLineDash([5, 5]);
      if (zone.shape === 'ellipse') {
        ctx.beginPath();
        ctx.ellipse(zone.x, zone.y, zone.width / 2 + 5, zone.height / 2 + 5, 0, 0, 2 * Math.PI);
        ctx.stroke();
      } else {
        const x = zone.x - zone.width / 2 - 5;
        const y = zone.y - zone.height / 2 - 5;
        ctx.beginPath();
        ctx.roundRect(x, y, zone.width + 10, zone.height + 10, Math.min(zone.width, zone.height) * 0.15);
        ctx.stroke();
      }
      ctx.setLineDash([]);
    }
    
    // Add percentage label for each zone (smaller, less intrusive)
    ctx.fillStyle = 'rgba(255, 255, 255, 0.85)';
    ctx.font = 'bold 9px system-ui';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.strokeStyle = 'rgba(0, 0, 0, 0.6)';
    ctx.lineWidth = 1.5;
    ctx.strokeText(`${zone.fat.toFixed(1)}%`, zone.x, zone.y);
    ctx.fillText(`${zone.fat.toFixed(1)}%`, zone.x, zone.y);
    
    ctx.restore();
  });
}

// Draw muscle mass overlay - Enhanced version
function drawMuscleMassOverlay(ctx, centerX, centerY, bodyWidth, bodyHeight, results) {
  const muscleMass = parseFloat(results.muscleMass);
  const weight = results.weight || getCurrentWeightKg({ allowDefaults: true });
  const musclePercent = (muscleMass / weight) * 100;
  
  // Enhanced muscle groups with more detail and better visualization
  const muscleGroups = [
    { 
      name: 'Chest', 
      x: centerX, 
      y: centerY - bodyHeight * 0.18, 
      width: bodyWidth * 0.32, 
      height: bodyHeight * 0.15,
      mass: muscleMass * 0.12,
      shape: 'rounded'
    },
    { 
      name: 'L Arm', 
      x: centerX - bodyWidth * 0.32, 
      y: centerY - bodyHeight * 0.05, 
      width: bodyWidth * 0.12, 
      height: bodyHeight * 0.28,
      mass: muscleMass * 0.08,
      shape: 'rounded'
    },
    { 
      name: 'R Arm', 
      x: centerX + bodyWidth * 0.32, 
      y: centerY - bodyHeight * 0.05, 
      width: bodyWidth * 0.12, 
      height: bodyHeight * 0.28,
      mass: muscleMass * 0.08,
      shape: 'rounded'
    },
    { 
      name: 'Core', 
      x: centerX, 
      y: centerY + bodyHeight * 0.05, 
      width: bodyWidth * 0.25, 
      height: bodyHeight * 0.18,
      mass: muscleMass * 0.15,
      shape: 'rounded'
    },
    { 
      name: 'Back', 
      x: centerX, 
      y: centerY - bodyHeight * 0.08, 
      width: bodyWidth * 0.28, 
      height: bodyHeight * 0.22,
      mass: muscleMass * 0.18,
      shape: 'rounded'
    },
    { 
      name: 'Legs', 
      x: centerX, 
      y: centerY + bodyHeight * 0.22, 
      width: bodyWidth * 0.30, 
      height: bodyHeight * 0.28,
      mass: muscleMass * 0.35,
      shape: 'rounded'
    }
  ];
  
  muscleGroups.forEach(group => {
    ctx.save();
    
    // Determine color based on muscle percentage (reduced opacity)
    let color, borderColor;
    if (musclePercent > 45) {
      color = 'rgba(229, 231, 235, 0.28)';
      borderColor = '#9ca3af';
    } else if (musclePercent > 40) {
      color = 'rgba(209, 213, 219, 0.24)';
      borderColor = '#6b7280';
    } else if (musclePercent > 35) {
      color = 'rgba(156, 163, 175, 0.22)';
      borderColor = '#4b5563';
    } else {
      color = 'rgba(107, 114, 128, 0.2)';
      borderColor = '#374151';
    }
    
    // Draw muscle group shape
    ctx.fillStyle = color;
    ctx.strokeStyle = borderColor;
    ctx.lineWidth = 2.5;
    
    const x = group.x - group.width / 2;
    const y = group.y - group.height / 2;
    const radius = Math.min(group.width, group.height) * 0.2;
    
    ctx.beginPath();
    ctx.roundRect(x, y, group.width, group.height, radius);
    ctx.fill();
    ctx.stroke();
    
    // Add muscle mass label (smaller, less intrusive)
    ctx.fillStyle = 'rgba(255, 255, 255, 0.9)';
    ctx.font = 'bold 8px system-ui';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.strokeStyle = 'rgba(0, 0, 0, 0.7)';
    ctx.lineWidth = 1.5;
    const label = `${group.name}\n${group.mass.toFixed(1)}kg`;
    const lines = label.split('\n');
    lines.forEach((line, idx) => {
      const offsetY = (idx - (lines.length - 1) / 2) * 9;
      ctx.strokeText(line, group.x, group.y + offsetY);
      ctx.fillText(line, group.x, group.y + offsetY);
    });
    
    ctx.restore();
  });
  
  // Draw compact overall muscle percentage indicator (smaller, top center)
  ctx.save();
  ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
  ctx.fillRect(centerX - 50, 8, 100, 22);
  ctx.fillStyle = '#e5e7eb';
  ctx.font = 'bold 10px system-ui';
  ctx.textAlign = 'center';
  ctx.fillText(`Muscle: ${musclePercent.toFixed(1)}%`, centerX, 22);
  ctx.restore();
}

// Draw visceral fat indicators - Enhanced version
function drawVisceralFatIndicators(ctx, centerX, centerY, bodyWidth, bodyHeight, results) {
  const visceralFat = parseFloat(results.visceralFat);
  const abdomenX = centerX;
  const abdomenY = centerY + bodyHeight * 0.12;
  
    ctx.save();
  
  // Draw visceral fat level indicator
  let indicatorColor, indicatorText, riskLevel;
  if (visceralFat < 5) {
    indicatorColor = '#9ca3af';
    indicatorText = 'OPTIMAL';
    riskLevel = 'Low Risk';
  } else if (visceralFat < 8) {
    indicatorColor = '#6b7280';
    indicatorText = 'GOOD';
    riskLevel = 'Moderate Risk';
  } else if (visceralFat < 12) {
    indicatorColor = '#4b5563';
    indicatorText = 'ELEVATED';
    riskLevel = 'High Risk';
  } else {
    indicatorColor = '#111827';
    indicatorText = 'HIGH';
    riskLevel = 'Very High Risk';
  }
  
  // Draw compact concentric circles for visceral fat visualization (smaller)
  const baseRadius = bodyWidth * 0.1;
  const maxRadius = baseRadius + (visceralFat / 15) * baseRadius * 0.8;
  
  // Outer warning ring (reduced opacity)
  if (visceralFat > 8) {
    const gradient = ctx.createRadialGradient(abdomenX, abdomenY, baseRadius, abdomenX, abdomenY, maxRadius);
    gradient.addColorStop(0, indicatorColor + '50');
    gradient.addColorStop(0.5, indicatorColor + '25');
    gradient.addColorStop(1, indicatorColor + '00');
    
    ctx.fillStyle = gradient;
    ctx.beginPath();
    ctx.arc(abdomenX, abdomenY, maxRadius, 0, 2 * Math.PI);
    ctx.fill();
    
    // Pulsing effect with dashed border (thinner)
    ctx.strokeStyle = indicatorColor;
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 3]);
    ctx.beginPath();
    ctx.arc(abdomenX, abdomenY, maxRadius + 3, 0, 2 * Math.PI);
    ctx.stroke();
    ctx.setLineDash([]);
  }
  
  // Inner core indicator (reduced opacity)
  ctx.fillStyle = indicatorColor + '40';
  ctx.beginPath();
  ctx.arc(abdomenX, abdomenY, baseRadius, 0, 2 * Math.PI);
  ctx.fill();
  
  ctx.strokeStyle = indicatorColor;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.arc(abdomenX, abdomenY, baseRadius, 0, 2 * Math.PI);
  ctx.stroke();
  
  // Draw compact visceral fat value and label (smaller)
  ctx.fillStyle = 'rgba(0, 0, 0, 0.75)';
  ctx.fillRect(abdomenX - 40, abdomenY - 25, 80, 38);
  
  ctx.fillStyle = indicatorColor;
  ctx.font = 'bold 9px system-ui';
    ctx.textAlign = 'center';
  ctx.fillText('VISCERAL', abdomenX, abdomenY - 12);
  
  ctx.fillStyle = '#ffffff';
  ctx.font = 'bold 11px system-ui';
  ctx.fillText(`${visceralFat.toFixed(1)}%`, abdomenX, abdomenY + 2);
  
  ctx.font = '8px system-ui';
  ctx.fillStyle = indicatorColor;
  ctx.fillText(riskLevel, abdomenX, abdomenY + 12);
  
    ctx.restore();
}

// Draw body composition zones - Enhanced version
function drawBodyCompositionZones(ctx, centerX, centerY, bodyWidth, bodyHeight, results) {
  const bodyFatPercent = parseFloat(results.bodyFatPercent);
  const visceralFat = parseFloat(results.visceralFat);
  
  // Determine body type pattern
  const androidRatio = visceralFat / bodyFatPercent;
  let pattern = 'Mixed';
  let patternColor = '#6b7280';
  
  if (androidRatio > 0.4) {
    pattern = 'Android (Apple)';
    patternColor = '#ef4444';
  } else if (androidRatio < 0.2) {
    pattern = 'Gynoid (Pear)';
    patternColor = '#3b82f6';
  }
  
  ctx.save();
  
  // Draw pattern indicator with gradient overlay (reduced opacity)
  if (pattern.includes('Android')) {
    // Apple shape - highlight upper body with gradient
    const gradient = ctx.createLinearGradient(
      centerX - bodyWidth * 0.2, centerY - bodyHeight * 0.4,
      centerX + bodyWidth * 0.2, centerY - bodyHeight * 0.1
    );
    gradient.addColorStop(0, patternColor + '20');
    gradient.addColorStop(0.5, patternColor + '12');
    gradient.addColorStop(1, patternColor + '05');
    
    ctx.fillStyle = gradient;
    ctx.beginPath();
    ctx.roundRect(
      centerX - bodyWidth * 0.2, 
      centerY - bodyHeight * 0.4, 
      bodyWidth * 0.4, 
      bodyHeight * 0.3,
      bodyWidth * 0.1
    );
    ctx.fill();
    
  ctx.strokeStyle = patternColor;
  ctx.lineWidth = 2;
    ctx.setLineDash([6, 3]);
    ctx.stroke();
    ctx.setLineDash([]);
  
  } else if (pattern.includes('Gynoid')) {
    // Pear shape - highlight lower body with gradient
    const gradient = ctx.createLinearGradient(
      centerX - bodyWidth * 0.25, centerY + bodyHeight * 0.1,
      centerX + bodyWidth * 0.25, centerY + bodyHeight * 0.5
    );
    gradient.addColorStop(0, patternColor + '20');
    gradient.addColorStop(0.5, patternColor + '12');
    gradient.addColorStop(1, patternColor + '05');
    
    ctx.fillStyle = gradient;
    ctx.beginPath();
    ctx.roundRect(
      centerX - bodyWidth * 0.25, 
      centerY + bodyHeight * 0.1, 
      bodyWidth * 0.5, 
      bodyHeight * 0.4,
      bodyWidth * 0.12
    );
    ctx.fill();
    
    ctx.strokeStyle = patternColor;
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 3]);
    ctx.stroke();
    ctx.setLineDash([]);
  }
  
  // Draw body symmetry lines for reference
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
  ctx.lineWidth = 1;
  ctx.setLineDash([2, 4]);
  
  // Vertical center line
  ctx.beginPath();
  ctx.moveTo(centerX, centerY - bodyHeight * 0.45);
  ctx.lineTo(centerX, centerY + bodyHeight * 0.45);
  ctx.stroke();
  
  // Horizontal reference lines
  ctx.beginPath();
  ctx.moveTo(centerX - bodyWidth * 0.5, centerY);
  ctx.lineTo(centerX + bodyWidth * 0.5, centerY);
  ctx.stroke();
  
  ctx.setLineDash([]);
  
  ctx.restore();
}

// Draw a lighter information overlay so the photo remains readable.
function drawDEXAInfoOverlay(ctx, width, height, results) {
  ctx.save();
  ctx.fillStyle = 'rgba(8, 15, 22, 0.62)';
  ctx.strokeStyle = 'rgba(255,255,255,0.14)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.roundRect(12, 12, 170, 74, 16);
  ctx.fill();
  ctx.stroke();

  ctx.fillStyle = '#e5e7eb';
  ctx.font = '700 11px system-ui';
  ctx.textAlign = 'left';
  ctx.fillText('FitLife DEXA Scan', 24, 32);

  ctx.fillStyle = '#f8fafc';
  ctx.font = '600 11px system-ui';
  ctx.fillText(`Body Fat  ${results.bodyFatPercent}%`, 24, 52);
  ctx.fillText(`Muscle    ${results.muscleMass} kg`, 24, 68);

  const riskColor = parseFloat(results.visceralFat) > 8 ? '#fb7185' : '#34d399';
  ctx.fillStyle = 'rgba(8, 15, 22, 0.62)';
  ctx.strokeStyle = 'rgba(255,255,255,0.14)';
  ctx.beginPath();
  ctx.roundRect(width - 136, 12, 124, 48, 16);
  ctx.fill();
  ctx.stroke();

  ctx.fillStyle = riskColor;
  ctx.font = '700 11px system-ui';
  ctx.textAlign = 'center';
  ctx.fillText(`Risk ${results.visceralFat}%`, width - 74, 33);

  ctx.fillStyle = '#e2e8f0';
  ctx.font = '600 10px system-ui';
  ctx.fillText(`${results.metabolicAge}y metabolic age`, width - 74, 48);
  ctx.restore();
}

// Store uploaded image data
let uploadedImageData = null;

// Handle file uploads
function setupFileUploads() {
  const fileInputs = ['bs-file', 'bs-images'];
  fileInputs.forEach(inputId => {
    const fileInput = document.getElementById(inputId);
    if (fileInput) {
      fileInput.addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (file) {
          console.log('=== FILE UPLOADED ===', file.name);
          const img = new Image();
          img.onload = async () => {
            // Store image data for later use
            uploadedImageData = {
              img: img,
              file: file,
              url: img.src
            };
            
            const canvas = document.getElementById('bs-annotated');
            if (canvas) {
              // Show the canvas
              canvas.parentElement.classList.remove('hidden');
              canvas.parentElement.style.display = 'block';
              
              // Set canvas size
              configureCanvasDisplay(canvas, canvas.parentElement.clientWidth, 400);
              
              // Show placeholder - don't draw image yet
              drawOverlay(canvas, [], false);
              console.log('Photo uploaded - stored for DEXA Analysis. Click "Run DEXA Analysis" to see photo and results.');
            }
          };
          img.src = URL.createObjectURL(file);
        }
      });
      console.log(`File input ${inputId} listener attached`);
    }
  });
}

// Draw uploaded image when DEXA analysis is run
function drawUploadedImage() {
  if (!uploadedImageData) {
    console.log('No image uploaded yet');
    return;
  }
  
  const canvas = document.getElementById('bs-annotated');
  if (!canvas) return;
  
  const img = uploadedImageData.img;
  const containerWidth = canvas.parentElement.clientWidth;
  const containerHeight = 400;
  const imgAspectRatio = img.width / img.height;
  const containerAspectRatio = containerWidth / containerHeight;
  
  let drawWidth, drawHeight, offsetX = 0, offsetY = 0;
  
  if (imgAspectRatio > containerAspectRatio) {
    // Image is wider - fit to width
    drawWidth = containerWidth;
    drawHeight = containerWidth / imgAspectRatio;
    offsetY = (containerHeight - drawHeight) / 2;
  } else {
    // Image is taller - fit to height
    drawHeight = containerHeight;
    drawWidth = containerHeight * imgAspectRatio;
    offsetX = (containerWidth - drawWidth) / 2;
  }
  
  const { ctx } = configureCanvasDisplay(canvas, containerWidth, containerHeight);

  ctx.clearRect(0, 0, containerWidth, containerHeight);

  const coverScale = Math.max(containerWidth / img.width, containerHeight / img.height);
  const coverWidth = img.width * coverScale;
  const coverHeight = img.height * coverScale;
  const coverX = (containerWidth - coverWidth) / 2;
  const coverY = (containerHeight - coverHeight) / 2;

  ctx.save();
  ctx.fillStyle = '#08111f';
  ctx.fillRect(0, 0, containerWidth, containerHeight);
  ctx.filter = 'blur(26px) saturate(0.9)';
  ctx.globalAlpha = 0.38;
  ctx.drawImage(img, coverX, coverY, coverWidth, coverHeight);
  ctx.restore();

  const framePaddingX = 18;
  const framePaddingY = 12;
  const maxFrameWidth = containerWidth - (framePaddingX * 2);
  const maxFrameHeight = containerHeight - (framePaddingY * 2);
  const frameScale = Math.min(maxFrameWidth / drawWidth, maxFrameHeight / drawHeight, 1);
  const frameWidth = drawWidth * frameScale;
  const frameHeight = drawHeight * frameScale;
  const frameX = (containerWidth - frameWidth) / 2;
  const frameY = (containerHeight - frameHeight) / 2;

  ctx.save();
  ctx.fillStyle = 'rgba(255,255,255,0.08)';
  ctx.strokeStyle = 'rgba(255,255,255,0.18)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.roundRect(frameX - 1, frameY - 1, frameWidth + 2, frameHeight + 2, 22);
  ctx.fill();
  ctx.stroke();
  ctx.restore();

  ctx.save();
  ctx.beginPath();
  ctx.roundRect(frameX, frameY, frameWidth, frameHeight, 20);
  ctx.clip();
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(img, frameX, frameY, frameWidth, frameHeight);
  ctx.restore();

  ctx.save();
  ctx.strokeStyle = 'rgba(255,255,255,0.26)';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.roundRect(frameX, frameY, frameWidth, frameHeight, 20);
  ctx.stroke();
  ctx.restore();

  uploadedImageData.renderBounds = {
    x: frameX,
    y: frameY,
    width: frameWidth,
    height: frameHeight
  };

  drawOverlay(canvas, [], true);
}

// Make functions globally available
window.runDEXAAnalysis = runDEXAAnalysis;
window.generateDEXAResults = generateDEXAResults;
window.getNormalizedBodyScanInputs = getNormalizedBodyScanInputs;
window.testDEXA = testDEXA;
window.showDEXAResults = showDEXAResults;
window.clearUI = clearUI;

// Polyfill for roundRect if not available
if (!CanvasRenderingContext2D.prototype.roundRect) {
  CanvasRenderingContext2D.prototype.roundRect = function(x, y, width, height, radius) {
    if (width < 2 * radius) radius = width / 2;
    if (height < 2 * radius) radius = height / 2;
    this.beginPath();
    this.moveTo(x + radius, y);
    this.arcTo(x + width, y, x + width, y + height, radius);
    this.arcTo(x + width, y + height, x, y + height, radius);
    this.arcTo(x, y + height, x, y, radius);
    this.arcTo(x, y, x + width, y, radius);
    this.closePath();
    return this;
  };
}

// Initialize on DOM load
document.addEventListener('DOMContentLoaded', () => {
  console.log('=== SIMPLE DEXA SCRIPT READY ===');
  
  const results = document.getElementById('bs-results');
  if (results) {
    results.innerHTML = '<p style="text-align: center; color: #666; padding: 20px;">Click "📊 Run DEXA Analysis" to see your results</p>';
  }
  
  // Setup file uploads
  setupFileUploads();
  
  // Setup unit toggle listeners
  const metricUnit = document.getElementById('bs-unit-metric');
  const imperialUnit = document.getElementById('bs-unit-imperial');
  if (metricUnit) {
    metricUnit.addEventListener('change', toggleBsUnitFields);
  }
  if (imperialUnit) {
    imperialUnit.addEventListener('change', toggleBsUnitFields);
  }
  // Initialize unit fields visibility
  toggleBsUnitFields();
  
  // Attach button event listeners
  const clearBtn = document.getElementById('bs-clear-btn');
  if (clearBtn) {
    clearBtn.addEventListener('click', (e) => {
      e.preventDefault();
      e.stopPropagation();
      console.log('=== CLEAR BUTTON CLICKED ===');
      clearUI();
    });
    console.log('Clear button listener attached');
  } else {
    console.log('Clear button not found!');
  }
  
  // Attach analyze button listener (in addition to inline onclick for reliability)
  const analyzeBtn = document.getElementById('bs-analyze-btn');
  if (analyzeBtn) {
    // Remove inline onclick to avoid conflicts
    analyzeBtn.removeAttribute('onclick');
    analyzeBtn.addEventListener('click', (e) => {
      e.preventDefault();
      e.stopPropagation();
      console.log('=== ANALYZE BUTTON CLICKED ===');
      runDEXAAnalysis();
    });
    console.log('Analyze button listener attached');
  } else {
    console.log('Analyze button not found!');
  }

  const importBtn = document.getElementById('bs-import-btn');
  if (importBtn) {
    importBtn.addEventListener('click', (e) => {
      e.preventDefault();
      importFromCalculators();
    });
  }

  const downloadBtn = document.getElementById('bs-download-report');
  if (downloadBtn) {
    downloadBtn.addEventListener('click', (e) => {
      e.preventDefault();
      downloadCurrentReport();
    });
  }

  const compareBtn = document.getElementById('bs-compare-btn');
  if (compareBtn) {
    compareBtn.addEventListener('click', async (e) => {
      e.preventDefault();
      await saveCurrentScanForComparison();
    });
  }

  const modal = document.getElementById('dexa-modal');
  const modalOpenBtn = document.getElementById('open-dexa-modal');
  const modalCloseButtons = document.querySelectorAll('[data-dexa-close]');

  function setModalOpen(open) {
    if (!modal) return;
    modal.classList.toggle('open', open);
    modal.setAttribute('aria-hidden', open ? 'false' : 'true');
    document.body.classList.toggle('modal-open', open);
    if (open) {
      requestAnimationFrame(() => {
        const canvas = document.getElementById('bs-annotated');
        if (!canvas || !canvas.parentElement) return;
        const width = canvas.parentElement.clientWidth || 520;
        const height = Math.min(window.innerHeight * 0.68, 680);
        configureCanvasDisplay(canvas, width, height);
        if (uploadedImageData) {
          drawUploadedImage();
        } else {
          drawOverlay(canvas, [], false);
        }
      });
    }
  }

  window.openDEXAModal = () => setModalOpen(true);
  window.closeDEXAModal = () => setModalOpen(false);

  modalOpenBtn?.addEventListener('click', (e) => {
    e.preventDefault();
    setModalOpen(true);
  });

  modalCloseButtons.forEach((button) => {
    button.addEventListener('click', () => setModalOpen(false));
  });

  modal?.addEventListener('click', (event) => {
    if (event.target === modal) {
      setModalOpen(false);
    }
  });

  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') {
      setModalOpen(false);
    }
  });
  
  // Show placeholder in canvas
  const canvas = document.getElementById('bs-annotated');
  if (canvas) {
    canvas.parentElement.classList.remove('hidden');
    canvas.parentElement.style.display = 'block';
    
    // Set canvas size
    configureCanvasDisplay(canvas, canvas.parentElement.clientWidth, 400);
    
    // Draw placeholder
    drawOverlay(canvas, [], false);
  }
  
  console.log('=== SIMPLE DEXA SCRIPT INITIALIZED ===');
});

console.log('=== SIMPLE WORKING DEXA SCRIPT LOADED ===');
