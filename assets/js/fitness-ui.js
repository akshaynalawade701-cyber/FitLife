console.log('=== FITNESS UI LOADING ===');

// Fitness Planning UI Components
class FitnessUI {
  constructor() {
    this.planner = window.fitnessPlanner;
    this.tracker = window.progressTracker;
    this.currentPlan = null;
    this.currentDietResults = null;
    this.currentObjectiveOptions = null;
  }

  // Initialize fitness planning interface
  initialize() {
    this.createFitnessTabs();
    this.setupEventListeners();
    this.loadUserGoals();
    this.loadDietPlan();
    this.loadWorkoutPlan();
    this.loadProgressTracking();
    this.loadGoalsSettings();
  }

  // Create fitness planning tabs
  createFitnessTabs() {
    const appContainer = document.getElementById('fitness-planning-app');
    if (!appContainer) return;

    appContainer.innerHTML = `
      <div class="fitness-tabs">
        <div class="tab-buttons">
          <button class="tab-btn active" data-tab="diet">🍽️ Diet Plan</button>
          <button class="tab-btn" data-tab="workout">💪 Workout Plan</button>
          <button class="tab-btn" data-tab="progress">📈 Progress Tracking</button>
          <button class="tab-btn" data-tab="goals">🎯 Goals & Settings</button>
        </div>
        
        <div class="tab-content">
          <div id="diet-tab" class="tab-panel active">
            <div id="diet-plan-container"></div>
          </div>
          
          <div id="workout-tab" class="tab-panel">
            <div style="margin-bottom: 15px; text-align: right;">
              <button id="refresh-workout-plan" class="btn btn-ghost" style="font-size: 0.9rem;">🔄 Refresh Plan</button>
            </div>
            <div id="workout-plan-container"></div>
          </div>
          
          <div id="progress-tab" class="tab-panel">
            <div id="progress-container"></div>
          </div>
          
          <div id="goals-tab" class="tab-panel">
            <div id="goals-container"></div>
          </div>
        </div>
      </div>
    `;

    this.setupTabSwitching();
    this.setupRefreshButton();
    
    // Clear workout plan container initially - don't show until DEXA is run
    setTimeout(() => {
      const workoutContainer = document.getElementById('workout-plan-container');
      if (workoutContainer && !workoutContainer.innerHTML.includes('DEXA')) {
        workoutContainer.innerHTML = '<p style="text-align: center; padding: 40px; color: #666;">⚠️ Please run DEXA analysis first to generate your personalized workout plan.</p>';
      }
    }, 100);
  }

  // Setup refresh button for workout plan
  setupRefreshButton() {
    const refreshBtn = document.getElementById('refresh-workout-plan');
    if (refreshBtn) {
      refreshBtn.addEventListener('click', () => {
        console.log('=== REFRESHING WORKOUT PLAN ===');
        this.loadWorkoutPlan();
      });
    }
  }

  // Setup tab switching
  setupTabSwitching() {
    const tabButtons = document.querySelectorAll('.tab-btn');
    const tabPanels = document.querySelectorAll('.tab-panel');

    tabButtons.forEach(button => {
      button.addEventListener('click', () => {
        const targetTab = button.dataset.tab;
        
        // Update active button
        tabButtons.forEach(btn => btn.classList.remove('active'));
        button.classList.add('active');
        
        // Update active panel
        tabPanels.forEach(panel => panel.classList.remove('active'));
        document.getElementById(`${targetTab}-tab`).classList.add('active');
        
        // Load tab content
        this.loadTabContent(targetTab);
      });
    });
  }

  // Load content for specific tab
  loadTabContent(tabName) {
    switch(tabName) {
      case 'diet':
        this.loadDietPlan();
        break;
      case 'workout':
        // Always refresh workout plan when tab is clicked to get latest DEXA results
        this.loadWorkoutPlan();
        break;
      case 'progress':
        this.loadProgressTracking();
        break;
      case 'goals':
        this.loadGoalsSettings();
        break;
    }
  }

  // Load diet plan interface
  loadDietPlan() {
    const container = document.getElementById('diet-plan-container');
    if (!container) return;

    // Get current DEXA results
    const dexaResults = this.getCurrentDEXAResults();
    if (!dexaResults) {
      container.innerHTML = '<p>Please run DEXA analysis first to generate diet plan.</p>';
      return;
    }

    const goals = this.getUserGoals();
    const dietPlan = this.planner.generateDietPlan(dexaResults, goals);
    this.currentDietResults = dexaResults;
    const objectiveOptions = dietPlan.objectiveOptions || this.planner.getObjectiveCalorieOptions(dexaResults, dietPlan.calorieProfile);
    this.currentObjectiveOptions = objectiveOptions;
    const focusLabels = {
      fat_loss: 'Fat Loss',
      gain_muscle: 'Muscle Gain',
      recomp: 'Body Recomposition',
      maintain: 'Maintenance'
    };
    const selectedGoal = goals.objective || dietPlan.nutritionFocus.recommendation;
    const selectedGoalLabel = focusLabels[selectedGoal] || selectedGoal;
    
    container.innerHTML = `
      <div class="diet-plan-header">
        <h3>🍽️ Nutrition Targets</h3>
        <div class="calorie-summary">
          <div class="calorie-card">
            <h4>${dietPlan.calorieProfile.maintenanceCalories}</h4>
            <p>Maintenance Calories</p>
          </div>
          <div class="macro-breakdown">
            <div class="macro-item">
              <span class="macro-label">Suggested Path</span>
              <span class="macro-value">${selectedGoalLabel}</span>
            </div>
            <div class="macro-item">
              <span class="macro-label">Activity Used</span>
              <span class="macro-value">${dietPlan.calorieProfile.activityLevel} (${dietPlan.calorieProfile.activityMultiplier}x)</span>
            </div>
          </div>
        </div>
        <div class="diet-calculation-note" style="margin-top: 14px; color: var(--muted);">
          Built from DEXA-estimated lean mass <strong>${dietPlan.calorieProfile.leanMassKg} kg</strong> and BMR <strong>${dietPlan.calorieProfile.bmr}</strong>. ${dietPlan.nutritionFocus.reason}
        </div>
      </div>

      <div class="diet-plan-header" style="margin-top: 16px; background: var(--card); border: 1px solid var(--border);">
        <h4 style="margin-bottom: 14px;">Choose Your Goal</h4>
        <div class="macro-breakdown" style="display:grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px;">
          ${[
            ['fat_loss', 'Fat Loss'],
            ['recomp', 'Recomp'],
            ['gain_muscle', 'Muscle Gain']
          ].map(([key, label]) => `
            <label class="diet-goal-option" style="display:block; border: 1px solid ${selectedGoal === key ? 'var(--accent-600)' : 'var(--border)'}; border-radius: 14px; padding: 14px; background: ${selectedGoal === key ? 'color-mix(in oklab, var(--bg-elev) 90%, var(--accent-100) 10%)' : 'var(--card)'}; color: var(--text); cursor: pointer;">
              <input type="radio" name="diet-goal-choice" value="${key}" ${selectedGoal === key ? 'checked' : ''} style="margin-right: 8px;">
              <strong>${label}</strong>
              <div style="margin-top: 8px; color: var(--text);">${objectiveOptions[key].calories} kcal/day</div>
              <div class="muted" style="margin-top: 4px;">${objectiveOptions[key].label}</div>
            </label>
          `).join('')}
        </div>
      </div>

      <div class="diet-plan-header" style="margin-top: 16px; background: var(--card); border: 1px solid var(--border);">
        <h4 style="margin-bottom: 12px;">Suggested Macros for ${selectedGoalLabel} at ${dietPlan.targetCalories} Calories</h4>
        <div class="macro-breakdown">
          <div class="macro-item">
            <span class="macro-label">Protein</span>
            <span class="macro-value">${dietPlan.macros.protein.grams}g (${dietPlan.macros.protein.percentage}%)</span>
          </div>
          <div class="macro-item">
            <span class="macro-label">Carbs</span>
            <span class="macro-value">${dietPlan.macros.carbs.grams}g (${dietPlan.macros.carbs.percentage}%)</span>
          </div>
          <div class="macro-item">
            <span class="macro-label">Fat</span>
            <span class="macro-value">${dietPlan.macros.fat.grams}g (${dietPlan.macros.fat.percentage}%)</span>
          </div>
        </div>
        <p class="muted" style="margin-top: 10px;">${dietPlan.calorieProfile.isManualTarget ? 'You are using a custom calorie target.' : `This target comes directly from the ${selectedGoalLabel.toLowerCase()} goal you selected above.`}</p>
      </div>

      <div class="diet-macro-controls card" style="margin-top: 16px;">
        <h4>⚙️ Macro Strategy</h4>
        <div class="grid cols-2">
          <label>Suggested split
            <select id="diet-macro-preset">
              <option value="recommended" ${!goals.macroPreset || goals.macroPreset === 'recommended' ? 'selected' : ''}>Recommended from DEXA</option>
              <option value="higher_protein" ${goals.macroPreset === 'higher_protein' ? 'selected' : ''}>Higher Protein</option>
              <option value="balanced_performance" ${goals.macroPreset === 'balanced_performance' ? 'selected' : ''}>Balanced Performance</option>
              <option value="higher_carb" ${goals.macroPreset === 'higher_carb' ? 'selected' : ''}>Higher Carb Training</option>
              <option value="lower_carb" ${goals.macroPreset === 'lower_carb' ? 'selected' : ''}>Lower Carb</option>
              <option value="custom" ${goals.customMacroMode === 'custom' ? 'selected' : ''}>Custom Ratio</option>
            </select>
          </label>
          <label>Suggested target calories
            <input type="number" id="diet-target-calories" value="${dietPlan.targetCalories}" min="1200" step="25">
          </label>
        </div>
        <div id="diet-custom-ratios" class="${goals.customMacroMode === 'custom' ? '' : 'hidden'}" style="margin-top: 12px;">
          <div class="grid cols-3">
            <label>Protein %
              <input type="number" id="diet-protein-ratio" min="10" max="60" value="${goals.customMacroRatio?.protein || 30}">
            </label>
            <label>Carbs %
              <input type="number" id="diet-carbs-ratio" min="10" max="70" value="${goals.customMacroRatio?.carbs || 40}">
            </label>
            <label>Fat %
              <input type="number" id="diet-fat-ratio" min="10" max="50" value="${goals.customMacroRatio?.fat || 30}">
            </label>
          </div>
          <p class="muted" style="margin-top: 8px;">Custom ratios should total 100%.</p>
        </div>
        <div class="row" style="margin-top: 12px;">
          <button id="apply-diet-macros-btn" class="btn btn-primary">Apply Macro Split</button>
        </div>
      </div>
      
      <div class="weekly-meal-plan">
        <h4>📅 7-Day Meal Plan</h4>
        <div class="horizontal-plan-rail">
          <div class="meal-plan-grid meal-plan-grid-horizontal">
            ${this.generateMealPlanHTML(dietPlan.weeklyPlan)}
          </div>
        </div>
      </div>

      <div class="weekly-meal-plan">
        <h4>📈 Expected Progress</h4>
        <div class="expected-results-grid">
          ${this.generateDietProgressHTML(selectedGoal, dietPlan, dexaResults)}
        </div>
      </div>
      
      <div class="diet-recommendations">
        <h4>💡 Recommendations</h4>
        <ul>
          ${dietPlan.recommendations.map(rec => `<li>${rec}</li>`).join('')}
        </ul>
      </div>
    `;

    this.currentPlan = dietPlan;
    this.setupDietPlanControls();
    if (window.latestDEXAResults && typeof window.showDEXAResults === 'function') {
      window.showDEXAResults(window.latestDEXAResults);
    }
  }

  // Generate meal plan HTML
  generateMealPlanHTML(weeklyPlan) {
    return weeklyPlan.map(day => `
      <div class="day-plan detailed-day-plan">
        <h5>${day.day}</h5>
        <div class="meals detailed-meals">
          ${['breakfast', 'lunch', 'dinner', 'snacks'].map((mealKey) => {
            const meal = day.meals[mealKey];
            const labelMap = {
              breakfast: 'Breakfast',
              lunch: 'Lunch',
              dinner: 'Dinner',
              snacks: 'Snack'
            };
            return `
              <div class="meal">
                <h6>${labelMap[mealKey]}</h6>
                <p><strong>${meal.name}</strong></p>
                <p class="meal-macro-line">
                  <span>${meal.calories} cal</span>
                  <span>${meal.protein}g protein</span>
                  <span>${meal.carbs}g carbs</span>
                  <span>${meal.fat}g fat</span>
                </p>
                <p class="ingredients">Serving: ${meal.servingMultiplier || 1}x</p>
                <div class="meal-portion-list">
                  ${meal.portionDetails.map((item) => `
                    <div class="meal-portion-row">
                      <span>${item.ingredient}</span>
                      <strong>${item.quantity}</strong>
                    </div>
                  `).join('')}
                </div>
              </div>
            `;
          }).join('')}
        </div>
        <div class="day-total">
          <strong>Total: ${day.totalCalories} calories</strong>
          <p class="ingredients">Macros: ${day.macros.protein}g P / ${day.macros.carbs}g C / ${day.macros.fat}g F</p>
        </div>
      </div>
    `).join('');
  }

  generateDietProgressHTML(goal, dietPlan, dexaResults) {
    const maintenance = dietPlan.calorieProfile.maintenanceCalories;
    const targetCalories = dietPlan.targetCalories;
    const dailyDelta = targetCalories - maintenance;
    const currentWeight = parseFloat(dexaResults.weight);
    const currentBodyFat = parseFloat(dexaResults.bodyFatPercent);
    const currentFatMass = parseFloat(dexaResults.fatMass || ((currentWeight * currentBodyFat) / 100));
    const currentMuscle = parseFloat(dexaResults.muscleMass);
    const currentVisceral = parseFloat(dexaResults.visceralFat);

    const timeframes = [4, 8, 12];
    return timeframes.map((weeks) => {
      const weeklyCalorieChange = (dailyDelta * 7 * weeks) / 7700;
      let fatChangeKg = 0;
      let muscleChangeKg = 0;
      let visceralChange = 0;

      if (goal === 'fat_loss') {
        fatChangeKg = Math.max(0, Math.abs(weeklyCalorieChange) * 0.82);
        muscleChangeKg = Math.max(-0.6, -0.05 * weeks);
        visceralChange = Math.min(currentVisceral * 0.45, 0.12 * weeks * Math.max(1, currentVisceral / 6));
      } else if (goal === 'gain_muscle') {
        muscleChangeKg = Math.min(0.35 * weeks, Math.max(0.8, weeklyCalorieChange * 0.55));
        fatChangeKg = Math.max(0.15 * (weeks / 4), weeklyCalorieChange * 0.35);
        visceralChange = Math.max(0, Math.min(0.8, fatChangeKg * 0.15));
      } else {
        fatChangeKg = Math.max(0.4, 0.18 * weeks);
        muscleChangeKg = Math.max(0.3, 0.12 * weeks);
        visceralChange = Math.min(currentVisceral * 0.25, 0.08 * weeks * Math.max(1, currentVisceral / 6));
      }

      const projectedWeight = currentWeight - fatChangeKg + muscleChangeKg;
      const projectedBodyFat = Math.max(6, ((currentFatMass - fatChangeKg) / projectedWeight) * 100);
      const projectedMuscle = currentMuscle + muscleChangeKg;
      const projectedVisceral = Math.max(1, currentVisceral - visceralChange);

      return `
      <div class="expected-result-card">
        <h5>${weeks} weeks</h5>
        <ul>
          <li>${goal === 'gain_muscle' ? 'Expected lean mass gain' : goal === 'fat_loss' ? 'Expected fat loss' : 'Expected fat loss'}: <strong>${goal === 'gain_muscle' ? `+${muscleChangeKg.toFixed(1)} kg` : `-${fatChangeKg.toFixed(1)} kg`}</strong></li>
          <li>Expected body fat change: <strong>${projectedBodyFat < currentBodyFat ? '-' : '+'}${Math.abs(currentBodyFat - projectedBodyFat).toFixed(1)}%</strong> to about <strong>${projectedBodyFat.toFixed(1)}%</strong></li>
          <li>Expected muscle mass: <strong>${projectedMuscle.toFixed(1)} kg</strong> (${muscleChangeKg >= 0 ? '+' : ''}${muscleChangeKg.toFixed(1)} kg)</li>
          <li>Expected visceral fat: <strong>${projectedVisceral.toFixed(1)}%</strong> (${projectedVisceral < currentVisceral ? '-' : '+'}${Math.abs(currentVisceral - projectedVisceral).toFixed(1)}%)</li>
          <li>Expected scale weight: <strong>${projectedWeight.toFixed(1)} kg</strong> (${projectedWeight >= currentWeight ? '+' : ''}${(projectedWeight - currentWeight).toFixed(1)} kg)</li>
        </ul>
      </div>
    `;
    }).join('');
  }

  getWorkoutDaySummary(workout) {
    if (workout.type === 'Rest Day') {
      return workout.description;
    }
    return (workout.exercises || [])
      .filter((exercise) => exercise.name !== 'Warm-up' && exercise.name !== 'Cool-down')
      .slice(0, 3)
      .map((exercise) => exercise.name)
      .join(', ');
  }

  generateTwelveWeekRoadmapHTML(progressionPlan, weeklyPlan) {
    const dayLabels = {
      monday: 'Mon',
      tuesday: 'Tue',
      wednesday: 'Wed',
      thursday: 'Thu',
      friday: 'Fri',
      saturday: 'Sat',
      sunday: 'Sun'
    };

    const weeklySplit = Object.entries(weeklyPlan)
      .map(([day, workout]) => `${dayLabels[day]}: ${workout.type}`)
      .join(' | ');

    return progressionPlan.phases.map((phase) => `
      <div class="roadmap-week-card">
        <div class="roadmap-week-top">
          <h5>Week ${phase.week}</h5>
          <span>${phase.intensity}</span>
        </div>
        <p class="roadmap-week-focus">${phase.focus}</p>
        <p class="roadmap-week-split">${weeklySplit}</p>
        <ul class="roadmap-week-changes">
          ${phase.changes.slice(0, 3).map((change) => `<li>${change}</li>`).join('')}
        </ul>
      </div>
    `).join('');
  }

  setupDietPlanControls() {
    const presetSelect = document.getElementById('diet-macro-preset');
    const applyButton = document.getElementById('apply-diet-macros-btn');
    const customRatios = document.getElementById('diet-custom-ratios');
    const goalOptions = document.querySelectorAll('input[name="diet-goal-choice"]');
    const targetCaloriesInput = document.getElementById('diet-target-calories');

    if (presetSelect && customRatios) {
      presetSelect.addEventListener('change', () => {
        customRatios.classList.toggle('hidden', presetSelect.value !== 'custom');
        if (presetSelect.value !== 'custom') {
          this.applyDietMacroSettings();
        }
      });
    }

    goalOptions.forEach((option) => {
      option.addEventListener('change', () => {
        const goals = this.getEffectiveGoals();
        const optionMap = this.currentObjectiveOptions || {};
        const nextCalories = optionMap[option.value]?.calories || targetCaloriesInput?.value || null;
        const updatedGoals = {
          ...goals,
          objective: option.value,
          dietTargetCalories: nextCalories,
          dietTargetMode: 'goal',
          macroPreset: 'recommended',
          customMacroMode: 'recommended'
        };
        localStorage.setItem('fitlife_goals', JSON.stringify(updatedGoals));
        this.loadDietPlan();
      });
    });

    if (applyButton) {
      applyButton.addEventListener('click', () => this.applyDietMacroSettings());
    }

    if (targetCaloriesInput) {
      targetCaloriesInput.addEventListener('change', () => this.applyDietMacroSettings());
    }

    ['diet-protein-ratio', 'diet-carbs-ratio', 'diet-fat-ratio'].forEach((id) => {
      const input = document.getElementById(id);
      if (!input) return;
      input.addEventListener('change', () => {
        if (document.getElementById('diet-macro-preset')?.value === 'custom') {
          this.applyDietMacroSettings();
        }
      });
    });
  }

  // Load workout plan interface
  loadWorkoutPlan() {
    const container = document.getElementById('workout-plan-container');
    if (!container) return;

    const normalizedInputs = window.getNormalizedBodyScanInputs ? window.getNormalizedBodyScanInputs() : null;
    if (!normalizedInputs?.ok) {
      container.innerHTML = '<p style="text-align: center; padding: 40px; color: #666;">⚠️ Please enter your details (age, height, weight) and run DEXA analysis first to generate a personalized workout plan.</p>';
      return;
    }

    const hasResults = !!window.latestDEXAResults;
    if (!hasResults) {
      container.innerHTML = '<p style="text-align: center; padding: 40px; color: #666;">⚠️ Please click "📊 Run DEXA Analysis" button first to generate your personalized workout plan based on your body composition.</p>';
      return;
    }

    const dexaResults = this.getCurrentDEXAResults();
    if (!dexaResults || !dexaResults.bodyFatPercent) {
      container.innerHTML = '<p style="text-align: center; padding: 40px; color: #666;">⚠️ Please run DEXA analysis first to generate workout plan.</p>';
      return;
    }
    
    // Validate that we have actual DEXA results, not just form values
    if (!window.generateDEXAResults) {
      container.innerHTML = '<p style="text-align: center; padding: 40px; color: #666;">⚠️ DEXA analysis system not loaded. Please refresh the page and run DEXA analysis.</p>';
      return;
    }

    // Log for debugging - show what we're using
    console.log('=== GENERATING WORKOUT PLAN ===', {
      bodyFat: dexaResults.bodyFatPercent,
      muscleMass: dexaResults.muscleMass,
      visceralFat: dexaResults.visceralFat,
      healthScore: dexaResults.healthScore,
      weight: dexaResults.weight,
      height: dexaResults.height,
      age: dexaResults.age,
      sex: dexaResults.sex,
      usingStoredResults: !!window.latestDEXAResults
    });
    
    // Verify the body fat percentage matches what's displayed
    const displayedBF = parseFloat(dexaResults.bodyFatPercent);
    console.log('Body Fat from DEXA Results:', displayedBF);
    
    const workoutPlan = this.planner.generateWorkoutPlan(dexaResults, this.getUserGoals());
    
    console.log('=== WORKOUT PLAN GENERATED ===', {
      fitnessLevel: workoutPlan.fitnessLevel,
      intensity: workoutPlan.workoutIntensity,
      mondayType: workoutPlan.weeklyPlan.monday?.type,
      mondayExercises: workoutPlan.weeklyPlan.monday?.exercises?.map(e => e.name),
      exercisesCount: workoutPlan.weeklyPlan.monday?.exercises?.length
    });
    
    container.innerHTML = `
      <div class="workout-plan-header">
        <h3>💪 Personalized Workout Plan</h3>
        <p>Based on your DEXA analysis: <strong>${dexaResults.bodyFatPercent}%</strong> body fat, <strong>${dexaResults.muscleMass}kg</strong> muscle mass, <strong>${dexaResults.visceralFat}%</strong> visceral fat</p>
        <div>
          <strong>Fitness Level:</strong> ${workoutPlan.fitnessLevel.charAt(0).toUpperCase() + workoutPlan.fitnessLevel.slice(1)} | 
          <strong>Intensity:</strong> ${workoutPlan.workoutIntensity.charAt(0).toUpperCase() + workoutPlan.workoutIntensity.slice(1)} |
          <strong>Health Score:</strong> ${dexaResults.healthScore}/100
        </div>
      </div>
      
      <div class="weekly-workout-plan">
        <h4>📅 Weekly Schedule</h4>
        <div class="horizontal-plan-rail">
          <div class="workout-schedule">
            ${this.generateWorkoutScheduleHTML(workoutPlan.weeklyPlan)}
          </div>
        </div>
      </div>
      
      ${workoutPlan.progressionPlan ? `
      <div class="progression-plan">
        <h4>📈 Full 12-Week Roadmap</h4>
        <div class="horizontal-plan-rail">
          <div class="roadmap-week-grid">
            ${this.generateTwelveWeekRoadmapHTML(workoutPlan.progressionPlan, workoutPlan.weeklyPlan)}
          </div>
        </div>
      </div>
      ` : ''}
      
      ${workoutPlan.expectedResults ? `
      <div class="expected-results">
        <h4>🎯 Expected Results</h4>
        <div class="expected-results-grid">
          ${Object.entries(workoutPlan.expectedResults).map(([timeframe, results]) => `
            <div class="expected-result-card">
              <h5>${timeframe}</h5>
              <ul>
                ${results.map(result => `<li>${result}</li>`).join('')}
              </ul>
            </div>
          `).join('')}
        </div>
      </div>
      ` : ''}
      
      <div class="workout-recommendations">
        <h4>💡 Detailed Recommendations</h4>
        <div>
          ${workoutPlan.recommendations.map(rec => `
            <div class="recommendation-card priority-${rec.priority.toLowerCase()}">
              <div class="recommendation-header">
                <strong class="recommendation-priority priority-${rec.priority.toLowerCase()}">${rec.priority} Priority</strong>
                <span class="recommendation-category">${rec.category}</span>
              </div>
              <p class="recommendation-text">${rec.recommendation}</p>
              <p class="recommendation-reason">${rec.reason}</p>
            </div>
          `).join('')}
        </div>
      </div>
    `;
  }

  // Generate workout schedule HTML
  generateWorkoutScheduleHTML(weeklyPlan) {
    const days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];
    
    return Object.entries(weeklyPlan).map(([day, workout]) => {
      const dayIndex = Object.keys(weeklyPlan).indexOf(day);
      const dayName = days[dayIndex];
      
      if (workout.type === 'Rest Day') {
        return `
          <div class="workout-day">
            <h5>${dayName}</h5>
            <div class="rest-day">
              <h6>🛌 ${workout.type}</h6>
              <p>${workout.description}</p>
              ${workout.activities ? `<p class="compact-workout-list">Optional: ${workout.activities.map(act => `${act.name} (${act.duration})`).join(' • ')}</p>` : ''}
            </div>
          </div>
        `;
      }
      
      return `
        <div class="workout-day">
          <h5>${dayName}</h5>
          <div class="workout-details">
            <div class="workout-meta">
              <div class="workout-meta-left">
                <h6>${workout.type}</h6>
                <p>${workout.focus}</p>
              </div>
              <div class="workout-meta-right">
                <span class="difficulty-badge">${workout.difficulty}</span>
                <p>⏱️ ${workout.duration}</p>
              </div>
            </div>
            <p class="compact-workout-list"><strong>Key work:</strong> ${this.getWorkoutDaySummary(workout)}</p>
            <details class="workout-detail-toggle">
              <summary>View full session</summary>
              ${workout.warmup ? `
                <div class="warmup-section">
                  <strong>Warm-up (${workout.warmup.duration}):</strong>
                  <p class="compact-workout-list">${workout.warmup.activities.join(' • ')}</p>
                </div>
              ` : ''}
              <div class="exercises">
                ${workout.exercises.map((ex, idx) => {
                  if (ex.name === 'Warm-up' || ex.name === 'Cool-down') {
                    return `
                      <div class="exercise">
                        <strong>${ex.name}</strong> - ${ex.duration} (${ex.intensity} intensity)
                        ${ex.notes ? `<p>${ex.notes}</p>` : ''}
                      </div>
                    `;
                  }
                  if (ex.exercises) {
                    return `
                      <div class="exercise">
                        <strong>${ex.name}:</strong>
                        <ul style="margin: 5px 0 0 20px; font-size: 0.9em;">
                          ${ex.exercises.map(e => `<li>${e.name} - ${e.reps || e.duration} ${e.sets ? `(${e.sets} sets)` : ''}</li>`).join('')}
                        </ul>
                      </div>
                    `;
                  }
                  return `
                    <div class="exercise">
                      <strong>${idx + 1}. ${ex.name}</strong>
                      <div class="exercise-content">
                        <div class="exercise-info">
                          ${ex.targetMuscles ? `<p class="target-muscles">Target: ${ex.targetMuscles.join(', ')}</p>` : ''}
                          ${ex.notes ? `<p>${ex.notes}</p>` : ''}
                          ${ex.progression ? `<div class="progression-note">Progression: ${ex.progression}</div>` : ''}
                        </div>
                        <div class="exercise-meta">
                          ${ex.sets ? `<span class="sets-reps">${ex.sets} sets</span>` : ''}
                          ${ex.reps ? `<span>${ex.reps} reps</span>` : ''}
                          ${ex.duration ? `<span>${ex.duration}</span>` : ''}
                          ${ex.rest ? `<span class="rest-time">Rest: ${ex.rest}</span>` : ''}
                        </div>
                      </div>
                    </div>
                  `;
                }).join('')}
              </div>
              ${workout.cooldown ? `
                <div class="cooldown-section">
                  <strong>Cool-down (${workout.cooldown.duration}):</strong>
                  <p class="compact-workout-list">${workout.cooldown.activities.join(' • ')}</p>
                </div>
              ` : ''}
            </details>
          </div>
        </div>
      `;
    }).join('');
  }

  // Load progress tracking interface
  loadProgressTracking() {
    const container = document.getElementById('progress-container');
    if (!container) return;

    const scanHistory = this.tracker.getScanHistory();
    const progressReport = this.tracker.generateProgressReport();
    
    container.innerHTML = `
      <div class="progress-header">
        <h3>📈 Progress Tracking</h3>
        <button id="add-scan-btn" class="btn btn-primary">📸 Add New Scan</button>
      </div>
      
      <div class="progress-summary">
        <h4>📊 Progress Summary</h4>
        ${this.generateProgressSummaryHTML(progressReport)}
      </div>
      
      <div class="scan-history">
        <h4>📅 Scan History</h4>
        <div class="scan-list">
          ${this.generateScanHistoryHTML(scanHistory)}
        </div>
      </div>
      
      <div class="photo-comparison">
        <h4>🖼️ Photo Comparison</h4>
        <div id="photo-comparison-container"></div>
      </div>
    `;

    this.setupProgressEventListeners();
  }

  // Generate progress summary HTML
  generateProgressSummaryHTML(report) {
    if (report.scans < 2) {
      return '<p>Need at least 2 scans to show progress summary.</p>';
    }

    const { comparison } = report;
    
    return `
      <div class="progress-metrics">
        <div class="metric-card ${comparison.weightChange > 0 ? 'positive' : 'negative'}">
          <h5>Weight</h5>
          <span>${comparison.weightChange > 0 ? '+' : ''}${comparison.weightChange.toFixed(1)} kg</span>
        </div>
        <div class="metric-card ${comparison.bodyFatChange < 0 ? 'positive' : 'negative'}">
          <h5>Body Fat</h5>
          <span>${comparison.bodyFatChange > 0 ? '+' : ''}${comparison.bodyFatChange.toFixed(1)}%</span>
        </div>
        <div class="metric-card ${comparison.muscleChange > 0 ? 'positive' : 'negative'}">
          <h5>Muscle Mass</h5>
          <span>${comparison.muscleChange > 0 ? '+' : ''}${comparison.muscleChange.toFixed(1)} kg</span>
        </div>
        <div class="metric-card">
          <h5>Time Span</h5>
          <span>${comparison.timeSpan} days</span>
        </div>
      </div>
      
      <div class="progress-recommendations">
        <h5>💡 Recommendations</h5>
        <ul>
          ${comparison.recommendations.map(rec => `<li>${rec.message}</li>`).join('')}
        </ul>
      </div>
    `;
  }

  // Generate scan history HTML
  generateScanHistoryHTML(scanHistory) {
    if (scanHistory.length === 0) {
      return '<p>No scans yet. Add your first scan to start tracking progress!</p>';
    }

    return scanHistory.map(scan => `
      <div class="scan-item">
        <div class="scan-date">${new Date(scan.date).toLocaleDateString()}</div>
        <div class="scan-metrics">
          <span>Weight: ${scan.weight}kg</span>
          <span>Body Fat: ${scan.bodyFatPercent}%</span>
          <span>Muscle: ${scan.muscleMass}kg</span>
        </div>
        <div class="scan-actions">
          <button class="btn btn-sm" onclick="fitnessUI.compareWithLatest(${scan.id})">Compare</button>
        </div>
      </div>
    `).join('');
  }

  // Load goals and settings interface
  loadGoalsSettings() {
    const container = document.getElementById('goals-container');
    if (!container) return;

    container.innerHTML = `
      <div class="goals-header">
        <h3>🎯 Goals & Settings</h3>
        <p>Customize your fitness journey</p>
      </div>
      
      <div class="goals-form">
        <h4>Fitness Goals</h4>
        <div class="form-group">
          <label>Primary Objective</label>
          <select id="fitness-objective">
            <option value="lose_fat">Lose Fat</option>
            <option value="gain_muscle">Gain Muscle</option>
            <option value="maintain">Maintain Current</option>
            <option value="recomp">Body Recomposition</option>
          </select>
        </div>
        
        <div class="form-group">
          <label>Activity Level</label>
          <select id="fitness-activity-level">
            <option value="sedentary">Sedentary (Little/No Exercise)</option>
            <option value="light">Light (1-3 days/week)</option>
            <option value="moderate" selected>Moderate (3-5 days/week)</option>
            <option value="active">Active (6-7 days/week)</option>
            <option value="very_active">Very Active (2x/day)</option>
          </select>
        </div>
        
        <div class="form-group">
          <label>Target Weight (kg)</label>
          <input type="number" id="target-weight" placeholder="Enter target weight">
        </div>
        
        <div class="form-group">
          <label>Target Body Fat %</label>
          <input type="number" id="target-bodyfat" placeholder="Enter target body fat %">
        </div>
        
        <button id="save-goals-btn" class="btn btn-primary">💾 Save Goals</button>
      </div>
      
      <div class="current-goals">
        <h4>Current Goals</h4>
        <div id="current-goals-display"></div>
      </div>
    `;

    this.setupGoalsEventListeners();
  }

  // Setup event listeners
  setupEventListeners() {
    // Add generate fitness plan button listener
    const generateBtn = document.getElementById('generate-fitness-plan');
    if (generateBtn) {
      generateBtn.addEventListener('click', () => {
        this.loadDietPlan();
        this.loadWorkoutPlan();
        this.loadProgressTracking();
        this.loadGoalsSettings();
      });
    }
  }

  // Setup progress tracking event listeners
  setupProgressEventListeners() {
    const addScanBtn = document.getElementById('add-scan-btn');
    if (addScanBtn) {
      addScanBtn.addEventListener('click', () => this.showAddScanModal());
    }
  }

  // Setup goals event listeners
  setupGoalsEventListeners() {
    const saveGoalsBtn = document.getElementById('save-goals-btn');
    if (saveGoalsBtn) {
      saveGoalsBtn.addEventListener('click', () => this.saveUserGoals());
    }
  }

  // Get current DEXA results
  getCurrentDEXAResults() {
    const normalizedInputs = window.getNormalizedBodyScanInputs ? window.getNormalizedBodyScanInputs() : null;

    // First, try to use the latest stored DEXA results (from when analysis was run)
    if (window.latestDEXAResults && normalizedInputs?.ok) {
      console.log('=== USING STORED DEXA RESULTS ===', window.latestDEXAResults);
      return {
        ...window.latestDEXAResults,
        weight: normalizedInputs.weightKg,
        height: normalizedInputs.heightCm,
        age: normalizedInputs.age,
        sex: normalizedInputs.sex
      };
    }
    
    // Fallback: Always use the actual DEXA results generator - it uses the accurate formula
    if (window.generateDEXAResults && normalizedInputs?.ok) {
      try {
        const results = window.generateDEXAResults({ inputData: normalizedInputs });
        console.log('=== DEXA RESULTS RETRIEVED FROM GENERATOR ===', results);
        return {
          ...results,
          weight: normalizedInputs.weightKg,
          height: normalizedInputs.heightCm,
          age: normalizedInputs.age,
          sex: normalizedInputs.sex
        };
      } catch (error) {
        console.log('Error getting DEXA results:', error);
      }
    }
    
    // Return null if no valid data - don't use fallback calculations
    console.log('=== NO DEXA RESULTS AVAILABLE ===');
    return null;
  }

  // Get user goals
  getUserGoals() {
    const defaults = {
      objective: 'maintain',
      activityLevel: 'moderate',
      targetWeight: null,
      targetBodyFat: null,
      macroPreset: 'recommended',
      customMacroMode: 'recommended',
      customMacroRatio: { protein: 30, carbs: 40, fat: 30 },
      dietTargetCalories: null,
      dietTargetMode: 'goal'
    };
    const saved = localStorage.getItem('fitlife_goals');
    if (!saved) return defaults;
    try {
      const parsed = JSON.parse(saved);
      return {
        ...defaults,
        ...parsed,
        customMacroRatio: {
          ...defaults.customMacroRatio,
          ...(parsed.customMacroRatio || {})
        }
      };
    } catch (error) {
      console.error('Failed to parse saved goals:', error);
      return defaults;
    }
  }

  mapCalculatorActivityToGoal(activityValue) {
    const value = parseFloat(activityValue);
    if (!Number.isFinite(value)) return null;
    if (Math.abs(value - 1.2) < 0.001) return 'sedentary';
    if (Math.abs(value - 1.375) < 0.001) return 'light';
    if (Math.abs(value - 1.55) < 0.001) return 'moderate';
    if (Math.abs(value - 1.725) < 0.001) return 'active';
    if (Math.abs(value - 1.9) < 0.001) return 'very_active';
    return null;
  }

  getEffectiveGoals() {
    const goals = this.getUserGoals();
    const calculatorActivity = document.getElementById('activity-level')?.value;
    const mappedActivity = this.mapCalculatorActivityToGoal(calculatorActivity);
    if (mappedActivity) {
      return {
        ...goals,
        activityLevel: mappedActivity
      };
    }
    return goals;
  }

  // Save user goals
  saveUserGoals() {
    const existingGoals = this.getUserGoals();
    const goals = {
      objective: document.getElementById('fitness-objective')?.value || 'maintain',
      activityLevel: document.getElementById('fitness-activity-level')?.value || 'moderate',
      targetWeight: parseFloat(document.getElementById('target-weight')?.value) || null,
      targetBodyFat: parseFloat(document.getElementById('target-bodyfat')?.value) || null,
      macroPreset: existingGoals.macroPreset || 'recommended',
      customMacroMode: existingGoals.customMacroMode || 'recommended',
      customMacroRatio: existingGoals.customMacroRatio || { protein: 30, carbs: 40, fat: 30 },
      dietTargetCalories: existingGoals.dietTargetCalories || null,
      dietTargetMode: existingGoals.dietTargetMode || 'goal'
    };

    localStorage.setItem('fitlife_goals', JSON.stringify(goals));
    this.loadUserGoals();
    alert('Goals saved successfully!');
  }

  // Load user goals into form
  loadUserGoals() {
    const goals = this.getEffectiveGoals();
    
    const objectiveSelect = document.getElementById('fitness-objective');
    const activitySelect = document.getElementById('fitness-activity-level');
    const targetWeightInput = document.getElementById('target-weight');
    const targetBodyFatInput = document.getElementById('target-bodyfat');
    
    if (objectiveSelect) objectiveSelect.value = goals.objective;
    if (activitySelect) activitySelect.value = goals.activityLevel;
    if (targetWeightInput) targetWeightInput.value = goals.targetWeight || '';
    if (targetBodyFatInput) targetBodyFatInput.value = goals.targetBodyFat || '';
  }

  applyDietMacroSettings() {
    if (!this.currentDietResults) return;

    const goals = this.getEffectiveGoals();
    const preset = document.getElementById('diet-macro-preset')?.value || 'recommended';
    const targetCalories = parseFloat(document.getElementById('diet-target-calories')?.value);
    const proteinRatio = parseFloat(document.getElementById('diet-protein-ratio')?.value);
    const carbsRatio = parseFloat(document.getElementById('diet-carbs-ratio')?.value);
    const fatRatio = parseFloat(document.getElementById('diet-fat-ratio')?.value);
    const selectedGoal = document.querySelector('input[name="diet-goal-choice"]:checked')?.value;
    const goalCalories = this.currentObjectiveOptions?.[selectedGoal || goals.objective]?.calories || null;
    const normalizedTargetCalories = Number.isFinite(targetCalories) ? Math.round(targetCalories) : null;
    const isManualTarget = Number.isFinite(normalizedTargetCalories) && Number.isFinite(goalCalories)
      ? normalizedTargetCalories !== Math.round(goalCalories)
      : Number.isFinite(normalizedTargetCalories);

    const updatedGoals = {
      ...goals,
      objective: selectedGoal || goals.objective,
      macroPreset: preset,
      customMacroMode: preset === 'custom' ? 'custom' : 'recommended',
      dietTargetCalories: isManualTarget ? normalizedTargetCalories : goalCalories,
      dietTargetMode: isManualTarget ? 'manual' : 'goal'
    };

    if (preset === 'custom') {
      const total = proteinRatio + carbsRatio + fatRatio;
      if (Math.round(total) !== 100) {
        alert('Custom macro ratios must total 100%.');
        return;
      }
      updatedGoals.customMacroRatio = {
        protein: proteinRatio,
        carbs: carbsRatio,
        fat: fatRatio
      };
    }

    localStorage.setItem('fitlife_goals', JSON.stringify(updatedGoals));
    this.loadDietPlan();
  }

  // Show add scan modal
  showAddScanModal() {
    // This would show a modal for adding new scans
    alert('Add scan functionality will be implemented in the next step!');
  }

  // Compare with latest scan
  compareWithLatest(scanId) {
    const latest = this.tracker.getLatestScan();
    if (!latest) return;

    const comparison = this.tracker.compareScans(scanId, latest.id);
    if (!comparison) return;

    // Show comparison results
    alert(`Comparison Results:\nWeight Change: ${comparison.weightChange}kg\nBody Fat Change: ${comparison.bodyFatChange}%\nMuscle Change: ${comparison.muscleChange}kg`);
  }
}

// Initialize global fitness UI
window.fitnessUI = new FitnessUI();

console.log('=== FITNESS UI LOADED ===');
