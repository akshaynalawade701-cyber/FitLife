console.log('=== DIET & WORKOUT PLANNER LOADING ===');

// Diet and Workout Planner System
class FitnessPlanner {
  constructor() {
    this.mealDatabase = this.initializeMealDatabase();
    this.workoutDatabase = this.initializeWorkoutDatabase();
    this.userProgress = this.loadUserProgress();
  }

  // Initialize comprehensive meal database
  initializeMealDatabase() {
    return {
      breakfast: [
        {
          name: "Protein Power Bowl",
          calories: 450,
          protein: 35,
          carbs: 25,
          fat: 20,
          ingredients: ["Greek yogurt", "Berries", "Nuts", "Honey", "Chia seeds"],
          portions: ["200 g", "80 g", "20 g", "10 g", "10 g"],
          prepTime: "5 min",
          category: "high-protein"
        },
        {
          name: "Avocado Toast Deluxe",
          calories: 380,
          protein: 18,
          carbs: 32,
          fat: 22,
          ingredients: ["Whole grain bread", "Avocado", "Eggs", "Tomatoes", "Spinach"],
          portions: ["2 slices", "100 g", "2 whole", "80 g", "40 g"],
          prepTime: "10 min",
          category: "balanced"
        },
        {
          name: "Oatmeal Powerhouse",
          calories: 420,
          protein: 22,
          carbs: 45,
          fat: 15,
          ingredients: ["Steel cut oats", "Protein powder", "Banana", "Almonds", "Cinnamon"],
          portions: ["60 g", "30 g", "1 medium", "20 g", "2 g"],
          prepTime: "8 min",
          category: "high-protein"
        },
        {
          name: "Green Smoothie Bowl",
          calories: 350,
          protein: 20,
          carbs: 40,
          fat: 12,
          ingredients: ["Spinach", "Banana", "Protein powder", "Almond milk", "Granola"],
          portions: ["40 g", "1 medium", "25 g", "250 ml", "35 g"],
          prepTime: "7 min",
          category: "low-calorie"
        }
      ],
      lunch: [
        {
          name: "Grilled Chicken Salad",
          calories: 480,
          protein: 42,
          carbs: 20,
          fat: 25,
          ingredients: ["Chicken breast", "Mixed greens", "Cherry tomatoes", "Cucumber", "Olive oil"],
          portions: ["160 g", "90 g", "80 g", "80 g", "15 ml"],
          prepTime: "15 min",
          category: "high-protein"
        },
        {
          name: "Quinoa Buddha Bowl",
          calories: 520,
          protein: 28,
          carbs: 55,
          fat: 18,
          ingredients: ["Quinoa", "Chickpeas", "Sweet potato", "Broccoli", "Tahini"],
          portions: ["180 g cooked", "120 g", "150 g", "100 g", "20 g"],
          prepTime: "20 min",
          category: "balanced"
        },
        {
          name: "Salmon & Veggies",
          calories: 460,
          protein: 38,
          carbs: 25,
          fat: 22,
          ingredients: ["Salmon fillet", "Asparagus", "Brown rice", "Lemon", "Herbs"],
          portions: ["160 g", "120 g", "140 g cooked", "1/2 piece", "5 g"],
          prepTime: "18 min",
          category: "high-protein"
        },
        {
          name: "Mediterranean Wrap",
          calories: 420,
          protein: 25,
          carbs: 35,
          fat: 20,
          ingredients: ["Whole wheat wrap", "Hummus", "Feta cheese", "Olives", "Cucumber"],
          portions: ["1 large", "50 g", "40 g", "30 g", "60 g"],
          prepTime: "12 min",
          category: "balanced"
        }
      ],
      dinner: [
        {
          name: "Turkey Meatballs & Zoodles",
          calories: 450,
          protein: 40,
          carbs: 15,
          fat: 25,
          ingredients: ["Ground turkey", "Zucchini noodles", "Marinara sauce", "Parmesan", "Basil"],
          portions: ["180 g", "180 g", "120 g", "15 g", "5 g"],
          prepTime: "25 min",
          category: "low-carb"
        },
        {
          name: "Baked Cod & Sweet Potato",
          calories: 480,
          protein: 35,
          carbs: 40,
          fat: 18,
          ingredients: ["Cod fillet", "Sweet potato", "Green beans", "Garlic", "Olive oil"],
          portions: ["170 g", "220 g", "120 g", "8 g", "10 ml"],
          prepTime: "30 min",
          category: "balanced"
        },
        {
          name: "Beef Stir Fry",
          calories: 520,
          protein: 45,
          carbs: 25,
          fat: 28,
          ingredients: ["Lean beef", "Broccoli", "Bell peppers", "Soy sauce", "Ginger"],
          portions: ["170 g", "120 g", "100 g", "15 ml", "8 g"],
          prepTime: "20 min",
          category: "high-protein"
        },
        {
          name: "Vegetarian Curry",
          calories: 380,
          protein: 22,
          carbs: 45,
          fat: 15,
          ingredients: ["Chickpeas", "Coconut milk", "Cauliflower", "Spinach", "Curry spices"],
          portions: ["160 g", "120 ml", "160 g", "50 g", "8 g"],
          prepTime: "25 min",
          category: "plant-based"
        }
      ],
      snacks: [
        {
          name: "Greek Yogurt Parfait",
          calories: 180,
          protein: 15,
          carbs: 20,
          fat: 4,
          ingredients: ["Greek yogurt", "Berries", "Granola"],
          portions: ["170 g", "70 g", "25 g"],
          prepTime: "3 min",
          category: "high-protein"
        },
        {
          name: "Almond & Apple",
          calories: 200,
          protein: 6,
          carbs: 25,
          fat: 12,
          ingredients: ["Almonds", "Apple", "Cinnamon"],
          portions: ["28 g", "1 medium", "2 g"],
          prepTime: "2 min",
          category: "balanced"
        },
        {
          name: "Protein Smoothie",
          calories: 220,
          protein: 25,
          carbs: 15,
          fat: 6,
          ingredients: ["Protein powder", "Almond milk", "Banana", "Peanut butter"],
          portions: ["30 g", "250 ml", "1/2 medium", "12 g"],
          prepTime: "5 min",
          category: "high-protein"
        },
        {
          name: "Veggie Hummus",
          calories: 150,
          protein: 8,
          carbs: 18,
          fat: 6,
          ingredients: ["Carrots", "Cucumber", "Hummus", "Bell peppers"],
          portions: ["80 g", "80 g", "45 g", "70 g"],
          prepTime: "5 min",
          category: "low-calorie"
        }
      ]
    };
  }

  // Initialize workout database
  initializeWorkoutDatabase() {
    return {
      strength: [
        {
          name: "Upper Body Power",
          duration: "45 min",
          difficulty: "Intermediate",
          exercises: [
            { name: "Bench Press", sets: 4, reps: "8-10", rest: "2 min" },
            { name: "Pull-ups", sets: 3, reps: "6-8", rest: "2 min" },
            { name: "Shoulder Press", sets: 3, reps: "10-12", rest: "90 sec" },
            { name: "Bent-over Rows", sets: 3, reps: "10-12", rest: "90 sec" },
            { name: "Bicep Curls", sets: 3, reps: "12-15", rest: "60 sec" },
            { name: "Tricep Dips", sets: 3, reps: "10-12", rest: "60 sec" }
          ],
          targetMuscles: ["Chest", "Back", "Shoulders", "Arms"],
          equipment: ["Barbell", "Dumbbells", "Pull-up bar"]
        },
        {
          name: "Lower Body Blast",
          duration: "50 min",
          difficulty: "Intermediate",
          exercises: [
            { name: "Squats", sets: 4, reps: "10-12", rest: "2 min" },
            { name: "Deadlifts", sets: 4, reps: "8-10", rest: "2 min" },
            { name: "Lunges", sets: 3, reps: "12 each leg", rest: "90 sec" },
            { name: "Leg Press", sets: 3, reps: "12-15", rest: "90 sec" },
            { name: "Calf Raises", sets: 4, reps: "15-20", rest: "60 sec" },
            { name: "Glute Bridges", sets: 3, reps: "15-20", rest: "60 sec" }
          ],
          targetMuscles: ["Quads", "Hamstrings", "Glutes", "Calves"],
          equipment: ["Barbell", "Dumbbells", "Leg press machine"]
        },
        {
          name: "Full Body Circuit",
          duration: "40 min",
          difficulty: "Beginner",
          exercises: [
            { name: "Push-ups", sets: 3, reps: "10-15", rest: "60 sec" },
            { name: "Bodyweight Squats", sets: 3, reps: "15-20", rest: "60 sec" },
            { name: "Plank", sets: 3, reps: "30-45 sec", rest: "60 sec" },
            { name: "Lunges", sets: 3, reps: "10 each leg", rest: "60 sec" },
            { name: "Mountain Climbers", sets: 3, reps: "20-30", rest: "60 sec" },
            { name: "Burpees", sets: 3, reps: "8-12", rest: "90 sec" }
          ],
          targetMuscles: ["Full Body"],
          equipment: ["Bodyweight only"]
        }
      ],
      cardio: [
        {
          name: "HIIT Cardio Blast",
          duration: "25 min",
          difficulty: "High",
          exercises: [
            { name: "Sprint Intervals", sets: 8, reps: "30 sec on, 30 sec off", rest: "30 sec" },
            { name: "Jump Squats", sets: 4, reps: "45 sec", rest: "15 sec" },
            { name: "High Knees", sets: 4, reps: "30 sec", rest: "30 sec" },
            { name: "Burpees", sets: 4, reps: "45 sec", rest: "15 sec" }
          ],
          targetMuscles: ["Cardiovascular", "Full Body"],
          equipment: ["None"]
        },
        {
          name: "Steady State Cardio",
          duration: "45 min",
          difficulty: "Low",
          exercises: [
            { name: "Treadmill Walk/Run", sets: 1, reps: "45 min", rest: "None" }
          ],
          targetMuscles: ["Cardiovascular", "Legs"],
          equipment: ["Treadmill or Outdoor"]
        }
      ],
      flexibility: [
        {
          name: "Yoga Flow",
          duration: "30 min",
          difficulty: "Beginner",
          exercises: [
            { name: "Sun Salutation", sets: 3, reps: "5 rounds", rest: "None" },
            { name: "Warrior Poses", sets: 2, reps: "Hold 30 sec each", rest: "None" },
            { name: "Downward Dog", sets: 3, reps: "Hold 45 sec", rest: "None" },
            { name: "Child's Pose", sets: 3, reps: "Hold 30 sec", rest: "None" }
          ],
          targetMuscles: ["Full Body Flexibility"],
          equipment: ["Yoga mat"]
        }
      ]
    };
  }

  // Generate personalized diet plan
  generateDietPlan(dexaResults, goals = {}) {
    const { weight, height, age, sex, bodyFatPercent, muscleMass, fatMass, visceralFat } = dexaResults;
    const bodyFat = parseFloat(bodyFatPercent);
    const totalWeight = parseFloat(weight);
    const leanMassKg = Math.max(
      parseFloat(muscleMass || 0) + parseFloat(dexaResults.boneMass || 0) + parseFloat(dexaResults.waterMass || 0) * 0.27,
      fatMass ? totalWeight - parseFloat(fatMass) : totalWeight * (1 - (bodyFat / 100))
    );

    const calorieProfile = this.calculateDEXACalorieTargets({
      weight: totalWeight,
      height: parseFloat(height),
      age: parseFloat(age),
      sex,
      leanMassKg,
      bodyFatPercent: bodyFat,
      visceralFat: parseFloat(visceralFat),
      goals
    });

    const nutritionFocus = this.determineNutritionFocus(dexaResults, goals, calorieProfile);
    const objectiveOptions = this.getObjectiveCalorieOptions(dexaResults, calorieProfile);
    const selectedObjective = goals.objective && objectiveOptions[goals.objective]
      ? goals.objective
      : nutritionFocus.recommendation;

    if (objectiveOptions[selectedObjective]) {
      nutritionFocus.recommendation = selectedObjective;
      nutritionFocus.suggestedCalories = objectiveOptions[selectedObjective].calories;
      nutritionFocus.suggestedLabel = objectiveOptions[selectedObjective].label;
    }
    if (
      goals.dietTargetMode === 'manual' &&
      goals.dietTargetCalories &&
      Number.isFinite(parseFloat(goals.dietTargetCalories))
    ) {
      calorieProfile.targetCalories = Math.round(parseFloat(goals.dietTargetCalories));
      calorieProfile.isManualTarget = true;
    } else {
      calorieProfile.targetCalories = nutritionFocus.suggestedCalories;
      calorieProfile.isManualTarget = false;
    }

    this.finalizeCalorieProfile(calorieProfile, goals);

    const macroPreset = this.resolveMacroPreset(goals, bodyFat);
    const macros = this.calculateMacros(calorieProfile.targetCalories, {
      goals,
      bodyFatPercent: bodyFat,
      leanMassKg,
      weightKg: totalWeight,
      preset: macroPreset,
      customRatio: goals.customMacroRatio || null
    });

    const weeklyPlan = this.generateWeeklyMealPlan(calorieProfile.targetCalories, macros);
    
    return {
      targetCalories: calorieProfile.targetCalories,
      calorieProfile,
      nutritionFocus,
      objectiveOptions,
      macros,
      weeklyPlan,
      twelveWeekNutritionPlan: this.generateTwelveWeekNutritionPlan(dexaResults, goals, {
        targetCalories: calorieProfile.targetCalories,
        calorieProfile,
        nutritionFocus,
        macroPreset,
        macros
      }),
      macroPreset,
      recommendations: this.generateDietRecommendations(dexaResults, goals, calorieProfile, macros)
    };
  }

  // Calculate BMR using Mifflin-St Jeor Equation
  calculateBMR(weight, height, age, sex) {
    if (sex === 'male') {
      return 10 * weight + 6.25 * height - 5 * age + 5;
    } else {
      return 10 * weight + 6.25 * height - 5 * age - 161;
    }
  }

  // Calculate TDEE based on activity level
  calculateTDEE(bmr, activityLevel) {
    const multipliers = {
      'sedentary': 1.2,
      'light': 1.375,
      'moderate': 1.55,
      'active': 1.725,
      'very_active': 1.9
    };
    return bmr * (multipliers[activityLevel] || 1.55);
  }

  getActivityMultiplier(activityLevel) {
    const multipliers = {
      'sedentary': 1.2,
      'light': 1.375,
      'moderate': 1.55,
      'active': 1.725,
      'very_active': 1.9
    };
    return multipliers[activityLevel] || 1.55;
  }

  calculateDEXACalorieTargets({ weight, height, age, sex, leanMassKg, bodyFatPercent, visceralFat, goals }) {
    const mifflinBmr = this.calculateBMR(weight, height, age, sex);
    const katchBmr = 370 + (21.6 * leanMassKg);
    const baseBmr = Math.round((mifflinBmr * 0.35) + (katchBmr * 0.65));
    const activityLevel = goals.activityLevel || 'moderate';
    const activityMultiplier = this.getActivityMultiplier(activityLevel);
    const maintenanceCalories = Math.round(this.calculateTDEE(baseBmr, activityLevel));

    let adjustment = 0;
    let strategy = 'maintenance';

    if (goals.objective === 'lose_fat') {
      const deficitPercent = bodyFatPercent >= 28 || visceralFat >= 8 ? 0.20 : bodyFatPercent >= 22 ? 0.17 : 0.14;
      adjustment = -Math.round(maintenanceCalories * deficitPercent);
      strategy = 'fat-loss deficit';
    } else if (goals.objective === 'gain_muscle') {
      adjustment = Math.round(maintenanceCalories * 0.08);
      strategy = 'lean-gain surplus';
    } else if (goals.objective === 'recomp') {
      adjustment = bodyFatPercent >= 20 ? -Math.round(maintenanceCalories * 0.08) : 0;
      strategy = bodyFatPercent >= 20 ? 'recomp slight deficit' : 'recomp maintenance';
    }

    let targetCalories = maintenanceCalories + adjustment;

    if (goals.targetWeight && goals.targetWeight < weight) {
      targetCalories = Math.min(targetCalories, maintenanceCalories - 250);
    }
    if (goals.targetWeight && goals.targetWeight > weight && goals.objective !== 'lose_fat') {
      targetCalories = Math.max(targetCalories, maintenanceCalories + 150);
    }

    targetCalories = Math.max(Math.round(baseBmr * 1.15), targetCalories);

    return {
      bmr: baseBmr,
      activityLevel,
      activityMultiplier,
      maintenanceCalories,
      targetCalories,
      leanMassKg: Math.round(leanMassKg * 10) / 10,
      strategy,
      isManualTarget: false
    };
  }

  finalizeCalorieProfile(calorieProfile, goals) {
    const deltaFromMaintenance = calorieProfile.targetCalories - calorieProfile.maintenanceCalories;
    calorieProfile.deltaFromMaintenance = deltaFromMaintenance;

    if (deltaFromMaintenance > 0) {
      calorieProfile.direction = 'surplus';
      calorieProfile.directionLabel = `${deltaFromMaintenance} calorie surplus`;
    } else if (deltaFromMaintenance < 0) {
      calorieProfile.direction = 'deficit';
      calorieProfile.directionLabel = `${Math.abs(deltaFromMaintenance)} calorie deficit`;
    } else {
      calorieProfile.direction = 'maintenance';
      calorieProfile.directionLabel = 'maintenance calories';
    }

    if (calorieProfile.isManualTarget) {
      calorieProfile.strategy = `manual target set at ${calorieProfile.directionLabel}`;
    }

    calorieProfile.objectiveConsistency = 'aligned';
    if (goals.objective === 'lose_fat' && deltaFromMaintenance >= 0) {
      calorieProfile.objectiveConsistency = 'not_aligned';
    } else if (goals.objective === 'gain_muscle' && deltaFromMaintenance <= 0) {
      calorieProfile.objectiveConsistency = 'not_aligned';
    }
  }

  determineNutritionFocus(dexaResults, goals, calorieProfile) {
    const bodyFat = parseFloat(dexaResults.bodyFatPercent);
    const visceralFat = parseFloat(dexaResults.visceralFat);
    const muscleMass = parseFloat(dexaResults.muscleMass);
    const weight = parseFloat(dexaResults.weight);
    const muscleRatio = muscleMass / Math.max(weight, 1);

    let recommendation = 'maintain';
    let reason = 'Your current composition looks balanced enough to maintain while improving training quality.';

    if (bodyFat >= 24 || visceralFat >= 7) {
      recommendation = 'fat_loss';
      reason = 'Your body fat and visceral fat markers suggest cutting body fat first will likely improve health and body composition the fastest.';
    } else if (muscleRatio < 0.4 || muscleMass < 30) {
      recommendation = 'gain_muscle';
      reason = 'Your lean mass markers suggest adding muscle should be the main priority right now.';
    } else if (bodyFat >= 18 && muscleRatio < 0.45) {
      recommendation = 'recomp';
      reason = 'You appear suited for recomposition: trimming fat slowly while building or preserving muscle.';
    }

    if (goals.objective && goals.objective !== 'maintain') {
      recommendation = goals.objective;
      reason = `Your saved goal is set to ${goals.objective.replace('_', ' ')}.`;
    }

    const recommendationMap = {
      lose_fat: 'fat_loss',
      gain_muscle: 'gain_muscle',
      recomp: 'recomp',
      maintain: 'maintain'
    };
    const normalizedRecommendation = recommendationMap[recommendation] || recommendation;

    return {
      recommendation: normalizedRecommendation,
      reason,
      suggestedCalories: calorieProfile.maintenanceCalories,
      suggestedLabel: 'maintenance calories'
    };
  }

  getObjectiveCalorieOptions(dexaResults, calorieProfile) {
    const bodyFat = parseFloat(dexaResults.bodyFatPercent);
    const visceralFat = parseFloat(dexaResults.visceralFat);

    const fatLossDeficit = bodyFat >= 24 || visceralFat >= 7 ? 350 : 250;
    const recompDeficit = bodyFat >= 18 ? 120 : 0;
    const muscleGainSurplus = bodyFat >= 22 ? 120 : 180;

    return {
      fat_loss: {
        calories: Math.max(calorieProfile.bmr + 150, calorieProfile.maintenanceCalories - fatLossDeficit),
        label: `${fatLossDeficit} calorie deficit`
      },
      recomp: {
        calories: Math.max(calorieProfile.bmr + 100, calorieProfile.maintenanceCalories - recompDeficit),
        label: recompDeficit > 0 ? `${recompDeficit} calorie slight deficit` : 'maintenance calories'
      },
      gain_muscle: {
        calories: calorieProfile.maintenanceCalories + muscleGainSurplus,
        label: `${muscleGainSurplus} calorie surplus`
      }
    };
  }

  resolveMacroPreset(goals, bodyFatPercent) {
    if (goals.macroPreset === 'recommended') {
      return goals.objective === 'gain_muscle'
        ? 'higher_carb'
        : goals.objective === 'lose_fat' || bodyFatPercent >= 20
          ? 'higher_protein'
          : 'balanced_performance';
    }
    if (goals.customMacroMode === 'custom' && goals.customMacroRatio) {
      return 'custom';
    }
    if (goals.macroPreset && goals.macroPreset !== 'recommended') {
      return goals.macroPreset;
    }
    if (goals.objective === 'gain_muscle') return 'higher_carb';
    if (goals.objective === 'lose_fat' || bodyFatPercent >= 20) return 'higher_protein';
    if (goals.objective === 'recomp') return 'balanced_performance';
    return 'balanced_performance';
  }

  getMacroRatioPreset(preset) {
    const presets = {
      higher_protein: { protein: 40, carbs: 25, fat: 35 },
      balanced_performance: { protein: 30, carbs: 40, fat: 30 },
      higher_carb: { protein: 25, carbs: 50, fat: 25 },
      lower_carb: { protein: 40, carbs: 20, fat: 40 }
    };
    return presets[preset] || presets.balanced_performance;
  }

  // Calculate macro distribution
  calculateMacros(calories, options) {
    const {
      goals,
      bodyFatPercent,
      leanMassKg,
      weightKg,
      preset,
      customRatio
    } = options;

    const proteinFloorPerKg = goals.objective === 'gain_muscle' ? 2.0 : goals.objective === 'lose_fat' ? 2.2 : 1.9;
    const proteinFloor = Math.round(Math.max(weightKg * proteinFloorPerKg, leanMassKg * 2.4));
    const fatFloor = Math.round(Math.max(weightKg * 0.7, (calories * 0.22) / 9));

    const selectedRatio = preset === 'custom' && customRatio
      ? customRatio
      : this.getMacroRatioPreset(preset);

    let proteinCalories = Math.round(calories * (selectedRatio.protein / 100));
    let fatCalories = Math.round(calories * (selectedRatio.fat / 100));
    let carbCalories = calories - proteinCalories - fatCalories;

    let proteinGrams = Math.round(proteinCalories / 4);
    let fatGrams = Math.round(fatCalories / 9);

    if (proteinGrams < proteinFloor) {
      proteinGrams = proteinFloor;
      proteinCalories = proteinGrams * 4;
    }

    if (fatGrams < fatFloor) {
      fatGrams = fatFloor;
      fatCalories = fatGrams * 9;
    }

    carbCalories = calories - proteinCalories - fatCalories;
    if (carbCalories < weightKg * 2 * 4) {
      const minimumCarbCalories = Math.round(weightKg * 2 * 4);
      const difference = minimumCarbCalories - carbCalories;
      fatCalories = Math.max((fatFloor * 9), fatCalories - difference);
      fatGrams = Math.round(fatCalories / 9);
      carbCalories = calories - proteinCalories - fatCalories;
    }

    const carbGrams = Math.max(0, Math.round(carbCalories / 4));
    const normalizedCalories = proteinCalories + fatCalories + (carbGrams * 4);

    return {
      protein: { grams: proteinGrams, calories: proteinCalories, percentage: Math.round((proteinCalories / normalizedCalories) * 100) },
      carbs: { grams: carbGrams, calories: carbGrams * 4, percentage: Math.round(((carbGrams * 4) / normalizedCalories) * 100) },
      fat: { grams: fatGrams, calories: fatCalories, percentage: Math.round((fatCalories / normalizedCalories) * 100) },
      preset,
      ratio: {
        protein: Math.round((proteinCalories / normalizedCalories) * 100),
        carbs: Math.round(((carbGrams * 4) / normalizedCalories) * 100),
        fat: Math.round((fatCalories / normalizedCalories) * 100)
      }
    };
  }

  // Generate 7-day meal plan with variations
  generateWeeklyMealPlan(targetCalories, macros, startOffset = 0) {
    const weeklyPlan = [];
    
    for (let day = 1; day <= 7; day++) {
      const rotationOffset = (startOffset + day - 1) % 4;
      const dayPlan = {
        day: `Day ${day}`,
        meals: {
          breakfast: this.selectMeal('breakfast', {
            targetCalories: targetCalories * 0.25,
            targetProtein: macros.protein.grams * 0.28,
            targetCarbs: macros.carbs.grams * 0.28,
            targetFat: macros.fat.grams * 0.22,
            rotationOffset
          }),
          lunch: this.selectMeal('lunch', {
            targetCalories: targetCalories * 0.32,
            targetProtein: macros.protein.grams * 0.32,
            targetCarbs: macros.carbs.grams * 0.30,
            targetFat: macros.fat.grams * 0.28,
            rotationOffset
          }),
          dinner: this.selectMeal('dinner', {
            targetCalories: targetCalories * 0.31,
            targetProtein: macros.protein.grams * 0.30,
            targetCarbs: macros.carbs.grams * 0.27,
            targetFat: macros.fat.grams * 0.30,
            rotationOffset
          }),
          snacks: this.selectMeal('snacks', {
            targetCalories: targetCalories * 0.12,
            targetProtein: macros.protein.grams * 0.10,
            targetCarbs: macros.carbs.grams * 0.15,
            targetFat: macros.fat.grams * 0.20,
            rotationOffset
          })
        },
        totalCalories: 0,
        macros: { protein: 0, carbs: 0, fat: 0 }
      };
      
      // Calculate totals
      Object.values(dayPlan.meals).forEach(meal => {
        dayPlan.totalCalories += meal.calories;
        dayPlan.macros.protein += meal.protein;
        dayPlan.macros.carbs += meal.carbs;
        dayPlan.macros.fat += meal.fat;
      });
      
      weeklyPlan.push(dayPlan);
    }
    
    return weeklyPlan;
  }

  generateTwelveWeekNutritionPlan(dexaResults, goals, baseDietPlan) {
    const goal = goals.objective || baseDietPlan.nutritionFocus.recommendation || 'maintain';
    const baseCalories = baseDietPlan.targetCalories;
    const weightKg = parseFloat(dexaResults.weight);
    const leanMassKg = parseFloat(baseDietPlan.calorieProfile.leanMassKg);
    const weeks = [];

    for (let week = 1; week <= 12; week++) {
      let calorieAdjustment = 0;
      let weeklyFocus = 'Keep intake steady and build consistency';
      let checkIn = 'Review body weight, waist, energy, and training performance at the end of the week.';

      if (goal === 'lose_fat') {
        calorieAdjustment = week >= 5 && week <= 8 ? -50 : week >= 9 ? -100 : 0;
        weeklyFocus = week <= 4
          ? 'Lock in consistency and hunger control'
          : week <= 8
            ? 'Protect protein while tightening calories slightly'
            : 'Maintain fat-loss pace without sacrificing recovery';
        checkIn = week <= 4
          ? 'If weight is not trending down, tighten portions slightly next week.'
          : 'If energy drops hard or recovery slips, hold calories steady for one week.';
      } else if (goal === 'gain_muscle') {
        calorieAdjustment = week >= 5 && week <= 8 ? 75 : week >= 9 ? 125 : 0;
        weeklyFocus = week <= 4
          ? 'Build appetite rhythm and training consistency'
          : week <= 8
            ? 'Push performance and progressive overload'
            : 'Support higher output with a slightly larger surplus';
        checkIn = week <= 4
          ? 'If scale weight is flat and strength is not improving, add a little more food.'
          : 'If body fat is climbing too quickly, pull calories back by 50-100.';
      } else if (goal === 'recomp') {
        calorieAdjustment = week >= 9 ? -50 : 0;
        weeklyFocus = week <= 4
          ? 'Stay close to maintenance and improve food quality'
          : week <= 8
            ? 'Push protein and training quality'
            : 'Tighten calories slightly only if body fat is not improving';
        checkIn = 'Use mirror, waist, and gym performance together instead of scale weight alone.';
      }

      const targetCalories = Math.max(1200, Math.round(baseCalories + calorieAdjustment));
      const macros = this.calculateMacros(targetCalories, {
        goals,
        bodyFatPercent: parseFloat(dexaResults.bodyFatPercent),
        leanMassKg,
        weightKg,
        preset: baseDietPlan.macroPreset,
        customRatio: goals.customMacroRatio || null
      });

      weeks.push({
        week,
        targetCalories,
        macros,
        hydrationMl: Math.round(weightKg * 35),
        weeklyFocus,
        checkIn,
        mealTiming: goal === 'gain_muscle'
          ? 'Front-load carbs around training and keep protein evenly spread across 4 meals.'
          : goal === 'lose_fat'
            ? 'Keep protein high at every meal and place most carbs around training or your most active hours.'
            : 'Distribute protein evenly and keep carbs higher near training sessions.',
        weeklyPlan: this.generateWeeklyMealPlan(targetCalories, macros, week - 1)
      });
    }

    return weeks;
  }

  // Select appropriate meal based on calorie target
  selectMeal(mealType, target) {
    const availableMeals = this.mealDatabase[mealType];
    const scoredMeals = availableMeals.map((meal, index) => {
      const calorieScore = Math.abs(meal.calories - target.targetCalories);
      const proteinScore = Math.abs(meal.protein - target.targetProtein) * 8;
      const carbScore = Math.abs(meal.carbs - target.targetCarbs) * 3;
      const fatScore = Math.abs(meal.fat - target.targetFat) * 4;
      return {
        meal,
        index,
        score: calorieScore + proteinScore + carbScore + fatScore
      };
    }).sort((a, b) => a.score - b.score);

    const selected = scoredMeals[target.rotationOffset % Math.min(2, scoredMeals.length) || 0]?.meal || scoredMeals[0].meal;
    const servingMultiplier = Math.min(1.35, Math.max(0.85, target.targetCalories / selected.calories));

    return {
      ...selected,
      servingMultiplier: Math.round(servingMultiplier * 100) / 100,
      calories: Math.round(selected.calories * servingMultiplier),
      protein: Math.round(selected.protein * servingMultiplier),
      carbs: Math.round(selected.carbs * servingMultiplier),
      fat: Math.round(selected.fat * servingMultiplier),
      portionDetails: this.scaleIngredientPortions(selected, servingMultiplier)
    };
  }

  scaleIngredientPortions(meal, multiplier) {
    if (!meal.portions || !Array.isArray(meal.portions)) {
      return meal.ingredients.map((ingredient) => ({ ingredient, quantity: '' }));
    }

    return meal.ingredients.map((ingredient, index) => ({
      ingredient,
      quantity: this.scaleQuantityString(meal.portions[index] || '', multiplier)
    }));
  }

  scaleQuantityString(quantity, multiplier) {
    if (!quantity) return '';
    const match = quantity.match(/^(\d+(?:\.\d+)?)(?:\/(\d+))?\s*(.*)$/);
    if (!match) return quantity;

    let baseValue = parseFloat(match[1]);
    if (match[2]) {
      baseValue = baseValue / parseFloat(match[2]);
    }

    const unit = match[3] || '';
    const scaled = baseValue * multiplier;
    const rounded = scaled >= 10 ? Math.round(scaled) : Math.round(scaled * 10) / 10;
    return `${rounded} ${unit}`.trim();
  }

  // Generate detailed personalized workout plan based on DEXA results
  generateWorkoutPlan(dexaResults, goals = {}) {
    const { bodyFatPercent, muscleMass, visceralFat, healthScore, bodyComposition, muscleQuality } = dexaResults;
    const bfPercent = parseFloat(bodyFatPercent);
    const visFat = parseFloat(visceralFat);
    const muscle = parseFloat(muscleMass);
    
    // Determine fitness level and workout intensity
    const fitnessLevel = this.determineFitnessLevel(bfPercent, healthScore, visFat);
    const workoutIntensity = this.determineWorkoutIntensity(bfPercent, visFat, goals);
    
    // Generate comprehensive weekly plan
    const weeklyPlan = this.generateDetailedWeeklyPlan(dexaResults, fitnessLevel, workoutIntensity, goals);
    
    // Generate progression plan
    const progressionPlan = this.generateProgressionPlan(fitnessLevel, bfPercent, visFat);
    
    // Generate specific recommendations
    const recommendations = this.generateDetailedWorkoutRecommendations(dexaResults, goals);
    
    return {
      weeklyPlan,
      progressionPlan,
      recommendations,
      fitnessLevel,
      workoutIntensity,
      expectedResults: this.generateExpectedResults(dexaResults, goals)
    };
  }
  
  // Determine fitness level based on DEXA results
  determineFitnessLevel(bodyFatPercent, healthScore, visceralFat) {
    if (bodyFatPercent > 30 || healthScore < 50 || visceralFat > 12) {
      return 'beginner'; // Start with beginner if obese or poor health
    } else if (bodyFatPercent > 25 || healthScore < 65 || visceralFat > 8) {
      return 'beginner-intermediate';
    } else if (bodyFatPercent > 20 || healthScore < 75) {
      return 'intermediate';
    } else {
      return 'advanced';
    }
  }
  
  // Determine workout intensity
  determineWorkoutIntensity(bodyFatPercent, visceralFat, goals) {
    if (bodyFatPercent > 30 || visceralFat > 12) {
      return 'moderate'; // Start moderate for obese individuals
    } else if (goals.objective === 'lose_fat' && bodyFatPercent > 20) {
      return 'high'; // High intensity for fat loss
    } else if (goals.objective === 'gain_muscle') {
      return 'moderate-high';
    } else {
      return 'moderate';
    }
  }
  
  // Generate detailed weekly workout plan - Highly customized based on DEXA results
  generateDetailedWeeklyPlan(dexaResults, fitnessLevel, intensity, goals) {
    const { bodyFatPercent, visceralFat, muscleMass, healthScore, bodyComposition } = dexaResults;
    const bfPercent = parseFloat(bodyFatPercent);
    const visFat = parseFloat(visceralFat);
    const muscle = parseFloat(muscleMass);
    const health = parseFloat(healthScore);
    
    // Determine workout focus based on comprehensive analysis
    const focusFatLoss = bfPercent > 20 || visFat > 8;
    const focusCore = visFat > 8 || bfPercent > 25;
    const isObese = bfPercent > 30;
    const isOverweight = bfPercent > 25 && bfPercent <= 30;
    const needsMuscleBuilding = muscle < 35 || (muscle / parseFloat(dexaResults.weight || 70)) < 0.4;
    const isBeginner = fitnessLevel === 'beginner' || health < 60;
    
    const plan = {};
    
    // Customize workout structure based on body composition
    if (isObese || isBeginner) {
      // For obese/beginner: More cardio, lighter weights, more recovery
      plan.monday = this.generateUpperBodyWorkout(fitnessLevel, 'moderate', true, focusCore, isObese, needsMuscleBuilding);
      plan.tuesday = this.generateCardioWorkout(fitnessLevel, 'moderate', true, focusCore, false, isObese);
      plan.wednesday = this.generateLowerBodyWorkout(fitnessLevel, 'moderate', true, focusCore, isObese, needsMuscleBuilding);
      plan.thursday = this.generateRecoveryWorkout(fitnessLevel, focusCore, isObese);
      plan.friday = this.generateUpperBodyWorkout(fitnessLevel, 'moderate', true, focusCore, isObese, needsMuscleBuilding);
      plan.saturday = this.generateCardioWorkout(fitnessLevel, 'moderate', true, focusCore, true, isObese);
    } else if (isOverweight || focusFatLoss) {
      // For overweight/fat loss focus: Balanced cardio and strength
      plan.monday = this.generateUpperBodyWorkout(fitnessLevel, intensity, true, focusCore, false, needsMuscleBuilding);
      plan.tuesday = this.generateCardioWorkout(fitnessLevel, intensity === 'high' ? 'high' : 'moderate', true, focusCore, false, false);
      plan.wednesday = this.generateLowerBodyWorkout(fitnessLevel, intensity, true, focusCore, false, needsMuscleBuilding);
      plan.thursday = this.generateRecoveryWorkout(fitnessLevel, focusCore, false);
      plan.friday = this.generateFullBodyWorkout(fitnessLevel, intensity, true, focusCore, false, needsMuscleBuilding);
      plan.saturday = this.generateCardioWorkout(fitnessLevel, intensity === 'high' ? 'high' : 'moderate', true, focusCore, true, false);
    } else if (needsMuscleBuilding) {
      // For muscle building: More strength, less cardio
      plan.monday = this.generateUpperBodyWorkout(fitnessLevel, 'moderate-high', false, focusCore, false, true);
      plan.tuesday = this.generateCardioWorkout(fitnessLevel, 'moderate', false, focusCore, false, false);
      plan.wednesday = this.generateLowerBodyWorkout(fitnessLevel, 'moderate-high', false, focusCore, false, true);
      plan.thursday = this.generateRecoveryWorkout(fitnessLevel, focusCore, false);
      plan.friday = this.generateUpperBodyWorkout(fitnessLevel, 'moderate-high', false, focusCore, false, true);
      plan.saturday = this.generateLowerBodyWorkout(fitnessLevel, 'moderate-high', false, focusCore, false, true);
    } else {
      // For fit individuals: Balanced advanced program
      plan.monday = this.generateUpperBodyWorkout(fitnessLevel, intensity, false, focusCore, false, false);
      plan.tuesday = this.generateCardioWorkout(fitnessLevel, intensity, false, focusCore, false, false);
      plan.wednesday = this.generateLowerBodyWorkout(fitnessLevel, intensity, false, focusCore, false, false);
      plan.thursday = this.generateRecoveryWorkout(fitnessLevel, focusCore, false);
      plan.friday = this.generateFullBodyWorkout(fitnessLevel, intensity, false, focusCore, false, false);
      plan.saturday = this.generateCardioWorkout(fitnessLevel, intensity, false, focusCore, true, false);
    }
    
    // Sunday - Rest Day (customized based on fitness level)
    plan.sunday = {
      type: 'Rest Day',
      description: isObese || isBeginner 
        ? 'Complete rest or very light activity - your body needs recovery'
        : 'Active recovery - light walking, stretching, or complete rest',
      duration: isObese || isBeginner ? '20-30 min' : '30-60 min',
      activities: isObese || isBeginner ? [
        { name: 'Gentle Stretching', duration: '10-15 min', intensity: 'Very Low' },
        { name: 'Light Walking (if feeling up to it)', duration: '10-15 min', intensity: 'Very Low' }
      ] : [
        { name: 'Light Walking', duration: '20-30 min', intensity: 'Low' },
        { name: 'Stretching', duration: '15-20 min', intensity: 'Low' },
        { name: 'Foam Rolling', duration: '10-15 min', intensity: 'Low' }
      ]
    };
    
    return plan;
  }
  
  // Generate upper body workout - Highly customized
  generateUpperBodyWorkout(level, intensity, fatLoss, coreFocus, isObese, needsMuscle) {
    // Adjust sets/reps based on multiple factors
    let sets, reps, rest;
    
    if (isObese) {
      sets = 2; // Lower volume for obese individuals
      reps = '8-10';
      rest = '90-120 sec';
    } else if (level === 'beginner') {
      sets = fatLoss ? 3 : 3;
      reps = fatLoss ? '12-15' : '10-12';
      rest = fatLoss ? '60-90 sec' : '90 sec';
    } else if (level === 'beginner-intermediate') {
      sets = fatLoss ? 3 : 4;
      reps = fatLoss ? '12-15' : '10-12';
      rest = fatLoss ? '60-90 sec' : '90-120 sec';
    } else if (level === 'intermediate') {
      sets = fatLoss ? 4 : 4;
      reps = fatLoss ? '10-12' : '8-10';
      rest = fatLoss ? '60-90 sec' : '90-120 sec';
    } else {
      sets = fatLoss ? 4 : 5;
      reps = fatLoss ? '10-12' : '6-8';
      rest = fatLoss ? '60-90 sec' : '90-120 sec';
    }
    
    // Select exercises based on fitness level and goals
    const exercises = [];
    
    // Chest exercises - vary by level
    if (isObese || level === 'beginner') {
      exercises.push({
        name: 'Wall Push-ups',
        sets: sets,
        reps: '10-15',
        rest: '60 sec',
        targetMuscles: ['Chest', 'Shoulders', 'Triceps'],
        notes: 'Start against wall, progress to incline, then floor',
        progression: 'Move to lower incline, then floor push-ups'
      });
    } else if (level === 'beginner-intermediate') {
      exercises.push({
        name: 'Incline Push-ups',
        sets: sets,
        reps: reps,
        rest: rest,
        targetMuscles: ['Chest', 'Shoulders', 'Triceps'],
        notes: 'Use bench or elevated surface, focus on form',
        progression: 'Lower the incline or move to floor push-ups'
      });
    } else {
      exercises.push({
        name: needsMuscle ? 'Bench Press' : 'Push-ups',
        sets: sets,
        reps: reps,
        rest: rest,
        targetMuscles: ['Chest', 'Shoulders', 'Triceps'],
        notes: needsMuscle ? 'Use barbell or dumbbells, focus on progressive overload' : 'Full range of motion',
        progression: needsMuscle ? 'Increase weight weekly' : 'Add weight vest or elevate feet'
      });
    }
    
    // Back exercises - vary by level
    if (isObese || level === 'beginner') {
      exercises.push({
        name: 'Resistance Band Rows',
        sets: sets,
        reps: '12-15',
        rest: '60 sec',
        targetMuscles: ['Back', 'Biceps'],
        notes: 'Sit or stand, pull band to chest, control the movement',
        progression: 'Use stronger band or add resistance'
      });
    } else if (level === 'beginner-intermediate') {
      exercises.push({
        name: 'Assisted Pull-ups or Lat Pulldowns',
        sets: sets - 1,
        reps: '8-10',
        rest: rest,
        targetMuscles: ['Back', 'Biceps'],
        notes: 'Use resistance bands or assisted machine',
        progression: 'Reduce assistance or increase weight'
      });
    } else {
      exercises.push({
        name: needsMuscle ? 'Pull-ups' : 'Bent-over Rows',
        sets: sets - 1,
        reps: level === 'advanced' ? '6-8' : '8-10',
        rest: rest,
        targetMuscles: ['Back', 'Biceps'],
        notes: needsMuscle ? 'Full pull-ups, add weight when ready' : 'Use dumbbells or barbell',
        progression: needsMuscle ? 'Add weight belt' : 'Increase weight'
      });
    }
    
    // Shoulder exercises
    if (isObese) {
      exercises.push({
        name: 'Wall Angels',
        sets: 2,
        reps: '10-12',
        rest: '60 sec',
        targetMuscles: ['Shoulders', 'Upper Back'],
        notes: 'Stand against wall, slide arms up and down',
        progression: 'Move away from wall slightly'
      });
    } else {
      exercises.push({
        name: level === 'beginner' ? 'Dumbbell Shoulder Press (Light)' : 'Shoulder Press',
        sets: sets,
        reps: reps,
        rest: rest,
        targetMuscles: ['Shoulders', 'Triceps'],
        notes: level === 'beginner' ? 'Start with 2-5kg dumbbells' : 'Use dumbbells or barbell',
        progression: 'Increase weight gradually'
      });
    }
    
    // Additional back exercise for muscle building
    if (needsMuscle && !isObese) {
      exercises.push({
        name: 'Bent-over Rows',
        sets: sets,
        reps: reps,
        rest: rest,
        targetMuscles: ['Back', 'Biceps'],
        notes: 'Keep back straight, core engaged',
        progression: 'Increase weight'
      });
    }
    
    // Arm exercises - only if not obese (reduce volume for obese)
    if (!isObese) {
      exercises.push({
        name: 'Bicep Curls',
        sets: sets - 1,
        reps: fatLoss ? '12-15' : '10-12',
        rest: '60 sec',
        targetMuscles: ['Biceps'],
        notes: 'Control the movement, avoid swinging',
        progression: 'Increase weight or reps'
      });
      
      exercises.push({
        name: level === 'beginner' ? 'Tricep Dips (Assisted)' : 'Tricep Dips',
        sets: sets - 1,
        reps: '10-12',
        rest: '60 sec',
        targetMuscles: ['Triceps'],
        notes: level === 'beginner' ? 'Use bench with knees bent' : 'Full dips',
        progression: 'Straighten legs or add weight'
      });
    }
    
    // Add core exercises if focus is on core
    if (coreFocus) {
      exercises.push({
        name: 'Plank',
        sets: 3,
        reps: '30-60 sec',
        rest: '60 sec',
        targetMuscles: ['Core', 'Abs'],
        notes: 'Keep body straight, engage core',
        progression: 'Increase hold time'
      });
      exercises.push({
        name: 'Russian Twists',
        sets: 3,
        reps: '20-30 each side',
        rest: '60 sec',
        targetMuscles: ['Core', 'Obliques'],
        notes: 'Add weight for progression',
        progression: 'Add weight or increase reps'
      });
    }
    
    return {
      type: 'Upper Body Strength',
      duration: isObese ? '30-40 min' : '45-60 min',
      difficulty: level,
      focus: fatLoss ? 'Fat Loss + Strength' : needsMuscle ? 'Muscle Building' : 'Strength & Conditioning',
      exercises: exercises,
      warmup: {
        duration: isObese ? '5 min' : '5-10 min',
        activities: isObese 
          ? ['Arm circles', 'Shoulder rolls', 'Gentle stretching']
          : ['Arm circles', 'Shoulder rolls', 'Light cardio', 'Dynamic stretching']
      },
      cooldown: {
        duration: isObese ? '5 min' : '5-10 min',
        activities: ['Static stretching', 'Foam rolling upper body']
      }
    };
  }
  
  // Generate lower body workout - Highly customized
  generateLowerBodyWorkout(level, intensity, fatLoss, coreFocus, isObese, needsMuscle) {
    // Adjust sets/reps based on multiple factors
    let sets, reps, rest;
    
    if (isObese) {
      sets = 2; // Lower volume, protect joints
      reps = '8-10';
      rest = '90-120 sec';
    } else if (level === 'beginner') {
      sets = fatLoss ? 3 : 3;
      reps = fatLoss ? '15-20' : '10-12';
      rest = fatLoss ? '60-90 sec' : '90 sec';
    } else if (level === 'beginner-intermediate') {
      sets = fatLoss ? 3 : 4;
      reps = fatLoss ? '12-15' : '10-12';
      rest = fatLoss ? '60-90 sec' : '90-120 sec';
    } else if (level === 'intermediate') {
      sets = fatLoss ? 4 : 4;
      reps = fatLoss ? '12-15' : '8-10';
      rest = fatLoss ? '60-90 sec' : '90-120 sec';
    } else {
      sets = fatLoss ? 4 : 5;
      reps = fatLoss ? '10-12' : '6-8';
      rest = fatLoss ? '60-90 sec' : '90-120 sec';
    }
    
    const exercises = [];
    
    // Squat variations based on fitness level
    if (isObese) {
      exercises.push({
        name: 'Chair-Assisted Squats',
        sets: sets,
        reps: '8-10',
        rest: '90 sec',
        targetMuscles: ['Quads', 'Glutes', 'Core'],
        notes: 'Use chair for support, sit down and stand up slowly',
        progression: 'Reduce chair assistance, then remove chair'
      });
    } else if (level === 'beginner') {
      exercises.push({
        name: 'Bodyweight Squats',
        sets: sets,
        reps: reps,
        rest: rest,
        targetMuscles: ['Quads', 'Glutes', 'Core'],
        notes: 'Focus on form, depth, and control',
        progression: 'Add weight or increase depth'
      });
    } else {
      exercises.push({
        name: needsMuscle ? 'Barbell Squats' : 'Weighted Squats',
        sets: sets,
        reps: reps,
        rest: rest,
        targetMuscles: ['Quads', 'Glutes', 'Core'],
        notes: needsMuscle ? 'Use barbell, focus on progressive overload' : 'Use dumbbells or kettlebell',
        progression: 'Increase weight weekly'
      });
    }
    
    // Lunge variations
    if (isObese) {
      exercises.push({
        name: 'Static Lunges (Holding Support)',
        sets: sets - 1,
        reps: '6-8 each leg',
        rest: '90 sec',
        targetMuscles: ['Quads', 'Glutes'],
        notes: 'Hold onto wall or chair for balance, focus on form',
        progression: 'Reduce support, then remove'
      });
    } else {
      exercises.push({
        name: level === 'beginner' ? 'Reverse Lunges' : 'Walking Lunges',
        sets: sets,
        reps: `${reps} each leg`,
        rest: rest,
        targetMuscles: ['Quads', 'Glutes', 'Hamstrings'],
        notes: level === 'beginner' ? 'Easier on knees, step back' : 'Forward movement, add weight',
        progression: 'Add weight or increase distance'
      });
    }
    
    // Deadlift variations - skip for obese, protect lower back
    if (!isObese && level !== 'beginner') {
      exercises.push({
        name: needsMuscle ? 'Romanian Deadlifts' : 'Good Mornings',
        sets: sets - 1,
        reps: '10-12',
        rest: rest,
        targetMuscles: ['Hamstrings', 'Glutes', 'Lower Back'],
        notes: needsMuscle ? 'Use barbell or dumbbells' : 'Bodyweight, focus on hip hinge',
        progression: 'Increase weight or add resistance'
      });
    }
    
    // Leg press/step-ups
    if (isObese) {
      exercises.push({
        name: 'Seated Leg Extensions (Light)',
        sets: sets,
        reps: '10-12',
        rest: '90 sec',
        targetMuscles: ['Quads'],
        notes: 'Use machine, light weight, protect knees',
        progression: 'Gradually increase weight'
      });
    } else if (level === 'beginner') {
      exercises.push({
        name: 'Step-ups',
        sets: sets,
        reps: reps,
        rest: rest,
        targetMuscles: ['Quads', 'Glutes'],
        notes: 'Use step or bench, bodyweight only',
        progression: 'Increase step height or add weight'
      });
    } else {
      exercises.push({
        name: needsMuscle ? 'Leg Press' : 'Step-ups',
        sets: sets,
        reps: reps,
        rest: rest,
        targetMuscles: ['Quads', 'Glutes'],
        notes: needsMuscle ? 'Use machine, progressive overload' : 'Add weight to step-ups',
        progression: needsMuscle ? 'Increase weight' : 'Increase step height or weight'
      });
    }
    
    // Calf raises - only if not obese (reduce joint stress)
    if (!isObese) {
      exercises.push({
        name: 'Calf Raises',
        sets: 4,
        reps: '15-20',
        rest: '60 sec',
        targetMuscles: ['Calves'],
        notes: 'Full range of motion',
        progression: 'Add weight or single leg'
      });
    }
    
    // Glute bridges - important for everyone
    exercises.push({
      name: isObese ? 'Glute Bridges (Assisted)' : 'Glute Bridges',
      sets: sets,
      reps: isObese ? '10-12' : '15-20',
      rest: '60 sec',
      targetMuscles: ['Glutes', 'Hamstrings'],
      notes: isObese ? 'Focus on squeezing glutes, use support if needed' : 'Squeeze glutes at top',
      progression: isObese ? 'Remove assistance' : 'Single leg or add weight'
    });
    
    // Add core exercises if focus is on core
    if (coreFocus) {
      exercises.push({
        name: 'Dead Bug',
        sets: 3,
        reps: '12-15 each side',
        rest: '60 sec',
        targetMuscles: ['Core', 'Deep Abs'],
        notes: 'Keep lower back pressed to floor',
        progression: 'Add resistance band'
      });
      exercises.push({
        name: 'Bicycle Crunches',
        sets: 3,
        reps: '20-30 each side',
        rest: '60 sec',
        targetMuscles: ['Core', 'Obliques'],
        notes: 'Slow and controlled',
        progression: 'Increase reps or add weight'
      });
    }
    
    return {
      type: 'Lower Body Strength',
      duration: isObese ? '30-40 min' : '45-60 min',
      difficulty: level,
      focus: fatLoss ? 'Fat Loss + Strength' : needsMuscle ? 'Muscle Building' : 'Strength & Conditioning',
      exercises: exercises,
      warmup: {
        duration: isObese ? '5 min' : '5-10 min',
        activities: isObese
          ? ['Leg swings', 'Hip circles', 'Gentle stretching']
          : ['Leg swings', 'Hip circles', 'Light cardio', 'Dynamic stretching']
      },
      cooldown: {
        duration: isObese ? '5 min' : '5-10 min',
        activities: ['Static stretching', 'Foam rolling legs']
      }
    };
  }
  
  // Generate cardio workout - Highly customized
  generateCardioWorkout(level, intensity, fatLoss, coreFocus, isWeekend = false, isObese = false) {
    // Adjust duration based on fitness level and body composition
    let duration;
    if (isObese) {
      duration = '20-30 min'; // Shorter for obese individuals
    } else if (level === 'beginner') {
      duration = fatLoss ? '30-40 min' : '25-35 min';
    } else if (level === 'beginner-intermediate') {
      duration = fatLoss ? '35-45 min' : '30-40 min';
    } else {
      duration = fatLoss ? '45-60 min' : '30-45 min';
    }
    
    let exercises = [];
    
    // For obese individuals - low impact only
    if (isObese) {
      exercises = [
        {
          name: 'Warm-up',
          duration: '5 min',
          intensity: 'Low',
          notes: 'Gentle walking, arm circles'
        },
        {
          name: 'Brisk Walking',
          duration: '15-20 min',
          intensity: 'Low-Moderate',
          targetMuscles: ['Cardiovascular', 'Legs'],
          notes: 'Comfortable pace, can hold conversation, protect joints',
          progression: 'Gradually increase pace or duration by 2-3 min weekly'
        },
        {
          name: 'Seated Marching',
          duration: '5 min',
          intensity: 'Low',
          targetMuscles: ['Cardiovascular', 'Legs'],
          notes: 'Sit in chair, lift knees alternately, low impact'
        },
        {
          name: 'Cool-down',
          duration: '5 min',
          intensity: 'Low',
          notes: 'Gentle walking and stretching'
        }
      ];
    } else if (fatLoss && intensity === 'high' && level !== 'beginner') {
      // HIIT for fat loss (not for beginners or obese)
      exercises = [
        {
          name: 'Warm-up',
          duration: '5 min',
          intensity: 'Low',
          notes: 'Light jogging or fast walking'
        },
        {
          name: level === 'intermediate' ? 'Sprint Intervals' : 'High-Intensity Intervals',
          sets: level === 'intermediate' ? 6 : 8,
          reps: level === 'intermediate' ? '30 sec sprint, 90 sec walk' : '30 sec sprint, 60 sec walk',
          rest: '60 sec',
          intensity: 'High',
          targetMuscles: ['Cardiovascular', 'Full Body'],
          notes: 'Maximum effort during sprints'
        },
        {
          name: level === 'intermediate' ? 'Squat Jumps' : 'Jump Squats',
          sets: 3,
          reps: level === 'intermediate' ? '30 sec' : '45 sec',
          rest: '30 sec',
          intensity: 'High',
          targetMuscles: ['Legs', 'Cardiovascular'],
          notes: 'Explosive movement, land softly'
        },
        {
          name: 'Burpees',
          sets: level === 'intermediate' ? 3 : 4,
          reps: level === 'intermediate' ? '20 sec' : '30 sec',
          rest: '30 sec',
          intensity: 'High',
          targetMuscles: ['Full Body', 'Cardiovascular'],
          notes: 'Full range of motion'
        },
        {
          name: 'Mountain Climbers',
          sets: 3,
          reps: '30-45 sec',
          rest: '15 sec',
          intensity: 'High',
          targetMuscles: ['Core', 'Cardiovascular'],
          notes: 'Fast pace, keep core engaged'
        },
        {
          name: 'Cool-down',
          duration: '5 min',
          intensity: 'Low',
          notes: 'Light walking and stretching'
        }
      ];
    } else {
      // Steady state cardio - varies by level
      const cardioType = isObese ? 'Brisk Walking' :
                        level === 'beginner' ? 'Brisk Walking' :
                        level === 'beginner-intermediate' ? 'Walking/Jogging Intervals' :
                        fatLoss ? 'Running/Jogging' : 'Moderate Cardio';
      
      exercises = [
        {
          name: 'Warm-up',
          duration: '5 min',
          intensity: 'Low',
          notes: 'Light walking or jogging'
        },
        {
          name: cardioType,
          duration: duration,
          intensity: level === 'beginner' ? 'Low-Moderate' : 'Moderate',
          targetMuscles: ['Cardiovascular', 'Legs'],
          notes: level === 'beginner' 
            ? 'Comfortable pace, can hold conversation'
            : level === 'beginner-intermediate'
            ? 'Alternate 2 min walk, 1 min jog'
            : 'Maintain steady pace, can talk but slightly breathless',
          progression: 'Increase pace or duration gradually'
        },
        {
          name: 'Cool-down',
          duration: '5 min',
          intensity: 'Low',
          notes: 'Light walking and stretching'
        }
      ];
    }
    
    // Add core exercises if focus is on core
    if (coreFocus) {
      exercises.push({
        name: 'Core Finisher',
        exercises: [
          { name: 'Plank Hold', duration: '30-60 sec', sets: 3 },
          { name: 'Side Plank', duration: '30-45 sec each side', sets: 2 },
          { name: 'Flutter Kicks', reps: '20-30', sets: 3 }
        ],
        notes: 'Focus on form and breathing'
      });
    }
    
    return {
      type: fatLoss && intensity === 'high' ? 'HIIT Cardio' : 'Steady State Cardio',
      duration: duration,
      difficulty: level,
      focus: fatLoss ? 'Fat Loss' : 'Cardiovascular Health',
      exercises: exercises,
      warmup: {
        duration: '5 min',
        activities: ['Light movement', 'Dynamic stretching']
      },
      cooldown: {
        duration: '5-10 min',
        activities: ['Light walking', 'Static stretching']
      }
    };
  }
  
  // Generate full body workout - Highly customized
  generateFullBodyWorkout(level, intensity, fatLoss, coreFocus, isObese, needsMuscle) {
    let sets, reps;
    
    if (isObese) {
      sets = 2;
      reps = '8-10';
    } else if (level === 'beginner') {
      sets = 3;
      reps = fatLoss ? '12-15' : '10-12';
    } else {
      sets = 4;
      reps = fatLoss ? '12-15' : '10-12';
    }
    
    const exercises = [];
    
    // Squat variation
    if (isObese) {
      exercises.push({
        name: 'Chair-Assisted Squats',
        sets: sets,
        reps: '8-10',
        rest: '90 sec',
        targetMuscles: ['Full Body'],
        notes: 'Use chair for support, compound movement'
      });
    } else {
      exercises.push({
        name: level === 'beginner' ? 'Bodyweight Squats' : 'Squat to Press',
        sets: sets,
        reps: reps,
        rest: '60-90 sec',
        targetMuscles: ['Full Body'],
        notes: level === 'beginner' ? 'Focus on form' : 'Compound movement, great for calorie burn'
      });
    }
    
    // Deadlift - skip for obese
    if (!isObese && level !== 'beginner') {
      exercises.push({
        name: 'Deadlift',
        sets: sets,
        reps: '8-10',
        rest: '90 sec',
        targetMuscles: ['Back', 'Glutes', 'Hamstrings'],
        notes: 'Focus on form, start light'
      });
    }
    
    // Push-ups
    exercises.push({
      name: isObese ? 'Wall Push-ups' : level === 'beginner' ? 'Incline Push-ups' : 'Push-ups',
      sets: sets,
      reps: reps,
      rest: '60 sec',
      targetMuscles: ['Chest', 'Shoulders', 'Triceps'],
      notes: isObese ? 'Against wall, progress to incline' : level === 'beginner' ? 'Incline push-ups' : 'Full push-ups'
    });
    
    // Rows
    exercises.push({
      name: isObese ? 'Resistance Band Rows' : 'Bent-over Rows',
      sets: sets,
      reps: reps,
      rest: '60-90 sec',
      targetMuscles: ['Back', 'Biceps'],
      notes: isObese ? 'Seated or standing with band' : 'Keep core engaged'
    });
    
    // Lunges
    if (!isObese) {
      exercises.push({
        name: level === 'beginner' ? 'Reverse Lunges' : 'Lunges',
        sets: sets,
        reps: `${reps} each leg`,
        rest: '60 sec',
        targetMuscles: ['Legs', 'Glutes'],
        notes: 'Alternate legs'
      });
    }
    
    // Core
    exercises.push({
      name: isObese ? 'Seated Knee Lifts' : 'Plank',
      sets: 3,
      reps: isObese ? '10-12' : '30-60 sec',
      rest: '60 sec',
      targetMuscles: ['Core'],
      notes: isObese ? 'Sit, lift knees, engage core' : 'Keep body straight'
    });
    
    // High intensity finisher - only if not obese and fat loss focus
    if (fatLoss && !isObese && level !== 'beginner') {
      exercises.push({
        name: level === 'beginner-intermediate' ? 'Modified Burpees' : 'Burpees',
        sets: 3,
        reps: level === 'beginner-intermediate' ? '8-10' : '10-15',
        rest: '60 sec',
        targetMuscles: ['Full Body', 'Cardiovascular'],
        notes: level === 'beginner-intermediate' ? 'Step back instead of jump' : 'High intensity finisher'
      });
    }
    
    return {
      type: 'Full Body Circuit',
      duration: isObese ? '30-40 min' : '40-50 min',
      difficulty: level,
      focus: fatLoss ? 'Fat Loss Circuit' : 'Full Body Strength',
      exercises: exercises,
      warmup: {
        duration: '5-10 min',
        activities: ['Full body dynamic warm-up', 'Light cardio']
      },
      cooldown: {
        duration: '5-10 min',
        activities: ['Full body stretching', 'Foam rolling']
      }
    };
  }
  
  // Generate recovery workout - Customized
  generateRecoveryWorkout(level, coreFocus, isObese) {
    const exercises = [];
    
    if (isObese || level === 'beginner') {
      exercises.push({
        name: 'Gentle Stretching',
        duration: '15-20 min',
        intensity: 'Very Low',
        notes: 'Seated or standing stretches, hold 20-30 seconds each'
      });
      exercises.push({
        name: 'Breathing Exercises',
        duration: '10 min',
        intensity: 'Very Low',
        notes: 'Deep breathing, focus on relaxation'
      });
    } else {
      exercises.push({
        name: 'Yoga Flow',
        duration: '20-30 min',
        intensity: 'Low',
        notes: 'Gentle movements, focus on flexibility'
      });
      exercises.push({
        name: 'Static Stretching',
        duration: '15-20 min',
        intensity: 'Low',
        notes: 'Hold each stretch 30-60 seconds'
      });
      exercises.push({
        name: 'Foam Rolling',
        duration: '15-20 min',
        intensity: 'Low',
        notes: 'Focus on tight areas'
      });
    }
    
    if (coreFocus) {
      exercises.push({
        name: 'Gentle Core Work',
        exercises: isObese ? [
          { name: 'Seated Cat-Cow', reps: '10-15', sets: 2 },
          { name: 'Seated Twists', reps: '10 each side', sets: 2 }
        ] : [
          { name: 'Cat-Cow Stretch', reps: '10-15', sets: 2 },
          { name: 'Child\'s Pose', duration: '1-2 min' },
          { name: 'Gentle Twists', reps: '10 each side', sets: 2 }
        ],
        notes: 'Low intensity core mobility'
      });
    }
    
    return {
      type: 'Active Recovery',
      duration: isObese ? '20-30 min' : '30-45 min',
      difficulty: 'Low',
      focus: 'Recovery & Flexibility',
      exercises: exercises
    };
  }
  
  // Generate progression plan
  generateProgressionPlan(fitnessLevel, bodyFatPercent, visceralFat) {
    const weeks = [];
    const isObese = bodyFatPercent > 30;
    const hasHighVisceralFat = visceralFat > 8;
    
    for (let week = 1; week <= 12; week++) {
      const weekPlan = {
        week: week,
        focus: '',
        changes: [],
        intensity: '',
        notes: ''
      };
      
      if (week <= 2) {
        weekPlan.focus = 'Building Foundation';
        weekPlan.intensity = 'Low-Moderate';
        weekPlan.changes = [
          'Focus on learning proper form',
          'Start with lighter weights or bodyweight',
          'Build consistency with 3-4 workouts per week'
        ];
        weekPlan.notes = 'This phase is about establishing routine and proper movement patterns.';
      } else if (week <= 4) {
        weekPlan.focus = 'Increasing Volume';
        weekPlan.intensity = 'Moderate';
        weekPlan.changes = [
          'Increase sets by 1 per exercise',
          'Add 2-5 reps to each set',
          'Reduce rest time by 10-15 seconds'
        ];
        weekPlan.notes = 'Gradually increase workout volume while maintaining good form.';
      } else if (week <= 8) {
        weekPlan.focus = 'Progressive Overload';
        weekPlan.intensity = 'Moderate-High';
        weekPlan.changes = [
          'Increase weight by 2.5-5kg (or 5-10%)',
          'Add 1-2 more exercises per session',
          'Increase cardio duration by 5-10 minutes'
        ];
        weekPlan.notes = 'Focus on getting stronger and building endurance.';
      } else {
        weekPlan.focus = 'Peak Performance';
        weekPlan.intensity = 'High';
        weekPlan.changes = [
          'Continue progressive overload',
          'Add advanced variations',
          'Increase workout frequency if ready'
        ];
        weekPlan.notes = 'Push your limits while maintaining proper form and recovery.';
      }
      
      if (isObese && week <= 4) {
        weekPlan.changes.push('Focus on low-impact exercises to protect joints');
        weekPlan.changes.push('Gradually increase cardio duration');
      }
      
      if (hasHighVisceralFat) {
        weekPlan.changes.push('Emphasize core and abdominal exercises');
        weekPlan.changes.push('Include daily core work (5-10 minutes)');
      }
      
      weeks.push(weekPlan);
    }
    
    return {
      duration: '12 weeks',
      phases: weeks,
      generalGuidelines: [
        'Track your progress weekly',
        'Adjust based on how your body responds',
        'Prioritize recovery and sleep (7-9 hours)',
        'Stay consistent - consistency beats intensity',
        'Listen to your body and rest when needed'
      ]
    };
  }
  
  // Generate expected results
  generateExpectedResults(dexaResults, goals) {
    const { bodyFatPercent, visceralFat, healthScore } = dexaResults;
    const bfPercent = parseFloat(bodyFatPercent);
    const visFat = parseFloat(visceralFat);
    
    const results = {
      '4 weeks': [],
      '8 weeks': [],
      '12 weeks': []
    };
    
    if (goals.objective === 'lose_fat' || bfPercent > 20) {
      results['4 weeks'].push('2-4% body fat reduction');
      results['4 weeks'].push('2-5kg weight loss');
      results['8 weeks'].push('4-8% body fat reduction');
      results['8 weeks'].push('5-10kg weight loss');
      results['12 weeks'].push('8-12% body fat reduction');
      results['12 weeks'].push('10-15kg weight loss');
      
      if (visFat > 8) {
        results['4 weeks'].push('1-2% visceral fat reduction');
        results['8 weeks'].push('2-4% visceral fat reduction');
        results['12 weeks'].push('4-6% visceral fat reduction');
      }
    }
    
    if (goals.objective === 'gain_muscle' || parseFloat(dexaResults.muscleMass) < 40) {
      results['4 weeks'].push('1-2kg muscle gain');
      results['8 weeks'].push('2-4kg muscle gain');
      results['12 weeks'].push('4-6kg muscle gain');
    }
    
    results['4 weeks'].push('Improved cardiovascular fitness');
    results['4 weeks'].push('Increased energy levels');
    results['8 weeks'].push('Better sleep quality');
    results['8 weeks'].push('Improved mood and mental health');
    results['12 weeks'].push('Significant health score improvement');
    results['12 weeks'].push('Reduced risk of chronic diseases');
    
    return results;
  }
  
  // Generate detailed workout recommendations
  generateDetailedWorkoutRecommendations(dexaResults, goals) {
    const { bodyFatPercent, visceralFat, muscleMass, healthScore, bodyComposition } = dexaResults;
    const bfPercent = parseFloat(bodyFatPercent);
    const visFat = parseFloat(visceralFat);
    const muscle = parseFloat(muscleMass);
    const recommendations = [];
    
    // Body fat specific recommendations
    if (bfPercent > 30) {
      recommendations.push({
        priority: 'High',
        category: 'Fat Loss',
        recommendation: 'Focus on 4-5 cardio sessions per week (30-60 min each) combined with strength training 3-4 times per week',
        reason: 'High body fat percentage requires consistent cardio and strength training for optimal fat loss'
      });
      recommendations.push({
        priority: 'High',
        category: 'Exercise Selection',
        recommendation: 'Prioritize compound movements (squats, deadlifts, push-ups) as they burn more calories',
        reason: 'Compound exercises engage multiple muscle groups, increasing calorie expenditure'
      });
    } else if (bfPercent > 25) {
      recommendations.push({
        priority: 'High',
        category: 'Fat Loss',
        recommendation: 'Include 3-4 HIIT sessions per week (20-30 min) plus strength training 3 times per week',
        reason: 'HIIT is highly effective for fat loss while preserving muscle mass'
      });
    } else if (bfPercent > 20) {
      recommendations.push({
        priority: 'Medium',
        category: 'Fat Loss',
        recommendation: 'Maintain 3-4 cardio sessions per week with progressive strength training',
        reason: 'Continue fat loss while building muscle for better body composition'
      });
    }
    
    // Visceral fat specific recommendations
    if (visFat > 12) {
      recommendations.push({
        priority: 'Critical',
        category: 'Core & Visceral Fat',
        recommendation: 'Include daily core exercises (10-15 min) focusing on deep core activation',
        reason: 'High visceral fat requires targeted core work and overall fat loss'
      });
      recommendations.push({
        priority: 'High',
        category: 'Cardio',
        recommendation: 'Prioritize moderate-intensity steady-state cardio (45-60 min) 4-5 times per week',
        reason: 'Steady-state cardio is effective for reducing visceral fat'
      });
    } else if (visFat > 8) {
      recommendations.push({
        priority: 'High',
        category: 'Core & Visceral Fat',
        recommendation: 'Add core exercises to every workout (5-10 min)',
        reason: 'Targeted core work helps reduce visceral fat accumulation'
      });
    }
    
    // Muscle mass recommendations
    if (muscle < 35) {
      recommendations.push({
        priority: 'High',
        category: 'Muscle Building',
        recommendation: 'Focus on progressive overload - increase weight or reps weekly',
        reason: 'Low muscle mass requires consistent strength training with progressive overload'
      });
      recommendations.push({
        priority: 'High',
        category: 'Recovery',
        recommendation: 'Allow 48-72 hours between training the same muscle groups',
        reason: 'Adequate recovery is essential for muscle growth'
      });
    }
    
    // Health score recommendations
    if (healthScore < 60) {
      recommendations.push({
        priority: 'Critical',
        category: 'Overall Health',
        recommendation: 'Start with low-impact exercises and gradually increase intensity',
        reason: 'Low health score indicates need for gradual progression to avoid injury'
      });
      recommendations.push({
        priority: 'High',
        category: 'Consistency',
        recommendation: 'Focus on consistency over intensity - 3-4 workouts per week minimum',
        reason: 'Building consistent habits is more important than intense sporadic workouts'
      });
    }
    
    // General recommendations
    recommendations.push({
      priority: 'Medium',
      category: 'Recovery',
      recommendation: 'Ensure 7-9 hours of sleep per night for optimal recovery and fat loss',
      reason: 'Sleep is crucial for muscle recovery and hormone regulation'
    });
    
    recommendations.push({
      priority: 'Medium',
      category: 'Hydration',
      recommendation: 'Drink 3-4 liters of water daily, especially around workouts',
      reason: 'Proper hydration supports performance and recovery'
    });
    
    recommendations.push({
      priority: 'Low',
      category: 'Tracking',
      recommendation: 'Track your workouts, weight, and measurements weekly to monitor progress',
      reason: 'Tracking helps identify what\'s working and allows for program adjustments'
    });
    
    return recommendations;
  }


  // Generate diet recommendations
  generateDietRecommendations(dexaResults, goals, calorieProfile, macros) {
    const { bodyFatPercent, muscleMass, visceralFat } = dexaResults;
    const recommendations = [];
    recommendations.push(`Target calories are ${calorieProfile.targetCalories}, which is a ${calorieProfile.directionLabel} from your estimated maintenance of ${calorieProfile.maintenanceCalories}.`);
    recommendations.push(`Protein is set high at ${macros.protein.grams}g to support lean mass retention and recovery.`);

    if (goals.objective === 'lose_fat' && calorieProfile.deltaFromMaintenance >= 0) {
      recommendations.push('For fat loss, calories should usually sit below maintenance, not above it. A 200-500 calorie deficit is the more typical target.');
    }
    
    if (parseFloat(bodyFatPercent) > 20) {
      recommendations.push("Keep most carbs around training and bias the rest of the day toward lean protein, vegetables, and high-fiber carbs.");
      recommendations.push("Aim for 25-35g fiber daily from vegetables, fruit, legumes, and whole grains.");
    }
    
    if (parseFloat(visceralFat) > 5) {
      recommendations.push("Limit processed foods, alcohol, and added sugars to help reduce visceral fat.");
      recommendations.push("Increase omega-3 intake from fish, flax, chia, or walnuts 3-4 times per week.");
    }
    
    if (parseFloat(muscleMass) < 40) {
      recommendations.push("Split protein across 4 meals so each meal contributes meaningfully to muscle retention.");
      recommendations.push("Place one protein-rich meal or shake within 1-2 hours after training.");
    }
    
    recommendations.push("Stay hydrated with roughly 35-40 ml of water per kg body weight daily.");
    recommendations.push("Use the macro controls above if you want a different carb-fat emphasis while keeping calories tighter.");
    
    return recommendations;
  }

  // Generate workout recommendations
  generateWorkoutRecommendations(dexaResults, goals) {
    const { bodyFatPercent, muscleMass, overallHealth } = dexaResults;
    const recommendations = [];
    
    if (parseFloat(bodyFatPercent) > 20) {
      recommendations.push("Include 3-4 cardio sessions per week");
      recommendations.push("Focus on compound movements for maximum calorie burn");
    }
    
    if (parseFloat(muscleMass) < 40) {
      recommendations.push("Prioritize progressive overload in strength training");
      recommendations.push("Allow 48-72 hours between muscle group training");
    }
    
    recommendations.push("Warm up for 5-10 minutes before each workout");
    recommendations.push("Cool down and stretch after each session");
    recommendations.push("Listen to your body and adjust intensity as needed");
    
    return recommendations;
  }

  // Load user progress from localStorage
  loadUserProgress() {
    const saved = localStorage.getItem('fitlife_progress');
    return saved ? JSON.parse(saved) : {
      scans: [],
      currentWeight: null,
      currentBodyFat: null,
      goals: {}
    };
  }

  // Save user progress to localStorage
  saveUserProgress() {
    localStorage.setItem('fitlife_progress', JSON.stringify(this.userProgress));
  }

  // Add new scan to progress tracking
  addScan(scanData) {
    const scan = {
      id: Date.now(),
      date: new Date().toISOString(),
      ...scanData
    };
    
    this.userProgress.scans.push(scan);
    this.userProgress.currentWeight = scanData.weight;
    this.userProgress.currentBodyFat = scanData.bodyFatPercent;
    
    this.saveUserProgress();
    return scan;
  }

  // Compare scans for progress analysis
  compareScans(scan1, scan2) {
    const weightChange = scan2.weight - scan1.weight;
    const bodyFatChange = scan2.bodyFatPercent - scan1.bodyFatPercent;
    const muscleChange = scan2.muscleMass - scan1.muscleMass;
    
    const progress = {
      weightChange,
      bodyFatChange,
      muscleChange,
      timeSpan: Math.round((new Date(scan2.date) - new Date(scan1.date)) / (1000 * 60 * 60 * 24)), // days
      recommendations: []
    };
    
    // Generate recommendations based on progress
    if (bodyFatChange > 2) {
      progress.recommendations.push("Consider reducing caloric intake by 200-300 calories");
      progress.recommendations.push("Increase cardio frequency to 4-5 times per week");
    } else if (bodyFatChange < -1) {
      progress.recommendations.push("Great progress! Maintain current approach");
    }
    
    if (muscleChange < -1) {
      progress.recommendations.push("Increase protein intake and strength training frequency");
    } else if (muscleChange > 1) {
      progress.recommendations.push("Excellent muscle gain! Continue current program");
    }
    
    return progress;
  }
}

// Initialize global fitness planner
window.fitnessPlanner = new FitnessPlanner();

console.log('=== DIET & WORKOUT PLANNER LOADED ===');
