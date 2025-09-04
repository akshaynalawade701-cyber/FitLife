if ('serviceWorker' in navigator) { navigator.serviceWorker.getRegistrations().then(rs => rs.forEach(r => r.unregister())); }
(function() {
  'use strict';

  // Utilities
  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
  const clamp = (n, min, max) => Math.max(min, Math.min(max, n));
  const fmt = (n, digits = 0) => Number.isFinite(n) ? n.toFixed(digits) : '-';
  const todayISO = () => new Date().toISOString().slice(0, 10);

  // Theme toggle
  const themeToggle = $('#theme-toggle');
  const storedTheme = localStorage.getItem('fitlife_theme');
  if (storedTheme) document.documentElement.dataset.theme = storedTheme;
  themeToggle?.addEventListener('click', () => {
    const next = document.documentElement.dataset.theme === 'light' ? 'dark' : 'light';
    document.documentElement.dataset.theme = next;
    localStorage.setItem('fitlife_theme', next);
  });

  // Mobile nav
  const mobileBtn = $('#mobile-menu-btn');
  const nav = $('#site-nav');
  mobileBtn?.addEventListener('click', () => {
    const open = nav.classList.toggle('open');
    mobileBtn.setAttribute('aria-expanded', String(open));
  });

  // Footer year
  const yearEl = $('#year');
  if (yearEl) yearEl.textContent = String(new Date().getFullYear());

  // --- Calculators ---
  function toggleUnitFields(prefix) {
    const metric = $(`#${prefix}-unit-metric`).checked;
    $(`#${prefix}-metric-fields`).classList.toggle('hidden', !metric);
    $(`#${prefix}-imperial-fields`).classList.toggle('hidden', metric);
  }

  // BMI
  function calcBMI() {
    const metric = $('#bmi-unit-metric').checked;
    let heightM, weightKg;
    if (metric) {
      const cm = parseFloat($('#bmi-height-cm').value);
      const kg = parseFloat($('#bmi-weight').value);
      if (!cm || !kg) return setBMIResult('Enter height and weight');
      heightM = cm / 100;
      weightKg = kg;
    } else {
      const ft = parseFloat($('#bmi-feet').value) || 0;
      const inch = parseFloat($('#bmi-inches').value) || 0;
      const lb = parseFloat($('#bmi-pounds').value);
      if ((!ft && !inch) || !lb) return setBMIResult('Enter height and weight');
      heightM = (((ft * 12) + inch) * 2.54) / 100;
      weightKg = lb * 0.45359237;
    }
    const bmi = weightKg / (heightM * heightM);
    let cat = 'Normal';
    if (bmi < 18.5) cat = 'Underweight';
    else if (bmi < 25) cat = 'Normal';
    else if (bmi < 30) cat = 'Overweight';
    else cat = 'Obese';
    setBMIResult(`BMI: ${fmt(bmi, 1)} (${cat})`);
  }
  function setBMIResult(text) { $('#bmi-result').textContent = text; }

  // BMR/TDEE
  function calcBMR() {
    const metric = $('#bmr-unit-metric').checked;
    const sex = $('#bmr-sex').value;
    const age = parseFloat($('#bmr-age').value);
    const activity = parseFloat($('#activity-level').value);
    if (!age) return setBmrResults('Enter your age');

    let heightCm, weightKg;
    if (metric) {
      heightCm = parseFloat($('#bmr-height-cm').value);
      weightKg = parseFloat($('#bmr-weight-kg').value);
      if (!heightCm || !weightKg) return setBmrResults('Enter height and weight');
    } else {
      const ft = parseFloat($('#bmr-feet').value) || 0;
      const inch = parseFloat($('#bmr-inches').value) || 0;
      const lb = parseFloat($('#bmr-pounds').value);
      if ((!ft && !inch) || !lb) return setBmrResults('Enter height and weight');
      heightCm = ((ft * 12) + inch) * 2.54;
      weightKg = lb * 0.45359237;
    }

    const bmr = (10 * weightKg) + (6.25 * heightCm) - (5 * age) + (sex === 'male' ? 5 : -161);
    const tdee = bmr * activity;
    setBmrResults(`BMR: ${fmt(bmr, 0)} kcal/day`, `TDEE: ${fmt(tdee, 0)} kcal/day`);
  }
  function setBmrResults(bmrText, tdeeText) {
    $('#bmr-result').textContent = bmrText || '';
    $('#tdee-result').textContent = tdeeText || '';
  }

  // Setup toggles and buttons
  ['bmi', 'bmr'].forEach(prefix => {
    $(`#${prefix}-unit-metric`)?.addEventListener('change', () => toggleUnitFields(prefix));
    $(`#${prefix}-unit-imperial`)?.addEventListener('change', () => toggleUnitFields(prefix));
    toggleUnitFields(prefix);
  });
  $('#bmi-calc-btn')?.addEventListener('click', calcBMI);
  $('#bmr-calc-btn')?.addEventListener('click', calcBMR);

  // --- Workout Builder ---
  const EXERCISES = {
    chest: {
      bodyweight: ['Push-up', 'Incline push-up', 'Decline push-up'],
      dumbbells: ['DB bench press', 'DB incline press', 'DB fly'],
      gym: ['Barbell bench press', 'Incline bench press', 'Cable fly']
    },
    back: {
      bodyweight: ['Inverted row', 'Superman', 'Prone Y-T-W'],
      dumbbells: ['DB row', 'DB pullover', 'DB deadlift'],
      gym: ['Lat pulldown', 'Seated row', 'Barbell row']
    },
    legs: {
      bodyweight: ['Bodyweight squat', 'Reverse lunge', 'Glute bridge'],
      dumbbells: ['Goblet squat', 'DB lunge', 'DB RDL'],
      gym: ['Back squat', 'Leg press', 'Romanian deadlift']
    },
    shoulders: {
      bodyweight: ['Pike push-up', 'Wall walk', 'Handstand hold'],
      dumbbells: ['DB shoulder press', 'Lateral raise', 'Rear delt raise'],
      gym: ['Overhead press', 'Cable lateral raise', 'Face pull']
    },
    arms: {
      bodyweight: ['Diamond push-up', 'Chin-up hold', 'Bench dip'],
      dumbbells: ['DB curl', 'DB hammer curl', 'DB triceps extension'],
      gym: ['Cable curl', 'EZ-bar curl', 'Cable pushdown']
    },
    core: {
      bodyweight: ['Plank', 'Dead bug', 'Hollow body hold'],
      dumbbells: ['DB side bend', 'Weighted sit-up', 'Russian twist'],
      gym: ['Cable crunch', 'Hanging knee raise', 'Ab rollout']
    },
    fullbody: {
      bodyweight: ['Burpee', 'Bear crawl', 'Lunge to knee drive'],
      dumbbells: ['DB thruster', 'DB snatch (alt)', 'DB clean'],
      gym: ['Barbell complex', 'Kettlebell swing', 'Row + push-up circuit']
    }
  };

  function makeSplit(days) {
    switch (days) {
      case 3: return [['push'], ['pull'], ['legs']];
      case 4: return [['upper'], ['lower'], ['upper'], ['lower']];
      case 5: return [['chest','triceps'], ['back','biceps'], ['legs'], ['shoulders','core'], ['fullbody']];
      case 6: return [['push'], ['pull'], ['legs'], ['push'], ['pull'], ['legs']];
      default: return [['fullbody']];
    }
  }

  function resolveGroups(tag) {
    if (tag === 'push') return ['chest', 'shoulders', 'arms'];
    if (tag === 'pull') return ['back', 'arms'];
    if (tag === 'upper') return ['chest', 'back', 'shoulders', 'arms'];
    if (tag === 'lower') return ['legs', 'core'];
    if (tag === 'chest') return ['chest'];
    if (tag === 'back') return ['back'];
    if (tag === 'legs') return ['legs'];
    if (tag === 'shoulders') return ['shoulders'];
    if (tag === 'arms') return ['arms'];
    if (tag === 'core') return ['core'];
    if (tag === 'fullbody') return ['fullbody'];
    return ['fullbody'];
  }

  function generatePlan({ goal, days, equipment }) {
    const split = makeSplit(days);
    const plan = [];
    const setsReps = goal === 'muscle' ? { sets: 4, reps: '8–12' }
                    : goal === 'fatloss' ? { sets: 3, reps: '12–15' }
                    : goal === 'endurance' ? { sets: 3, reps: '15–20' }
                    : { sets: 3, reps: '10–12' };

    split.forEach((dayTags, idx) => {
      const groups = dayTags.flatMap(resolveGroups);
      const uniqueGroups = [...new Set(groups)];
      const exercises = [];
      uniqueGroups.forEach(group => {
        const list = (EXERCISES[group]?.[equipment] || []).slice(0, 3);
        list.forEach(name => exercises.push({ name, group, sets: setsReps.sets, reps: setsReps.reps }));
      });
      plan.push({ day: `Day ${idx + 1}`, tags: dayTags.join(', '), exercises });
    });
    return plan;
  }

  function renderPlan(plan) {
    const container = $('#wb-plan-container');
    container.innerHTML = '';
    plan.forEach(d => {
      const el = document.createElement('div');
      el.className = 'plan-day';
      el.innerHTML = `<h4>${d.day} · <span class="muted">${d.tags}</span></h4>
        <table><thead><tr><th>Exercise</th><th>Group</th><th>Sets</th><th>Reps</th></tr></thead>
        <tbody>
        ${d.exercises.map(x => `<tr><td>${x.name}</td><td>${x.group}</td><td>${x.sets}</td><td>${x.reps}</td></tr>`).join('')}
        </tbody></table>`;
      container.appendChild(el);
    });
  }

  function savePlan(plan) {
    localStorage.setItem('fitlife_plan', JSON.stringify(plan));
  }
  function loadPlan() {
    try { return JSON.parse(localStorage.getItem('fitlife_plan') || 'null'); } catch { return null; }
  }

  $('#wb-generate-btn')?.addEventListener('click', () => {
    const goal = $('#wb-goal').value;
    const days = parseInt($('#wb-days').value, 10);
    const equipment = $('#wb-equipment').value;
    const plan = generatePlan({ goal, days, equipment });
    renderPlan(plan);
  });
  $('#wb-save-btn')?.addEventListener('click', () => {
    const nodes = $$('#wb-plan-container .plan-day');
    if (!nodes.length) return alert('Generate a plan first');
    // Rebuild from DOM for simplicity
    const plan = nodes.map((node, idx) => {
      const tags = node.querySelector('h4 .muted').textContent;
      const rows = Array.from(node.querySelectorAll('tbody tr'));
      const exercises = rows.map(r => ({
        name: r.children[0].textContent,
        group: r.children[1].textContent,
        sets: r.children[2].textContent,
        reps: r.children[3].textContent
      }));
      return { day: `Day ${idx + 1}`, tags, exercises };
    });
    savePlan(plan);
    alert('Plan saved locally');
  });
  $('#wb-print-btn')?.addEventListener('click', () => window.print());
  $('#wb-reset-btn')?.addEventListener('click', () => { $('#wb-plan-container').innerHTML = ''; });
  // Load saved plan if exists
  const saved = loadPlan();
  if (saved) renderPlan(saved);

  // --- Nutrition Tracker ---
  const STORAGE_KEY = 'fitlife_nutrition_v1';
  function loadNutrition() {
    try { return JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}'); } catch { return {}; }
  }
  function saveNutrition(data) {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
  }
  let nutrition = loadNutrition();

  const dateEl = $('#nt-date');
  if (dateEl) dateEl.value = todayISO();

  function getDayData(date) {
    if (!nutrition[date]) nutrition[date] = { goals: { calories: 0, protein: 0, carbs: 0, fat: 0 }, meals: [] };
    return nutrition[date];
  }

  function updateProgressBars(day) {
    const totals = day.meals.reduce((acc, m) => ({
      calories: acc.calories + (Number(m.calories) || 0),
      protein: acc.protein + (Number(m.protein) || 0),
      carbs: acc.carbs + (Number(m.carbs) || 0),
      fat: acc.fat + (Number(m.fat) || 0)
    }), { calories: 0, protein: 0, carbs: 0, fat: 0 });

    const calGoal = Math.max(0, Number(day.goals.calories) || 0);
    const pGoal = Math.max(0, Number(day.goals.protein) || 0);
    const cGoal = Math.max(0, Number(day.goals.carbs) || 0);
    const fGoal = Math.max(0, Number(day.goals.fat) || 0);

    $('#nt-total-calories').textContent = `${totals.calories} / ${calGoal}`;
    $('#nt-total-protein').textContent = `${totals.protein}g / ${pGoal}g`;
    $('#nt-total-carbs').textContent = `${totals.carbs}g / ${cGoal}g`;
    $('#nt-total-fat').textContent = `${totals.fat}g / ${fGoal}g`;

    $('#nt-calories-bar').style.width = `${clamp(calGoal ? (totals.calories / calGoal) * 100 : 0, 0, 100)}%`;
    $('#nt-protein-bar').style.width = `${clamp(pGoal ? (totals.protein / pGoal) * 100 : 0, 0, 100)}%`;
    $('#nt-carbs-bar').style.width = `${clamp(cGoal ? (totals.carbs / cGoal) * 100 : 0, 0, 100)}%`;
    $('#nt-fat-bar').style.width = `${clamp(fGoal ? (totals.fat / fGoal) * 100 : 0, 0, 100)}%`;
  }

  function renderMealsTable(day) {
    const tbody = $('#nt-meals-table-body');
    tbody.innerHTML = day.meals.map(m => `<tr>
      <td>${m.name}</td>
      <td>${m.calories}</td>
      <td>${m.protein}g</td>
      <td>${m.carbs}g</td>
      <td>${m.fat}g</td>
      <td><button data-id="${m.id}" class="btn btn-ghost del-meal">✕</button></td>
    </tr>`).join('');
    $$('.del-meal', tbody).forEach(btn => btn.addEventListener('click', () => {
      const id = btn.getAttribute('data-id');
      day.meals = day.meals.filter(x => x.id !== id);
      saveNutrition(nutrition);
      renderMealsTable(day);
      updateProgressBars(day);
    }));
  }

  function refreshNutrition() {
    const date = $('#nt-date')?.value || todayISO();
    const day = getDayData(date);
    $('#nt-calorie-goal').value = day.goals.calories || '';
    $('#nt-protein-goal').value = day.goals.protein || '';
    $('#nt-carbs-goal').value = day.goals.carbs || '';
    $('#nt-fat-goal').value = day.goals.fat || '';
    renderMealsTable(day);
    updateProgressBars(day);
  }

  $('#nt-date')?.addEventListener('change', refreshNutrition);
  $('#nt-save-goals-btn')?.addEventListener('click', () => {
    const date = $('#nt-date').value;
    const day = getDayData(date);
    day.goals.calories = Number($('#nt-calorie-goal').value) || 0;
    day.goals.protein = Number($('#nt-protein-goal').value) || 0;
    day.goals.carbs = Number($('#nt-carbs-goal').value) || 0;
    day.goals.fat = Number($('#nt-fat-goal').value) || 0;
    saveNutrition(nutrition);
    updateProgressBars(day);
  });

  $('#nt-add-meal-btn')?.addEventListener('click', () => {
    const date = $('#nt-date').value;
    const day = getDayData(date);
    const meal = {
      id: crypto.randomUUID(),
      name: $('#nt-meal-name').value || 'Meal',
      calories: Number($('#nt-calories').value) || 0,
      protein: Number($('#nt-protein').value) || 0,
      carbs: Number($('#nt-carbs').value) || 0,
      fat: Number($('#nt-fat').value) || 0
    };
    day.meals.push(meal);
    saveNutrition(nutrition);
    // Clear inputs
    ['nt-meal-name','nt-calories','nt-protein','nt-carbs','nt-fat'].forEach(id => { const el = $('#'+id); if (el) el.value = ''; });
    renderMealsTable(day);
    updateProgressBars(day);
  });

  $('#nt-reset-day-btn')?.addEventListener('click', () => {
    const date = $('#nt-date').value;
    nutrition[date] = { goals: { calories: 0, protein: 0, carbs: 0, fat: 0 }, meals: [] };
    saveNutrition(nutrition);
    refreshNutrition();
  });

  $('#nt-export-btn')?.addEventListener('click', () => {
    const blob = new Blob([JSON.stringify(nutrition, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = 'fitlife-nutrition.json';
    document.body.appendChild(a); a.click(); a.remove();
    URL.revokeObjectURL(url);
  });

  $('#nt-import-input')?.addEventListener('change', (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const data = JSON.parse(String(reader.result || '{}'));
        if (typeof data === 'object') {
          nutrition = { ...nutrition, ...data };
          saveNutrition(nutrition);
          refreshNutrition();
          alert('Imported');
        }
      } catch { alert('Invalid file'); }
    };
    reader.readAsText(file);
  });

  refreshNutrition();

  // --- Contact form ---
  $('#contact-form')?.addEventListener('submit', (e) => {
    e.preventDefault();
    const name = $('#cf-name').value.trim();
    const email = $('#cf-email').value.trim();
    const message = $('#cf-message').value.trim();
    const status = $('#cf-status');
    if (!name || !email || !message) { status.textContent = 'Please complete all fields.'; return; }
    status.textContent = 'Thanks! We will get back to you soon.';
    e.target.reset();
  });

  // --- Service worker registration ---
  if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
      // Use relative path so it works under GitHub Pages project subpath
      navigator.serviceWorker.register('sw.js', { scope: './' }).catch(() => {});
    });
  }
})();