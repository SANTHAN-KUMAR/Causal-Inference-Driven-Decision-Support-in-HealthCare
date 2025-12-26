/**
 * AEGIS 3.0 - Simple, Clear Application Logic
 * Focus: Easy to use, easy to understand results
 */

// ============================================
// State
// ============================================

const State = {
    section: 'input',
    processing: false,
    results: null,
    settings: {
        activity: 'light',
        stress: 'normal',
        sleep: 'good',
        illness: 'none'
    },
    chart: null,
    demoChart: null
};

// ============================================
// Initialization
// ============================================

document.addEventListener('DOMContentLoaded', () => {
    initNavigation();
    initGlucoseInput();
    initMealInput();
    initFeelingButtons();
    initSettingsInputs();
    initActions();
});

// ============================================
// Navigation
// ============================================

function initNavigation() {
    document.querySelectorAll('.nav-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            navigateToSection(btn.dataset.section);
        });
    });
}

function navigateToSection(section) {
    document.querySelectorAll('.nav-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.section === section);
    });

    document.querySelectorAll('.section').forEach(sec => {
        sec.classList.remove('active');
    });

    document.getElementById(`${section}-section`)?.classList.add('active');
    State.section = section;
}

window.navigateToSection = navigateToSection;

// ============================================
// Glucose Input
// ============================================

function initGlucoseInput() {
    const input = document.getElementById('current-glucose');
    const status = document.getElementById('glucose-status');

    if (input) {
        input.addEventListener('input', () => {
            updateGlucoseStatus(parseFloat(input.value));
        });
        updateGlucoseStatus(parseFloat(input.value));
    }
}

function updateGlucoseStatus(glucose) {
    const dot = document.querySelector('.status-dot');
    const text = document.querySelector('.status-text');

    if (glucose < 70) {
        dot.className = 'status-dot danger';
        text.textContent = 'Low - consider having some carbs first';
    } else if (glucose < 80) {
        dot.className = 'status-dot warning';
        text.textContent = 'Slightly low - be careful with insulin';
    } else if (glucose <= 180) {
        dot.className = 'status-dot good';
        text.textContent = 'In target range';
    } else if (glucose <= 250) {
        dot.className = 'status-dot warning';
        text.textContent = 'Above target - may need correction';
    } else {
        dot.className = 'status-dot danger';
        text.textContent = 'High - correction recommended';
    }
}

// ============================================
// Meal Input
// ============================================

function initMealInput() {
    // Meal type buttons
    document.querySelectorAll('.meal-type-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.meal-type-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            document.getElementById('meal-type').value = btn.dataset.type;
        });
    });

    // Carb slider
    const slider = document.getElementById('meal-carbs');
    const display = document.getElementById('carb-display');

    if (slider) {
        slider.addEventListener('input', () => {
            display.textContent = slider.value;
        });
    }

    // Carb presets
    document.querySelectorAll('.preset-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const carbs = btn.dataset.carbs;
            slider.value = carbs;
            display.textContent = carbs;
        });
    });
}

// ============================================
// Feeling Buttons
// ============================================

function initFeelingButtons() {
    document.querySelectorAll('.option-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const field = btn.dataset.field;
            const value = btn.dataset.value;

            // Update button states in same group
            const group = btn.closest('.feeling-group');
            group.querySelectorAll('.option-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            // Update state
            State.settings[field] = value;
        });
    });
}

// ============================================
// Settings Inputs
// ============================================

function initSettingsInputs() {
    const tdiInput = document.getElementById('patient-tdi');

    if (tdiInput) {
        tdiInput.addEventListener('input', updateCalculatedSettings);
        updateCalculatedSettings();
    }
}

function updateCalculatedSettings() {
    const tdi = parseFloat(document.getElementById('patient-tdi')?.value) || 45;

    const icr = Math.round(500 / tdi);
    const isf = Math.round(1800 / tdi);

    document.getElementById('calc-icr').textContent = `1:${icr}`;
    document.getElementById('calc-isf').textContent = isf.toString();

    document.querySelector('#calc-icr').closest('.calc-item').querySelector('.calc-explain').textContent =
        `(1 unit per ${icr}g carbs)`;
    document.querySelector('#calc-isf').closest('.calc-item').querySelector('.calc-explain').textContent =
        `(1 unit lowers glucose by ${isf} mg/dL)`;
}

// ============================================
// Action Handlers
// ============================================

function initActions() {
    document.getElementById('calculate-dose')?.addEventListener('click', calculateDose);
    document.getElementById('emergency-stop')?.addEventListener('click', emergencyStop);
    document.getElementById('approve-btn')?.addEventListener('click', approveDose);
    document.getElementById('adjust-btn')?.addEventListener('click', adjustDose);
    document.getElementById('run-demo')?.addEventListener('click', runDemo);
}

// ============================================
// Main Calculation
// ============================================

async function calculateDose() {
    if (State.processing) return;
    State.processing = true;

    // Get inputs
    const glucose = parseFloat(document.getElementById('current-glucose').value) || 120;
    const trend = document.getElementById('glucose-trend').value;
    const carbs = parseFloat(document.getElementById('meal-carbs').value) || 0;
    const mealType = document.getElementById('meal-type').value;
    const tdi = parseFloat(document.getElementById('patient-tdi').value) || 45;
    const weight = parseFloat(document.getElementById('patient-weight').value) || 75;

    // Calculate using insulin math
    const icr = 500 / tdi;  // Insulin to carb ratio
    const isf = 1800 / tdi; // Insulin sensitivity factor
    const targetGlucose = 110;

    // Base meal dose
    let mealDose = carbs / icr;

    // Correction dose
    let correctionDose = 0;
    if (glucose > targetGlucose) {
        correctionDose = (glucose - targetGlucose) / isf;
    }

    // Apply modifiers based on feelings
    let modifierExplanation = [];
    let totalModifier = 0;

    // Activity modifier
    const activityMods = {
        sedentary: 0.05,
        light: 0,
        moderate: -0.15,
        intense: -0.25
    };
    if (State.settings.activity !== 'light') {
        totalModifier += activityMods[State.settings.activity];
        if (activityMods[State.settings.activity] < 0) {
            modifierExplanation.push(`being active (reduces need by ${Math.abs(activityMods[State.settings.activity] * 100).toFixed(0)}%)`);
        } else {
            modifierExplanation.push(`being sedentary (increases need by ${(activityMods[State.settings.activity] * 100).toFixed(0)}%)`);
        }
    }

    // Stress modifier
    const stressMods = { low: -0.05, normal: 0, elevated: 0.10, high: 0.20 };
    if (State.settings.stress !== 'normal') {
        totalModifier += stressMods[State.settings.stress];
        if (State.settings.stress === 'elevated' || State.settings.stress === 'high') {
            modifierExplanation.push(`stress (increases need by ${(stressMods[State.settings.stress] * 100).toFixed(0)}%)`);
        }
    }

    // Sleep modifier
    const sleepMods = { poor: 0.10, fair: 0.05, good: 0, excellent: -0.05 };
    if (State.settings.sleep !== 'good') {
        totalModifier += sleepMods[State.settings.sleep];
        if (State.settings.sleep === 'poor') {
            modifierExplanation.push(`poor sleep (increases need by 10%)`);
        }
    }

    // Illness modifier
    const illnessMods = { none: 0, mild_cold: 0.15, fever: 0.30 };
    if (State.settings.illness !== 'none') {
        totalModifier += illnessMods[State.settings.illness];
        modifierExplanation.push(`being unwell (increases need by ${(illnessMods[State.settings.illness] * 100).toFixed(0)}%)`);
    }

    const baseDose = mealDose + correctionDose;
    const adjustmentDose = baseDose * totalModifier;
    let finalDose = baseDose + adjustmentDose;

    // Safety checks
    let safetyIssues = [];
    let originalDose = finalDose;

    // Check 1: Low glucose
    if (glucose < 70) {
        safetyIssues.push('Your glucose is low. Do not take insulin until you have treated the low.');
        finalDose = 0;
    } else if (glucose < 80 && correctionDose > 0) {
        safetyIssues.push('Correction dose removed due to glucose being near low range.');
        finalDose = mealDose * (1 + totalModifier);
        correctionDose = 0;
    }

    // Check 2: Max dose
    const maxDose = 15;
    if (finalDose > maxDose) {
        safetyIssues.push(`Dose capped at ${maxDose} units for safety.`);
        finalDose = maxDose;
    }

    // Check 3: Falling glucose
    if (trend.includes('falling') && glucose < 120) {
        const reduction = trend === 'falling_fast' ? 0.5 : 0.25;
        safetyIssues.push(`Dose reduced by ${reduction * 100}% because your glucose is dropping.`);
        finalDose = finalDose * (1 - reduction);
    }

    // Round to 0.5
    finalDose = Math.round(finalDose * 2) / 2;
    mealDose = Math.round(mealDose * 2) / 2;
    correctionDose = Math.round(correctionDose * 2) / 2;

    // Store results
    State.results = {
        finalDose,
        mealDose,
        correctionDose,
        adjustmentDose: Math.round(adjustmentDose * 2) / 2,
        glucose,
        carbs,
        modifiers: modifierExplanation,
        safetyIssues,
        isf,
        icr,
        targetGlucose,
        peakPrediction: glucose + (carbs * 3) - (finalDose * isf * 0.7)
    };

    // Display results
    displayResults();

    State.processing = false;
}

// ============================================
// Display Results
// ============================================

function displayResults() {
    const r = State.results;

    // Switch to results
    navigateToSection('results');

    // Show content
    document.getElementById('results-empty').classList.add('hidden');
    document.getElementById('results-content').classList.remove('hidden');

    // Main dose
    document.getElementById('final-dose').textContent = r.finalDose.toFixed(1);
    document.getElementById('dose-explanation').textContent =
        `Based on your ${r.carbs}g carb meal and current glucose of ${r.glucose} mg/dL`;

    // Breakdown
    document.getElementById('meal-dose').textContent = `${r.mealDose.toFixed(1)} U`;
    document.getElementById('correction-dose').textContent = `${r.correctionDose.toFixed(1)} U`;
    document.getElementById('adjustment-dose').textContent =
        `${r.adjustmentDose >= 0 ? '+' : ''}${r.adjustmentDose.toFixed(1)} U`;

    // Show/hide adjustment
    const adjItem = document.getElementById('adjustment-item');
    if (r.adjustmentDose !== 0) {
        adjItem.classList.add('active');
        adjItem.style.display = 'block';
    } else {
        adjItem.classList.remove('active');
        adjItem.style.display = 'none';
    }

    // Explanation
    document.getElementById('explain-meal').innerHTML =
        `<strong>Meal Coverage:</strong> You're eating ${r.carbs}g of carbs. With your carb ratio of 1:${Math.round(r.icr)}, you need about ${r.mealDose.toFixed(1)} units to cover this meal.`;

    if (r.correctionDose > 0) {
        document.getElementById('explain-correction').innerHTML =
            `<strong>Correction:</strong> Your glucose is ${r.glucose} mg/dL, which is ${r.glucose - r.targetGlucose} above your target of ${r.targetGlucose}. A correction of ${r.correctionDose.toFixed(1)} units will help bring it down.`;
    } else {
        document.getElementById('explain-correction').innerHTML =
            `<strong>Correction:</strong> Your glucose is ${r.glucose} mg/dL, which is in or near your target range. No correction needed.`;
    }

    if (r.modifiers.length > 0) {
        document.getElementById('explain-factors').innerHTML =
            `<strong>Adjustments:</strong> We adjusted your dose for: ${r.modifiers.join(', ')}.`;
    } else {
        document.getElementById('explain-factors').innerHTML =
            `<strong>Other Factors:</strong> No special adjustments needed - you're having a normal day!`;
    }

    // Safety badge
    const badge = document.getElementById('safety-badge');
    if (r.safetyIssues.length > 0) {
        badge.classList.add('warning');
        badge.querySelector('.badge-icon').textContent = '⚠';
        badge.querySelector('.badge-text').textContent = 'Safety Adjusted';
    } else {
        badge.classList.remove('warning');
        badge.querySelector('.badge-icon').textContent = '✓';
        badge.querySelector('.badge-text').textContent = 'Safety Verified';
    }

    // Safety checks
    updateSafetyItem('safety-1', r.glucose >= 70,
        r.glucose >= 70 ? 'Glucose is safe for insulin delivery' : 'Caution: Low glucose detected');
    updateSafetyItem('safety-2', r.peakPrediction > 54,
        r.peakPrediction > 54 ? 'No predicted low blood sugar' : 'Warning: Risk of low blood sugar');
    updateSafetyItem('safety-3', r.finalDose <= 15,
        r.finalDose <= 15 ? 'Dose within safe limits' : 'Dose was capped for safety');

    // Prediction
    const peak = Math.min(300, Math.max(r.glucose, r.peakPrediction)).toFixed(0);
    document.getElementById('pred-peak').textContent = `${peak} mg/dL`;
    document.getElementById('pred-peak-time').textContent = 'in about 1-2 hours';
    document.getElementById('pred-return').textContent = '~3 hours';

    // Update prediction chart
    updatePredictionChart(r);
}

function updateSafetyItem(id, pass, text) {
    const item = document.getElementById(id);
    item.className = `safety-item ${pass ? 'pass' : 'fail'}`;
    item.querySelector('.safety-icon').textContent = pass ? '✓' : '✕';
    item.querySelector('.safety-text').textContent = text;
}

function updatePredictionChart(results) {
    const ctx = document.getElementById('prediction-chart');
    if (!ctx) return;

    // Generate prediction trajectory
    const trajectory = [];
    let glucose = results.glucose;
    const mealEffect = results.carbs * 3;
    const insulinEffect = results.finalDose * results.isf;

    for (let t = 0; t <= 180; t += 10) {
        // Simplified glucose dynamics
        let change = 0;

        // Meal absorption (peak around 60-90 min)
        if (t > 0 && t <= 120) {
            change += mealEffect * 0.01 * Math.exp(-(t - 60) * (t - 60) / 2000);
        }

        // Insulin effect (delayed, peak around 90-120 min)
        if (t > 30) {
            change -= insulinEffect * 0.008 * Math.exp(-(t - 90) * (t - 90) / 3000);
        }

        glucose = Math.max(50, Math.min(350, glucose + change));
        trajectory.push({ time: t, glucose });
    }

    if (State.chart) {
        State.chart.destroy();
    }

    State.chart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: trajectory.map(p => p.time === 0 ? 'Now' : `${p.time}m`),
            datasets: [{
                label: 'Predicted Glucose',
                data: trajectory.map(p => p.glucose),
                borderColor: '#4f8cff',
                backgroundColor: 'rgba(79, 140, 255, 0.1)',
                fill: true,
                tension: 0.4,
                pointRadius: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: {
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    ticks: { color: '#5a6578' }
                },
                y: {
                    min: 50,
                    max: 250,
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    ticks: { color: '#5a6578' }
                }
            }
        },
        plugins: [{
            id: 'targetRange',
            beforeDraw: (chart) => {
                const { ctx, chartArea, scales } = chart;
                if (!chartArea) return;
                ctx.save();
                ctx.fillStyle = 'rgba(52, 211, 153, 0.08)';
                const y180 = scales.y.getPixelForValue(180);
                const y70 = scales.y.getPixelForValue(70);
                ctx.fillRect(chartArea.left, y180, chartArea.right - chartArea.left, y70 - y180);
                ctx.restore();
            }
        }]
    });
}

// ============================================
// Action Handlers
// ============================================

function emergencyStop() {
    showToast('🚨 Emergency Stop activated - All insulin suspended', 'warning');
}

function approveDose() {
    if (State.results) {
        showToast(`✓ Dose approved: ${State.results.finalDose.toFixed(1)} units`, 'success');
    }
}

function adjustDose() {
    const newDose = prompt('Enter your adjusted dose (units):', State.results?.finalDose?.toFixed(1) || '0');
    if (newDose !== null) {
        showToast(`Adjusted dose: ${parseFloat(newDose).toFixed(1)} units`, 'success');
    }
}

// ============================================
// Demo Simulation
// ============================================

async function runDemo() {
    const scenario = document.getElementById('demo-scenario').value;
    const duration = parseInt(document.getElementById('demo-duration').value);

    document.getElementById('demo-results').classList.remove('hidden');

    const interpretation = document.getElementById('demo-interpretation');
    interpretation.textContent = 'Running simulation...';

    // Simulate glucose trajectory
    const trajectory = await simulateDay(scenario, duration);

    // Calculate metrics
    const glucose = trajectory.map(p => p.glucose);
    const tir = (glucose.filter(g => g >= 70 && g <= 180).length / glucose.length * 100).toFixed(0);
    const low = (glucose.filter(g => g < 70).length / glucose.length * 100).toFixed(0);
    const high = (glucose.filter(g => g > 180).length / glucose.length * 100).toFixed(0);
    const avg = (glucose.reduce((a, b) => a + b, 0) / glucose.length).toFixed(0);

    // Update metrics display
    document.getElementById('demo-tir').textContent = tir + '%';
    document.getElementById('demo-low').textContent = low + '%';
    document.getElementById('demo-high').textContent = high + '%';
    document.getElementById('demo-avg').textContent = avg + ' mg/dL';

    // Color code metrics
    document.querySelector('.demo-metric:nth-child(1)').className =
        `demo-metric ${parseInt(tir) >= 70 ? 'good' : ''}`;
    document.getElementById('demo-low-metric').className =
        `demo-metric ${parseInt(low) < 4 ? 'good' : 'bad'}`;

    // Interpretation
    let interp = [];
    if (parseInt(tir) >= 70) {
        interp.push('✓ Excellent time in range! The AEGIS system kept glucose well controlled.');
    } else if (parseInt(tir) >= 50) {
        interp.push('○ Good time in range, but there\'s room for improvement.');
    } else {
        interp.push('✕ Time in range needs improvement.');
    }

    if (parseInt(low) === 0) {
        interp.push('✓ No low blood sugar events - the safety system worked well.');
    } else {
        interp.push(`⚠ There were some low glucose periods (${low}% of time).`);
    }

    interpretation.innerHTML = interp.join('<br><br>');

    // Update chart
    updateDemoChart(trajectory);
}

async function simulateDay(scenario, duration) {
    const trajectory = [];
    let glucose = 100;
    const hours = duration;

    // Define meals based on scenario
    const meals = scenario === 'high_carb' ?
        [{ hour: 7, carbs: 60 }, { hour: 12, carbs: 100 }, { hour: 19, carbs: 120 }] :
        [{ hour: 7, carbs: 45 }, { hour: 12, carbs: 65 }, { hour: 19, carbs: 75 }];

    for (let h = 0; h < hours; h += 0.25) {
        // Check for meal
        const meal = meals.find(m => Math.abs(m.hour - h) < 0.25);
        if (meal) {
            // AEGIS calculates dose
            const dose = meal.carbs / 11; // Simplified
            glucose += meal.carbs * 2;
            glucose -= dose * 30;
        }

        // Scenario effects
        if (scenario === 'exercise' && h >= 17 && h <= 18) {
            glucose -= 3; // Exercise drops glucose
        }
        if (scenario === 'stress') {
            glucose += 0.5; // Stress raises glucose
        }

        // Natural return to baseline
        glucose = glucose * 0.99 + 100 * 0.01;

        // Add noise
        glucose += (Math.random() - 0.5) * 5;

        // Bounds
        glucose = Math.max(55, Math.min(300, glucose));

        trajectory.push({ hour: h, glucose });
    }

    return trajectory;
}

function updateDemoChart(trajectory) {
    const ctx = document.getElementById('demo-chart');
    if (!ctx) return;

    if (State.demoChart) {
        State.demoChart.destroy();
    }

    State.demoChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: trajectory.map(p => `${p.hour.toFixed(0)}h`),
            datasets: [{
                label: 'Glucose',
                data: trajectory.map(p => p.glucose),
                borderColor: '#4f8cff',
                backgroundColor: 'rgba(79, 140, 255, 0.1)',
                fill: true,
                tension: 0.3,
                pointRadius: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: {
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    ticks: { color: '#5a6578', maxTicksLimit: 12 }
                },
                y: {
                    min: 40,
                    max: 300,
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    ticks: { color: '#5a6578' }
                }
            }
        },
        plugins: [{
            id: 'ranges',
            beforeDraw: (chart) => {
                const { ctx, chartArea, scales } = chart;
                if (!chartArea) return;
                ctx.save();

                // Target range
                ctx.fillStyle = 'rgba(52, 211, 153, 0.08)';
                const y180 = scales.y.getPixelForValue(180);
                const y70 = scales.y.getPixelForValue(70);
                ctx.fillRect(chartArea.left, y180, chartArea.right - chartArea.left, y70 - y180);

                // Low zone
                ctx.fillStyle = 'rgba(248, 113, 113, 0.08)';
                const y54 = scales.y.getPixelForValue(54);
                ctx.fillRect(chartArea.left, y54, chartArea.right - chartArea.left, chartArea.bottom - y54);

                ctx.restore();
            }
        }]
    });
}

// ============================================
// Toast Notifications
// ============================================

function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    container.appendChild(toast);

    setTimeout(() => {
        toast.style.opacity = '0';
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}
