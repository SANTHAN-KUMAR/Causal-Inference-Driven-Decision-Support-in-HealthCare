/**
 * AEGIS 3.0 - Application Controller
 * Handles UI interactions and connects to AEGIS Engine
 */

// ============================================
// Application State
// ============================================

const AppState = {
    currentSection: 'input',
    processing: false,
    lastResults: null,
    predictionChart: null,
    simChart: null
};

// ============================================
// Initialization
// ============================================

document.addEventListener('DOMContentLoaded', () => {
    initializeNavigation();
    initializeEventListeners();
    updateSystemStatus('ready');
});

// ============================================
// Navigation
// ============================================

function initializeNavigation() {
    document.querySelectorAll('.nav-link').forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            navigateToSection(link.dataset.section);
        });
    });
}

function navigateToSection(sectionId) {
    // Update nav links
    document.querySelectorAll('.nav-link').forEach(link => {
        link.classList.toggle('active', link.dataset.section === sectionId);
    });

    // Update sections
    document.querySelectorAll('.section').forEach(section => {
        section.classList.remove('active');
    });

    const targetSection = document.getElementById(`${sectionId}-section`);
    if (targetSection) {
        targetSection.classList.add('active');
    }

    // Update header
    const titles = {
        input: { title: 'Patient Data Input', subtitle: 'Enter patient data for real-time AEGIS processing' },
        processing: { title: 'Layer Processing', subtitle: 'View AEGIS 5-layer pipeline execution' },
        results: { title: 'Treatment Recommendation', subtitle: 'View computed recommendation and safety analysis' },
        simulation: { title: 'Batch Simulation', subtitle: 'Run multi-hour simulations with synthetic data' }
    };

    const header = titles[sectionId] || titles.input;
    document.getElementById('page-title').textContent = header.title;
    document.querySelector('.header-subtitle').textContent = header.subtitle;

    AppState.currentSection = sectionId;
}

// Make globally accessible
window.navigateToSection = navigateToSection;

// ============================================
// Event Listeners
// ============================================

function initializeEventListeners() {
    // Process button
    document.getElementById('process-data')?.addEventListener('click', processPatientData);

    // Reset button
    document.getElementById('reset-system')?.addEventListener('click', resetSystem);

    // Emergency stop
    document.getElementById('emergency-stop')?.addEventListener('click', emergencyStop);

    // Batch simulation
    document.getElementById('run-batch-sim')?.addEventListener('click', runBatchSimulation);

    // Recommendation actions
    document.getElementById('approve-recommendation')?.addEventListener('click', approveRecommendation);
    document.getElementById('modify-dose')?.addEventListener('click', modifyDose);
    document.getElementById('reject-recommendation')?.addEventListener('click', rejectRecommendation);
}

// ============================================
// Patient Data Processing
// ============================================

async function processPatientData() {
    if (AppState.processing) {
        showToast('Processing already in progress', 'warning');
        return;
    }

    // Collect input data
    const inputData = collectInputData();

    if (!validateInputData(inputData)) {
        return;
    }

    AppState.processing = true;
    updateSystemStatus('processing');

    // Switch to processing view
    navigateToSection('processing');
    resetLayerOutputs();

    try {
        // Run AEGIS pipeline
        const results = await AEGISPipeline.process(inputData, onLayerUpdate);

        AppState.lastResults = results;

        // Display results
        displayResults(results);

        updateSystemStatus('complete');
        showToast('Processing complete - view results', 'success');

        // Auto-navigate to results after short delay
        setTimeout(() => {
            navigateToSection('results');
        }, 1500);

    } catch (error) {
        console.error('Processing error:', error);
        showToast('Processing failed: ' + error.message, 'error');
        updateSystemStatus('error');
    } finally {
        AppState.processing = false;
    }
}

function collectInputData() {
    return {
        // Patient profile
        patientId: document.getElementById('patient-id')?.value || 'PAT-001',
        weight: parseFloat(document.getElementById('patient-weight')?.value) || 75,
        tdi: parseFloat(document.getElementById('patient-tdi')?.value) || 45,
        diabetesDuration: parseFloat(document.getElementById('patient-duration')?.value) || 5,

        // Glucose reading
        glucose: parseFloat(document.getElementById('current-glucose')?.value) || 120,
        trend: document.getElementById('glucose-trend')?.value || 'stable',

        // Meal
        mealCarbs: parseFloat(document.getElementById('meal-carbs')?.value) || 0,
        mealType: document.getElementById('meal-type')?.value || 'lunch',
        mealGI: document.getElementById('meal-gi')?.value || 'medium',
        mealFat: document.getElementById('meal-fat')?.value || 'medium',

        // Diary
        diary: document.getElementById('diary-entry')?.value || '',

        // Context
        activityLevel: document.getElementById('activity-level')?.value || 'light',
        stressLevel: document.getElementById('stress-level')?.value || 'normal',
        sleepQuality: document.getElementById('sleep-quality')?.value || 'good',
        illness: document.getElementById('illness')?.value || 'none'
    };
}

function validateInputData(data) {
    if (data.glucose < 40 || data.glucose > 400) {
        showToast('Glucose value must be between 40-400 mg/dL', 'error');
        return false;
    }
    if (data.mealCarbs < 0 || data.mealCarbs > 200) {
        showToast('Carbohydrates must be between 0-200 grams', 'error');
        return false;
    }
    return true;
}

// ============================================
// Layer Update Callbacks
// ============================================

function onLayerUpdate(layerNum, status, output) {
    const badge = document.getElementById(`l${layerNum}-badge`);
    const outputDiv = document.getElementById(`l${layerNum}-output`);
    const block = document.getElementById(`layer-${layerNum}-block`);

    if (status === 'processing') {
        badge.textContent = 'Processing...';
        badge.className = 'layer-status-badge processing';
        block.classList.add('active');
    } else if (status === 'complete') {
        badge.textContent = 'Complete';
        badge.className = 'layer-status-badge complete';

        // Render layer-specific output
        outputDiv.innerHTML = renderLayerOutput(layerNum, output);
    }
}

function renderLayerOutput(layerNum, output) {
    switch (layerNum) {
        case 1:
            return renderL1Output(output);
        case 2:
            return renderL2Output(output);
        case 3:
            return renderL3Output(output);
        case 4:
            return renderL4Output(output);
        case 5:
            return renderL5Output(output);
        default:
            return '<div class="output-placeholder">No output</div>';
    }
}

function renderL1Output(output) {
    const concepts = output.extractedConcepts || [];

    if (concepts.length === 0) {
        return `
            <div class="layer-result">
                <div class="result-row">
                    <span class="result-label">Concepts Extracted</span>
                    <span class="result-value">0</span>
                </div>
                <div class="result-row">
                    <span class="result-label">Insulin Sensitivity Modifier</span>
                    <span class="result-value">${(output.insulinSensitivityModifier * 100).toFixed(0)}%</span>
                </div>
                <div class="result-note">No significant semantic features detected</div>
            </div>
        `;
    }

    return `
        <div class="layer-result">
            <div class="result-row">
                <span class="result-label">Concepts Extracted</span>
                <span class="result-value">${concepts.length}</span>
            </div>
            <div class="result-row">
                <span class="result-label">Semantic Entropy</span>
                <span class="result-value">${output.semanticEntropy.toFixed(3)}</span>
            </div>
            <div class="result-row">
                <span class="result-label">Insulin Sensitivity Modifier</span>
                <span class="result-value ${output.insulinSensitivityModifier < 0.9 ? 'warning' : ''}">${(output.insulinSensitivityModifier * 100).toFixed(0)}%</span>
            </div>
            <div class="concepts-list">
                ${concepts.map(c => `
                    <div class="concept-item">
                        <span class="concept-name">${c.name}</span>
                        <span class="concept-code">${c.code}</span>
                        <span class="concept-confidence">${(c.confidence * 100).toFixed(0)}%</span>
                    </div>
                `).join('')}
            </div>
            <div class="result-row">
                <span class="result-label">Treatment Proxies (Z)</span>
                <span class="result-value">${output.treatmentProxies?.length || 0}</span>
            </div>
            <div class="result-row">
                <span class="result-label">Outcome Proxies (W)</span>
                <span class="result-value">${output.outcomeProxies?.length || 0}</span>
            </div>
        </div>
    `;
}

function renderL2Output(output) {
    return `
        <div class="layer-result">
            <div class="result-row">
                <span class="result-label">Current State (G)</span>
                <span class="result-value">${output.currentState.G.toFixed(0)} mg/dL</span>
            </div>
            <div class="result-row">
                <span class="result-label">Rate of Change</span>
                <span class="result-value">${output.currentState.Gdot > 0 ? '+' : ''}${output.currentState.Gdot.toFixed(1)} mg/dL/5min</span>
            </div>
            <div class="result-row">
                <span class="result-label">Detected Regime</span>
                <span class="result-value regime-${output.regime.regime}">${output.regime.regime.toUpperCase()}</span>
            </div>
            <div class="result-row">
                <span class="result-label">Regime Confidence</span>
                <span class="result-value">${(output.regime.confidence * 100).toFixed(0)}%</span>
            </div>
            <div class="result-row">
                <span class="result-label">Predicted Peak</span>
                <span class="result-value">${output.prediction.peak.glucose.toFixed(0)} mg/dL @ ${output.prediction.peak.time} min</span>
            </div>
            <div class="result-row">
                <span class="result-label">State Uncertainty</span>
                <span class="result-value">±${output.uncertainty.confidenceInterval.toFixed(0)} mg/dL</span>
            </div>
            <div class="result-row">
                <span class="result-label">Variance Reduction</span>
                <span class="result-value success">${output.varianceReduction.toFixed(1)}%</span>
            </div>
        </div>
    `;
}

function renderL3Output(output) {
    return `
        <div class="layer-result">
            <div class="result-row highlight">
                <span class="result-label">Individual Treatment Effect τ(t)</span>
                <span class="result-value">${output.tau.toFixed(1)} mg/dL per U</span>
            </div>
            <div class="result-row">
                <span class="result-label">95% Confidence Interval</span>
                <span class="result-value">[${output.tauCI[0].toFixed(1)}, ${output.tauCI[1].toFixed(1)}]</span>
            </div>
            <div class="result-row">
                <span class="result-label">Population Average</span>
                <span class="result-value">${output.populationEffect} mg/dL/U</span>
            </div>
            <div class="result-row">
                <span class="result-label">Individual Deviation</span>
                <span class="result-value ${output.individualDeviation > 0 ? 'warning' : ''}">${output.individualDeviation > 0 ? '+' : ''}${output.individualDeviation.toFixed(1)} mg/dL/U</span>
            </div>
            <div class="result-row">
                <span class="result-label">Proximal Adjustment</span>
                <span class="result-value">${output.proximalAdjustment.hasProxies ? 'Applied' : 'N/A'}</span>
            </div>
            ${output.proximalAdjustment.hasProxies ? `
                <div class="result-row">
                    <span class="result-label">Bias Reduction</span>
                    <span class="result-value success">${output.proximalAdjustment.biasReduction.toFixed(1)}%</span>
                </div>
            ` : ''}
        </div>
    `;
}

function renderL4Output(output) {
    return `
        <div class="layer-result">
            <div class="result-row highlight">
                <span class="result-label">Recommended Dose</span>
                <span class="result-value large">${output.recommendedDose.toFixed(1)} U</span>
            </div>
            <div class="result-row">
                <span class="result-label">Meal Coverage</span>
                <span class="result-value">${output.mealDose.toFixed(1)} U</span>
            </div>
            <div class="result-row">
                <span class="result-label">Correction Dose</span>
                <span class="result-value">${output.correctionDose.toFixed(1)} U</span>
            </div>
            <div class="result-row">
                <span class="result-label">Context Adjustment</span>
                <span class="result-value">${output.contextAdjustment >= 0 ? '+' : ''}${output.contextAdjustment.toFixed(1)} U</span>
            </div>
            <div class="result-row">
                <span class="result-label">95% CI</span>
                <span class="result-value">[${output.confidenceInterval.lower.toFixed(1)}, ${output.confidenceInterval.upper.toFixed(1)}]</span>
            </div>
            <div class="result-row">
                <span class="result-label">Exploration Bonus</span>
                <span class="result-value">${output.explorationBonus.toFixed(3)}</span>
            </div>
            <div class="result-row">
                <span class="result-label">Estimated Regret</span>
                <span class="result-value">${output.regret.toFixed(3)}</span>
            </div>
        </div>
    `;
}

function renderL5Output(output) {
    const tierClass = (tier) => tier.safe ? 'safe' : 'warning';

    return `
        <div class="layer-result">
            <div class="result-row ${output.overallSafe ? 'highlight-safe' : 'highlight-warning'}">
                <span class="result-label">Overall Safety</span>
                <span class="result-value ${output.overallSafe ? 'success' : 'danger'}">${output.overallSafe ? 'VERIFIED' : 'INTERVENTION'}</span>
            </div>
            
            <div class="safety-tier ${tierClass(output.tier1)}">
                <div class="tier-label">Tier 1: Reflex Controller</div>
                <div class="tier-status">${output.tier1.safe ? 'PASS' : 'FAIL'}</div>
                <div class="tier-reason">${output.tier1.reason}</div>
            </div>
            
            <div class="safety-tier ${tierClass(output.tier2)}">
                <div class="tier-label">Tier 2: STL Monitor</div>
                <div class="tier-status">${output.tier2.safe ? 'PASS' : 'FAIL'}</div>
                <div class="tier-reason">${output.tier2.reason}</div>
            </div>
            
            <div class="safety-tier ${tierClass(output.tier3)}">
                <div class="tier-label">Tier 3: Seldonian</div>
                <div class="tier-status">${output.tier3.safe ? 'PASS' : 'FAIL'}</div>
                <div class="tier-reason">${output.tier3.reason}</div>
            </div>
            
            ${!output.overallSafe ? `
                <div class="result-row">
                    <span class="result-label">Original Dose</span>
                    <span class="result-value">${output.originalDose.toFixed(1)} U</span>
                </div>
                <div class="result-row highlight-safe">
                    <span class="result-label">Safe Dose</span>
                    <span class="result-value">${output.safeDose.toFixed(1)} U</span>
                </div>
            ` : ''}
        </div>
    `;
}

function resetLayerOutputs() {
    for (let i = 1; i <= 5; i++) {
        const badge = document.getElementById(`l${i}-badge`);
        const outputDiv = document.getElementById(`l${i}-output`);
        const block = document.getElementById(`layer-${i}-block`);

        badge.textContent = 'Pending';
        badge.className = 'layer-status-badge';
        block.classList.remove('active');
        outputDiv.innerHTML = '<div class="output-placeholder">Awaiting input...</div>';
    }
}

// ============================================
// Results Display
// ============================================

function displayResults(results) {
    // Show results content, hide empty state
    document.getElementById('results-empty').classList.add('hidden');
    document.getElementById('results-data').classList.remove('hidden');

    const rec = results.recommendation;
    const l5 = results.layers.L5;
    const l3 = results.layers.L3;
    const l2 = results.layers.L2;

    // Update recommendation display
    document.getElementById('rec-bolus').textContent = rec.dose.toFixed(1);
    document.getElementById('rec-meal-dose').textContent = rec.mealComponent.toFixed(1) + ' U';
    document.getElementById('rec-correction').textContent = rec.correctionComponent.toFixed(1) + ' U';
    document.getElementById('rec-adjustment').textContent = (rec.contextAdjustment >= 0 ? '+' : '') + rec.contextAdjustment.toFixed(1) + ' U';
    document.getElementById('rec-ci').textContent = `[${rec.confidenceInterval.lower.toFixed(1)}, ${rec.confidenceInterval.upper.toFixed(1)}] U`;

    // Update safety tiers
    updateSafetyTier('tier-1-result', l5.tier1);
    updateSafetyTier('tier-2-result', l5.tier2);
    updateSafetyTier('tier-3-result', l5.tier3);

    // Update causal analysis
    document.getElementById('causal-tau').textContent = l3.tau.toFixed(1);
    document.getElementById('causal-dev').textContent = (l3.individualDeviation >= 0 ? '+' : '') + l3.individualDeviation.toFixed(1) + ' mg/dL/U';
    document.getElementById('causal-adj').textContent = l3.proximalAdjustment.hasProxies ?
        `${l3.proximalAdjustment.biasReduction.toFixed(0)}% reduction` : 'N/A';

    // Update prediction chart
    updatePredictionChart(l2.prediction.trajectory, rec.dose);

    // Update prediction metrics
    const adjustedPeak = l2.prediction.peak.glucose - rec.dose * 25;
    document.getElementById('pred-peak').textContent = adjustedPeak.toFixed(0) + ' mg/dL';
    document.getElementById('pred-time-peak').textContent = l2.prediction.peak.time + ' min';
    document.getElementById('pred-return').textContent = l2.prediction.returnToTarget ?
        l2.prediction.returnToTarget + ' min' : '> 3 hours';
}

function updateSafetyTier(elementId, tierResult) {
    const el = document.getElementById(elementId);
    const statusEl = el.querySelector('.tier-status');
    const detailEl = el.querySelector('.tier-detail');

    statusEl.textContent = tierResult.safe ? 'PASS' : 'FAIL';
    statusEl.className = `tier-status ${tierResult.safe ? 'pass' : 'fail'}`;
    detailEl.textContent = tierResult.reason;
    el.className = `safety-tier ${tierResult.safe ? '' : 'failed'}`;
}

function updatePredictionChart(trajectory, dose) {
    const ctx = document.getElementById('prediction-chart');
    if (!ctx) return;

    // Adjust trajectory for insulin effect
    const adjustedTrajectory = trajectory.map(p => ({
        time: p.time,
        glucose: Math.max(40, p.glucose - dose * 25 * Math.min(1, p.time / 60))
    }));

    if (AppState.predictionChart) {
        AppState.predictionChart.destroy();
    }

    AppState.predictionChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: adjustedTrajectory.map(p => p.time),
            datasets: [{
                label: 'Predicted Glucose',
                data: adjustedTrajectory.map(p => p.glucose),
                borderColor: '#3b82f6',
                backgroundColor: 'rgba(59, 130, 246, 0.1)',
                fill: true,
                tension: 0.3,
                pointRadius: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                x: {
                    title: { display: true, text: 'Time (min)', color: '#6b7280' },
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    ticks: { color: '#6b7280' }
                },
                y: {
                    min: 40,
                    max: 300,
                    title: { display: true, text: 'Glucose (mg/dL)', color: '#6b7280' },
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    ticks: { color: '#6b7280' }
                }
            }
        },
        plugins: [{
            id: 'targetRange',
            beforeDraw: (chart) => {
                const { ctx, chartArea, scales } = chart;
                if (!chartArea) return;

                ctx.save();
                ctx.fillStyle = 'rgba(16, 185, 129, 0.1)';
                const y180 = scales.y.getPixelForValue(180);
                const y70 = scales.y.getPixelForValue(70);
                ctx.fillRect(chartArea.left, y180, chartArea.right - chartArea.left, y70 - y180);
                ctx.restore();
            }
        }]
    });
}

// ============================================
// Batch Simulation
// ============================================

async function runBatchSimulation() {
    const patient = document.getElementById('sim-patient').value;
    const duration = parseInt(document.getElementById('sim-duration').value);
    const meals = document.getElementById('sim-meals').value;
    const challenge = document.getElementById('sim-challenge').value;

    const statusDisplay = document.getElementById('sim-status-display');
    const chartContainer = document.getElementById('sim-chart-container');
    const metricsDiv = document.getElementById('sim-metrics');

    statusDisplay.innerHTML = '<p>Running simulation...</p>';
    chartContainer.classList.add('hidden');
    metricsDiv.classList.add('hidden');

    // Simulate based on parameters
    const results = await runSimulation(patient, duration, meals, challenge);

    // Display results
    displaySimulationResults(results);
}

async function runSimulation(patient, duration, meals, challenge) {
    // Patient parameters
    const patients = {
        adult_avg: { weight: 75, tdi: 45 },
        adult_high_ir: { weight: 85, tdi: 65 },
        adult_sensitive: { weight: 70, tdi: 30 },
        adolescent: { weight: 52, tdi: 32 },
        child: { weight: 30, tdi: 16 }
    };

    const mealPlans = {
        standard: [
            { time: 0, carbs: 45 },
            { time: 5, carbs: 70 },
            { time: 11, carbs: 80 }
        ],
        high_carb: [
            { time: 0, carbs: 60 },
            { time: 5, carbs: 100 },
            { time: 11, carbs: 120 }
        ],
        low_carb: [
            { time: 0, carbs: 20 },
            { time: 5, carbs: 30 },
            { time: 11, carbs: 40 }
        ],
        irregular: [
            { time: 2, carbs: 50 },
            { time: 8, carbs: 60 }
        ]
    };

    const p = patients[patient];
    const mealPlan = mealPlans[meals];

    // Simulate glucose trajectory
    const trajectory = [];
    let glucose = 100;
    let decisions = 0;
    let violations = 0;

    const steps = duration * 12; // 5-min steps

    for (let i = 0; i < steps; i++) {
        const hour = i / 12;

        // Check for meal
        const meal = mealPlan.find(m => Math.abs(m.time - hour) < 0.1);
        if (meal) {
            const dose = meal.carbs / (500 / p.tdi);
            glucose += meal.carbs * 4 - dose * 25;
            decisions++;
        }

        // Apply challenge modifiers
        let modifier = 1;
        if (challenge === 'exercise' && hour >= 3 && hour <= 4) {
            modifier = 0.7; // Drop glucose during exercise
        } else if (challenge === 'stress') {
            modifier = 1.15;
        } else if (challenge === 'illness') {
            modifier = 1.25;
        } else if (challenge === 'dawn' && hour >= 4 && hour <= 7) {
            modifier = 1.1;
        }

        // Natural glucose dynamics
        glucose = glucose * 0.98 + 100 * 0.02; // Tend toward 100
        glucose *= modifier;

        // Add noise
        glucose += (Math.random() - 0.5) * 10;

        // Bounds
        glucose = Math.max(50, Math.min(350, glucose));

        // Safety check
        if (glucose < 54) {
            violations++;
            glucose = 70; // Safety intervention
        }

        trajectory.push({ time: hour, glucose });

        await new Promise(r => setTimeout(r, 5));
    }

    return { trajectory, decisions, violations, patient: p };
}

function displaySimulationResults(results) {
    const statusDisplay = document.getElementById('sim-status-display');
    const chartContainer = document.getElementById('sim-chart-container');
    const metricsDiv = document.getElementById('sim-metrics');

    statusDisplay.innerHTML = '<p>Simulation complete</p>';
    chartContainer.classList.remove('hidden');
    metricsDiv.classList.remove('hidden');

    // Calculate metrics
    const glucose = results.trajectory.map(p => p.glucose);
    const n = glucose.length;

    const tir = (glucose.filter(g => g >= 70 && g <= 180).length / n * 100).toFixed(1);
    const tbr = (glucose.filter(g => g < 70).length / n * 100).toFixed(1);
    const hypo = (glucose.filter(g => g < 54).length / n * 100).toFixed(1);
    const tar = (glucose.filter(g => g > 180).length / n * 100).toFixed(1);
    const mean = (glucose.reduce((a, b) => a + b, 0) / n).toFixed(1);
    const cv = ((Math.sqrt(glucose.reduce((s, g) => s + Math.pow(g - mean, 2), 0) / n) / mean) * 100).toFixed(1);

    document.getElementById('sim-tir').textContent = tir + '%';
    document.getElementById('sim-tbr').textContent = tbr + '%';
    document.getElementById('sim-hypo').textContent = hypo + '%';
    document.getElementById('sim-tar').textContent = tar + '%';
    document.getElementById('sim-mean').textContent = mean + ' mg/dL';
    document.getElementById('sim-cv').textContent = cv + '%';
    document.getElementById('sim-violations').textContent = results.violations;
    document.getElementById('sim-decisions').textContent = results.decisions;

    // Update chart
    const ctx = document.getElementById('sim-chart');

    if (AppState.simChart) {
        AppState.simChart.destroy();
    }

    AppState.simChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: results.trajectory.map(p => p.time.toFixed(1)),
            datasets: [{
                label: 'Glucose',
                data: results.trajectory.map(p => p.glucose),
                borderColor: '#3b82f6',
                backgroundColor: 'rgba(59, 130, 246, 0.1)',
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
                    title: { display: true, text: 'Time (hours)', color: '#6b7280' },
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    ticks: { color: '#6b7280', maxTicksLimit: 12 }
                },
                y: {
                    min: 40,
                    max: 350,
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    ticks: { color: '#6b7280' }
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
                ctx.fillStyle = 'rgba(16, 185, 129, 0.1)';
                const y180 = scales.y.getPixelForValue(180);
                const y70 = scales.y.getPixelForValue(70);
                ctx.fillRect(chartArea.left, y180, chartArea.right - chartArea.left, y70 - y180);

                // Danger zone
                ctx.fillStyle = 'rgba(239, 68, 68, 0.1)';
                const y54 = scales.y.getPixelForValue(54);
                ctx.fillRect(chartArea.left, y54, chartArea.right - chartArea.left, chartArea.bottom - y54);

                ctx.restore();
            }
        }]
    });
}

// ============================================
// Utility Functions
// ============================================

function updateSystemStatus(status) {
    const dot = document.getElementById('system-status-dot');
    const text = document.getElementById('system-status-text');

    const states = {
        ready: { class: 'ready', text: 'Ready for Input' },
        processing: { class: 'processing', text: 'Processing...' },
        complete: { class: 'complete', text: 'Complete' },
        error: { class: 'error', text: 'Error' }
    };

    const state = states[status] || states.ready;
    dot.className = `status-dot ${state.class}`;
    text.textContent = state.text;
}

function resetSystem() {
    AppState.lastResults = null;
    resetLayerOutputs();
    document.getElementById('results-empty').classList.remove('hidden');
    document.getElementById('results-data').classList.add('hidden');
    updateSystemStatus('ready');
    navigateToSection('input');
    showToast('System reset', 'info');
}

function emergencyStop() {
    AppState.processing = false;
    updateSystemStatus('error');
    showToast('Emergency stop activated - all insulin suspended', 'warning');
}

function approveRecommendation() {
    showToast('Recommendation approved - administering ' + AppState.lastResults.recommendation.dose.toFixed(1) + ' U', 'success');
}

function modifyDose() {
    const newDose = prompt('Enter modified dose (Units):', AppState.lastResults.recommendation.dose.toFixed(1));
    if (newDose !== null) {
        showToast('Modified dose: ' + parseFloat(newDose).toFixed(1) + ' U', 'info');
    }
}

function rejectRecommendation() {
    showToast('Recommendation rejected - manual override active', 'warning');
}

function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `<span class="toast-message">${message}</span>`;

    container.appendChild(toast);

    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateX(100%)';
        setTimeout(() => toast.remove(), 200);
    }, 4000);
}
