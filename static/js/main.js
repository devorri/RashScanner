/**
 * main.js - Client-side Application Logic for 100% AI Real-Time Skin Disease Scanner
 * Features:
 * - Full-Screen Real-Time AI Camera with live bounding boxes, HUD telemetry, and capture
 * - 100% AI Computer Vision classification across 10 unique conditions
 * - Modern interactive checkbox/radio forms
 * - Clinical patient charts database & reporting
 */

let currentImageSource = 'upload';
let selectedImageFile = null;
let currentImageBase64 = null;
let currentImageFilename = null;
let currentAiResults = [];
let realtimePollInterval = null;

// Initialize Page Defaults & Splash Loading Animation
document.addEventListener('DOMContentLoaded', () => {
    const dateInput = document.getElementById('date_of_assessment');
    if (dateInput) {
        dateInput.value = new Date().toISOString().split('T')[0];
    }

    startSplashAnimation();
    fetchActiveModelInfo();

    // Start 10,000mAh Power Bank Battery Monitor
    updateBatteryStatus();
    if (!batteryPollInterval) {
        batteryPollInterval = setInterval(updateBatteryStatus, 15000);
    }

    // Drag & Drop event wiring for Stage 5 dropzone
    const dropzone = document.getElementById('imageDropzone');
    if (dropzone) {
        dropzone.addEventListener('dragover', (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.add('dropzone-hover');
        });
        dropzone.addEventListener('dragleave', (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.remove('dropzone-hover');
        });
        dropzone.addEventListener('drop', (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.remove('dropzone-hover');
            const files = e.dataTransfer.files;
            if (files && files.length > 0) {
                loadImageFile(files[0]);
            }
        });
    }
});

// Splash Screen Loader Animation
function startSplashAnimation() {
    const progressBar = document.getElementById('splashProgressBar');
    const statusText = document.getElementById('splashStatusText');
    const actionArea = document.getElementById('splashActionArea');

    if (!progressBar) return;

    let progress = 0;
    const interval = setInterval(() => {
        progress += 10;
        progressBar.style.width = `${progress}%`;

        if (progress === 40) {
            statusText.innerText = "Loading 100% AI Real-Time Neural Model & 10 Conditions...";
        } else if (progress === 80) {
            statusText.innerText = "Edge Computer Vision Model Ready • Real-Time Engine Active.";
        } else if (progress >= 100) {
            clearInterval(interval);
            statusText.innerHTML = "<strong style='color: var(--success-color);'><i class='fa-solid fa-circle-check'></i> 100% AI Vision Engine Ready!</strong>";
            if (actionArea) actionArea.classList.remove('hidden');
        }
    }, 120);
}

// Screen Transitions
function showScreen(screenId) {
    ['splashScreen', 'loginScreen', 'hubScreen', 'appContainer'].forEach(id => {
        const el = document.getElementById(id);
        if (el) {
            el.classList.add('hidden');
            el.style.display = 'none';
        }
    });
    const target = document.getElementById(screenId);
    if (target) {
        target.classList.remove('hidden');
        target.style.display = (screenId === 'appContainer') ? 'block' : 'flex';
    }
}

// -----------------------------------------------------------------------------
// 10,000mAh Power Bank Battery Manager Logic
// -----------------------------------------------------------------------------
let batteryPollInterval = null;

async function updateBatteryStatus() {
    try {
        const res = await fetch('/api/system/status');
        const data = await res.json();
        if (data && data.success) {
            applyBatteryUI(data.percentage, data.hours_remaining, data.low_voltage_warning);
        }
    } catch (e) {
        console.log('[Battery] Local tracker active');
    }
}

function applyBatteryUI(percentage, hoursLeft, isLowVoltage) {
    const percentEls = document.querySelectorAll('.kiosk-battery-percent');
    const hoursEls = document.querySelectorAll('.kiosk-battery-hours');
    const iconEls = document.querySelectorAll('.kiosk-battery-icon');
    const pills = document.querySelectorAll('.battery-pill');

    percentEls.forEach(el => el.textContent = `${percentage}%`);
    hoursEls.forEach(el => el.textContent = `(~${hoursLeft}h)`);

    let iconClass = 'fa-solid fa-battery-full text-success';
    if (percentage <= 15 || isLowVoltage) {
        iconClass = 'fa-solid fa-battery-empty text-danger';
        pills.forEach(p => p.classList.add('battery-low-alert'));
    } else if (percentage <= 35) {
        iconClass = 'fa-solid fa-battery-quarter text-warning';
        pills.forEach(p => p.classList.remove('battery-low-alert'));
    } else if (percentage <= 65) {
        iconClass = 'fa-solid fa-battery-half text-warning';
        pills.forEach(p => p.classList.remove('battery-low-alert'));
    } else if (percentage <= 85) {
        iconClass = 'fa-solid fa-battery-three-quarters text-success';
        pills.forEach(p => p.classList.remove('battery-low-alert'));
    } else {
        iconClass = 'fa-solid fa-battery-full text-success';
        pills.forEach(p => p.classList.remove('battery-low-alert'));
    }

    iconEls.forEach(icon => {
        icon.className = `${iconClass} kiosk-battery-icon`;
    });

    const modalPercent = document.getElementById('modalBatteryPercent');
    const modalSubtext = document.getElementById('modalBatterySubtext');
    const modalIcon = document.getElementById('modalBatteryIcon');

    if (modalPercent) modalPercent.textContent = `${percentage}%`;
    if (modalSubtext) modalSubtext.textContent = isLowVoltage ?
        `⚠️ LOW VOLTAGE DETECTED! Connect Power Bank / Charger immediately.` :
        `Estimated Runtime Remaining: ~${hoursLeft} Hours`;
    if (modalIcon) modalIcon.className = `${iconClass}`;
}

function openBatteryModal() {
    const modal = document.getElementById('batteryModal');
    if (modal) {
        modal.classList.remove('hidden');
        modal.style.display = 'flex';
        updateBatteryStatus();
    }
}

function closeBatteryModal() {
    const modal = document.getElementById('batteryModal');
    if (modal) {
        modal.classList.add('hidden');
        modal.style.display = 'none';
    }
}

async function resetPowerBankTracker() {
    try {
        const res = await fetch('/api/system/battery/reset', { method: 'POST' });
        const data = await res.json();
        if (data && data.success) {
            applyBatteryUI(100, data.hours_remaining, false);
            alert("Power bank tracker reset to 100% (~6.0 hours remaining)!");
            closeBatteryModal();
        }
    } catch (e) {
        applyBatteryUI(100, 6.0, false);
        closeBatteryModal();
    }
}

// -----------------------------------------------------------------------------
// System Maintenance & Power Controls
// -----------------------------------------------------------------------------
function openMaintenanceModal() {
    const modal = document.getElementById('maintenanceModal');
    const authPanel = document.getElementById('maintenanceAuth');
    const actionsPanel = document.getElementById('maintenanceActions');
    const errBox = document.getElementById('maintenanceAuthError');
    const pwInput = document.getElementById('maintenancePassword');

    if (modal) {
        if (authPanel) { authPanel.classList.remove('hidden'); authPanel.style.display = ''; }
        if (actionsPanel) { actionsPanel.classList.add('hidden'); actionsPanel.style.display = 'none'; }
        if (errBox) { errBox.classList.add('hidden'); errBox.textContent = ''; }
        if (pwInput) pwInput.value = '';
        modal.classList.remove('hidden');
        modal.style.display = 'flex';
    }
}

function closeMaintenanceModal() {
    const modal = document.getElementById('maintenanceModal');
    if (modal) {
        modal.classList.add('hidden');
        modal.style.display = 'none';
    }
}

async function authenticateMaintenance() {
    const pw = (document.getElementById('maintenancePassword').value || '').trim();
    const errBox = document.getElementById('maintenanceAuthError');

    try {
        const res = await fetch('/api/login', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username: 'admin', password: pw })
        });
        const data = await res.json();

        if (data.success) {
            const authPanel = document.getElementById('maintenanceAuth');
            const actionsPanel = document.getElementById('maintenanceActions');
            if (authPanel) { authPanel.classList.add('hidden'); authPanel.style.display = 'none'; }
            if (actionsPanel) { actionsPanel.classList.remove('hidden'); actionsPanel.style.display = ''; }
            if (errBox) errBox.classList.add('hidden');
        } else {
            if (errBox) {
                errBox.textContent = 'Incorrect admin password.';
                errBox.classList.remove('hidden');
            }
        }
    } catch (e) {
        if (errBox) {
            errBox.textContent = 'Server connection error.';
            errBox.classList.remove('hidden');
        }
    }
}

function openPowerModal() {
    const modal = document.getElementById('powerModal');
    if (modal) {
        modal.classList.remove('hidden');
        modal.style.display = 'flex';
    }
}

function closePowerModal() {
    const modal = document.getElementById('powerModal');
    if (modal) {
        modal.classList.add('hidden');
        modal.style.display = 'none';
    }
}

async function executeSystemAction(action) {
    if (action === 'kill') {
        if (!confirm('Kill App & Exit Kiosk?\n\nThis will close the fullscreen browser and stop the server.')) return;
        closePowerModal();
        try { await fetch('/api/system/kill-kiosk', { method: 'POST' }); } catch (e) { }
        setTimeout(() => { try { window.close(); } catch (e) { } }, 500);
    } else if (action === 'shutdown') {
        if (!confirm('Safely Shut Down Raspberry Pi?')) return;
        closePowerModal();
        try { await fetch('/api/system/shutdown', { method: 'POST' }); } catch (e) { }
    } else if (action === 'reboot') {
        if (!confirm('Reboot Raspberry Pi?')) return;
        closePowerModal();
        try { await fetch('/api/system/reboot', { method: 'POST' }); } catch (e) { }
    }
}

function exitKioskMode() { executeSystemAction('kill'); }
function minimizeKiosk() {
    if (document.fullscreenElement) document.exitFullscreen().catch(() => { });
    alert('Exited fullscreen mode.');
}
function rebootKiosk() { executeSystemAction('reboot'); }

function proceedFromSplash() { showScreen('loginScreen'); }

async function performPortalLogin() {
    const u = document.getElementById('portalUsername').value;
    const p = document.getElementById('portalPassword').value;
    const alertBox = document.getElementById('portalLoginAlert');

    try {
        const res = await fetch('/api/login', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username: u, password: p })
        });
        const data = await res.json();
        if (data.success) {
            alertBox.classList.add('hidden');
            showScreen('hubScreen');
        } else {
            alertBox.innerText = data.message || "Invalid username or password";
            alertBox.classList.remove('hidden');
        }
    } catch (err) {
        alertBox.innerText = "Connection error. Login as Demo Clinician to test.";
        alertBox.classList.remove('hidden');
    }
}

function bypassLoginAsGuest() { showScreen('hubScreen'); }

function navigateToApp(tabName) {
    showScreen('appContainer');
    switchTab(tabName);
    if (tabName === 'assessment') {
        goToWizardStep(1);
    }
}

function returnToHub() { showScreen('hubScreen'); }
function logoutToLogin() { showScreen('loginScreen'); }

// -----------------------------------------------------------------------------
// 5-Stage Stepper Wizard Navigation
// -----------------------------------------------------------------------------
let currentWizardStep = 1;

function goToWizardStep(stepNum) {
    if (stepNum < 1 || stepNum > 5) return;
    currentWizardStep = stepNum;

    for (let i = 1; i <= 5; i++) {
        const stepEl = document.getElementById(`wizardStep${i}`);
        const pillEl = document.getElementById(`stepPill${i}`);

        if (stepEl) {
            stepEl.classList.add('hidden');
            stepEl.style.display = 'none';
        }
        if (pillEl) {
            pillEl.classList.remove('active');
        }
    }

    const targetStep = document.getElementById(`wizardStep${stepNum}`);
    const targetPill = document.getElementById(`stepPill${stepNum}`);

    if (targetStep) {
        targetStep.classList.remove('hidden');
        targetStep.style.display = 'block';
    }
    if (targetPill) {
        targetPill.classList.add('active');
    }

    window.scrollTo({ top: 150, behavior: 'smooth' });
}

function switchTab(tabName) {
    document.querySelectorAll('.tab-content').forEach(tab => tab.classList.add('hidden'));
    document.querySelectorAll('.nav-btn').forEach(btn => btn.classList.remove('active'));

    if (tabName === 'assessment') {
        document.getElementById('assessmentTab').classList.remove('hidden');
        document.getElementById('tabAssessmentBtn').classList.add('active');
    } else if (tabName === 'dashboard') {
        document.getElementById('dashboardTab').classList.remove('hidden');
        document.getElementById('tabDashboardBtn').classList.add('active');
        loadPatientsDashboard();
    }
}

// -----------------------------------------------------------------------------
// Form Interactive Checkboxes / Radio Handlers
// -----------------------------------------------------------------------------
function handleSexChange(radioInput) {
    document.querySelectorAll('input[name="sex_radio"]').forEach(r => {
        r.closest('.radio-card').classList.remove('active');
    });
    radioInput.closest('.radio-card').classList.add('active');
    const hiddenField = document.getElementById('sex');
    if (hiddenField) hiddenField.value = radioInput.value;
}

function handleLymphChange(radioInput) {
    document.querySelectorAll('input[name="lymph_radio"]').forEach(r => {
        r.closest('.radio-card').classList.remove('active');
    });
    radioInput.closest('.radio-card').classList.add('active');
    const hiddenField = document.getElementById('regional_lymph_nodes');
    if (hiddenField) hiddenField.value = radioInput.value;
}

function handleOnsetChange(radioInput) {
    document.querySelectorAll('input[name="onset_radio"]').forEach(r => {
        r.closest('.radio-card').classList.remove('active');
    });
    radioInput.closest('.radio-card').classList.add('active');
    const hiddenField = document.getElementById('onset');
    if (hiddenField) hiddenField.value = radioInput.value;
}

function handlePatternChange(radioInput) {
    document.querySelectorAll('input[name="pattern_radio"]').forEach(r => {
        r.closest('.radio-card').classList.remove('active');
    });
    radioInput.closest('.radio-card').classList.add('active');
    const hiddenField = document.getElementById('pattern');
    if (hiddenField) hiddenField.value = radioInput.value;
}

function handleProgressionChange(radioInput) {
    document.querySelectorAll('input[name="progression_radio"]').forEach(r => {
        r.closest('.radio-card').classList.remove('active');
    });
    radioInput.closest('.radio-card').classList.add('active');
    const hiddenField = document.getElementById('progression');
    if (hiddenField) hiddenField.value = radioInput.value;
}

function togglePillActive(checkbox) {
    const card = checkbox.closest('.checkbox-card');
    const group = checkbox.closest('.checkbox-pill-group');
    const isChecked = checkbox.checked;

    if (isChecked) {
        card.classList.add('active');
    } else {
        card.classList.remove('active');
    }

    // Update hidden field value with comma-separated selected values
    const checkedValues = [];
    group.querySelectorAll('input[type="checkbox"]:checked').forEach(cb => {
        checkedValues.push(cb.value);
    });

    if (group.id === 'onsetGroup') {
        document.getElementById('onset').value = checkedValues.join(', ') || 'Sudden/Acute';
    } else if (group.id === 'patternGroup') {
        document.getElementById('pattern').value = checkedValues.join(', ') || 'Constant';
    } else if (group.id === 'progressionGroup') {
        document.getElementById('progression').value = checkedValues.join(', ') || 'Static';
    }
}

function toggleSymptomChip(checkbox) {
    const label = checkbox.closest('.chip-checkbox');
    if (checkbox.checked) {
        label.classList.add('active');
    } else {
        label.classList.remove('active');
    }

    // Sync selected chips into associated_symptoms note field
    const selectedChips = [];
    document.querySelectorAll('.checkbox-chip-grid input[type="checkbox"]:checked').forEach(cb => {
        selectedChips.push(cb.value);
    });

    const symptomsInput = document.getElementById('associated_symptoms');
    if (symptomsInput) {
        const existingText = symptomsInput.value.trim();
        // Append or update text
        symptomsInput.value = selectedChips.join(', ');
    }
}

// -----------------------------------------------------------------------------
// FULL-SCREEN REAL-TIME AI WEBCAM SCANNER LOGIC
// -----------------------------------------------------------------------------
let clientMediaStream = null;
let currentFacingMode = 'user'; // 'user' (front/webcam) or 'environment' (back)
let availableVideoDevices = [];
let currentDeviceIndex = 0;
let reticleAnimFrameId = null;
let scanLineY = 0;
let scanLineDir = 1;

async function openCameraFullscreen(event) {
    if (event) event.stopPropagation();

    const modal = document.getElementById('cameraFullscreenModal');
    if (!modal) return;

    modal.classList.remove('hidden');
    modal.style.display = 'flex';

    // Query available video devices
    await updateAvailableVideoDevices();

    // Start Real-Time Webcam
    await startClientWebcam();
}

function closeCameraFullscreen() {
    const modal = document.getElementById('cameraFullscreenModal');
    if (!modal) return;

    // 1. Stop Webcam Tracks
    stopClientWebcamTracks();

    // 2. Stop Canvas Reticle Animation
    if (reticleAnimFrameId) {
        cancelAnimationFrame(reticleAnimFrameId);
        reticleAnimFrameId = null;
    }

    modal.classList.add('hidden');
    modal.style.display = 'none';
}

async function updateAvailableVideoDevices() {
    try {
        if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) return;
        const devices = await navigator.mediaDevices.enumerateDevices();
        availableVideoDevices = devices.filter(d => d.kind === 'videoinput');
    } catch (e) {
        availableVideoDevices = [];
    }
}

async function startClientWebcam() {
    const video = document.getElementById('clientWebcamVideo');
    const insecureNotice = document.getElementById('camInsecureNotice');
    const subInfo = document.getElementById('hudSubInfo');

    stopClientWebcamTracks();

    // Check if getUserMedia is supported in current context
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        console.warn("[Webcam] navigator.mediaDevices.getUserMedia is undefined.");
        if (insecureNotice) {
            insecureNotice.classList.remove('hidden');
            insecureNotice.style.display = 'block';
            const msg = document.getElementById('camInsecureMessage');
            if (msg) {
                msg.innerHTML = "Webcam blocked by browser security (HTTP LAN Origin).<br><small style='color:#94a3b8;'>Modern browsers require HTTPS or localhost to prompt for webcam.</small>";
            }
        }
        if (subInfo) subInfo.textContent = "Tap 'Open Native Camera Photo' below to snap a picture";
        return;
    }

    if (insecureNotice) insecureNotice.style.display = 'none';

    // Attempt preferred constraint first, with fallback to basic video
    let stream = null;
    const selectedDeviceId = (availableVideoDevices.length > 0 && availableVideoDevices[currentDeviceIndex])
        ? availableVideoDevices[currentDeviceIndex].deviceId
        : null;

    const attempts = [];
    if (selectedDeviceId) {
        attempts.push({ video: { deviceId: { exact: selectedDeviceId }, width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
        attempts.push({ video: { deviceId: { exact: selectedDeviceId } }, audio: false });
    }
    attempts.push({ video: { facingMode: currentFacingMode, width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
    attempts.push({ video: { facingMode: currentFacingMode }, audio: false });
    attempts.push({ video: true, audio: false });

    for (const constraints of attempts) {
        try {
            stream = await navigator.mediaDevices.getUserMedia(constraints);
            if (stream) break;
        } catch (err) {
            // Try next constraint
        }
    }

    if (!stream) {
        console.warn("[Webcam] Unable to start webcam on any constraint.");
        if (insecureNotice) {
            insecureNotice.classList.remove('hidden');
            insecureNotice.style.display = 'block';
            const msg = document.getElementById('camInsecureMessage');
            if (msg) {
                msg.innerHTML = "⚠️ Camera access was blocked or unavailable.<br><small>Please allow camera permissions in your browser URL bar.</small>";
            }
        }
        if (subInfo) subInfo.textContent = "Use 'Open Native Camera Photo' button below";
        return;
    }

    clientMediaStream = stream;

    if (video) {
        video.srcObject = stream;
        try {
            await video.play();
        } catch (playErr) {
            console.warn("[Webcam] video.play error:", playErr);
        }
        startReticleAnimation();
    }

    const deviceLabel = (availableVideoDevices.length > 0 && availableVideoDevices[currentDeviceIndex]?.label)
        ? availableVideoDevices[currentDeviceIndex].label
        : "Webcam";

    if (subInfo) subInfo.textContent = `${deviceLabel} Active • Position lesion in reticle`;
    const topCond = document.getElementById('hudTopCondition');
    if (topCond) topCond.textContent = "Position camera on skin lesion...";
    const topConf = document.getElementById('hudTopConfidence');
    if (topConf) topConf.textContent = "READY";
}

function stopClientWebcamTracks() {
    if (clientMediaStream) {
        clientMediaStream.getTracks().forEach(track => {
            try { track.stop(); } catch (e) {}
        });
        clientMediaStream = null;
    }
    const video = document.getElementById('clientWebcamVideo');
    if (video) video.srcObject = null;
}

async function flipWebcamFacingMode() {
    if (availableVideoDevices.length > 1) {
        currentDeviceIndex = (currentDeviceIndex + 1) % availableVideoDevices.length;
    } else {
        currentFacingMode = (currentFacingMode === 'user') ? 'environment' : 'user';
    }
    await startClientWebcam();
}

function triggerNativeCameraSnap(event) {
    if (event) event.stopPropagation();
    const input = document.getElementById('nativeCameraInput');
    if (input) input.click();
}

function handleNativeCameraCapture(event) {
    const files = event.target.files;
    if (files && files.length > 0) {
        closeCameraFullscreen();
        loadImageFile(files[0]);
        // Auto-run AI examination on captured photo
        setTimeout(() => {
            executeAiAnalysis();
        }, 300);
    }
}

// -----------------------------------------------------------------------------
// Sleek Futuristic Animated Reticle HUD for Client Camera
// -----------------------------------------------------------------------------
function startReticleAnimation() {
    const canvas = document.getElementById('webcamOverlayCanvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    function draw() {
        if (currentCameraSource !== 'webcam' || !clientMediaStream) {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            return;
        }

        const w = canvas.parentElement.clientWidth || window.innerWidth;
        const h = canvas.parentElement.clientHeight || window.innerHeight;
        if (canvas.width !== w || canvas.height !== h) {
            canvas.width = w;
            canvas.height = h;
        }

        ctx.clearRect(0, 0, w, h);

        const cx = w / 2;
        const cy = h / 2;
        const boxSize = Math.min(w, h) * 0.55;
        const half = boxSize / 2;
        const cornerLen = 28;

        // Draw Corner Target Brackets (Neon Cyan)
        ctx.strokeStyle = '#00e5ff';
        ctx.lineWidth = 3;
        ctx.shadowColor = '#00e5ff';
        ctx.shadowBlur = 8;

        // Top-Left
        ctx.beginPath();
        ctx.moveTo(cx - half, cy - half + cornerLen);
        ctx.lineTo(cx - half, cy - half);
        ctx.lineTo(cx - half + cornerLen, cy - half);
        ctx.stroke();

        // Top-Right
        ctx.beginPath();
        ctx.moveTo(cx + half - cornerLen, cy - half);
        ctx.lineTo(cx + half, cy - half);
        ctx.lineTo(cx + half, cy - half + cornerLen);
        ctx.stroke();

        // Bottom-Left
        ctx.beginPath();
        ctx.moveTo(cx - half, cy + half - cornerLen);
        ctx.lineTo(cx - half, cy + half);
        ctx.lineTo(cx - half + cornerLen, cy + half);
        ctx.stroke();

        // Bottom-Right
        ctx.beginPath();
        ctx.moveTo(cx + half - cornerLen, cy + half);
        ctx.lineTo(cx + half, cy + half);
        ctx.lineTo(cx + half, cy + half - cornerLen);
        ctx.stroke();

        // Center Crosshairs
        ctx.lineWidth = 1;
        ctx.strokeStyle = 'rgba(0, 229, 255, 0.4)';
        ctx.beginPath();
        ctx.moveTo(cx - 15, cy); ctx.lineTo(cx + 15, cy);
        ctx.moveTo(cx, cy - 15); ctx.lineTo(cx, cy + 15);
        ctx.stroke();

        // Animated Laser Scan Line
        scanLineY += 2.5 * scanLineDir;
        if (scanLineY > half) { scanLineY = half; scanLineDir = -1; }
        if (scanLineY < -half) { scanLineY = -half; scanLineDir = 1; }

        ctx.strokeStyle = 'rgba(0, 229, 255, 0.65)';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(cx - half + 8, cy + scanLineY);
        ctx.lineTo(cx + half - 8, cy + scanLineY);
        ctx.stroke();

        reticleAnimFrameId = requestAnimationFrame(draw);
    }

    if (reticleAnimFrameId) cancelAnimationFrame(reticleAnimFrameId);
    reticleAnimFrameId = requestAnimationFrame(draw);
}

// -----------------------------------------------------------------------------
// Telemetry Status Poller (Used when in Host Pi Camera Mode)
// -----------------------------------------------------------------------------
async function pollRealtimeStatus() {
    try {
        const res = await fetch('/api/realtime/status');
        const data = await res.json();

        if (data.success && data.telemetry) {
            const tel = data.telemetry;
            const topCondEl = document.getElementById('hudTopCondition');
            const topConfEl = document.getElementById('hudTopConfidence');
            const progressEl = document.getElementById('hudProgressBar');
            const subInfoEl = document.getElementById('hudSubInfo');

            if (tel.predictions && tel.predictions.length > 0) {
                const top = tel.predictions[0];
                const condClean = top.condition.replace(/_/g, ' ');
                const confPct = top.ai_confidence_pct;

                if (topCondEl) topCondEl.textContent = condClean;
                if (topConfEl) topConfEl.textContent = `${confPct}%`;
                if (progressEl) progressEl.style.width = `${Math.min(100, confPct)}%`;
                if (subInfoEl) subInfoEl.innerHTML = `<span style="color:#34d399;"><i class="fa-solid fa-circle-check"></i> Skin In Focus</span> • ${top.contagious} • ${top.severity}`;
            } else {
                if (topCondEl) topCondEl.textContent = tel.status || "Scanning skin lesion...";
                if (topConfEl) topConfEl.textContent = "--%";
                if (progressEl) progressEl.style.width = '0%';
                if (subInfoEl) subInfoEl.textContent = "Hold camera 4-8 inches from affected skin area";
            }
        }
    } catch (e) { }
}

// -----------------------------------------------------------------------------
// Unified Shutter Snapshot Capture & Instant AI Diagnosis
// -----------------------------------------------------------------------------
async function captureRealtimeSnapshot() {
    // 1. Shutter Flash Animation Effect
    const flashEl = document.getElementById('cameraShutterFlash');
    if (flashEl) {
        flashEl.classList.remove('hidden');
        flashEl.classList.add('flash-active');
        setTimeout(() => {
            flashEl.classList.remove('flash-active');
            flashEl.classList.add('hidden');
        }, 120);
    }

    try {
        const video = document.getElementById('clientWebcamVideo');
        if (!video || !video.videoWidth || !video.videoHeight) {
            alert('No active webcam video stream to capture. Please allow camera permissions or tap Direct Snap.');
            return;
        }

        const canvas = document.getElementById('captureCanvas') || document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        const imageBase64 = canvas.toDataURL('image/jpeg', 0.92);

        // Send base64 frame to 100% AI examination endpoint
        const formData = new FormData();
        const symptomsText = document.getElementById('associated_symptoms') ?
            document.getElementById('associated_symptoms').value.trim() : '';
        formData.append('associated_symptoms', symptomsText);
        formData.append('image_base64', imageBase64);

        const res = await fetch('/api/examine', { method: 'POST', body: formData });
        const data = await res.json();

        if (!data.success && data.error_type === 'quality_rejection') {
            alert('⚠️ ' + data.message);
            return;
        }

        if (data.success) {
            closeCameraFullscreen();

            currentImageFilename = data.image_filename;
            currentImageBase64 = null;
            selectedImageFile = null;
            currentAiResults = data.top_matches;

            const preview = document.getElementById('imagePreview');
            const previewBox = document.getElementById('imagePreviewBox');
            const dropzone = document.getElementById('imageDropzone');

            if (preview) preview.src = data.image_url;
            if (previewBox) { previewBox.classList.remove('hidden'); previewBox.style.display = ''; }
            if (dropzone) { dropzone.classList.add('hidden'); dropzone.style.display = 'none'; }

            renderAiResultsTable(data.top_matches);

            if (data.suggestions) {
                const set = (id, val) => { const el = document.getElementById(id); if (el) el.value = val || ''; };
                set('primary_diagnosis', data.suggestions.primary_diagnosis);
                set('ddx_1', data.suggestions.ddx_1);
                set('ddx_2', data.suggestions.ddx_2);
                set('ddx_3', data.suggestions.ddx_3);
            }

            alert(`✅ Snapshot diagnosed in ${data.inference_time_ms} ms!\nTop AI Match: ${data.suggestions.primary_diagnosis} (${data.top_score}%)`);
        } else {
            alert('Capture Error: ' + (data.message || 'Unknown error'));
        }
    } catch (e) {
        alert('Webcam Capture Failed: ' + e);
    }
}

// -----------------------------------------------------------------------------
// Image File Loader & Stage 5 Analysis
// -----------------------------------------------------------------------------
function loadImageFile(file) {
    if (!file || !file.type.startsWith('image/')) {
        alert('Please select a valid image file (JPG, PNG, WEBP).');
        return;
    }
    selectedImageFile = file;
    currentImageBase64 = null;
    currentImageFilename = null;

    const reader = new FileReader();
    reader.onload = (e) => {
        const preview = document.getElementById('imagePreview');
        const previewBox = document.getElementById('imagePreviewBox');
        const dropzone = document.getElementById('imageDropzone');
        if (preview) preview.src = e.target.result;
        if (previewBox) {
            previewBox.classList.remove('hidden');
            previewBox.style.display = '';
        }
        if (dropzone) {
            dropzone.classList.add('hidden');
            dropzone.style.display = 'none';
        }
    };
    reader.readAsDataURL(file);
}

function handleFileSelect(event) {
    const file = event.target.files && event.target.files[0];
    if (file) loadImageFile(file);
}

function clearImageSelection() {
    selectedImageFile = null;
    currentImageBase64 = null;
    currentImageFilename = null;

    const preview = document.getElementById('imagePreview');
    const previewBox = document.getElementById('imagePreviewBox');
    const dropzone = document.getElementById('imageDropzone');
    const fileInput = document.getElementById('imageFileInput');

    if (preview) preview.src = '';
    if (previewBox) { previewBox.classList.add('hidden'); previewBox.style.display = 'none'; }
    if (dropzone) { dropzone.classList.remove('hidden'); dropzone.style.display = ''; }
    if (fileInput) fileInput.value = '';
}

// -----------------------------------------------------------------------------
// 100% AI Vision Execution
// -----------------------------------------------------------------------------
async function executeAiAnalysis() {
    if (!selectedImageFile && !currentImageBase64 && !currentImageFilename) {
        alert('Please capture a camera frame or upload an image file first.');
        return;
    }

    const runBtn = document.querySelector('#wizardStep5 .btn-accent');
    if (runBtn) {
        runBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Running 100% AI Computer Vision...';
        runBtn.disabled = true;
    }

    try {
        const formData = new FormData();
        const symptomsText = document.getElementById('associated_symptoms') ?
            document.getElementById('associated_symptoms').value.trim() : '';
        formData.append('associated_symptoms', symptomsText);

        if (selectedImageFile) {
            formData.append('image_file', selectedImageFile);
        } else if (currentImageBase64) {
            formData.append('image_base64', currentImageBase64);
        } else if (currentImageFilename) {
            formData.append('image_filename', currentImageFilename);
        }

        const res = await fetch('/api/examine', { method: 'POST', body: formData });
        const data = await res.json();

        const noSkinBanner = document.getElementById('noSkinBanner');
        const redFlagsBanner = document.getElementById('redFlagsBanner');
        const redFlagsList = document.getElementById('redFlagsList');
        const lowMatchBanner = document.getElementById('lowMatchBanner');

        if (!data.success && data.error_type === "quality_rejection") {
            if (noSkinBanner) {
                const msgEl = document.getElementById('noSkinMessage');
                if (msgEl) msgEl.innerText = data.message;
                noSkinBanner.classList.remove('hidden');
            }
            alert('🛑 ' + data.message);
            return;
        }

        if (data.success) {
            if (noSkinBanner) noSkinBanner.classList.add('hidden');
            currentImageFilename = data.image_filename;
            currentAiResults = data.top_matches;

            // Red flags
            if (data.red_flags && data.red_flags.length > 0) {
                if (redFlagsList) redFlagsList.innerHTML = data.red_flags.map(f => `<li>${f}</li>`).join('');
                if (redFlagsBanner) redFlagsBanner.classList.remove('hidden');
            } else {
                if (redFlagsBanner) redFlagsBanner.classList.add('hidden');
            }

            // Low confidence banner
            if (lowMatchBanner) {
                if (data.is_low_confidence) lowMatchBanner.classList.remove('hidden');
                else lowMatchBanner.classList.add('hidden');
            }

            // Render table
            renderAiResultsTable(data.top_matches);

            // Auto-fill diagnosis
            if (data.suggestions) {
                const set = (id, val) => { const el = document.getElementById(id); if (el) el.value = val || ''; };
                set('primary_diagnosis', data.suggestions.primary_diagnosis);
                set('ddx_1', data.suggestions.ddx_1);
                set('ddx_2', data.suggestions.ddx_2);
                set('ddx_3', data.suggestions.ddx_3);
            }

            alert(`✅ 100% AI Analysis complete in ${data.inference_time_ms} ms! Diagnoses auto-populated.`);
        } else {
            alert('AI Examination Error: ' + (data.message || 'Unknown error'));
        }
    } catch (err) {
        alert('Server request failed: ' + err);
    } finally {
        if (runBtn) {
            runBtn.innerHTML = '<i class="fa-solid fa-brain"></i> Execute 100% AI Vision Analysis';
            runBtn.disabled = false;
        }
    }
}

function renderAiResultsTable(matches) {
    const aiResultsCard = document.getElementById('aiResultsCard');
    const tbody = document.getElementById('resultsTbody');
    if (!tbody) return;

    tbody.innerHTML = '';
    matches.forEach((match, idx) => {
        const condName = match.condition.replace(/_/g, ' ');
        const confPct = match.ai_confidence_pct;
        const contagious = match.contagious || 'Non-Contact';
        const isConfident = confPct >= 20;

        const badge = contagious === 'Contact'
            ? `<span class="badge-contact">⚠ Contact</span>`
            : `<span class="badge-noncontact">✓ Non-Contact</span>`;

        const tr = document.createElement('tr');
        tr.style.opacity = isConfident ? '1' : '0.65';
        tr.innerHTML = `
            <td>#${idx + 1}</td>
            <td>
                <strong>${condName}</strong>
                ${isConfident ? '' : '<span style="color:#fbbf24; font-size:0.75rem; margin-left:4px;">(Low Match)</span>'}
            </td>
            <td><span class="badge ${isConfident ? '' : 'badge-low'}">${confPct}%</span></td>
            <td>${badge}</td>
            <td><small class="text-muted">${match.severity}</small></td>
        `;
        tbody.appendChild(tr);
    });

    if (aiResultsCard) aiResultsCard.classList.remove('hidden');
}

// -----------------------------------------------------------------------------
// Save Patient Record
// -----------------------------------------------------------------------------
async function savePatientRecord() {
    const payload = {
        date_of_assessment: document.getElementById('date_of_assessment').value,
        assessed_by: document.getElementById('assessed_by').value,
        patient_name: document.getElementById('patient_name').value,
        age: parseInt(document.getElementById('age').value) || null,
        sex: document.getElementById('sex').value,
        background: document.getElementById('background').value,
        current_residence: document.getElementById('current_residence').value,
        onset: document.getElementById('onset').value,
        pattern: document.getElementById('pattern').value,
        progression: document.getElementById('progression').value,
        location: document.getElementById('location').value,
        provoking_relieving_factors: document.getElementById('provoking_relieving_factors').value,
        associated_symptoms: document.getElementById('associated_symptoms').value,
        treatment_history: document.getElementById('treatment_history').value,
        past_medical_history: document.getElementById('past_medical_history').value,
        family_history: document.getElementById('family_history').value,
        occupational_hobbies: document.getElementById('occupational_hobbies').value,
        allergies: document.getElementById('allergies').value,
        distribution: document.getElementById('distribution').value,
        color_discoloration: document.getElementById('skin_color_discoloration').value,
        morphology: document.getElementById('morphology').value,
        regional_lymph_nodes: document.getElementById('regional_lymph_nodes').value,
        primary_diagnosis: document.getElementById('primary_diagnosis').value,
        ddx_1: document.getElementById('ddx_1').value,
        ddx_2: document.getElementById('ddx_2').value,
        ddx_3: document.getElementById('ddx_3').value,
        plan_management: document.getElementById('plan_management').value,
        image_filename: currentImageFilename,
        ai_results: currentAiResults
    };

    try {
        const res = await fetch('/api/patients', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const data = await res.json();

        if (data.success) {
            alert(`Patient record #${data.patient_id} saved successfully!`);
            document.getElementById('assessmentForm').reset();
            clearImageSelection();
            const aiResultsCard = document.getElementById('aiResultsCard');
            if (aiResultsCard) aiResultsCard.classList.add('hidden');
            currentAiResults = [];
            goToWizardStep(1);
        } else {
            alert('Save Error: ' + data.message);
        }
    } catch (err) {
        alert('Failed to save record: ' + err);
    }
}

// -----------------------------------------------------------------------------
// Clinical Records Database & Reporting
// -----------------------------------------------------------------------------
async function loadPatientsDashboard() {
    const tbody = document.getElementById('patientsTbody');
    tbody.innerHTML = '<tr><td colspan="7">Loading patient records...</td></tr>';

    try {
        const response = await fetch('/api/patients');
        const data = await response.json();

        if (data.success) {
            tbody.innerHTML = '';
            const statTotal = document.getElementById('statTotalPatients');
            if (statTotal) statTotal.innerText = data.patients.length;

            if (data.patients.length === 0) {
                tbody.innerHTML = '<tr><td colspan="7">No patient assessment records found.</td></tr>';
                return;
            }

            data.patients.forEach(pt => {
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td>#${pt.id}</td>
                    <td>${pt.date_of_assessment || pt.created_at.split(' ')[0]}</td>
                    <td><strong>${pt.patient_name}</strong></td>
                    <td>${pt.age || 'N/A'} / ${pt.sex}</td>
                    <td><span class="badge">${pt.primary_diagnosis || 'Unspecified'}</span></td>
                    <td>${pt.assessed_by || 'Clinician'}</td>
                    <td>
                        <button class="btn btn-sm btn-secondary" onclick="viewPatientReport(${pt.id})"><i class="fa-solid fa-eye"></i> View</button>
                        <button class="btn btn-sm btn-outline" style="color:#ff1744;" onclick="deletePatientRecord(${pt.id})"><i class="fa-solid fa-trash"></i> Delete</button>
                    </td>
                `;
                tbody.appendChild(tr);
            });
        }
    } catch (err) {
        tbody.innerHTML = '<tr><td colspan="7">Failed to load patients.</td></tr>';
    }
}

async function viewPatientReport(patientId) {
    try {
        const response = await fetch(`/api/patients/${patientId}`);
        const data = await response.json();

        if (data.success) {
            const pt = data.patient;
            const imgHtml = pt.image_filename
                ? `<img src="/uploads/${pt.image_filename}" style="max-width:280px; border-radius:12px; border:1px solid rgba(255,255,255,0.15); margin-bottom:1rem;">`
                : '<div style="padding:1.5rem; background:rgba(0,0,0,0.3); border-radius:12px; border:1px solid rgba(255,255,255,0.1); text-align:center; color:var(--text-muted);"><i class="fa-solid fa-camera-slash" style="font-size:2rem; margin-bottom:0.5rem; display:block;"></i>No Rash Image Attached</div>';

            let topMatchesHtml = '';
            if (pt.ai_results && Array.isArray(pt.ai_results) && pt.ai_results.length > 0) {
                const rows = pt.ai_results.map((match, idx) => {
                    const condName = match.condition.replace(/_/g, ' ');
                    const confPct = match.ai_confidence_pct;
                    const contagious = match.contagious || 'Non-Contact';
                    const badge = contagious === 'Contact'
                        ? `<span class="badge-contact">⚠ Contact</span>`
                        : `<span class="badge-noncontact">✓ Non-Contact</span>`;

                    return `
                        <tr>
                            <td>#${idx + 1}</td>
                            <td><strong>${condName}</strong></td>
                            <td><span class="badge">${confPct}%</span></td>
                            <td>${badge}</td>
                            <td><small class="text-muted">${match.severity}</small></td>
                        </tr>
                    `;
                }).join('');

                topMatchesHtml = `
                    <div style="margin-top:1.5rem;">
                        <h4 style="margin-bottom:0.8rem;"><i class="fa-solid fa-brain text-cyan"></i> 100% AI Differential Diagnoses</h4>
                        <div class="table-responsive">
                            <table class="results-table clinical-table">
                                <thead>
                                    <tr>
                                        <th>Rank</th>
                                        <th>Condition</th>
                                        <th>Accuracy Level</th>
                                        <th>Transmission</th>
                                        <th>Severity</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    ${rows}
                                </tbody>
                            </table>
                        </div>
                    </div>
                `;
            }

            const modalContent = document.getElementById('patientReportContent');
            modalContent.innerHTML = `
                <div class="grid-2col" style="margin-bottom:1.5rem; gap:1.5rem;">
                    <div>
                        <h3 style="color:var(--text-primary); margin-bottom:0.5rem;"><i class="fa-solid fa-user-doctor text-cyan"></i> ${pt.patient_name}</h3>
                        <p><strong>Demographics:</strong> ${pt.age || 'N/A'} Years Old | Biological ${pt.sex} | ${pt.background || 'Unspecified Ethnicity'}</p>
                        <p><strong>Assessment Date:</strong> ${pt.date_of_assessment} | <strong>Assessed By:</strong> ${pt.assessed_by || 'Clinician'}</p>
                        <p style="margin-top:0.6rem;"><strong>Primary Diagnosis:</strong> <span class="badge badge-clinical" style="font-size:0.95rem;">${pt.primary_diagnosis || 'Unspecified'}</span></p>
                        <p><strong>Differential Diagnoses:</strong> 1. ${pt.ddx_1 || 'N/A'} | 2. ${pt.ddx_2 || 'N/A'} | 3. ${pt.ddx_3 || 'N/A'}</p>
                    </div>
                    <div>
                        ${imgHtml}
                    </div>
                </div>

                <div class="glass-card medical-card" style="margin-bottom:1rem; padding:1.2rem;">
                    <h4 style="margin-bottom:0.6rem; color:var(--primary-cyan);"><i class="fa-solid fa-notes-medical"></i> Clinical History & Symptoms</h4>
                    <div class="grid-2col-gap">
                        <div>
                            <p><strong>Onset & Course:</strong> ${pt.onset || 'N/A'} / ${pt.pattern || 'N/A'} (${pt.progression || 'N/A'})</p>
                            <p><strong>Anatomical Location:</strong> ${pt.location || 'N/A'}</p>
                            <p><strong>Associated Symptoms:</strong> ${pt.associated_symptoms || 'N/A'}</p>
                        </div>
                        <div>
                            <p><strong>Past Medical History:</strong> ${pt.past_medical_history || 'None'}</p>
                            <p><strong>Prior Treatments:</strong> ${pt.treatment_history || 'None'}</p>
                            <p><strong>Allergies:</strong> ${pt.allergies || 'NKDA'}</p>
                        </div>
                    </div>
                </div>

                ${topMatchesHtml}

                <div class="glass-card medical-card" style="margin-top:1.5rem; padding:1.2rem;">
                    <h4 style="margin-bottom:0.6rem; color:var(--success-color);"><i class="fa-solid fa-clipboard-check"></i> Clinical Management Strategy</h4>
                    <p>${pt.plan_management || 'Standard outpatient follow-up and monitoring.'}</p>
                </div>
            `;

            const modal = document.getElementById('patientReportModal');
            if (modal) {
                modal.classList.remove('hidden');
                modal.style.display = 'flex';
            }
        }
    } catch (err) {
        alert('Could not retrieve report: ' + err);
    }
}

function closeReportModal() {
    const modal = document.getElementById('patientReportModal');
    if (modal) {
        modal.classList.add('hidden');
        modal.style.display = 'none';
    }
}

async function deletePatientRecord(patientId) {
    if (confirm(`Are you sure you want to delete patient record #${patientId}?`)) {
        await fetch(`/api/patients/${patientId}`, { method: 'DELETE' });
        loadPatientsDashboard();
    }
}

// ------------------------------------------------------------------------------
// Multi-Model Preset Management & Hub / Header UI Sync
// ------------------------------------------------------------------------------
function updateModelLabelsUI(presetKey, numClasses, activeName) {
    const presetLabels = {
        'current': `Current (${numClasses || 12})`,
        'rash-22': `Rash-22 (${numClasses || 22})`,
        'rash-50': `Rash-50 (${numClasses || 50})`
    };
    const labelText = presetLabels[presetKey] || `${activeName || presetKey} (${numClasses || 10})`;

    // Update header label
    const headerLabel = document.getElementById('headerModelLabel');
    if (headerLabel) {
        headerLabel.innerText = labelText;
    }

    // Update assessment tab status text
    const assessmentLabel = document.getElementById('assessmentActiveModelText');
    if (assessmentLabel) {
        assessmentLabel.innerText = `${labelText} Conditions`;
    }

    // Highlight active card on hub
    document.querySelectorAll('.model-card').forEach(card => card.classList.remove('active'));
    const targetCard = document.getElementById(`landingCard_${presetKey}`);
    if (targetCard) {
        targetCard.classList.add('active');
    }

    // Highlight active item in dropdown
    document.querySelectorAll('.model-dropdown-item').forEach(item => {
        if (item.getAttribute('data-preset') === presetKey) {
            item.classList.add('active');
        } else {
            item.classList.remove('active');
        }
    });

    // Update hub status bar
    const statusText = document.getElementById('modelSwitchStatus');
    if (statusText) {
        statusText.innerHTML = `<span style="color: var(--success-color); font-weight: 600;"><i class="fa-solid fa-circle-check"></i> Active AI Model Preset: <strong>${labelText}</strong></span>`;
    }
}

async function fetchActiveModelInfo() {
    try {
        const res = await fetch('/api/model/active');
        const data = await res.json();
        if (data.success) {
            updateModelLabelsUI(data.active_preset, data.num_classes, data.active_name);
        }
    } catch (e) {
        console.warn('Failed to fetch model info:', e);
    }
}

async function selectLandingModel(presetKey) {
    await switchActiveModelUI(presetKey);
}

async function switchActiveModelUI(presetKey) {
    try {
        const res = await fetch('/api/model/switch', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ preset: presetKey })
        });
        const data = await res.json();
        if (data.success) {
            updateModelLabelsUI(presetKey, data.num_classes, data.active_name);
        } else {
            alert(data.message || 'Failed to switch model');
        }
    } catch (e) {
        alert('Error switching AI model: ' + e.message);
    }
}

function toggleModelDropdown(e) {
    if (e) e.stopPropagation();
    const menu = document.getElementById('modelDropdownMenu');
    if (menu) {
        menu.classList.toggle('show');
    }
}

function closeModelDropdown() {
    const menu = document.getElementById('modelDropdownMenu');
    if (menu) {
        menu.classList.remove('show');
    }
}

async function switchModelFromHeader(presetKey) {
    closeModelDropdown();
    await switchActiveModelUI(presetKey);
}

// Close model dropdown on document click outside
document.addEventListener('click', (e) => {
    const dropdown = document.getElementById('modelSwitcherDropdown');
    if (dropdown && !dropdown.contains(e.target)) {
        closeModelDropdown();
    }
});
