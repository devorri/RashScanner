/**
 * main.js - Client-side Application Logic for Local Multimodal Rash Scanner
 */

let currentImageSource = 'upload';
let selectedImageFile = null;
let currentImageBase64 = null;
let currentImageFilename = null;
let currentAiResults = [];

// Initialize Page Defaults & Splash Loading Animation
document.addEventListener('DOMContentLoaded', () => {
    // Set default assessment date to today
    const dateInput = document.getElementById('date_of_assessment');
    if (dateInput) {
        dateInput.value = new Date().toISOString().split('T')[0];
    }

    startSplashAnimation();

    // Start 10,000mAh Power Bank Battery Monitor
    updateBatteryStatus();
    if (!batteryPollInterval) {
        batteryPollInterval = setInterval(updateBatteryStatus, 15000);
    }

    // --- Drag & Drop event wiring for Stage 5 dropzone ---
    const dropzone = document.getElementById('imageDropzone');
    if (dropzone) {
        dropzone.addEventListener('dragover', (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.add('dropzone-hover');
        });
        dropzone.addEventListener('dragleave', (e) => {
            e.preventDefault();
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
            statusText.innerText = "Loading 50/50 Multimodal Vision & Clinical Matcher Engine...";
        } else if (progress === 80) {
            statusText.innerText = "Edge Neural Model Ready • System Operational.";
        } else if (progress >= 100) {
            clearInterval(interval);
            statusText.innerHTML = "<strong style='color: var(--success-color);'><i class='fa-solid fa-circle-check'></i> Edge Rashilience Engine Ready!</strong>";
            actionArea.classList.remove('hidden');
        }
    }, 150);
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
        console.log('[Battery] Using local tracker');
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
            alert("Power bank battery tracker reset to 100% (~6.0 hours remaining)!");
            closeBatteryModal();
        }
    } catch (e) {
        applyBatteryUI(100, 6.0, false);
        closeBatteryModal();
    }
}

// -----------------------------------------------------------------------------
// Kiosk Maintenance Mode (Password-Protected Exit/Minimize/Reboot)
// -----------------------------------------------------------------------------
function openMaintenanceModal() {
    const modal = document.getElementById('maintenanceModal');
    const authPanel = document.getElementById('maintenanceAuth');
    const actionsPanel = document.getElementById('maintenanceActions');
    const errBox = document.getElementById('maintenanceAuthError');
    const pwInput = document.getElementById('maintenancePassword');

    if (modal) {
        // Reset to password step
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
            // Show maintenance actions
            const authPanel = document.getElementById('maintenanceAuth');
            const actionsPanel = document.getElementById('maintenanceActions');
            if (authPanel) { authPanel.classList.add('hidden'); authPanel.style.display = 'none'; }
            if (actionsPanel) { actionsPanel.classList.remove('hidden'); actionsPanel.style.display = ''; }
            if (errBox) errBox.classList.add('hidden');
        } else {
            if (errBox) {
                errBox.textContent = 'Incorrect admin password. Access denied.';
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

async function exitKioskMode() {
    if (!confirm('Exit Kiosk Mode?\n\nChromium will close and you will see the Pi desktop.\nThe Flask server keeps running. Re-open Chromium or reboot to restart the kiosk.')) return;

    try {
        await fetch('/api/system/exit-kiosk', { method: 'POST' });
    } catch (e) {
        // Expected — the browser itself is being killed
    }
}

function minimizeKiosk() {
    // Try exiting fullscreen via Fullscreen API
    if (document.fullscreenElement) {
        document.exitFullscreen().catch(() => {});
    } else if (document.webkitFullscreenElement) {
        document.webkitExitFullscreen();
    }

    // Try to make window not full-size (works on some kiosk configs)
    try {
        window.resizeTo(800, 480);
        window.moveTo(50, 50);
    } catch (e) {}

    alert('Exited fullscreen mode. You may need to press F11 or Alt+Tab to access the desktop.');
}

async function rebootKiosk() {
    if (!confirm('Reboot the Raspberry Pi?\n\nThe system will shut down and restart. This takes about 30 seconds.')) return;

    try {
        await fetch('/api/system/restart-kiosk', { method: 'POST' });
    } catch (e) {}
}

function proceedFromSplash() {
    showScreen('loginScreen');
}

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
            alertBox.innerText = data.error || "Invalid username or password";
            alertBox.classList.remove('hidden');
        }
    } catch (err) {
        alertBox.innerText = "Connection error. Login as Guest to test.";
        alertBox.classList.remove('hidden');
    }
}

function bypassLoginAsGuest() {
    showScreen('hubScreen');
}

function navigateToApp(tabName) {
    showScreen('appContainer');
    switchTab(tabName);
    if (tabName === 'assessment') {
        goToWizardStep(1);
    }
}

function returnToHub() {
    showScreen('hubScreen');
}

function logoutToLogin() {
    showScreen('loginScreen');
}

// 5-Stage Stepper Wizard Navigation
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


// Tab Switching inside App Container
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

// Authentication Modal
function toggleAuthModal() {
    const modal = document.getElementById('loginModal');
    modal.classList.toggle('hidden');
}

async function performAdminLogin() {
    const usernameInput = document.getElementById('loginUsername').value;
    const passwordInput = document.getElementById('loginPassword').value;
    const alertBox = document.getElementById('loginAlert');

    try {
        const response = await fetch('/api/login', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username: usernameInput, password: passwordInput })
        });

        const data = await response.json();
        if (data.success) {
            document.getElementById('adminBadge').classList.remove('hidden');
            document.getElementById('authBtn').innerHTML = '<i class="fa-solid fa-right-from-bracket"></i> Logout';
            document.getElementById('authBtn').onclick = performLogout;
            toggleAuthModal();
            alert('Admin login successful!');
        } else {
            alertBox.innerText = data.message;
            alertBox.classList.remove('hidden');
        }
    } catch (err) {
        alertBox.innerText = 'Login server connection failed.';
        alertBox.classList.remove('hidden');
    }
}

async function performLogout() {
    await fetch('/api/logout', { method: 'POST' });
    document.getElementById('adminBadge').classList.add('hidden');
    document.getElementById('authBtn').innerHTML = '<i class="fa-solid fa-right-to-bracket"></i> Admin Login';
    document.getElementById('authBtn').onclick = toggleAuthModal;
    alert('Logged out.');
}

// ============================================================
// Stage 5: Image Mode Toggle (Upload vs Pi Camera)
// ============================================================
function setImageMode(mode) {
    currentImageSource = mode;
    document.getElementById('modeUploadBtn').classList.toggle('active', mode === 'upload');
    document.getElementById('modeCameraBtn').classList.toggle('active', mode === 'camera');

    const uploadContainer = document.getElementById('uploadContainer');
    const cameraContainer = document.getElementById('cameraContainer');

    if (mode === 'upload') {
        uploadContainer.classList.remove('hidden');
        uploadContainer.style.display = '';
        cameraContainer.classList.add('hidden');
        cameraContainer.style.display = 'none';
    } else {
        cameraContainer.classList.remove('hidden');
        cameraContainer.style.display = '';
        uploadContainer.classList.add('hidden');
        uploadContainer.style.display = 'none';
        startCameraFeed();
    }
}

// Core image loader — shared by file input onchange AND drag-drop
function loadImageFile(file) {
    if (!file || !file.type.startsWith('image/')) {
        alert('Please select a valid image file (JPG, PNG, WEBP).');
        return;
    }
    selectedImageFile = file;
    currentImageBase64 = null;

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

// Called by <input type="file" onchange="handleFileSelect(event)">
function handleFileSelect(event) {
    const file = event.target.files && event.target.files[0];
    if (file) loadImageFile(file);
}

// Clear selected image — show dropzone again
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

// Start webcam / Pi camera feed
let cameraStream = null;
async function startCameraFeed() {
    const video = document.getElementById('webcamFeed');
    if (!video) return;
    try {
        if (cameraStream) {
            cameraStream.getTracks().forEach(t => t.stop());
        }
        cameraStream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = cameraStream;
    } catch (err) {
        // Fallback to Pi Camera API
        try {
            const res = await fetch('/api/camera/snap');
            const data = await res.json();
            if (data.success) {
                currentImageBase64 = data.base64;
                currentImageFilename = data.filename;
                selectedImageFile = null;
                const preview = document.getElementById('imagePreview');
                const previewBox = document.getElementById('imagePreviewBox');
                const dropzone = document.getElementById('imageDropzone');
                if (preview) preview.src = data.base64;
                if (previewBox) { previewBox.classList.remove('hidden'); previewBox.style.display = ''; }
                if (dropzone) { dropzone.classList.add('hidden'); dropzone.style.display = 'none'; }
                alert('Pi Camera snapshot captured!');
            } else {
                alert('Camera Error: ' + data.message);
            }
        } catch (e) {
            alert('Could not access camera. Error: ' + e);
        }
    }
}

// Capture snapshot from webcam video element
function captureSnapshot() {
    const video = document.getElementById('webcamFeed');
    const canvas = document.getElementById('snapshotCanvas');
    if (!video || !canvas) return;

    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    canvas.getContext('2d').drawImage(video, 0, 0);

    const dataUrl = canvas.toDataURL('image/jpeg', 0.9);
    currentImageBase64 = dataUrl;
    selectedImageFile = null;

    const preview = document.getElementById('imagePreview');
    const previewBox = document.getElementById('imagePreviewBox');
    const dropzone = document.getElementById('imageDropzone');
    if (preview) preview.src = dataUrl;
    if (previewBox) { previewBox.classList.remove('hidden'); previewBox.style.display = ''; }
    if (dropzone) { dropzone.classList.add('hidden'); dropzone.style.display = 'none'; }

    // Stop camera after capture
    if (cameraStream) cameraStream.getTracks().forEach(t => t.stop());
    alert('Snapshot captured! Ready for AI analysis.');
}

// ============================================================
// Execute 50/50 AI Multimodal Fusion Examination
// ============================================================
async function executeAiAnalysis() {
    const symptomsText = document.getElementById('associated_symptoms') ?
        document.getElementById('associated_symptoms').value.trim() : '';

    if (!selectedImageFile && !currentImageBase64 && !currentImageFilename) {
        alert('Please upload or capture a rash image before running the AI analysis.');
        return;
    }

    const runBtn = document.querySelector('#wizardStep5 .btn-accent');
    if (runBtn) {
        runBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Running Multimodal Fusion Engine...';
        runBtn.disabled = true;
    }

    try {
        const formData = new FormData();

        // Auto-extract clinical keywords from ALL wizard stages for richer symptom fusion
        const fieldsToPull = [
            'associated_symptoms',       // Stage 3: explicit symptoms
            'onset',                     // Stage 2: Sudden/Acute vs Gradual/Chronic
            'pattern',                   // Stage 2: Constant vs Episodic
            'progression',               // Stage 2: Improving/Worsening/Static
            'location',                  // Stage 2: Anatomical location
            'provoking_relieving_factors',// Stage 2: Triggers
            'treatment_history',         // Stage 3: Prior treatments
            'past_medical_history',      // Stage 3: PMH
            'family_history',            // Stage 3: Family
            'drug_history',              // Stage 3: Drugs
            'allergies',                 // Stage 3: Allergies
            'distribution',              // Stage 4: Lesion distribution
            'skin_color_discoloration',  // Stage 4: Color
            'morphology',                // Stage 4: Primary morphology
            'lymph_nodes'                // Stage 4: Lymph nodes
        ];

        const allKeywords = [];
        fieldsToPull.forEach(fieldId => {
            const el = document.getElementById(fieldId);
            if (el) {
                const val = (el.value || '').trim();
                if (val && val !== 'No') allKeywords.push(val);
            }
        });

        const combinedSymptoms = allKeywords.join(', ');
        if (combinedSymptoms) formData.append('associated_symptoms', combinedSymptoms);

        if (selectedImageFile) {
            formData.append('image_file', selectedImageFile);
        } else if (currentImageBase64) {
            formData.append('image_base64', currentImageBase64);
        } else if (currentImageFilename) {
            formData.append('image_filename', currentImageFilename);
        }

        const response = await fetch('/api/examine', { method: 'POST', body: formData });
        const data = await response.json();

        const noSkinBanner = document.getElementById('noSkinBanner');
        const qualityBanner = document.getElementById('qualityWarningBanner');
        const qualityList = document.getElementById('qualityWarningsList');
        const redFlagsBanner = document.getElementById('redFlagsBanner');
        const redFlagsList = document.getElementById('redFlagsList');
        const lowMatchBanner = document.getElementById('lowMatchBanner');
        const aiResultsCard = document.getElementById('aiResultsCard');

        // Handle Rejection: Quality Gate (No Skin / Blurry / Bad Lighting)
        if (!data.success && data.error_type === "quality_rejection") {
            if (noSkinBanner) {
                const msgEl = document.getElementById('noSkinMessage');
                if (msgEl) msgEl.innerText = data.message;
                noSkinBanner.classList.remove('hidden');
            }
            if (qualityBanner) qualityBanner.classList.add('hidden');
            if (redFlagsBanner) redFlagsBanner.classList.add('hidden');
            if (lowMatchBanner) lowMatchBanner.classList.add('hidden');
            if (aiResultsCard) aiResultsCard.classList.add('hidden');

            alert('🛑 ' + data.message);
            return;
        }

        if (data.success) {
            if (noSkinBanner) noSkinBanner.classList.add('hidden');
            currentImageFilename = data.image_filename;
            currentAiResults = data.top_matches;

            // Quality / Blur Warnings
            if (data.quality && data.quality.warnings && data.quality.warnings.length > 0) {
                if (qualityList) {
                    qualityList.innerHTML = data.quality.warnings.map(w => `<li>${w}</li>`).join('');
                }
                if (qualityBanner) qualityBanner.classList.remove('hidden');
            } else {
                if (qualityBanner) qualityBanner.classList.add('hidden');
            }

            // Red flags banner
            if (data.red_flags && data.red_flags.length > 0) {
                if (redFlagsList) {
                    redFlagsList.innerHTML = data.red_flags.map(f => `<li>${f}</li>`).join('');
                }
                if (redFlagsBanner) redFlagsBanner.classList.remove('hidden');
            } else {
                if (redFlagsBanner) redFlagsBanner.classList.add('hidden');
            }

            // Low Match / Inconclusive Banner
            if (lowMatchBanner) {
                if (data.is_low_confidence) {
                    lowMatchBanner.innerHTML = `
                        <h4 style="color: #fbbf24; margin: 0 0 6px 0;">
                            <i class="fa-solid fa-triangle-exclamation"></i> Inconclusive Scan (Max Match: ${data.top_score}%)
                        </h4>
                        <p style="margin: 0 0 8px 0; font-size: 0.9rem; color: #f3f4f6;">
                            The AI visual score is too low (${data.top_score}% < 25% threshold) to identify a specific rash pattern. This is normal when scanning normal/healthy skin, an unindexed rash, or when no symptoms are provided in Stages 2–4.
                        </p>
                        <small style="color: #67e8f9;"><i class="fa-solid fa-lightbulb"></i> Tip: Enter clinical keywords (e.g. "itchy, scaly, red bumps") in Stage 3 to guide the AI match.</small>
                    `;
                    lowMatchBanner.classList.remove('hidden');
                } else {
                    lowMatchBanner.classList.add('hidden');
                }
            }

            // Top 10 results table
            const tbody = document.getElementById('resultsTbody');
            if (tbody) {
                tbody.innerHTML = '';
                data.top_matches.forEach((match, idx) => {
                    const condName = match.condition.replace(/_/g, ' ');
                    const matchPct = (match.final_score * 100).toFixed(1);
                    const vPct = (match.visual_score * 100).toFixed(1);
                    const sPct = (match.symptom_score * 100).toFixed(1);
                    const contagious = match.contagious || 'Unknown';
                    const isConfident = match.final_score >= 0.25;
                    const badge = contagious === 'Contact'
                        ? `<span class="badge-contact">⚠ Contact</span>`
                        : contagious === 'Non-Contact'
                            ? `<span class="badge-noncontact">✓ Non-Contact</span>`
                            : `<span class="badge-unknown">? Unknown</span>`;

                    const tr = document.createElement('tr');
                    tr.style.opacity = isConfident ? '1' : '0.65';
                    tr.innerHTML = `
                        <td>#${idx + 1}</td>
                        <td>
                            <strong>${condName}</strong>
                            ${isConfident ? '' : '<span style="color:#fbbf24; font-size:0.75rem; margin-left:4px;">(Low Match)</span>'}<br>
                            <small class="text-muted">${match.severity}</small>
                        </td>
                        <td><span class="badge ${isConfident ? '' : 'badge-low'}">${matchPct}%</span></td>
                        <td>${vPct}%</td>
                        <td>${sPct}%</td>
                        <td>${badge}</td>`;
                    tbody.appendChild(tr);
                });
            }

            if (aiResultsCard) aiResultsCard.classList.remove('hidden');

            // Auto-populate diagnosis fields
            if (data.suggestions) {
                const set = (id, val) => { const el = document.getElementById(id); if (el) el.value = val || ''; };
                set('primary_diagnosis', data.suggestions.primary_diagnosis);
                set('ddx_1', data.suggestions.ddx_1);
                set('ddx_2', data.suggestions.ddx_2);
                set('ddx_3', data.suggestions.ddx_3);
            }

            if (data.is_low_confidence) {
                alert(`⚠️ Inconclusive Scan (Top match only ${data.top_score}%).\n\nThe AI does not recognize a clear rash in this photo alone. Enter symptoms in Stage 3 for an accurate diagnosis!`);
            } else {
                alert(`✅ AI Analysis complete in ${data.inference_time_ms} ms! Top diagnoses auto-populated.`);
            }
        } else {
            alert('AI Examination Error: ' + (data.message || 'Unknown error'));
        }
    } catch (err) {
        alert('Server request failed: ' + err);
    } finally {
        if (runBtn) {
            runBtn.innerHTML = '<i class="fa-solid fa-brain"></i> Execute 50/50 AI Multimodal Fusion';
            runBtn.disabled = false;
        }
    }
}

// Legacy alias — kept for backwards compatibility
async function runAiExamination() { return executeAiAnalysis(); }

// Save Patient Record
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
        travel: document.getElementById('travel').value,
        drug_history: document.getElementById('drug_history').value,
        smoking_alcohol: document.getElementById('smoking_alcohol').value,
        allergies: document.getElementById('allergies').value,
        psychological_social: document.getElementById('psychological_social').value,
        distribution: document.getElementById('distribution').value,
        color_discoloration: document.getElementById('color_discoloration').value,
        morphology: document.getElementById('morphology').value,
        regional_lymph_nodes: document.getElementById('regional_lymph_nodes').value,
        primary_diagnosis: document.getElementById('primary_diagnosis').value,
        ddx_1: document.getElementById('ddx_1').value,
        ddx_2: document.getElementById('ddx_2').value,
        ddx_3: document.getElementById('ddx_3').value,
        plan_investigations: document.getElementById('plan_investigations').value,
        plan_management: document.getElementById('plan_management').value,
        plan_referral: document.getElementById('plan_referral').value,
        image_filename: currentImageFilename,
        ai_results: currentAiResults
    };

    try {
        const response = await fetch('/api/patients', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const data = await response.json();
        if (data.success) {
            alert(`Patient record #${data.patient_id} saved successfully!`);
            document.getElementById('assessmentForm').reset();
            // Reset wizard image upload state
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

// Load Admin Dashboard Patients Table
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
        tbody.innerHTML = '<tr><td colspan="7">Failed to load patients from database.</td></tr>';
    }
}

// Filter Patients Table
function filterPatientsTable() {
    const query = document.getElementById('searchInput').value.toLowerCase();
    const rows = document.querySelectorAll('#patientsTbody tr');

    rows.forEach(row => {
        const text = row.innerText.toLowerCase();
        row.style.display = text.includes(query) ? '' : 'none';
    });
}

// View Full Patient Report Modal
async function viewPatientReport(patientId) {
    try {
        const response = await fetch(`/api/patients/${patientId}`);
        const data = await response.json();

        if (data.success) {
            const pt = data.patient;
            const imgHtml = pt.image_filename
                ? `<img src="/uploads/${pt.image_filename}" style="max-width:280px; border-radius:12px; border:1px solid rgba(255,255,255,0.15); margin-bottom:1rem;">`
                : '<div style="padding:1.5rem; background:rgba(0,0,0,0.3); border-radius:12px; border:1px solid rgba(255,255,255,0.1); text-align:center; color:var(--text-muted);"><i class="fa-solid fa-camera-slash" style="font-size:2rem; margin-bottom:0.5rem; display:block;"></i>No Rash Image Attached</div>';

            // Parse AI results
            let topMatchesHtml = '';
            if (pt.ai_results && Array.isArray(pt.ai_results) && pt.ai_results.length > 0) {
                const rows = pt.ai_results.map((match, idx) => {
                    const condName = match.condition.replace(/_/g, ' ');
                    const matchPct = (match.final_score * 100).toFixed(1);
                    const vPct = (match.visual_score * 100).toFixed(1);
                    const sPct = (match.symptom_score * 100).toFixed(1);
                    const contagious = match.contagious || 'Unknown';
                    const badge = contagious === 'Contact'
                        ? `<span class="badge-contact">⚠ Contact</span>`
                        : contagious === 'Non-Contact'
                            ? `<span class="badge-noncontact">✓ Non-Contact</span>`
                            : `<span class="badge-unknown">? Unknown</span>`;

                    return `
                        <tr>
                            <td>#${idx + 1}</td>
                            <td><strong>${condName}</strong> <br><small class="text-muted">${match.severity}</small></td>
                            <td><span class="badge">${matchPct}%</span></td>
                            <td>${vPct}%</td>
                            <td>${sPct}%</td>
                            <td>${badge}</td>
                        </tr>
                    `;
                }).join('');

                topMatchesHtml = `
                    <div style="margin-top:1.5rem;">
                        <h4 style="margin-bottom:0.8rem;"><i class="fa-solid fa-brain text-cyan"></i> Top 10 Multimodal AI Differential Diagnoses</h4>
                        <div class="table-responsive">
                            <table class="results-table clinical-table">
                                <thead>
                                    <tr>
                                        <th>Rank</th>
                                        <th>Condition</th>
                                        <th>Match %</th>
                                        <th>Vision %</th>
                                        <th>Symptom %</th>
                                        <th>Transmission</th>
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
                    <h4 style="margin-bottom:0.6rem; color:var(--success-color);"><i class="fa-solid fa-clipboard-check"></i> Clinical Management & Treatment Strategy</h4>
                    <p>${pt.plan_management || pt.plan_investigations || 'Standard outpatient follow-up and symptom monitoring.'}</p>
                </div>
            `;

            document.getElementById('patientReportModal').classList.remove('hidden');
        }
    } catch (err) {
        alert('Could not retrieve report: ' + err);
    }
}


function closeReportModal() {
    document.getElementById('patientReportModal').classList.add('hidden');
}

// Delete Patient Record
async function deletePatientRecord(patientId) {
    if (confirm(`Are you sure you want to delete patient record #${patientId}?`)) {
        await fetch(`/api/patients/${patientId}`, { method: 'DELETE' });
        loadPatientsDashboard();
    }
}
