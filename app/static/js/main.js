// F1 2026 Race Predictor - JavaScript (FIXED + aligned with backend)

// ==================== GLOBAL STATE ====================
let appState = {
  races: [],
  drivers: [],
  modelLoaded: false
};

// ==================== API BASE URL ====================
const API_BASE = ''; // same-origin

// ==================== CONFIG ====================
const GRID_SIZE_DEFAULT = 22; // 11 teams, 22 drivers

// ==================== INITIALIZATION ====================
document.addEventListener('DOMContentLoaded', () => {
  console.log('🏎️ F1 2026 Race Predictor initialized');

  checkModelStatus();
  loadRaces();
  loadDrivers();
  setupEventListeners();
});

// ==================== MODEL STATUS ====================
async function checkModelStatus() {
  const statusBanner = document.getElementById('model-status');
  if (!statusBanner) return;

  try {
    const response = await fetch(`${API_BASE}/api/model-info`, { cache: 'no-store' });
    const data = await response.json();

    if (data && data.loaded) {
      appState.modelLoaded = true;

      const meta = data.metadata || {};
      const features = meta.features ?? 0;
      const drivers = meta.drivers ?? 0;

      statusBanner.innerHTML = `
        <div class="status-success">
          ✓ Model loaded successfully (${features} features, ${drivers} drivers)
        </div>
      `;
    } else {
      appState.modelLoaded = false;
      statusBanner.innerHTML = `
        <div class="status-error">
          ❌ Model not loaded. Please train the model first.
        </div>
      `;
    }

    validateGridInputs();
  } catch (error) {
    console.error('Error checking model status:', error);
    appState.modelLoaded = false;
    statusBanner.innerHTML = `
      <div class="status-error">
        ❌ Error connecting to server
      </div>
    `;
    validateGridInputs();
  }
}

// ==================== LOAD RACES ====================
async function loadRaces() {
  try {
    const response = await fetch(`${API_BASE}/api/races`, { cache: 'no-store' });
    const data = await response.json();

    const races = Array.isArray(data) ? data : (data.races || []);
    appState.races = races;

    populateRaceSelector(races);
    handleSprintBanner();
  } catch (error) {
    console.error('Error loading races:', error);
    showNotification('Error loading races', 'error');
  }
}

function populateRaceSelector(races) {
  const select = document.getElementById('race-select');
  if (!select) return;

  select.innerHTML = '<option value="">-- Select a race --</option>';

  races.forEach(r => {
    const option = document.createElement('option');

    // Backend expects race NAME
    option.value = r.name;

    // Determine sprint
    const isSprint = !!(r.is_sprint_race) || (String(r.format || '').toLowerCase() === 'sprint');
    const hasSprint = !!(r.has_sprint);

    // Visible sprint marker
    const dateText = r.formatted_date || r.date || '';
    const roundText = (r.round != null) ? `${r.round}. ` : '';
    const sprintTag = isSprint ? ' 🏁' : '';

    option.textContent = `${roundText}${r.name}${sprintTag} (${dateText})`;

    // Store sprint info for UI banner
    option.dataset.isSprint = String(isSprint);
    option.dataset.hasSprint = String(hasSprint);

    select.appendChild(option);
  });
}

// ==================== LOAD DRIVERS ====================
async function loadDrivers() {
  try {
    const response = await fetch(`${API_BASE}/api/drivers`, { cache: 'no-store' });
    const data = await response.json();

    const drivers = Array.isArray(data) ? data : (data.drivers || []);
    appState.drivers = drivers;

    populateGridInputs(drivers);
    validateGridInputs();
  } catch (error) {
    console.error('Error loading drivers:', error);
    showNotification('Error loading drivers', 'error');
  }
}

function populateGridInputs(drivers) {
  const container = document.getElementById('grid-container');
  if (!container) return;

  container.innerHTML = '';

  const sortedDrivers = [...drivers].sort((a, b) => {
    const ta = (a.team || '').toLowerCase();
    const tb = (b.team || '').toLowerCase();
    if (ta < tb) return -1;
    if (ta > tb) return 1;
    return (a.name || '').localeCompare(b.name || '');
  });

  sortedDrivers.forEach(driver => {
    const code = driver.code; // "NOR"
    if (!code) return;

    const gridItem = document.createElement('div');
    gridItem.className = 'grid-item';

    gridItem.innerHTML = `
      <div class="driver-header">
        <div class="driver-number">${driver.number ?? ''}</div>
        <div class="driver-info">
          <div class="driver-name">${driver.name ?? ''}</div>
          <div class="driver-team">${driver.team ?? ''}</div>
        </div>
      </div>

      <input
        type="number"
        class="grid-input"
        id="grid-${code}"
        data-driver-code="${code}"
        min="1"
        max="${GRID_SIZE_DEFAULT}"
        placeholder="Grid position"
      >
    `;

    container.appendChild(gridItem);
  });

  document.querySelectorAll('.grid-input').forEach(input => {
    input.addEventListener('input', validateGridInputs);
  });
}

// ==================== EVENT LISTENERS ====================
function setupEventListeners() {
  const defaultGridBtn = document.getElementById('load-default-grid');
  if (defaultGridBtn) defaultGridBtn.addEventListener('click', loadDefaultGrid);

  const predictBtn = document.getElementById('predict-btn');
  if (predictBtn) predictBtn.addEventListener('click', makePrediction);

  const raceSelect = document.getElementById('race-select');
  if (raceSelect) raceSelect.addEventListener('change', () => { handleSprintBanner(); validateGridInputs(); });
}

// ==================== LOAD DEFAULT GRID ====================
async function loadDefaultGrid() {
  try {
    const response = await fetch(`${API_BASE}/api/default-grid`, { cache: 'no-store' });
    const data = await response.json();

    if (data && data.status === 'success') {
      const gridPositions = data.grid_positions || {};

      Object.entries(gridPositions).forEach(([driverCode, position]) => {
        const input = document.getElementById(`grid-${driverCode}`);
        if (input) input.value = position;
      });

      validateGridInputs();
      showNotification('Default grid loaded', 'success');
    } else {
      showNotification(data?.message || 'Could not load default grid', 'error');
    }
  } catch (error) {
    console.error('Error loading default grid:', error);
    showNotification('Error loading default grid', 'error');
  }
}

// ==================== VALIDATE GRID INPUTS ====================
function validateGridInputs() {
  const inputs = document.querySelectorAll('.grid-input');
  const raceSelect = document.getElementById('race-select');

  // If drivers not loaded yet
  if (!inputs || inputs.length === 0) {
    disablePredictButton();
    return;
  }

  let allFilled = true;
  const values = [];

  inputs.forEach(input => {
    const value = parseInt(input.value, 10);
    if (!value || value < 1 || value > GRID_SIZE_DEFAULT) {
      allFilled = false;
    } else {
      values.push(value);
    }
  });

  const hasDuplicates = values.length !== new Set(values).size;
  const raceChosen = !!(raceSelect && raceSelect.value);

  if (allFilled && !hasDuplicates && raceChosen && appState.modelLoaded) {
    enablePredictButton();
  } else {
    disablePredictButton();
  }

  if (allFilled && hasDuplicates) {
    showNotification('Duplicate grid positions detected', 'warning');
  }
}

function enablePredictButton() {
  const btn = document.getElementById('predict-btn');
  if (btn) btn.disabled = false;
}

function disablePredictButton() {
  const btn = document.getElementById('predict-btn');
  if (btn) btn.disabled = true;
}

// ==================== MAKE PREDICTION ====================
async function makePrediction() {
  const race = document.getElementById('race-select')?.value;
  if (!race) {
    showNotification('Please select a race', 'warning');
    return;
  }

  const inputs = document.querySelectorAll('.grid-input');
  const gridPositions = {};

  inputs.forEach(input => {
    const driverCode = input.dataset.driverCode;
    const position = parseInt(input.value, 10);
    gridPositions[driverCode] = position;
  });

  showLoadingOverlay();

  try {
    const response = await fetch(`${API_BASE}/api/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        race: race,
        grid_positions: gridPositions
      })
    });

    const data = await response.json();

    if (data && data.status === 'success') {
      displayPredictions(data.predictions, data.race || race);
      showNotification('Predictions generated successfully', 'success');
    } else {
      showNotification(data?.message || 'Error making prediction', 'error');
    }
  } catch (error) {
    console.error('Error making prediction:', error);
    showNotification('Error making prediction', 'error');
  } finally {
    hideLoadingOverlay();
  }
}

// ==================== DISPLAY PREDICTIONS ====================
function displayPredictions(predictions, raceName) {
  const section = document.getElementById('predictions-section');
  const info = document.getElementById('predictions-info');
  const container = document.getElementById('predictions-container');

  if (!section || !info || !container) return;

  section.classList.remove('hidden');
  info.innerHTML = `<strong>🏁 ${raceName}</strong> - Predicted Finishing Order`;

  container.innerHTML = '';

  (predictions || []).forEach(prediction => {
    const item = document.createElement('div');
    item.className = `prediction-item position-${prediction.position}`;

    const positionChange = prediction.positionsGained ?? 0;
    const changeClass = positionChange > 0 ? 'positive' : positionChange < 0 ? 'negative' : '';
    const changeSymbol = positionChange > 0 ? '+' : '';

    item.innerHTML = `
      <div class="prediction-position">${prediction.position}</div>
      <div class="prediction-driver">
        <div class="prediction-driver-name">${prediction.driverName ?? ''}</div>
        <div class="prediction-driver-team">${prediction.team ?? ''}</div>
      </div>
      <div class="prediction-stats">
        <div>
          <span class="stat-label">Grid:</span>
          <span class="stat-value">P${prediction.gridPosition ?? ''}</span>
        </div>
        <div>
          <span class="stat-label">Change:</span>
          <span class="stat-value ${changeClass}">${changeSymbol}${positionChange}</span>
        </div>
      </div>
    `;

    container.appendChild(item);
  });

  section.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ==================== LOADING OVERLAY ====================
function showLoadingOverlay() {
  document.getElementById('loading-overlay')?.classList.remove('hidden');
}

function hideLoadingOverlay() {
  document.getElementById('loading-overlay')?.classList.add('hidden');
}

// ==================== NOTIFICATIONS ====================
function showNotification(message, type = 'info') {
  console.log(`[${type.toUpperCase()}] ${message}`);
}
    
// ==================== SPRINT BANNER ====================
function handleSprintBanner() {
  const raceSelect = document.getElementById('race-select');
  const sprintInfo = document.getElementById('sprintInfo');
  if (!raceSelect || !sprintInfo) return;

  const opt = raceSelect.options[raceSelect.selectedIndex];
  const isSprint = !!(opt && opt.dataset && opt.dataset.isSprint === 'true');

  if (isSprint) {
    sprintInfo.classList.remove('hidden');
  } else {
    sprintInfo.classList.add('hidden');
  }
}