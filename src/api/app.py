import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from src.models.predict import SymptomPredictor
from src.utils.helpers import setup_logger

logger = setup_logger("api")

app = FastAPI(
    title="Symptom-to-Disease Predictor API",
    description="API for predicting probable diseases from symptoms using machine learning.",
    version="1.0.0"
)

# Initialize predictor lazily
predictor = None

try:
    predictor = SymptomPredictor()
except Exception as e:
    logger.warning(f"Could not load SymptomPredictor at startup. Is the model trained? Error: {e}")

class SymptomRequest(BaseModel):
    symptoms: list[str]

@app.on_event("startup")
async def startup_event():
    global predictor
    if predictor is None:
        try:
            predictor = SymptomPredictor()
            logger.info("SymptomPredictor loaded successfully on startup.")
        except Exception as e:
            logger.error(f"Error loading predictor on startup: {e}")

HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Symptom-to-Disease Predictor</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-grad-1: #0f172a;
            --bg-grad-2: #1e1b4b;
            --glass-bg: rgba(30, 41, 59, 0.45);
            --glass-border: rgba(255, 255, 255, 0.08);
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --primary: #6366f1;
            --primary-hover: #4f46e5;
            --accent: #10b981;
            --accent-hover: #059669;
            --danger: #ef4444;
            --shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: 'Plus Jakarta Sans', sans-serif;
            background: linear-gradient(135deg, var(--bg-grad-1) 0%, var(--bg-grad-2) 100%);
            color: var(--text-primary);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: flex-start;
            padding: 2rem 1rem;
            overflow-x: hidden;
        }

        .container {
            width: 100%;
            max-width: 900px;
            margin: 0 auto;
        }

        header {
            text-align: center;
            margin-bottom: 2.5rem;
        }

        header h1 {
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(to right, #818cf8, #a78bfa, #f472b6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }

        header p {
            color: var(--text-secondary);
            font-size: 1.1rem;
        }

        .glass-card {
            background: var(--glass-bg);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--glass-border);
            border-radius: 24px;
            padding: 2.5rem;
            box-shadow: var(--shadow);
            margin-bottom: 2rem;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }

        .glass-card:hover {
            box-shadow: 0 12px 40px 0 rgba(99, 102, 241, 0.15);
        }

        .section-title {
            font-size: 1.3rem;
            font-weight: 600;
            margin-bottom: 1.5rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .search-container {
            position: relative;
            margin-bottom: 1.5rem;
        }

        .search-input {
            width: 100%;
            padding: 1.1rem 1.5rem;
            border-radius: 14px;
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid rgba(255, 255, 255, 0.1);
            color: var(--text-primary);
            font-family: inherit;
            font-size: 1.05rem;
            outline: none;
            transition: all 0.3s ease;
        }

        .search-input:focus {
            border-color: var(--primary);
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.25);
        }

        .autocomplete-dropdown {
            position: absolute;
            top: 100%;
            left: 0;
            right: 0;
            max-height: 250px;
            overflow-y: auto;
            background: #1e293b;
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-top: none;
            border-radius: 0 0 14px 14px;
            z-index: 10;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.5);
            display: none;
        }

        .autocomplete-item {
            padding: 0.9rem 1.5rem;
            cursor: pointer;
            transition: background 0.2s ease;
        }

        .autocomplete-item:hover {
            background: rgba(99, 102, 241, 0.2);
            color: var(--text-primary);
        }

        .selected-list {
            display: flex;
            flex-wrap: wrap;
            gap: 0.6rem;
            margin-bottom: 2rem;
            min-height: 40px;
            padding: 0.5rem;
            border-radius: 12px;
            background: rgba(15, 23, 42, 0.3);
        }

        .selected-placeholder {
            color: var(--text-secondary);
            font-size: 0.95rem;
            display: flex;
            align-items: center;
            justify-content: center;
            width: 100%;
            height: 100%;
        }

        .symptom-tag {
            background: rgba(99, 102, 241, 0.15);
            border: 1px solid rgba(99, 102, 241, 0.3);
            color: #a5b4fc;
            padding: 0.4rem 0.9rem;
            border-radius: 9999px;
            font-size: 0.9rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            animation: fadeIn 0.3s ease;
        }

        .symptom-tag button {
            background: none;
            border: none;
            color: #a5b4fc;
            cursor: pointer;
            font-size: 1rem;
            line-height: 1;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: color 0.2s ease;
        }

        .symptom-tag button:hover {
            color: var(--danger);
        }

        .btn-predict {
            width: 100%;
            padding: 1.1rem;
            background: linear-gradient(135deg, var(--primary) 0%, #7c3aed 100%);
            border: none;
            border-radius: 14px;
            color: white;
            font-family: inherit;
            font-size: 1.1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
        }

        .btn-predict:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(99, 102, 241, 0.45);
        }

        .btn-predict:active {
            transform: translateY(0);
        }

        .btn-predict:disabled {
            background: rgba(255, 255, 255, 0.1);
            color: var(--text-secondary);
            cursor: not-allowed;
            box-shadow: none;
            transform: none;
        }

        .results-section {
            display: none;
            animation: slideUp 0.5s cubic-bezier(0.16, 1, 0.3, 1);
        }

        .result-card {
            background: rgba(15, 23, 42, 0.4);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 16px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            display: flex;
            flex-direction: column;
            gap: 0.8rem;
            transition: transform 0.2s ease;
        }

        .result-card:hover {
            transform: scale(1.01);
            background: rgba(15, 23, 42, 0.5);
        }

        .result-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .result-disease-name {
            font-size: 1.2rem;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 0.6rem;
        }

        .rank-badge {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 28px;
            height: 28px;
            border-radius: 50%;
            font-size: 0.85rem;
            font-weight: 700;
        }

        .rank-1 { background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.4); }
        .rank-2 { background: rgba(148, 163, 184, 0.2); color: #cbd5e1; border: 1px solid rgba(148, 163, 184, 0.4); }
        .rank-3 { background: rgba(180, 83, 9, 0.2); color: #f97316; border: 1px solid rgba(180, 83, 9, 0.4); }

        .confidence-label {
            font-weight: 700;
            font-size: 1.1rem;
            color: var(--accent);
        }

        .progress-bar-container {
            width: 100%;
            height: 10px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 999px;
            overflow: hidden;
        }

        .progress-bar-fill {
            height: 100%;
            background: linear-gradient(90deg, var(--primary) 0%, var(--accent) 100%);
            border-radius: 999px;
            width: 0%;
            transition: width 1s cubic-bezier(0.16, 1, 0.3, 1);
        }

        .drawer-toggle {
            display: block;
            margin: 1.5rem auto 0;
            background: none;
            border: none;
            color: var(--primary);
            cursor: pointer;
            font-family: inherit;
            font-size: 0.95rem;
            font-weight: 500;
            text-decoration: underline;
            transition: color 0.2s ease;
        }

        .drawer-toggle:hover {
            color: #818cf8;
        }

        .all-symptoms-list {
            margin-top: 1.5rem;
            display: none;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 0.6rem;
            max-height: 300px;
            overflow-y: auto;
            padding: 1rem;
            border-radius: 14px;
            background: rgba(15, 23, 42, 0.5);
            border: 1px solid rgba(255, 255, 255, 0.05);
        }

        .symptom-item-btn {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.05);
            color: var(--text-secondary);
            padding: 0.5rem 0.8rem;
            border-radius: 8px;
            font-family: inherit;
            font-size: 0.85rem;
            text-align: left;
            cursor: pointer;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            transition: all 0.2s ease;
        }

        .symptom-item-btn:hover {
            background: rgba(99, 102, 241, 0.1);
            color: var(--text-primary);
            border-color: rgba(99, 102, 241, 0.2);
        }

        footer {
            margin-top: auto;
            padding: 2rem 0;
            text-align: center;
            color: var(--text-secondary);
            font-size: 0.8rem;
            max-width: 700px;
            line-height: 1.5;
        }

        .spinner {
            border: 3px solid rgba(255, 255, 255, 0.1);
            width: 24px;
            height: 24px;
            border-radius: 50%;
            border-left-color: white;
            animation: spin 1s linear infinite;
            display: none;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: scale(0.95); }
            to { opacity: 1; transform: scale(1); }
        }

        @keyframes slideUp {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        ::-webkit-scrollbar {
            width: 8px;
        }
        ::-webkit-scrollbar-track {
            background: rgba(0, 0, 0, 0.1);
            border-radius: 10px;
        }
        ::-webkit-scrollbar-thumb {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: rgba(255, 255, 255, 0.2);
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Symptom-to-Disease Predictor</h1>
            <p>Enter your symptoms to predict the most probable diagnoses</p>
        </header>

        <div class="glass-card">
            <div class="section-title">
                <span>🔍</span> Diagnostic Input
            </div>
            
            <div class="search-container">
                <input type="text" class="search-input" id="symptom-search" placeholder="Type a symptom (e.g. fever, headache, cough)..." autocomplete="off">
                <div class="autocomplete-dropdown" id="autocomplete-list"></div>
            </div>

            <div class="selected-list" id="selected-symptoms">
                <div class="selected-placeholder" id="placeholder-text">No symptoms added. Search and select symptoms above.</div>
            </div>

            <button class="btn-predict" id="btn-predict" disabled>
                <span id="btn-text">Predict Probable Diseases</span>
                <div class="spinner" id="btn-spinner"></div>
            </button>

            <button class="drawer-toggle" id="toggle-all-symptoms">Show all available symptoms</button>
            <div class="all-symptoms-list" id="all-symptoms-box"></div>
        </div>

        <div class="glass-card results-section" id="results-box">
            <div class="section-title" style="margin-bottom: 2rem;">
                <span>🏥</span> Probable Diagnoses (Top-3)
            </div>
            <div id="results-list"></div>
        </div>
    </div>

    <footer>
        <p><strong>Disclaimer:</strong> This tool is powered by a machine learning model for educational and demonstration purposes. It does not constitute professional medical advice, diagnosis, treatment, or clinical assessment. Always consult a qualified healthcare professional regarding any medical concerns or conditions.</p>
    </footer>

    <script>
        let availableSymptoms = [];
        let selectedSymptoms = new Set();

        const searchInput = document.getElementById('symptom-search');
        const autocompleteList = document.getElementById('autocomplete-list');
        const selectedContainer = document.getElementById('selected-symptoms');
        const placeholderText = document.getElementById('placeholder-text');
        const btnPredict = document.getElementById('btn-predict');
        const btnText = document.getElementById('btn-text');
        const btnSpinner = document.getElementById('btn-spinner');
        const resultsBox = document.getElementById('results-box');
        const resultsList = document.getElementById('results-list');
        const toggleAllBtn = document.getElementById('toggle-all-symptoms');
        const allSymptomsBox = document.getElementById('all-symptoms-box');

        async function fetchSymptoms() {
            try {
                const response = await fetch('/api/symptoms');
                if (response.ok) {
                    availableSymptoms = await response.json();
                    populateAllSymptoms();
                } else {
                    const err = await response.json();
                    console.error("Failed to load symptoms list:", err.detail);
                }
            } catch (error) {
                console.error("Failed to load symptoms list", error);
            }
        }

        function populateAllSymptoms() {
            allSymptomsBox.innerHTML = '';
            availableSymptoms.sort().forEach(symptom => {
                const btn = document.createElement('button');
                btn.className = 'symptom-item-btn';
                btn.innerText = symptom;
                btn.title = symptom;
                btn.onclick = () => {
                    addSymptom(symptom);
                    searchInput.value = '';
                };
                allSymptomsBox.appendChild(btn);
            });
        }

        toggleAllBtn.onclick = () => {
            if (allSymptomsBox.style.display === 'grid') {
                allSymptomsBox.style.display = 'none';
                toggleAllBtn.innerText = 'Show all available symptoms';
            } else {
                allSymptomsBox.style.display = 'grid';
                toggleAllBtn.innerText = 'Hide available symptoms';
            }
        };

        searchInput.addEventListener('input', function() {
            const val = this.value.trim().toLowerCase();
            autocompleteList.innerHTML = '';
            if (!val) {
                autocompleteList.style.display = 'none';
                return;
            }

            const matches = availableSymptoms.filter(s => s.toLowerCase().includes(val) && !selectedSymptoms.has(s)).slice(0, 10);
            
            if (matches.length === 0) {
                const item = document.createElement('div');
                item.className = 'autocomplete-item';
                item.style.color = '#94a3b8';
                item.innerText = 'No matches found';
                autocompleteList.appendChild(item);
            } else {
                matches.forEach(match => {
                    const item = document.createElement('div');
                    item.className = 'autocomplete-item';
                    
                    const startIdx = match.toLowerCase().indexOf(val);
                    const before = match.substring(0, startIdx);
                    const boldText = match.substring(startIdx, startIdx + val.length);
                    const after = match.substring(startIdx + val.length);
                    item.innerHTML = before + '<strong>' + boldText + '</strong>' + after;
                    
                    item.addEventListener('click', function() {
                        addSymptom(match);
                        searchInput.value = '';
                        autocompleteList.style.display = 'none';
                    });
                    autocompleteList.appendChild(item);
                });
            }
            autocompleteList.style.display = 'block';
        });

        document.addEventListener('click', function(e) {
            if (e.target !== searchInput && e.target !== autocompleteList) {
                autocompleteList.style.display = 'none';
            }
        });

        function addSymptom(symptom) {
            if (selectedSymptoms.has(symptom)) return;
            
            selectedSymptoms.add(symptom);
            updateSymptomTags();
        }

        function removeSymptom(symptom) {
            selectedSymptoms.delete(symptom);
            updateSymptomTags();
        }

        function updateSymptomTags() {
            const tags = selectedContainer.querySelectorAll('.symptom-tag');
            tags.forEach(t => t.remove());

            if (selectedSymptoms.size === 0) {
                placeholderText.style.display = 'flex';
                btnPredict.disabled = true;
            } else {
                placeholderText.style.display = 'none';
                btnPredict.disabled = false;

                selectedSymptoms.forEach(symptom => {
                    const tag = document.createElement('div');
                    tag.className = 'symptom-tag';
                    tag.textContent = symptom + ' ';
                    
                    const btn = document.createElement('button');
                    btn.innerHTML = '&times;';
                    btn.onclick = () => removeSymptom(symptom);
                    
                    tag.appendChild(btn);
                    selectedContainer.appendChild(tag);
                });
            }
        }

        btnPredict.onclick = async () => {
            if (selectedSymptoms.size === 0) return;

            btnPredict.disabled = true;
            btnText.style.display = 'none';
            btnSpinner.style.display = 'block';
            resultsBox.style.display = 'none';

            try {
                const response = await fetch('/api/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ symptoms: Array.from(selectedSymptoms) })
                });

                const predictions = await response.json();
                displayResults(predictions);
            } catch (error) {
                console.error("Prediction failed", error);
                alert("An error occurred while communicating with the server.");
                btnPredict.disabled = false;
                btnText.style.display = 'block';
                btnSpinner.style.display = 'none';
            }
        };

        function displayResults(predictions) {
            resultsList.innerHTML = '';
            
            predictions.forEach(p => {
                const card = document.createElement('div');
                card.className = 'result-card';
                
                const confidence = p.confidence;
                
                card.innerHTML = `
                    <div class="result-header">
                        <div class="result-disease-name">
                            <span class="rank-badge rank-${p.rank}">${p.rank}</span>
                            <span>${p.disease}</span>
                        </div>
                        <div class="confidence-label">${confidence.toFixed(2)}%</div>
                    </div>
                    <div class="progress-bar-container">
                        <div class="progress-bar-fill" id="fill-rank-${p.rank}"></div>
                    </div>
                `;
                
                resultsList.appendChild(card);
                
                setTimeout(() => {
                    const fill = document.getElementById(`fill-rank-${p.rank}`);
                    if (fill) fill.style.width = confidence + '%';
                }, 100);
            });

            resultsBox.style.display = 'block';
            resultsBox.scrollIntoView({ behavior: 'smooth' });

            btnPredict.disabled = false;
            btnText.style.display = 'block';
            btnSpinner.style.display = 'none';
        }

        fetchSymptoms();
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def serve_home():
    """Serves the visually stunning glassmorphic frontend checker."""
    return HTMLResponse(content=HTML_CONTENT)

@app.get("/api/symptoms")
async def get_symptoms():
    """Returns a list of all symptoms supported by the model."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Predictor model is not loaded. Please train the model first.")
    return predictor.feature_names

@app.post("/api/predict")
async def predict_diseases(request: SymptomRequest):
    """Predicts the top-3 probable diseases for the given symptom list."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Predictor model is not loaded. Please train the model first.")
    if not request.symptoms:
        raise HTTPException(status_code=400, detail="Please enter at least one symptom.")
    
    try:
        predictions = predictor.predict_top3(request.symptoms)
        return predictions
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
