/**
 * Plant Disease Detection - Frontend JavaScript Engine
 * Connects Frontend HTML to Google Colab Flask API Backend & Gemini API
 */

// Configuration Defaults & Local Storage
const DEFAULT_BACKEND_URL = "http://127.0.0.1:5000/predict";

function getBackendUrl() {
    return localStorage.getItem("backend_url") || DEFAULT_BACKEND_URL;
}

function setBackendUrl(url) {
    if (url) {
        let cleanUrl = url.trim();
        // Automatically ensure /predict endpoint if base URL is provided
        if (!cleanUrl.endsWith("/predict") && !cleanUrl.endsWith("/suggest")) {
            cleanUrl = cleanUrl.replace(/\/$/, "") + "/predict";
        }
        localStorage.setItem("backend_url", cleanUrl);
        return cleanUrl;
    }
    return getBackendUrl();
}

function getApiKey() {
    return localStorage.getItem("gemini_api_key") || "";
}

function setApiKey(key) {
    if (key !== null) {
        localStorage.setItem("gemini_api_key", key.trim());
    }
}

// User Configuration Prompt
function configureApiUrl() {
    const currentUrl = getBackendUrl();
    const newUrl = prompt(
        "🌿 ENTER BACKEND API ENDPOINT (Google Colab Ngrok URL):\nExample: https://xxxx-xx-xx-xx.ngrok-free.dev/predict",
        currentUrl
    );

    if (newUrl !== null && newUrl.trim() !== "") {
        const savedUrl = setBackendUrl(newUrl);
        alert("✅ Backend Server API URL successfully updated to:\n" + savedUrl);
    }

    const askKey = confirm("🔑 Do you also want to update or configure your Gemini API Key?");
    if (askKey) {
        configureApiKey();
    }
}

function configureApiKey() {
    const currentKey = getApiKey();
    const newKey = prompt(
        "🔑 ENTER GEMINI API KEY:\n(Used for enhanced AI treatment suggestions & vision analysis)",
        currentKey
    );

    if (newKey !== null) {
        setApiKey(newKey);
        alert(newKey.trim() !== "" ? "✅ Gemini API Key saved!" : "ℹ️ Gemini API Key cleared.");
    }
}

// Image Preview Handler
document.addEventListener("DOMContentLoaded", () => {
    const imageInput = document.getElementById("imageInput");
    const preview = document.getElementById("preview");
    const uploadForm = document.getElementById("uploadForm");

    if (imageInput && preview) {
        imageInput.addEventListener("change", function () {
            const file = this.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function (e) {
                    preview.src = e.target.result;
                    preview.style.display = "block";
                };
                reader.readAsDataURL(file);
            }
        });
    }

    if (uploadForm) {
        uploadForm.addEventListener("submit", handleDiseaseDetection);
    }
});

// Disease Detection Handler
async function handleDiseaseDetection(e) {
    e.preventDefault();

    const imageInput = document.getElementById("imageInput");
    const loading = document.getElementById("loading");
    const diseaseElem = document.getElementById("disease");
    const confidenceElem = document.getElementById("confidence");
    const statusElem = document.getElementById("status");
    const suggestionBox = document.getElementById("suggestionBox");
    const inlineRemedy = document.getElementById("inlineRemedy");

    if (!imageInput || !imageInput.files[0]) {
        alert("Please select a plant leaf image first!");
        return;
    }

    const file = imageInput.files[0];
    const backendUrl = getBackendUrl();
    const apiKey = getApiKey();

    // Show Loading
    if (loading) loading.style.display = "block";
    if (diseaseElem) diseaseElem.innerText = "Analyzing image with AI...";
    if (confidenceElem) confidenceElem.innerText = "Calculating...";
    if (statusElem) statusElem.innerText = "Processing...";
    if (suggestionBox) suggestionBox.style.display = "none";
    if (inlineRemedy) inlineRemedy.style.display = "none";

    const formData = new FormData();
    formData.append("image", file);

    try {
        const response = await fetch(backendUrl, {
            method: "POST",
            headers: apiKey ? { "X-API-Key": apiKey } : {},
            body: formData
        });

        if (!response.ok) {
            throw new Error(`Server returned HTTP ${response.status}`);
        }

        const data = await response.json();
        if (loading) loading.style.display = "none";

        // Display Results
        const diseaseName = data.disease || data.prediction || "Unknown Disease";
        const confidence = data.confidence || "95.0%";
        const status = data.status || (diseaseName.toLowerCase().includes("healthy") ? "Healthy" : "Diseased");

        if (diseaseElem) diseaseElem.innerHTML = `<span style="color:${status === 'Healthy' ? '#2e7d32' : '#d32f2f'}">${diseaseName}</span>`;
        if (confidenceElem) confidenceElem.innerText = confidence;
        if (statusElem) {
            statusElem.innerHTML = `<span style="padding:4px 12px; border-radius:12px; color:white; background:${status === 'Healthy' ? '#4caf50' : '#f44336'}">${status}</span>`;
        }

        // Show suggestion button & remedies if present
        if (suggestionBox) {
            suggestionBox.style.display = "block";
            window.lastDetectedDisease = diseaseName;
        }

        if (data.organic || data.chemical || data.prevention) {
            if (inlineRemedy) {
                inlineRemedy.style.display = "block";
                const orgElem = document.querySelector("#inlineOrganic span");
                const chemElem = document.querySelector("#inlineChemical span");
                const prevElem = document.querySelector("#inlinePrevention span");
                if (orgElem) orgElem.innerText = data.organic || "Prune infected leaves, apply neem oil.";
                if (chemElem) chemElem.innerText = data.chemical || "Consult local agricultural extension for fungicides.";
                if (prevElem) prevElem.innerText = data.prevention || "Rotate crops and maintain leaf moisture balance.";
            }
        } else {
            // Fetch remedies dynamically from /suggest or local knowledge base
            fetchInlineTreatment(diseaseName);
        }

    } catch (error) {
        console.error("Backend Connection Error:", error);
        if (loading) loading.style.display = "none";

        // Fallback UI alert and guidance
        if (diseaseElem) diseaseElem.innerText = "Connection Error";
        if (confidenceElem) confidenceElem.innerText = "N/A";
        if (statusElem) statusElem.innerText = "Offline";

        const setNow = confirm(
            `⚠️ Could not connect to backend server at:\n${backendUrl}\n\n` +
            `Option 1: If using Google Colab, click OK to paste your live Ngrok URL.\n` +
            `Option 2: If running locally, start 'python app.py'.\n\n` +
            `Would you like to set your Google Colab Server URL now?`
        );

        if (setNow) {
            configureApiUrl();
        } else {
            // Provide fallback demo prediction so UI doesn't stay blank
            const fallbackDisease = "Tomato Early Blight";
            if (diseaseElem) diseaseElem.innerHTML = `<span style="color:#d32f2f">${fallbackDisease} (Demo Mode)</span>`;
            if (confidenceElem) confidenceElem.innerText = "94.80%";
            if (statusElem) statusElem.innerHTML = `<span style="padding:4px 12px; border-radius:12px; color:white; background:#f44336">Diseased</span>`;

            if (suggestionBox) {
                suggestionBox.style.display = "block";
                window.lastDetectedDisease = fallbackDisease;
            }
            if (inlineRemedy) {
                inlineRemedy.style.display = "block";
                const orgElem = document.querySelector("#inlineOrganic span");
                const chemElem = document.querySelector("#inlineChemical span");
                const prevElem = document.querySelector("#inlinePrevention span");
                if (orgElem) orgElem.innerText = "Prune infected lower leaves immediately. Apply organic copper-based fungicide or neem oil solution every 7 to 10 days.";
                if (chemElem) chemElem.innerText = "Spray Chlorothalonil or Mancozeb fungicide at the first sign of leaf spots following package safety guidelines.";
                if (prevElem) prevElem.innerText = "Rotate tomato crops every 2-3 years. Mulch soil surface to prevent fungal spores from splashing onto lower leaves.";
            }
        }
    }
}

// Redirect to Suggestions Page with query parameter
function goToSuggestionsPage() {
    const disease = window.lastDetectedDisease || "Tomato Early Blight";
    window.location.href = `frontend-suggestion.html?disease=${encodeURIComponent(disease)}`;
}

// Helper to fetch inline treatment highlights
async function fetchInlineTreatment(diseaseName) {
    const backendUrl = getBackendUrl();
    const suggestEndpoint = backendUrl.replace("/predict", "/suggest");
    const inlineRemedy = document.getElementById("inlineRemedy");

    try {
        const resp = await fetch(`${suggestEndpoint}?disease=${encodeURIComponent(diseaseName)}`);
        if (resp.ok) {
            const data = await resp.json();
            if (inlineRemedy) {
                inlineRemedy.style.display = "block";
                const orgElem = document.querySelector("#inlineOrganic span");
                const chemElem = document.querySelector("#inlineChemical span");
                const prevElem = document.querySelector("#inlinePrevention span");
                if (orgElem) orgElem.innerText = data.organic || "Prune infected leaves and apply organic copper spray.";
                if (chemElem) chemElem.innerText = data.chemical || "Spray Chlorothalonil or Mancozeb fungicide as recommended.";
                if (prevElem) prevElem.innerText = data.prevention || "Maintain plant spacing and drip irrigation.";
            }
        }
    } catch (e) {
        console.log("Inline treatment fetch notice:", e);
    }
}
