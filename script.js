// Plant Disease Detection API Endpoint
// You can set localStorage.setItem("backend_url", "https://your-colab-ngrok-url.ngrok-free.app/predict")
const DEFAULT_API_URL = "http://127.0.0.1:5000/predict";

let currentDetectedDisease = "Tomato Early Blight";

function getApiUrl() {
    return localStorage.getItem("backend_url") || DEFAULT_API_URL;
}

function goToSuggestionsPage() {
    const targetUrl = "frontend-suggestion.html?disease=" + encodeURIComponent(currentDetectedDisease);
    window.location.href = targetUrl;
}

const form = document.getElementById("uploadForm");

if (form) {
    form.addEventListener("submit", async function(e) {
        e.preventDefault();

        const fileInput = document.getElementById("imageInput");
        const file = fileInput ? fileInput.files[0] : null;

        if (!file) {
            alert("Please select an image first.");
            return;
        }

        const formData = new FormData();
        formData.append("image", file);

        const loadingElem = document.getElementById("loading");
        if (loadingElem) loadingElem.style.display = "block";

        const currentApiUrl = getApiUrl();
        console.log("Connecting to API endpoint:", currentApiUrl);

        try {
            const response = await fetch(currentApiUrl, {
                method: "POST",
                body: formData
            });

            if (!response.ok) {
                throw new Error(`Server returned HTTP ${response.status}`);
            }

            const data = await response.json();

            if (data.error) {
                alert("Error: " + data.error);
                return;
            }

            const diseaseElem = document.getElementById("disease");
            const confidenceElem = document.getElementById("confidence");
            const statusElem = document.getElementById("status");

            if (diseaseElem) diseaseElem.innerHTML = data.disease || "--";
            if (confidenceElem) confidenceElem.innerHTML = data.confidence || "--";
            if (statusElem) statusElem.innerHTML = data.status || "--";

            currentDetectedDisease = data.disease || "Tomato Early Blight";

            // Display Suggestion Box & Button
            const suggestionBox = document.getElementById("suggestionBox");
            const askBtn = document.getElementById("askSuggestionBtn");
            const inlineRemedy = document.getElementById("inlineRemedy");

            if (suggestionBox) suggestionBox.style.display = "block";

            if (askBtn) {
                askBtn.innerHTML = `💡 Ask Suggestion for ${currentDetectedDisease}`;
            }

            // Populate inline remedies if available from backend
            if (data.suggestions && inlineRemedy) {
                inlineRemedy.style.display = "block";
                const orgElem = document.querySelector("#inlineOrganic span");
                const chemElem = document.querySelector("#inlineChemical span");
                const prevElem = document.querySelector("#inlinePrevention span");
                const fullLink = document.getElementById("fullSuggestionLink");

                if (orgElem) orgElem.innerHTML = data.suggestions.organic || "--";
                if (chemElem) chemElem.innerHTML = data.suggestions.chemical || "--";
                if (prevElem) prevElem.innerHTML = data.suggestions.prevention || "--";

                if (fullLink) {
                    fullLink.href = "frontend-suggestion.html?disease=" + encodeURIComponent(currentDetectedDisease);
                }
            }

        } catch (err) {
            console.error("Backend request failed:", err);
            alert("Backend connection failed. Please ensure your local Flask server or Google Colab Ngrok tunnel is running!\n\nTarget URL: " + currentApiUrl);
        } finally {
            if (loadingElem) loadingElem.style.display = "none";
        }
    });
}