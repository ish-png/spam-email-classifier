/* Spam Email Classifier - Dashboard & Client Script */

document.addEventListener('DOMContentLoaded', function () {
  // Initialize Chart.js charts if canvas exists
  const distCanvas = document.getElementById('distributionChart');
  if (distCanvas && window.Chart) {
    const spamCount = parseInt(distCanvas.dataset.spam || '0', 10);
    const hamCount = parseInt(distCanvas.dataset.ham || '0', 10);

    new Chart(distCanvas, {
      type: 'doughnut',
      data: {
        labels: ['Spam', 'Legitimate (Ham)'],
        datasets: [{
          data: [spamCount, hamCount],
          backgroundColor: ['#ef4444', '#22c55e'],
          borderWidth: 2,
          borderColor: '#ffffff'
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: 'bottom' }
        },
        cutout: '70%'
      }
    });
  }

  // Classifier form submission via AJAX for instant feedback
  const classifyForm = document.getElementById('classifyForm');
  if (classifyForm) {
    classifyForm.addEventListener('submit', async function (e) {
      e.preventDefault();
      const textArea = document.getElementById('emailText');
      const submitBtn = document.getElementById('submitBtn');
      const spinner = document.getElementById('btnSpinner');
      const resultArea = document.getElementById('resultArea');
      const errorAlert = document.getElementById('errorAlert');

      const messageText = textArea.value.trim();
      if (!messageText) {
        errorAlert.textContent = "Please enter an email or message before analyzing.";
        errorAlert.classList.remove('d-none');
        return;
      }

      errorAlert.classList.add('d-none');
      submitBtn.disabled = true;
      if (spinner) spinner.classList.remove('d-none');

      try {
        const response = await fetch('/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text: messageText })
        });

        const data = await response.json();

        if (!response.ok) {
          throw new Error(data.error || "Prediction request failed.");
        }

        // Render result card dynamically
        const isSpam = data.prediction === 'SPAM';
        const cardClass = isSpam ? 'result-card-spam' : 'result-card-ham';
        const badgeClass = isSpam ? 'badge-spam' : 'badge-ham';
        const barColor = isSpam ? 'bg-danger' : 'bg-success';

        let indicatorsHtml = '';
        if (data.key_indicators && data.key_indicators.length > 0) {
          indicatorsHtml = data.key_indicators.map(t => `<span class="indicator-tag">${t}</span>`).join(' ');
        } else {
          indicatorsHtml = '<span class="text-muted small">None identified</span>';
        }

        resultArea.innerHTML = `
          <div class="card ${cardClass} p-4 mt-4 shadow-sm">
            <div class="d-flex justify-content-between align-items-center mb-3">
              <div>
                <span class="badge ${badgeClass} fs-6 px-3 py-2">
                  <i class="bi ${isSpam ? 'bi-exclamation-triangle-fill' : 'bi-check-circle-fill'} me-1"></i>
                  ${data.prediction}
                </span>
              </div>
              <div class="text-end">
                <span class="fs-4 fw-bold ${isSpam ? 'text-danger' : 'text-success'}">${data.confidence}%</span>
                <span class="text-muted small d-block">Model Confidence</span>
              </div>
            </div>

            <!-- Confidence Progress Bar -->
            <div class="progress mb-3" style="height: 8px;">
              <div class="progress-bar ${barColor}" role="progressbar" style="width: ${data.confidence}%" aria-valuenow="${data.confidence}" aria-valuemin="0" aria-valuemax="100"></div>
            </div>

            <!-- Explanation -->
            <div class="mb-3">
              <h6 class="fw-bold mb-1"><i class="bi bi-info-circle me-1"></i> Linguistic Explanation</h6>
              <p class="text-secondary small mb-2">${data.explanation}</p>
              <div><strong class="small text-muted">Key TF-IDF Term Indicators:</strong><br>${indicatorsHtml}</div>
            </div>

            <hr class="my-2">

            <!-- Message Statistics -->
            <h6 class="fw-bold mb-2 small text-uppercase text-muted">Message Structural Statistics</h6>
            <div class="row g-2 text-center">
              <div class="col-4 col-md-2">
                <div class="p-2 border rounded bg-white">
                  <div class="fw-bold fs-6">${data.statistics.character_count}</div>
                  <div class="text-muted small" style="font-size: 0.75rem;">Characters</div>
                </div>
              </div>
              <div class="col-4 col-md-2">
                <div class="p-2 border rounded bg-white">
                  <div class="fw-bold fs-6">${data.statistics.word_count}</div>
                  <div class="text-muted small" style="font-size: 0.75rem;">Words</div>
                </div>
              </div>
              <div class="col-4 col-md-2">
                <div class="p-2 border rounded bg-white">
                  <div class="fw-bold fs-6">${data.statistics.urls_count}</div>
                  <div class="text-muted small" style="font-size: 0.75rem;">URLs</div>
                </div>
              </div>
              <div class="col-4 col-md-2">
                <div class="p-2 border rounded bg-white">
                  <div class="fw-bold fs-6">${data.statistics.numbers_count}</div>
                  <div class="text-muted small" style="font-size: 0.75rem;">Numbers</div>
                </div>
              </div>
              <div class="col-4 col-md-2">
                <div class="p-2 border rounded bg-white">
                  <div class="fw-bold fs-6">${data.statistics.special_chars_count}</div>
                  <div class="text-muted small" style="font-size: 0.75rem;">Special Chars</div>
                </div>
              </div>
              <div class="col-4 col-md-2">
                <div class="p-2 border rounded bg-white">
                  <div class="fw-bold fs-6">${data.statistics.exclamation_count}</div>
                  <div class="text-muted small" style="font-size: 0.75rem;">Exclamations (!)</div>
                </div>
              </div>
            </div>
          </div>
        `;
        resultArea.scrollIntoView({ behavior: 'smooth' });

      } catch (err) {
        errorAlert.textContent = err.message;
        errorAlert.classList.remove('d-none');
      } finally {
        submitBtn.disabled = false;
        if (spinner) spinner.classList.add('d-none');
      }
    });

    // Preset buttons helper
    window.loadPreset = function (text) {
      const textArea = document.getElementById('emailText');
      if (textArea) {
        textArea.value = text;
        textArea.focus();
      }
    };

    window.clearInput = function () {
      const textArea = document.getElementById('emailText');
      const resultArea = document.getElementById('resultArea');
      const errorAlert = document.getElementById('errorAlert');
      if (textArea) textArea.value = '';
      if (resultArea) resultArea.innerHTML = '';
      if (errorAlert) errorAlert.classList.add('d-none');
    };
  }
});
