document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const startButton = document.getElementById('startButton');
    const stopButton = document.getElementById('stopButton');
    const videoFeed = document.getElementById('videoFeed');
    const placeholderContainer = document.getElementById('placeholderContainer');
    const predictionResult = document.getElementById('predictionResult');
    const confidenceResult = document.getElementById('confidenceResult');
    const wordElements = document.querySelectorAll('.word');
    
    // Variables
    let isRunning = false;
    let predictionInterval;
    
    // Event Listeners
    startButton.addEventListener('click', startCamera);
    stopButton.addEventListener('click', stopCamera);
    
    // Functions
    function startCamera() {
        fetch('/start_webcam')
            .then(response => response.json())
            .then(data => {
                if (data.status === 'started' || data.status === 'already running') {
                    isRunning = true;
                    videoFeed.style.display = 'block';
                    placeholderContainer.style.display = 'none';
                    startButton.disabled = true;
                    stopButton.disabled = false;
                    videoFeed.src = "/video_feed";
                    
                    // Start polling for predictions
                    startPredictionPolling();
                }
            })
            .catch(error => {
                console.error('Error starting webcam:', error);
                showError('Failed to start camera. Please check your camera permissions.');
            });
    }
    
    function stopCamera() {
        fetch('/stop_webcam')
            .then(response => response.json())
            .then(data => {
                if (data.status === 'stopped') {
                    isRunning = false;
                    videoFeed.style.display = 'none';
                    placeholderContainer.style.display = 'block';
                    startButton.disabled = false;
                    stopButton.disabled = true;
                    videoFeed.src = "static/image/cam.png";
                    
                    // Stop polling for predictions
                    stopPredictionPolling();
                    
                    // Reset predictions
                    predictionResult.textContent = '-';
                    confidenceResult.textContent = '-';
                    resetHighlightedWords();
                }
            })
            .catch(error => {
                console.error('Error stopping webcam:', error);
            });
    }
    
    function startPredictionPolling() {
        // Clear any existing interval
        if (predictionInterval) {
            clearInterval(predictionInterval);
        }
        
        // Poll every 300ms for new predictions
        predictionInterval = setInterval(getPrediction, 300);
    }
    
    function stopPredictionPolling() {
        if (predictionInterval) {
            clearInterval(predictionInterval);
            predictionInterval = null;
        }
    }
    
    function getPrediction() {
        if (!isRunning) return;
        
        fetch('/get_prediction')
            .then(response => response.json())
            .then(data => {
                updatePredictionDisplay(data.prediction, data.probability);
            })
            .catch(error => {
                console.error('Error getting prediction:', error);
            });
    }
    
    function updatePredictionDisplay(prediction, probability) {
        // Update text displays
        predictionResult.textContent = prediction || 'No gesture detected';
        confidenceResult.textContent = probability ? `${probability}` : '-';
        
        // Highlight the corresponding word
        highlightWord(prediction);
    }
    
    function highlightWord(word) {
        // Reset all highlighted words
        resetHighlightedWords();
        
        // Find and highlight the matching word
        if (word && word !== '[no action]') {
            wordElements.forEach(element => {
                if (element.textContent.trim() === word.trim()) {
                    element.classList.add('highlighted');
                    
                    // Scroll to the highlighted word if it's not in view
                    // element.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                }
            });
        }
    }
    
    function resetHighlightedWords() {
        wordElements.forEach(element => {
            element.classList.remove('highlighted');
        });
    }
    
    function showError(message) {
        // Create error notification
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-notification';
        errorDiv.textContent = message;
        
        // Add to DOM
        document.body.appendChild(errorDiv);
        
        // Remove after 5 seconds
        setTimeout(() => {
            errorDiv.classList.add('fade-out');
            setTimeout(() => {
                document.body.removeChild(errorDiv);
            }, 500);
        }, 5000);
    }
    
    // Add click handlers to word elements
    // wordElements.forEach(element => {
    //     element.addEventListener('click', function() {
    //         // Display how to sign this word (this could be extended to show instructions or videos)
    //         const word = this.textContent.trim();
    //         showTip(`Try signing "${word}"`);
    //     });
    // });
    
    function showTip(message) {
        // Create tip notification
        const tipDiv = document.createElement('div');
        tipDiv.className = 'tip-notification';
        tipDiv.textContent = message;
        
        // Add to DOM
        document.body.appendChild(tipDiv);
        
        // Remove after 3 seconds
        setTimeout(() => {
            tipDiv.classList.add('fade-out');
            setTimeout(() => {
                document.body.removeChild(tipDiv);
            }, 500);
        }, 3000);
    }
    
    // Handle video feed errors
    videoFeed.addEventListener('error', function() {
        console.error('Video feed error:', this.error);
        showError('Video feed error. Please check your camera permissions and refresh the page.');
    });
});

// Add CSS for notifications
document.head.insertAdjacentHTML('beforeend', `
    <style>
        .error-notification, .tip-notification {
            position: fixed;
            top: 20px;
            left: 50%;
            transform: translateX(-50%);
            padding: 12px 20px;
            border-radius: 5px;
            color: white;
            font-weight: 500;
            z-index: 1000;
            animation: fadeIn 0.3s ease-out;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        }
        
        .error-notification {
            background-color: #f44336;
        }
        
        .tip-notification {
            background-color: #2196F3;
        }
        
        .fade-out {
            opacity: 0;
            transition: opacity 0.5s ease-out;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translate(-50%, -10px); }
            to { opacity: 1; transform: translate(-50%, 0); }
        }
    </style>
`);