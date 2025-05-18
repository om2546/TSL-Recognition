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
        if (probability > 0.7) {
            highlightWord(prediction);
        }
    }
    
    function highlightWord(word) {
        // Reset all highlighted words
        resetHighlightedWords();
        
        // Find and highlight the matching word
        if (word && word !== '[no action]') {
            wordElements.forEach(element => {
                if (element.textContent.trim() === word.trim()) {
                    element.classList.add('highlighted');
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
    wordElements.forEach(element => {
        element.addEventListener('click', function() {
            // Display how to sign this word (this could be extended to show instructions or videos)
            const word = this.textContent.trim();
            showPopup(word);
        });
    });
    function showPopup(word) {
        // Check if there's already a popup open
        const existingPopup = document.querySelector('.video-popup');
        if (existingPopup) {
            existingPopup.remove();
        }
        
        // Format the word for the file path (lowercase, replace spaces and slashes with underscores)
        const formattedWord = word.toLowerCase().replace(/[ /]/g, '_');
        
        // Ensure we're using the correct path based on Flask's URL structure
        // This uses the static folder configured in Flask
        const videoPath = `/static/video/${formattedWord}.mp4`;
        
        // Create popup container
        const popup = document.createElement('div');
        popup.className = 'video-popup';
        
        // Create popup content
        const popupContent = document.createElement('div');
        popupContent.className = 'popup-content';
        
        // Create title
        const title = document.createElement('h3');
        title.textContent = `How to sign: "${word}"`;
        
        // Create video element
        const video = document.createElement('video');
        video.controls = true;
        video.muted = false;
        video.playsInline = true;
        video.width = 480;  // Set explicit dimensions
        video.height = 360;
        video.className = 'sign-video';
        
        // Create source - placing the source element correctly
        const source = document.createElement('source');
        source.src = videoPath;
        source.type = 'video/mp4';
        video.appendChild(source);
        
        // Create play button for mobile/browser autoplay policy
        const playButton = document.createElement('button');
        playButton.className = 'play-button';
        playButton.textContent = 'Play Video';
        playButton.style.display = 'none'; // Hide initially, show only if needed
        
        // Create loading indicator
        const loadingIndicator = document.createElement('div');
        loadingIndicator.className = 'loading-indicator';
        loadingIndicator.innerHTML = 'Loading video...';

        // Create close button
        const closeButton = document.createElement('button');
        closeButton.className = 'close-button';
        closeButton.innerHTML = '&times;';
        closeButton.title = 'Close';
        
        // Add event handlers for video
        video.addEventListener('loadstart', function() {
            loadingIndicator.style.display = 'block';
        });
        
        video.addEventListener('canplay', function() {
            loadingIndicator.style.display = 'none';
            // Try to play automatically
            video.play()
                .catch(error => {
                    console.warn('Autoplay prevented:', error);
                    playButton.style.display = 'block';
                });
        });
        
        video.addEventListener('error', function(e) {
            console.error('Video error:', e);
            console.error('Error code:', video.error ? video.error.code : 'unknown');
            console.error('Error message:', video.error ? video.error.message : 'unknown');
            
            loadingIndicator.style.display = 'none';
            
            const errorMessage = document.createElement('div');
            errorMessage.className = 'video-error';
            errorMessage.innerHTML = `
                <p>Sorry, the video for "${word}" could not be loaded.</p>
                <p>Error: ${video.error ? video.error.message : 'Unknown error'}</p>
                <p>Please check if the video file exists at: ${videoPath}</p>
            `;
            
            // Replace video with error message
            if (video.parentNode) {
                video.parentNode.replaceChild(errorMessage, video);
                // Also remove the play button if visible
                if (playButton.parentNode) {
                    playButton.parentNode.removeChild(playButton);
                }
            }
        });
        
        // Handle play button click
        playButton.addEventListener('click', function() {
            video.play()
                .then(() => {
                    playButton.style.display = 'none';
                })
                .catch(error => {
                    console.error('Play failed:', error);
                });
        });
        
        // Assemble elements
        popupContent.appendChild(closeButton);
        popupContent.appendChild(title);
        popupContent.appendChild(loadingIndicator);
        popupContent.appendChild(video);
        popupContent.appendChild(playButton);
        
        popup.appendChild(popupContent);
        document.body.appendChild(popup);
        
        // Add close functionality
        closeButton.addEventListener('click', function() {
            // Pause video when closing
            video.pause();
            popup.classList.add('fade-out');
            setTimeout(() => {
                popup.remove();
            }, 300);
        });
        
        // Close when clicking outside the popup content
        popup.addEventListener('click', function(e) {
            if (e.target === popup) {
                // Pause video when closing
                video.pause();
                popup.classList.add('fade-out');
                setTimeout(() => {
                    popup.remove();
                }, 300);
            }
        });
        
        // Close on escape key
        const escHandler = function(e) {
            if (e.key === 'Escape') {
                const openPopup = document.querySelector('.video-popup');
                if (openPopup) {
                    // Pause video when closing
                    if (video) video.pause();
                    openPopup.classList.add('fade-out');
                    setTimeout(() => {
                        openPopup.remove();
                        // Clean up the event listener
                        document.removeEventListener('keydown', escHandler);
                    }, 300);
                }
            }
        };
        
        document.addEventListener('keydown', escHandler);
        
        // Add animation class after a small delay to trigger CSS transition
        setTimeout(() => {
            popup.classList.add('active');
        }, 10);
    }
    
    
    // Handle video feed errors
    videoFeed.addEventListener('error', function() {
        console.error('Video feed error:', this.error);
        showError('Video feed error. Please check your camera permissions and refresh the page.');
    });
});

