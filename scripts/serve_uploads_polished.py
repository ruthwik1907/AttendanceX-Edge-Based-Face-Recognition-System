from flask import Flask, request, render_template_string, redirect, url_for, flash, send_from_directory
from pathlib import Path
import uuid
from werkzeug.utils import secure_filename
import logging
import sys

app = Flask(__name__, static_folder='static')
app.config['SECRET_KEY'] = 'attendancex-secret-key'
UP = Path('data/enroll')
UP.mkdir(parents=True, exist_ok=True)


# --- NEW: Added CSS for Camera UI and Modal-like structure ---
HTML = '''
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AttendanceX — Enrollment Upload</title>
  <link rel="icon" type="image/png" href="/static/logo.png">
  <meta name="theme-color" content="#0b69ff">
  <style>
    :root{
      --rec-purple: #6A1B9A;
      --rec-gold: #FBC02D;
      --rec-cream: #FAF5EF;
      --rec-grey: #212121;
      --muted: rgba(33,33,33,0.56);
      --card: #ffffff;
      --glass: rgba(255,255,255,0.6);
      --border: rgba(33,33,33,0.06);
      --rec-green: #388E3C;
    }

    *{box-sizing:border-box}
    
    html,body{
      height:100%;
      margin:0;
      font-family:Inter,system-ui,-apple-system,'Segoe UI',Roboto,Helvetica,Arial;
      color:var(--rec-grey);
      background-color:var(--rec-cream);
      background-image:
        linear-gradient(45deg, transparent 48%, rgba(33,33,33,0.04) 49%, rgba(33,33,33,0.04) 51%, transparent 52%),
        linear-gradient(-45deg, transparent 48%, rgba(33,33,33,0.04) 49%, rgba(33,33,33,0.04) 51%, transparent 52%);
      background-size:28px 28px;
      background-position:0 0, 14px 14px;
      -webkit-font-smoothing:antialiased;
    }

    /* center everything vertically and horizontally */
    .viewport {
      min-height:100vh;
      display:flex;
      align-items:center;
      justify-content:center;
      padding:20px;
    }
    
    .wrap{width:100%; max-width:920px;padding:12px}
    .card{background:var(--card);border-radius:16px;padding:20px;box-shadow:0 12px 36px rgba(20,34,60,0.06);border:1px solid var(--border);overflow:hidden}
    
    header{display:flex;align-items:center;justify-content:space-between;gap:16px;margin-bottom:14px}
    .brand{display:flex;gap:12px;align-items:center}
    .logo{width:56px;height:56px;border-radius:10px;overflow:hidden;flex-shrink:0;display:inline-flex;align-items:center;justify-content:center;background:linear-gradient(180deg, var(--rec-purple), #4b1a7a);box-shadow:0 6px 18px rgba(106,27,154,0.08)}
    .logo img{width:100%;height:100%;object-fit:contain;display:block}
    h1{margin:0;font-size:20px;color:var(--rec-purple)}
    p.lead{margin:4px 0 0;color:var(--muted);font-size:13px}

    form{display:grid;gap:14px}
    label{font-weight:700;color:var(--rec-grey);font-size:13px;margin-bottom:6px;display:block}
    input[type=text]{width:100%;padding:12px 14px;border-radius:10px;border:1px solid var(--border);font-size:15px;background:#fff}
    .meta{font-size:13px;color:var(--muted)}

    .drop{border:2px dashed rgba(106,27,154,0.18);padding:14px;border-radius:12px;background:linear-gradient(180deg,#fff,#fbfdff);display:flex;flex-direction:column;gap:8px;align-items:center;justify-content:center;min-height:120px;cursor:pointer;transition:transform .15s ease, box-shadow .15s ease;box-shadow:0 4px 12px rgba(0,0,0,0.05)}
    .drop.dragover{box-shadow:0 12px 30px rgba(106,27,154,0.08);transform:translateY(-6px);border-color:var(--rec-gold)}
    .drop .hint{font-weight:700;color:var(--rec-purple)}
    .drop .sub{color:var(--muted);font-size:13px}

    /* Camera UI Styles */
    #video-container {
        position: relative;
        margin-bottom: 12px;
        border-radius: 12px;
        overflow: hidden;
        background: #000;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        width: 100%; /* Ensure responsiveness */
        max-width: 100%;
        height: 0; /* Aspect ratio trick */
        padding-bottom: 75%; /* 4:3 Aspect Ratio */
    }
    #video {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        object-fit: cover;
        transform: scaleX(-1); /* Flip video for typical front-facing camera mirror effect */
    }
    #status-overlay {
        position: absolute;
        inset: 0;
        background: rgba(0,0,0,0.5);
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 1.1rem;
        padding: 20px; /* Added padding for better mobile display */
        text-align: center;
        z-index: 10;
        opacity: 1;
        transition: opacity 0.3s ease;
    }
    #status-overlay.hidden { opacity: 0; pointer-events: none; }
    #canvas { display: none; }

    /* Button styles */
    .btn-group { display: flex; gap: 10px; justify-content: center; align-items: center; margin-top: 10px; }
    .btn{background:linear-gradient(90deg, var(--rec-purple), #4b1a7a);color:#fff;padding:12px 16px;border-radius:10px;border:none;font-weight:800;cursor:pointer;box-shadow:0 8px 20px rgba(106,27,154,0.12);transition:transform .12s ease, box-shadow .12s ease; display: inline-flex; align-items: center; gap: 8px;}
    .btn:hover{transform:translateY(-4px);box-shadow:0 20px 40px rgba(106,27,154,0.18)}
    .btn.ghost{background:transparent;border:1px solid var(--border);color:var(--rec-purple);font-weight:700}
    .btn.ghost:hover{background:rgba(106,27,154,0.04);border-color:var(--rec-purple)}
    .btn.camera-capture { background: var(--rec-green); box-shadow:0 8px 20px rgba(56,142,60,0.2); }
    .btn.camera-capture:hover { background: #2e7d32; box-shadow:0 20px 40px rgba(56,142,60,0.3); }
    .btn:disabled{ opacity: 0.5; cursor: not-allowed; transform: none; box-shadow: none; }

    .thumbs{display:grid;grid-template-columns:repeat(auto-fill,minmax(110px,1fr));gap:10px;margin-top:10px}
    .thumb{background:#fff;border-radius:10px;padding:8px;box-shadow:0 8px 18px rgba(15,23,42,0.04);display:flex;flex-direction:column;align-items:center;gap:8px;transition:transform .12s ease, box-shadow .12s ease;border:1px solid var(--border)}
    .thumb:hover{transform:translateY(-6px);box-shadow:0 18px 36px rgba(106,27,154,0.06)}
    .thumb img{width:100%;height:86px;object-fit:cover;border-radius:6px}
    .tname{font-size:12px;color:var(--muted);width:100%;text-align:center;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
    .remove{background:transparent;border:1px solid var(--border);padding:6px 8px;border-radius:8px;font-weight:700;color:var(--rec-purple);cursor:pointer;transition:all .12s ease}
    .remove:hover{background:rgba(106,27,154,0.06);border-color:var(--rec-purple)}

    .actions{display:flex;gap:10px;justify-content:flex-end;align-items:center}
    .footer{font-size:13px;color:var(--muted);text-align:center;margin-top:12px}

    .message{display:none;padding:12px;border-radius:10px;font-weight:700}
    .message.show{display:block}
    .message.error{background:rgba(106,27,154,0.08);color:var(--rec-purple);border:1px solid rgba(106,27,154,0.12)}
    .message.success{background:rgba(251,192,45,0.08);color:var(--rec-grey);border:1px solid rgba(251,192,45,0.12)}

    .uploading{position:fixed;inset:0;display:none;background:rgba(33,33,33,0.24);align-items:center;justify-content:center;z-index:60}
    .uploading.show{display:flex}
    .pill{background:var(--rec-gold);padding:10px 14px;border-radius:999px;font-weight:800;display:inline-flex;gap:8px;align-items:center;color:var(--rec-grey);font-size:14px}

    @media (max-width:700px){
      .thumbs{grid-template-columns:repeat(3,1fr)}
      .actions{flex-direction:column}
      .btn, .btn.ghost{width:100%}
      .drop{max-width:100%;padding:12px}
      .btn-group { flex-wrap: wrap; }
    }
    @media (max-width:420px){
      .thumbs{grid-template-columns:repeat(2,1fr)}
      header{flex-direction:column;align-items:flex-start;gap:10px}
      h1{font-size:18px}
      .logo{width:48px;height:48px}
    }
  </style>
</head>
<body>
  <div class="viewport">
    <div class="wrap">
      <div class="card" role="main" aria-labelledby="pageTitle">
        <header>
          <div class="brand" style="align-items:center">
            <div class="logo" aria-hidden="true">
              <img src="/static/logo.png" alt="Logo">
            </div>
            <div>
              <h1 id="pageTitle">AttendanceX — Student Enrollment</h1>
              <p class="lead">Upload clear face photos (recommended 30–40). Use front camera; vary angles & lighting.</p>
            </div>
          </div>

          <div style="text-align:right">
            <div style="font-size:13px;color:var(--muted);font-weight:700;padding:6px 10px;border-radius:8px;border:1px solid rgba(15,23,42,0.03);background:#fff">
              Local • Private
            </div>
          </div>
        </header>

        <div id="messageBox" class="message" role="status" aria-live="polite"></div>

        <form id="uploadForm" method="post" enctype="multipart/form-data" novalidate>
          <div>
            <label for="reg">Registration ID (Roll number — exactly 9 digits)</label>
            <input id="reg" name="reg" type="text" inputmode="numeric" pattern="\\d{9}" minlength="9" maxlength="9" required aria-required="true" autocomplete="off" placeholder="e.g. 220101001">
            <div class="meta">Files will be saved to <code>data/enroll/&lt;RegistrationID&gt;/</code></div>
          </div>

          <div>
            <label for="files">Photos</label>

            <!-- File Upload UI -->
            <div id="upload-ui">
                <div id="drop" class="drop" tabindex="0" role="button" aria-label="Add photos">
                  <div class="hint">Tap to browse or drop images here</div>
                  <div class="sub">Select multiple clear face photos — recommended 30–40.</div>
                </div>
                <button type="button" id="toggleToCamera" class="btn ghost" style="width:100%;margin-top:10px;">
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/><circle cx="12" cy="13" r="4"/></svg>
                    Capture 30 Photos with Camera (Optional)
                </button>
            </div>

            <!-- Camera UI -->
            <div id="camera-ui" style="display:none; text-align:center;">
                
                <!-- NEW Camera Selector Dropdown -->
                <div style="margin-bottom: 10px; text-align: left; max-width: 500px; margin: 0 auto 12px;">
                    <label for="camera-select" style="font-weight: 700; font-size: 13px; margin-bottom: 4px; display: block;">Select Camera</label>
                    <select id="camera-select" style="width: 100%; padding: 8px 10px; border-radius: 8px; border: 1px solid var(--border); font-size: 14px; background: #fff;"></select>
                </div>
                
                <div id="video-container">
                    <video id="video" autoplay playsinline></video>
                    <canvas id="canvas"></canvas>
                    <div id="status-overlay">Camera Loading...</div>
                </div>

                <div class="btn-group">
                    <button type="button" id="capture-btn" class="btn camera-capture" disabled>
                        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M12 8v8"/><path d="M8 12h8"/></svg>
                        Take Photo (<span id="photo-count-display">0</span> / 30)
                    </button>
                    <button type="button" id="finish-camera-btn" class="btn" disabled>
                        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>
                        Finish & Use Photos
                    </button>
                    <!-- RENAMED ID to reflect 'switch' action -->
                    <button type="button" id="switchToFileBtn" class="btn ghost">
                        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 6L6 18"/><path d="M6 6l12 12"/></svg>
                        Stop Camera & Switch to File Upload
                    </button>
                </div>
                <div class="meta" style="margin-top: 8px;">
                    Capture exactly **30** photos to enable the "Finish" button.
                </div>
            </div>

            <!-- Common display for selected files (from upload OR camera) -->
            <div id="thumbs" class="thumbs" aria-live="polite" aria-atomic="true" style="margin-top:10px"></div>
            <div class="meta" id="countInfo" style="margin-top:8px">No files selected</div>
          </div>

          <!-- Hidden native file input (still needed for JS interaction) -->
          <input id="files" name="files" type="file" accept="image/*" multiple style="display:none" aria-hidden="true">

          <div class="actions" style="margin-top:6px">
            <button type="button" id="clearBtn" class="btn ghost" aria-label="Clear selected files">Clear All Photos</button>
            <button type="submit" class="btn" aria-label="Upload files">Upload Enrollment Data</button>
          </div>
        </form>

        <div class="footer">Need help? Contact the admin. Files remain private on this machine.</div>
      </div>
    </div>
  </div>

  <div id="uploading" class="uploading" aria-hidden="true"><div class="pill">Uploading…</div></div>

  <script>
    (function(){
      // DOM Elements
      const drop = document.getElementById('drop');
      const fileInput = document.getElementById('files');
      const thumbs = document.getElementById('thumbs');
      const countInfo = document.getElementById('countInfo');
      const clearBtn = document.getElementById('clearBtn');
      const form = document.getElementById('uploadForm');
      const uploading = document.getElementById('uploading');
      const regInput = document.getElementById('reg');
      const messageBox = document.getElementById('messageBox');
      
      const uploadUI = document.getElementById('upload-ui');
      const cameraUI = document.getElementById('camera-ui');
      const toggleToCameraBtn = document.getElementById('toggleToCamera');
      const switchToFileBtn = document.getElementById('switchToFileBtn');
      
      const video = document.getElementById('video');
      const canvas = document.getElementById('canvas');
      const captureBtn = document.getElementById('capture-btn');
      const photoCountDisplay = document.getElementById('photo-count-display');
      const finishCameraBtn = document.getElementById('finish-camera-btn');
      const statusOverlay = document.getElementById('status-overlay');
      // NEW element for camera selection
      const cameraSelect = document.getElementById('camera-select');

      const TARGET_PHOTO_COUNT = 30;
      let filesState = []; // Holds all File objects (uploaded OR captured)
      let cameraFiles = []; // Holds captured File objects temporarily
      let cameraStream = null;
      let videoDevices = []; // Holds list of available video input devices

      // --- Utility Functions ---

      // Converts Base64 DataURL (from canvas) to a File object
      function dataURLtoFile(dataurl, filename) {
        let arr = dataurl.split(','),
            mime = arr[0].match(/:(.*?);/)[1],
            bstr = atob(arr[1]), 
            n = bstr.length, 
            u8arr = new Uint8Array(n);
        while(n--){
            u8arr[n] = bstr.charCodeAt(n);
        }
        return new File([u8arr], filename, {type:mime});
      }

      function human(n){ return n + (n===1? ' file' : ' files'); }

      function updateCount(){ 
        countInfo.textContent = filesState.length ? 
            human(filesState.length) + ' selected for upload.' : 
            'No files selected'; 
      }

      function showMessage(text, type='error'){
        console.log('UI MESSAGE (' + type.toUpperCase() + '):', text);
        messageBox.className = 'message show ' + (type === 'success' ? 'success' : 'error');
        messageBox.textContent = text;
        if(type === 'success'){ setTimeout(()=>{ messageBox.className = 'message'; }, 5000); }
      }

      function clearMessage(){ messageBox.className = 'message'; messageBox.textContent = ''; }

      // --- UI & Rendering ---

      function renderThumbs(){
        thumbs.innerHTML = '';
        filesState.forEach((f, i) => {
          const div = document.createElement('div'); div.className = 'thumb';
          const img = document.createElement('img');
          const name = document.createElement('div'); name.className = 'tname'; name.textContent = f.name;
          const rem = document.createElement('button'); rem.className = 'remove'; rem.textContent = 'Remove';
          rem.setAttribute('aria-label','Remove ' + f.name);
          
          rem.addEventListener('click', e => { 
            e.preventDefault(); 
            // Check if this file came from the camera session
            const cameraIndex = cameraFiles.findIndex(cf => cf === f); 
            if (cameraIndex > -1) {
                cameraFiles.splice(cameraIndex, 1);
            }
            filesState.splice(i,1); 
            renderThumbs(); 
            updateCount(); 
            updateCameraCaptureUI();
          });

          div.appendChild(img);
          div.appendChild(name);
          div.appendChild(rem);
          thumbs.appendChild(div);
          
          // Use FileReader only for files without an object URL (i.e., uploaded files)
          // For camera files, we can use the URL.createObjectURL or simply the FileReader
          const reader = new FileReader();
          reader.onload = e => img.src = e.target.result;
          reader.readAsDataURL(f);
        });
        updateCount();
      }

      function handleFiles(list){
        const arr = Array.from(list).filter(f=>f.type && f.type.startsWith('image/'));
        if(!arr.length) return;
        const existingKeys = new Set(filesState.map(f=>f.name + '|' + f.size));
        const toAdd = arr.filter(f=> !existingKeys.has(f.name + '|' + f.size));
        filesState = filesState.concat(toAdd);
        renderThumbs();
        console.log('Files added via upload:', toAdd.length);
      }
      
      function updateCameraCaptureUI() {
          const count = cameraFiles.length;
          photoCountDisplay.textContent = count;
          
          if (count >= TARGET_PHOTO_COUNT) {
              captureBtn.disabled = true;
              finishCameraBtn.disabled = false;
              captureBtn.textContent = 'Photo Limit Reached';
          } else {
              captureBtn.disabled = cameraStream === null;
              finishCameraBtn.disabled = true;
              captureBtn.textContent = 'Take Photo (' + count + ' / ' + TARGET_PHOTO_COUNT + ')';
          }
      }

      // --- Camera Logic ---

      function stopCamera(){
        if(cameraStream){
          console.log('Stopping camera stream.');
          cameraStream.getTracks().forEach(track => track.stop());
          cameraStream = null;
          video.srcObject = null;
        }
      }
      
      // Function to enumerate and populate the camera list
      async function getAndPopulateCameras() {
        if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) {
            console.warn("enumerateDevices() not supported.");
            cameraSelect.innerHTML = '<option value="">Device listing not supported</option>';
            cameraSelect.disabled = true;
            return;
        }

        try {
            const devices = await navigator.mediaDevices.enumerateDevices();
            videoDevices = devices.filter(device => device.kind === 'videoinput');
            
            cameraSelect.innerHTML = '';
            
            if (videoDevices.length === 0) {
                cameraSelect.innerHTML = '<option value="">No video input devices found</option>';
                cameraSelect.disabled = true;
                statusOverlay.textContent = 'No camera device found.';
                return;
            }

            // Start camera with the first device listed to ensure labels are populated
            const initialDeviceId = videoDevices[0].deviceId;
            await startCamera(initialDeviceId);
            
            // Re-enumerate to get proper device labels (often masked until permissions are granted)
            const devicesWithLabels = await navigator.mediaDevices.enumerateDevices();
            videoDevices = devicesWithLabels.filter(device => device.kind === 'videoinput');
            
            cameraSelect.innerHTML = '';
            videoDevices.forEach((device, index) => {
                const option = document.createElement('option');
                option.value = device.deviceId;
                // Use default label if device.label is empty
                option.textContent = device.label || `Camera ${index + 1}`; 
                option.selected = (device.deviceId === initialDeviceId);
                cameraSelect.appendChild(option);
            });
            cameraSelect.disabled = false;


        } catch (err) {
            console.error("Error during camera setup (initial start or enumeration):", err);
            cameraSelect.innerHTML = '<option value="">Error setting up camera</option>';
            cameraSelect.disabled = true;
            // The specific error message from startCamera will handle the UI message
        }
      }

      // Updated startCamera to accept a deviceId
      async function startCamera(deviceId) {
        stopCamera(); // Stop any existing stream first

        statusOverlay.textContent = 'Camera Loading...';
        statusOverlay.classList.remove('hidden');
        captureBtn.disabled = true;

        if (typeof window.isSecureContext === 'boolean' && !window.isSecureContext) {
            const secureError = 'ERROR: Camera access requires HTTPS (secure connection). Please run this application on a server with SSL.';
            statusOverlay.textContent = secureError;
            showMessage('Camera access failed: The browser requires a secure (HTTPS) connection to access your camera.', 'error');
            console.error(secureError);
            return;
        }
        
        // Define constraints using the provided deviceId
        const constraints = {
            video: {
                deviceId: deviceId ? { exact: deviceId } : undefined,
                // We no longer rely on facingMode as we have a list of deviceIds,
                // but we keep the audio: false
            }, 
            audio: false 
        };
        
        console.log('Attempting to start camera with deviceId:', deviceId, 'Constraints:', constraints);

        try {
          // Request stream using constraints
          const stream = await navigator.mediaDevices.getUserMedia(constraints);
          cameraStream = stream;
          video.srcObject = stream;
          video.onloadedmetadata = () => {
            console.log('Camera metadata loaded successfully.');
            statusOverlay.classList.add('hidden');
            updateCameraCaptureUI();
          };
          video.onerror = () => {
             const streamError = 'Error: Video stream failed.';
             statusOverlay.textContent = streamError;
             captureBtn.disabled = true;
             console.error(streamError);
          };

        } catch (err) {
          console.error("Camera stream failed: ", err.name, err);
          let errorMessage = 'ERROR: Camera stream failed. Access denied or unavailable.';

          if (err.name === 'NotAllowedError') {
             errorMessage = 'Access denied: You must grant permission for the browser to use your camera.';
          } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
             errorMessage = 'No camera found: Please ensure a camera is connected and enabled.';
          } else if (err.name === 'SecurityError') {
             errorMessage = 'Security Error: Camera access blocked. Ensure you are using a secure context (HTTPS).';
          } else if (err.name === 'ConstraintNotSatisfiedError') {
             errorMessage = 'Camera constraints could not be satisfied (e.g., trying to use rear camera that does not exist).';
          }
          
          statusOverlay.textContent = errorMessage;
          captureBtn.disabled = true;
          showMessage('Camera stream failed: ' + errorMessage, 'error');
        }
      }

      function capturePhoto() {
          if (!cameraStream || cameraFiles.length >= TARGET_PHOTO_COUNT) return;

          console.log('Capturing photo:', cameraFiles.length + 1);

          // Set canvas dimensions to video dimensions for a clean capture
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
          
          const ctx = canvas.getContext('2d');
          
          // Apply the flip transform to the canvas context before drawing
          ctx.translate(canvas.width, 0);
          ctx.scale(-1, 1);
          
          // Draw the video frame onto the canvas
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
          
          // Reset transform for potential future use (though not needed here)
          ctx.setTransform(1, 0, 0, 1, 0, 0);

          // Get the image data as a Base64 encoded JPEG
          const dataURL = canvas.toDataURL('image/jpeg', 0.9);
          
          // Convert DataURL to a File object
          const fileName = `cam_photo_${new Date().getTime()}.jpeg`;
          const photoFile = dataURLtoFile(dataURL, fileName);
          
          cameraFiles.push(photoFile);
          
          // Immediately add to filesState and re-render
          filesState.push(photoFile);
          renderThumbs(); 
          
          updateCameraCaptureUI();
      }
      
      function toggleUI(mode) {
          clearMessage();
          if (mode === 'camera') {
              console.log('Switching to Camera mode.');
              uploadUI.style.display = 'none';
              cameraUI.style.display = 'block';
              
              if (!cameraStream) {
                  // Only enumerate and start the camera if it's not already running
                  getAndPopulateCameras(); 
              } else {
                  // If switching back from upload and stream is still active
                  updateCameraCaptureUI();
              }

              // Hide the file input as it's not relevant here
              fileInput.style.display = 'none'; 

              // Reset cameraFiles to track only current captures
              cameraFiles = filesState.filter(f => f.name.startsWith('cam_photo_'));

          } else if (mode === 'upload') {
              console.log('Switching to Upload mode.');
              stopCamera();
              uploadUI.style.display = 'block';
              cameraUI.style.display = 'none';
              // Restore the hidden file input for drag/drop to function
              fileInput.style.display = 'none'; // Stays hidden, interaction via 'drop' div

          }
      }

      function finishCameraSession() {
          console.log('Finishing camera session.');
          if (cameraFiles.length !== TARGET_PHOTO_COUNT) {
              showMessage('You must capture exactly ' + TARGET_PHOTO_COUNT + ' photos before finishing.', 'error');
              return;
          }
          
          showMessage(
              'Successfully captured ' + TARGET_PHOTO_COUNT + ' photos. They are now included in the final upload list below.',
              'success'
          );
          toggleUI('upload');
          updateCount();
      }


      // --- Event Listeners ---
      
      // File Upload Listeners (Original functionality)
      drop.addEventListener('click', ()=> fileInput.click());
      drop.addEventListener('keydown', e => { if(e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput.click(); } });
      fileInput.addEventListener('change', ()=> { handleFiles(fileInput.files); fileInput.value=''; });

      ['dragenter','dragover'].forEach(ev => drop.addEventListener(ev, e=>{ e.preventDefault(); e.stopPropagation(); drop.classList.add('dragover'); }));
      ['dragleave','drop'].forEach(ev => drop.addEventListener(ev, e=>{ e.preventDefault(); e.stopPropagation(); drop.classList.remove('dragover'); }));
      drop.addEventListener('drop', e=>{ if(e.dataTransfer) handleFiles(e.dataTransfer.files); });

      clearBtn.addEventListener('click', ()=>{ 
          console.log('Clearing all files.');
          filesState=[]; 
          cameraFiles=[];
          renderThumbs(); 
          clearMessage(); 
          updateCameraCaptureUI(); // Update camera button if in camera mode
      });
      
      // Camera Listeners (New functionality)
      toggleToCameraBtn.addEventListener('click', () => toggleUI('camera'));
      switchToFileBtn.addEventListener('click', () => toggleUI('upload')); 
      captureBtn.addEventListener('click', capturePhoto);
      finishCameraBtn.addEventListener('click', finishCameraSession);
      
      // NEW: Camera selection change listener
      cameraSelect.addEventListener('change', () => {
          console.log('Camera selected:', cameraSelect.value);
          // Stop current stream and start a new one with the selected device ID
          startCamera(cameraSelect.value);
      });


      // Form Submission Logic
      form.addEventListener('submit', async e=>{
        e.preventDefault();
        stopCamera(); // Ensure camera stops if still running on submission
        clearMessage();
        
        const reg = regInput.value.trim();
        const valid = /^[0-9]{9}$/.test(reg);
        
        if(!valid){ showMessage('Registration ID must be exactly 9 digits (numbers only).','error'); regInput.focus(); return; }
        if(!filesState.length){ showMessage('Please select or capture at least one image.','error'); return; }
        
        console.log('Starting upload for Registration ID:', reg, 'Files:', filesState.length);

        const fd = new FormData();
        fd.append('reg', reg);
        filesState.forEach((f)=> fd.append('files', f, f.name));
        
        uploading.classList.add('show'); uploading.setAttribute('aria-hidden','false');

        try {
          const resp = await fetch('', { method:'POST', body: fd });
          uploading.classList.remove('show'); uploading.setAttribute('aria-hidden','true');
          if(resp.ok){
            showMessage('Uploaded ' + filesState.length + ' file' + (filesState.length !== 1 ? 's' : '') + ' for ' + reg + '.', 'success');
            console.log('Upload successful.');
            filesState = []; cameraFiles = []; renderThumbs(); // Clear state on success
          } else {
            const text = await resp.text(); 
            showMessage(text || 'Upload failed', 'error');
            console.error('Upload failed with status', resp.status, 'Response:', text);
          }
        } catch (err) {
          uploading.classList.remove('show'); uploading.setAttribute('aria-hidden','true');
          showMessage('Network error. Please try again.', 'error');
          console.error('Upload network error:', err);
        }
      });
      
      // Initial render call
      renderThumbs();
      
    })();
  </script>
</body>
</html>
'''

@app.route('/', methods=['GET','POST'])
def upload():
    if request.method == 'POST':
        reg = request.form.get('reg','').strip()
        if not reg:
            logging.error('Upload failed: Registration ID required.')
            return 'Registration ID required', 400
        # server-side validation: exactly 9 digits
        if not (reg.isdigit() and len(reg) == 9):
            logging.error(f'Upload failed for {reg}: Invalid Registration ID format.')
            return 'Registration ID must be exactly 9 digits (numbers only).', 400
        
        files = request.files.getlist('files')
        if not files:
            logging.error(f'Upload failed for {reg}: No files received.')
            return 'No files', 400
            
        outdir = UP / reg
        outdir.mkdir(parents=True, exist_ok=True)
        saved = 0
        
        for f in files:
            # Generate a secure and unique filename, especially for camera files (which have simple filenames)
            filename = secure_filename(f.filename)
            # Ensure a unique base filename if secure_filename is empty (e.g., for some captured files)
            if not filename:
                filename = f'image_{uuid.uuid4().hex[:8]}.jpg'
                
            # Final destination path with UUID to prevent naming collisions
            dest = outdir / f'{reg}_{uuid.uuid4().hex[:8]}_{filename}'
            
            try:
                # Use a simpler filename for saving if the provided filename is too long or complex
                f.save(str(dest))
                saved += 1
            except Exception as e:
                logging.error(f"Error saving file {f.filename} for {reg}: {e}")
                # We log the error but allow the process to continue for other files

        logging.info(f"Successfully saved {saved} files for Registration ID {reg} in {outdir}")
        flash(f'Uploaded {saved} file{"s" if saved != 1 else ""} for {reg}')
        return redirect(url_for('upload'))
    return render_template_string(HTML)

@app.route('/static/<path:p>')
def static_file(p):
    # This route is a placeholder, as the user likely has a static/logo.png
    static_dir = Path(app.static_folder)
    target = static_dir / p
    # Placeholder image if the actual logo.png is missing
    if not target.exists():
        # Using a simple placeholder URL that doesn't rely on local files
        return redirect(f'https://placehold.co/56x56/6A1B9A/FFFFFF?text=Logo', code=302)
        
    return send_from_directory(str(static_dir), p)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=5000)
    args = parser.parse_args()

    app.run(host=args.host, port=args.port)