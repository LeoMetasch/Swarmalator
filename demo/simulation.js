/**
 * Swarmalator Interactive Simulation
 * JavaScript implementation based on the Python Swarm class
 */

class Swarmalator {
    constructor(params = {}) {
        this.N = params.N || 100;
        this.dt = params.dt || 0.1;
        this.J = params.J ?? 0.9;
        this.K = params.K ?? 0;
        this.chirality = params.chirality || false;
        this.phaseCoupling = params.phaseCoupling || false;
        this.freqMode = params.freqMode || 'zero';
        this.eps = 1e-12;

        // Predator parameters
        this.predatorEnabled = false;
        this.predX = 0;
        this.predY = 0;
        this.huntingStrength = params.huntingStrength ?? 1.0;
        this.predSpeed = params.predSpeed ?? 1.0;

        this.initialize();
    }

    initialize() {
        // Positions
        this.x = new Float64Array(this.N);
        this.y = new Float64Array(this.N);
        // Phases
        this.theta = new Float64Array(this.N);
        // Natural frequencies
        this.omega = new Float64Array(this.N);
        // Velocities (for chirality)
        this.vx = new Float64Array(this.N);
        this.vy = new Float64Array(this.N);

        // Random initialization
        for (let i = 0; i < this.N; i++) {
            this.x[i] = (Math.random() * 2 - 1);
            this.y[i] = (Math.random() * 2 - 1);
            this.theta[i] = (Math.random() * 2 - 1) * Math.PI;
        }

        // Initialize natural frequencies based on mode
        this.initOmega();

        // Compute phase coupling Q matrices
        this.computeQ();

        // Initialize velocities if chirality is enabled
        if (this.chirality) {
            this.updateVelocities();
        }
    }

    initOmega() {
        switch (this.freqMode) {
            case 'zero':
                this.omega.fill(0);
                break;
            case 'uniform':
                this.omega.fill(1);
                break;
            case 'bimodal':
                const half = Math.floor(this.N / 2);
                for (let i = 0; i < this.N; i++) {
                    this.omega[i] = i < half ? 1 : -1;
                }
                break;
            case 'random':
                for (let i = 0; i < this.N; i++) {
                    this.omega[i] = Math.random() * 2 - 1;
                }
                break;
        }
    }

    computeQ() {
        // Q_x and Q_theta matrices for phase coupling
        // These depend on the sign of omega_i - omega_j
        this.Q_x = [];
        this.Q_theta = [];

        if (!this.phaseCoupling || this.freqMode === 'zero') {
            // No phase coupling - use zeros
            for (let i = 0; i < this.N; i++) {
                this.Q_x.push(new Float64Array(this.N));
                this.Q_theta.push(new Float64Array(this.N));
            }
            return;
        }

        for (let i = 0; i < this.N; i++) {
            this.Q_x.push(new Float64Array(this.N));
            this.Q_theta.push(new Float64Array(this.N));

            const omega_norm_i = this.omega[i] === 0 ? 1 : Math.abs(this.omega[i]);
            const s_i = this.omega[i] / omega_norm_i;

            for (let j = 0; j < this.N; j++) {
                const omega_norm_j = this.omega[j] === 0 ? 1 : Math.abs(this.omega[j]);
                const s_j = this.omega[j] / omega_norm_j;
                const diff_sign = Math.abs(s_j - s_i);

                this.Q_x[i][j] = (Math.PI / 2) * diff_sign;
                this.Q_theta[i][j] = (Math.PI / 4) * diff_sign;
            }
        }
    }

    updateVelocities() {
        for (let i = 0; i < this.N; i++) {
            this.vx[i] = this.omega[i] * Math.cos(this.theta[i] + Math.PI / 2);
            this.vy[i] = this.omega[i] * Math.sin(this.theta[i] + Math.PI / 2);
        }
    }

    /**
     * Calculate the correlation order parameter S± (max of S+ and S-)
     * S measures correlation between spatial position angle (φ) and phase (θ)
     */
    correlationOrderParameter() {
        let sumPlusReal = 0, sumPlusImag = 0;
        let sumMinusReal = 0, sumMinusImag = 0;

        for (let i = 0; i < this.N; i++) {
            const phi = Math.atan2(this.y[i], this.x[i]);
            const plusAngle = phi + this.theta[i];
            const minusAngle = phi - this.theta[i];

            sumPlusReal += Math.cos(plusAngle);
            sumPlusImag += Math.sin(plusAngle);
            sumMinusReal += Math.cos(minusAngle);
            sumMinusImag += Math.sin(minusAngle);
        }

        const sPlus = Math.sqrt(sumPlusReal * sumPlusReal + sumPlusImag * sumPlusImag) / this.N;
        const sMinus = Math.sqrt(sumMinusReal * sumMinusReal + sumMinusImag * sumMinusImag) / this.N;

        return Math.max(sPlus, sMinus);
    }

    /**
     * Calculate velocity order parameters
     * @param {Float64Array} xPrev - previous x positions
     * @param {Float64Array} yPrev - previous y positions
     * @param {Float64Array} thetaPrev - previous phases
     * @returns {Object} { V: mean spatial velocity, Omega: mean phase velocity }
     */
    calculateVelocityOrderParameter(xPrev, yPrev, thetaPrev) {
        let vSum = 0;
        let omegaSum = 0;

        for (let i = 0; i < this.N; i++) {
            // Spatial velocity
            const dx = this.x[i] - xPrev[i];
            const dy = this.y[i] - yPrev[i];
            vSum += Math.sqrt(dx * dx + dy * dy);

            // Phase velocity (accounting for wraparound)
            let dtheta = this.theta[i] - thetaPrev[i];
            dtheta = ((dtheta + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;
            omegaSum += Math.abs(dtheta) / this.dt;
        }

        return {
            V: vSum / this.N,
            Omega: omegaSum / this.N
        };
    }

    /**
     * Calculate synchrony order parameter R (global phase synchronization)
     * R = 1 means all oscillators have the same phase
     */
    calculateSynchronyOrder() {
        let sumReal = 0, sumImag = 0;
        for (let i = 0; i < this.N; i++) {
            sumReal += Math.cos(this.theta[i]);
            sumImag += Math.sin(this.theta[i]);
        }
        return Math.sqrt(sumReal * sumReal + sumImag * sumImag) / this.N;
    }

    /**
     * Classify the current state based on order parameters
     */
    classifyState(S, V, Omega) {
        if (S > 0.9 && V < 0.01 && Omega < 0.01) {
            return "Static Phase Wave";
        } else if (S > 0.1 && V >= 0.01 && Omega >= 0.01) {
            return "Active Phase Wave";
        } else if (S > 0.1 && V < 0.01 && Omega < 0.01) {
            return "Splintered Phase Wave";
        } else if (S <= 0.1) {
            const R = this.calculateSynchronyOrder();
            if (R > 0.9) {
                return "Static Sync";
            } else {
                return "Static Async";
            }
        } else {
            return "Transitioning";
        }
    }

    step() {
        const N = this.N;
        const eps = this.eps;

        // Compute pairwise quantities
        const xdot = new Float64Array(N);
        const ydot = new Float64Array(N);
        const thetadot = new Float64Array(N);

        for (let i = 0; i < N; i++) {
            let sumX = 0, sumY = 0, sumTheta = 0;

            for (let j = 0; j < N; j++) {
                if (i === j) continue;

                const dX = this.x[j] - this.x[i];
                const dY = this.y[j] - this.y[i];
                const dist2 = dX * dX + dY * dY;
                const dist = Math.sqrt(dist2);

                const dTheta = this.theta[j] - this.theta[i];
                const Q_x_ij = this.Q_x[i][j];
                const Q_theta_ij = this.Q_theta[i][j];

                const c = Math.cos(dTheta - Q_x_ij);
                const s = Math.sin(dTheta - Q_theta_ij);

                const coef = (1 + this.J * c) / (dist + eps) - 1 / (dist2 + eps);

                sumX += dX * coef;
                sumY += dY * coef;
                sumTheta += this.K * s / (dist + eps);
            }

            xdot[i] = sumX / N;
            ydot[i] = sumY / N;
            thetadot[i] = this.omega[i] + sumTheta / N;
        }

        // Add velocity from chirality
        if (this.chirality) {
            for (let i = 0; i < N; i++) {
                xdot[i] += this.vx[i];
                ydot[i] += this.vy[i];
            }
        }

        // Predator dynamics
        if (this.predatorEnabled) {
            // Find the closest swarmalator
            let minDistSq = Infinity;
            let nearestIdx = 0;
            for (let i = 0; i < N; i++) {
                const dx = this.x[i] - this.predX;
                const dy = this.y[i] - this.predY;
                const distSq = dx * dx + dy * dy;
                if (distSq < minDistSq) {
                    minDistSq = distSq;
                    nearestIdx = i;
                }
            }

            // Target the nearest swarmalator
            const targetX = this.x[nearestIdx];
            const targetY = this.y[nearestIdx];

            // Predator chases the nearest swarmalator
            const huntDx = targetX - this.predX;
            const huntDy = targetY - this.predY;
            const huntDist = Math.sqrt(huntDx * huntDx + huntDy * huntDy) + eps;

            this.predX += (huntDx / huntDist) * this.predSpeed * this.dt;
            this.predY += (huntDy / huntDist) * this.predSpeed * this.dt;

            // Agents are repelled from predator (inverse-square force)
            for (let i = 0; i < N; i++) {
                const predDx = this.x[i] - this.predX;
                const predDy = this.y[i] - this.predY;
                const dPred2 = predDx * predDx + predDy * predDy + eps;
                const dPred = Math.sqrt(dPred2);

                const repulsionMag = this.huntingStrength / dPred2;

                xdot[i] += (predDx / dPred) * repulsionMag;
                ydot[i] += (predDy / dPred) * repulsionMag;
            }
        }

        // Euler update
        for (let i = 0; i < N; i++) {
            this.x[i] += xdot[i] * this.dt;
            this.y[i] += ydot[i] * this.dt;
            this.theta[i] += thetadot[i] * this.dt;

            // Wrap theta to [-π, π)
            this.theta[i] = ((this.theta[i] + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;
        }

        // Update velocities for next step
        if (this.chirality) {
            this.updateVelocities();
        }
    }

    // Spawn predator at given coordinates
    spawnPredator(x, y) {
        this.predatorEnabled = true;
        this.predX = x;
        this.predY = y;
    }

    // Remove predator from simulation
    removePredator() {
        this.predatorEnabled = false;
    }
}

// ============ UI Controller ============

class SimulationController {
    constructor() {
        this.canvas = document.getElementById('canvas');
        this.ctx = this.canvas.getContext('2d');
        this.isPlaying = false;
        this.stepCount = 0;
        this.animationId = null;

        // Order parameter plot canvases
        this.sPlotCanvas = document.getElementById('sPlot');
        this.vPlotCanvas = document.getElementById('vPlot');
        this.omegaPlotCanvas = document.getElementById('omegaPlot');
        this.rPlotCanvas = document.getElementById('rPlot');

        this.sPlotCtx = this.sPlotCanvas.getContext('2d');
        this.vPlotCtx = this.vPlotCanvas.getContext('2d');
        this.omegaPlotCtx = this.omegaPlotCanvas.getContext('2d');
        this.rPlotCtx = this.rPlotCanvas.getContext('2d');

        // History arrays for time-series (max 200 points)
        this.maxHistory = 200;
        this.sHistory = [];
        this.vHistory = [];
        this.omegaHistory = [];
        this.rHistory = [];

        // Previous state for velocity calculations
        this.prevX = null;
        this.prevY = null;
        this.prevTheta = null;

        // Camera state
        this.zoom = 1.0;
        this.cameraX = 0; // Camera center in simulation coordinates
        this.cameraY = 0;
        this.autoTrack = false;

        // Initialize simulation with default params
        this.initSimulation();

        // Bind UI elements
        this.bindControls();

        // Bind canvas click for predator spawn
        this.bindCanvasClick();

        // Bind camera controls
        this.bindCameraControls();

        // Initial render
        this.render();
        this.updateOrderParameters();
    }

    initSimulation() {
        this.sim = new Swarmalator({
            N: parseInt(document.getElementById('nSlider').value),
            dt: parseFloat(document.getElementById('dtSlider').value),
            J: parseFloat(document.getElementById('jSlider').value),
            K: parseFloat(document.getElementById('kSlider').value),
            chirality: document.getElementById('chiralityToggle').checked,
            phaseCoupling: document.getElementById('phaseCouplingToggle').checked,
            freqMode: document.getElementById('freqModeSelect').value
        });
        this.stepCount = 0;
        document.getElementById('stepCount').textContent = '0';
    }

    bindControls() {
        // Play/Pause button
        document.getElementById('playPauseBtn').addEventListener('click', () => this.togglePlay());

        // Reset button
        document.getElementById('resetBtn').addEventListener('click', () => this.reset());

        // Sliders with live update
        const jSlider = document.getElementById('jSlider');
        const kSlider = document.getElementById('kSlider');
        const nSlider = document.getElementById('nSlider');
        const dtSlider = document.getElementById('dtSlider');

        jSlider.addEventListener('input', (e) => {
            document.getElementById('jValue').textContent = parseFloat(e.target.value).toFixed(2);
            this.sim.J = parseFloat(e.target.value);
        });

        kSlider.addEventListener('input', (e) => {
            document.getElementById('kValue').textContent = parseFloat(e.target.value).toFixed(2);
            this.sim.K = parseFloat(e.target.value);
        });

        nSlider.addEventListener('input', (e) => {
            document.getElementById('nValue').textContent = e.target.value;
        });

        // N slider - update particle count without full reset
        nSlider.addEventListener('change', (e) => {
            const newN = parseInt(e.target.value);
            this.updateParticleCount(newN);
        });

        dtSlider.addEventListener('input', (e) => {
            document.getElementById('dtValue').textContent = parseFloat(e.target.value).toFixed(2);
            this.sim.dt = parseFloat(e.target.value);
        });

        // Hunting strength slider
        const huntingStrengthSlider = document.getElementById('huntingStrengthSlider');
        huntingStrengthSlider.addEventListener('input', (e) => {
            document.getElementById('huntingStrengthValue').textContent = parseFloat(e.target.value).toFixed(1);
            this.sim.huntingStrength = parseFloat(e.target.value);
        });

        // Toggles - update live without reset
        document.getElementById('chiralityToggle').addEventListener('change', (e) => {
            this.sim.chirality = e.target.checked;
            if (this.sim.chirality) {
                this.sim.updateVelocities();
            } else {
                this.sim.vx.fill(0);
                this.sim.vy.fill(0);
            }
        });

        document.getElementById('phaseCouplingToggle').addEventListener('change', (e) => {
            this.sim.phaseCoupling = e.target.checked;
            this.sim.computeQ();
        });

        document.getElementById('freqModeSelect').addEventListener('change', (e) => {
            this.sim.freqMode = e.target.value;
            this.sim.initOmega();
            this.sim.computeQ();
            if (this.sim.chirality) {
                this.sim.updateVelocities();
            }
        });

        // Presets
        document.querySelectorAll('.btn-preset').forEach(btn => {
            btn.addEventListener('click', (e) => this.applyPreset(e.target.dataset.preset));
        });
    }

    bindCanvasClick() {
        this.canvas.addEventListener('click', (e) => {
            const rect = this.canvas.getBoundingClientRect();
            const clickX = e.clientX - rect.left;
            const clickY = e.clientY - rect.top;

            // Convert screen coordinates to simulation coordinates (accounting for zoom and pan)
            const baseScale = this.canvas.width / 5;
            const scale = baseScale * this.zoom;
            const offsetX = this.canvas.width / 2;
            const offsetY = this.canvas.height / 2;

            // Screen to sim: reverse the camera transform
            const simX = (clickX - offsetX) / scale + this.cameraX;
            const simY = -(clickY - offsetY) / scale + this.cameraY;

            // Spawn predator at clicked location
            this.sim.spawnPredator(simX, simY);
            this.render();
        });

        // Right-click to remove predator
        this.canvas.addEventListener('contextmenu', (e) => {
            e.preventDefault();
            this.sim.removePredator();
            this.render();
        });
    }

    bindCameraControls() {
        // Zoom slider
        const zoomSlider = document.getElementById('zoomSlider');
        zoomSlider.addEventListener('input', (e) => {
            this.zoom = parseFloat(e.target.value);
            document.getElementById('zoomValue').textContent = this.zoom.toFixed(1);
            this.render();
        });

        // Auto-track toggle
        document.getElementById('autoTrackToggle').addEventListener('change', (e) => {
            this.autoTrack = e.target.checked;
            if (this.autoTrack) {
                this.updateCameraTracking();
                this.render();
            }
        });

        // Reset camera button
        document.getElementById('resetCameraBtn').addEventListener('click', () => {
            this.zoom = 1.0;
            this.cameraX = 0;
            this.cameraY = 0;
            this.autoTrack = false;
            document.getElementById('zoomSlider').value = '1';
            document.getElementById('zoomValue').textContent = '1.0';
            document.getElementById('autoTrackToggle').checked = false;
            this.render();
        });

        // Mouse wheel zoom
        this.canvas.addEventListener('wheel', (e) => {
            e.preventDefault();
            const zoomDelta = e.deltaY < 0 ? 0.1 : -0.1;
            this.zoom = Math.max(0.1, Math.min(3.0, this.zoom + zoomDelta));
            document.getElementById('zoomSlider').value = this.zoom.toString();
            document.getElementById('zoomValue').textContent = this.zoom.toFixed(1);
            this.render();
        });
    }

    updateCameraTracking() {
        if (!this.autoTrack) return;

        // Calculate center of mass
        let comX = 0, comY = 0;
        for (let i = 0; i < this.sim.N; i++) {
            comX += this.sim.x[i];
            comY += this.sim.y[i];
        }
        comX /= this.sim.N;
        comY /= this.sim.N;

        // Smoothly interpolate camera position
        const smoothing = 0.1;
        this.cameraX += (comX - this.cameraX) * smoothing;
        this.cameraY += (comY - this.cameraY) * smoothing;
    }

    updateParticleCount(newN) {
        const oldN = this.sim.N;

        if (newN === oldN) return;

        // Create new arrays
        const newX = new Float64Array(newN);
        const newY = new Float64Array(newN);
        const newTheta = new Float64Array(newN);
        const newOmega = new Float64Array(newN);
        const newVx = new Float64Array(newN);
        const newVy = new Float64Array(newN);

        // Copy existing particles (up to min of old and new)
        const copyCount = Math.min(oldN, newN);
        for (let i = 0; i < copyCount; i++) {
            newX[i] = this.sim.x[i];
            newY[i] = this.sim.y[i];
            newTheta[i] = this.sim.theta[i];
            newOmega[i] = this.sim.omega[i];
            newVx[i] = this.sim.vx[i];
            newVy[i] = this.sim.vy[i];
        }

        // Add new particles if increasing
        for (let i = oldN; i < newN; i++) {
            newX[i] = Math.random() * 2 - 1;
            newY[i] = Math.random() * 2 - 1;
            newTheta[i] = (Math.random() * 2 - 1) * Math.PI;
        }

        // Update simulation
        this.sim.N = newN;
        this.sim.x = newX;
        this.sim.y = newY;
        this.sim.theta = newTheta;
        this.sim.omega = newOmega;
        this.sim.vx = newVx;
        this.sim.vy = newVy;

        // Reinitialize omega and Q matrices for new size
        this.sim.initOmega();
        this.sim.computeQ();

        if (this.sim.chirality) {
            this.sim.updateVelocities();
        }

        this.render();
    }

    applyPreset(preset) {
        const presets = {
            'static': { J: 1.0, K: 0.5, chirality: false, phaseCoupling: false, freqMode: 'zero' },
            'phase-wave': { J: 0.1, K: 1.0, chirality: false, phaseCoupling: false, freqMode: 'zero' },
            'chimera': { J: 0.9, K: 0.0, chirality: true, phaseCoupling: true, freqMode: 'bimodal' },
            'chaos': { J: -0.5, K: -0.5, chirality: true, phaseCoupling: false, freqMode: 'random' }
        };

        const p = presets[preset];
        if (!p) return;

        // Update UI
        document.getElementById('jSlider').value = p.J;
        document.getElementById('jValue').textContent = p.J.toFixed(2);
        document.getElementById('kSlider').value = p.K;
        document.getElementById('kValue').textContent = p.K.toFixed(2);
        document.getElementById('chiralityToggle').checked = p.chirality;
        document.getElementById('phaseCouplingToggle').checked = p.phaseCoupling;
        document.getElementById('freqModeSelect').value = p.freqMode;

        this.reset();
    }

    togglePlay() {
        this.isPlaying = !this.isPlaying;
        const btn = document.getElementById('playPauseBtn');
        const icon = document.getElementById('playIcon');

        if (this.isPlaying) {
            btn.innerHTML = '<span id="playIcon">⏸</span> Pause';
            btn.classList.add('playing');
            this.animate();
        } else {
            btn.innerHTML = '<span id="playIcon">▶</span> Play';
            btn.classList.remove('playing');
            if (this.animationId) {
                cancelAnimationFrame(this.animationId);
            }
        }
    }

    reset() {
        this.isPlaying = false;
        const btn = document.getElementById('playPauseBtn');
        btn.innerHTML = '<span id="playIcon">▶</span> Play';
        btn.classList.remove('playing');

        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
        }

        // Clear history arrays
        this.sHistory = [];
        this.vHistory = [];
        this.omegaHistory = [];
        this.rHistory = [];
        this.prevX = null;
        this.prevY = null;
        this.prevTheta = null;

        this.initSimulation();
        this.render();
        this.updateOrderParameters();
    }

    animate() {
        if (!this.isPlaying) return;

        // Store previous state for velocity calculations
        this.prevX = new Float64Array(this.sim.x);
        this.prevY = new Float64Array(this.sim.y);
        this.prevTheta = new Float64Array(this.sim.theta);

        // Single step per frame for smoother animation
        this.sim.step();
        this.stepCount++;

        // Update camera tracking if enabled
        this.updateCameraTracking();

        document.getElementById('stepCount').textContent = this.stepCount;
        this.render();
        this.updateOrderParameters();

        this.animationId = requestAnimationFrame(() => this.animate());
    }

    updateOrderParameters() {
        // Calculate order parameters
        const S = this.sim.correlationOrderParameter();
        const R = this.sim.calculateSynchronyOrder();

        let V = 0, Omega = 0;
        if (this.prevX && this.prevY && this.prevTheta) {
            const velocities = this.sim.calculateVelocityOrderParameter(this.prevX, this.prevY, this.prevTheta);
            V = velocities.V;
            Omega = velocities.Omega;
        }

        // Update displayed values
        document.getElementById('sValue').textContent = S.toFixed(3);
        document.getElementById('vValue').textContent = V.toFixed(3);
        document.getElementById('omegaValue').textContent = Omega.toFixed(3);
        document.getElementById('rValue').textContent = R.toFixed(3);

        // Classify and update state badge
        const state = this.sim.classifyState(S, V, Omega);
        const stateBadge = document.getElementById('stateValue');
        stateBadge.textContent = state;

        // Remove all state classes and add appropriate one
        stateBadge.className = 'state-badge';
        const stateClass = state.toLowerCase().replace(/ /g, '-');
        stateBadge.classList.add(stateClass);

        // Update history arrays
        this.sHistory.push(S);
        this.vHistory.push(V);
        this.omegaHistory.push(Omega);
        this.rHistory.push(R);

        // Trim to max history
        if (this.sHistory.length > this.maxHistory) this.sHistory.shift();
        if (this.vHistory.length > this.maxHistory) this.vHistory.shift();
        if (this.omegaHistory.length > this.maxHistory) this.omegaHistory.shift();
        if (this.rHistory.length > this.maxHistory) this.rHistory.shift();

        // Draw plots
        this.drawPlot(this.sPlotCtx, this.sPlotCanvas, this.sHistory, '#58a6ff', 0, 1);
        this.drawPlot(this.vPlotCtx, this.vPlotCanvas, this.vHistory, '#3fb950', 0, null);
        this.drawPlot(this.omegaPlotCtx, this.omegaPlotCanvas, this.omegaHistory, '#a371f7', 0, null);
        this.drawPlot(this.rPlotCtx, this.rPlotCanvas, this.rHistory, '#d29922', 0, 1);
    }

    drawPlot(ctx, canvas, data, color, minVal, maxVal) {
        const width = canvas.width;
        const height = canvas.height;

        // Clear
        ctx.fillStyle = '#0d1117';
        ctx.fillRect(0, 0, width, height);

        if (data.length < 2) return;

        // Auto-scale if maxVal is null
        let yMin = minVal;
        let yMax = maxVal;
        if (yMax === null) {
            yMax = Math.max(...data) * 1.1 || 1;
        }

        // Draw line
        ctx.beginPath();
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;

        for (let i = 0; i < data.length; i++) {
            const x = (i / (this.maxHistory - 1)) * width;
            const y = height - ((data[i] - yMin) / (yMax - yMin)) * height;

            if (i === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        }
        ctx.stroke();

        // Draw glow effect
        ctx.strokeStyle = color;
        ctx.globalAlpha = 0.3;
        ctx.lineWidth = 4;
        ctx.stroke();
        ctx.globalAlpha = 1;
    }

    render() {
        const ctx = this.ctx;
        const width = this.canvas.width;
        const height = this.canvas.height;

        // Clear with dark background
        ctx.fillStyle = '#0d1117';
        ctx.fillRect(0, 0, width, height);

        // Draw subtle grid (fixed on screen, affected by zoom for visual feedback)
        ctx.strokeStyle = 'rgba(48, 54, 61, 0.5)';
        ctx.lineWidth = 1;
        const gridSize = 60 * this.zoom;
        const gridOffsetX = (width / 2 - this.cameraX * (width / 5) * this.zoom) % gridSize;
        const gridOffsetY = (height / 2 + this.cameraY * (height / 5) * this.zoom) % gridSize;

        for (let x = gridOffsetX; x <= width; x += gridSize) {
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, height);
            ctx.stroke();
        }
        for (let y = gridOffsetY; y <= height; y += gridSize) {
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(width, y);
            ctx.stroke();
        }

        // Coordinate transform with zoom and camera offset
        // Base scale: maps simulation unit to pixels
        const baseScale = width / 5;
        const scale = baseScale * this.zoom;
        const offsetX = width / 2;
        const offsetY = height / 2;

        // Helper function to convert sim coords to screen coords
        const toScreenX = (simX) => (simX - this.cameraX) * scale + offsetX;
        const toScreenY = (simY) => -(simY - this.cameraY) * scale + offsetY;

        // Draw particles - simple circles without glow
        for (let i = 0; i < this.sim.N; i++) {
            const screenX = toScreenX(this.sim.x[i]);
            const screenY = toScreenY(this.sim.y[i]);

            // Skip particles far off-screen for performance
            if (screenX < -20 || screenX > width + 20 || screenY < -20 || screenY > height + 20) continue;

            // Map phase to color (HSL color wheel)
            const hue = ((this.sim.theta[i] + Math.PI) / (2 * Math.PI)) * 360;

            ctx.beginPath();
            ctx.arc(screenX, screenY, Math.max(2, 4 * this.zoom), 0, Math.PI * 2);
            ctx.fillStyle = `hsl(${hue}, 80%, 55%)`;
            ctx.fill();
        }

        // Draw predator if enabled
        if (this.sim.predatorEnabled) {
            const predScreenX = toScreenX(this.sim.predX);
            const predScreenY = toScreenY(this.sim.predY);

            // Predator glow effect
            const glowRadius = 16 * this.zoom;
            ctx.beginPath();
            ctx.arc(predScreenX, predScreenY, glowRadius, 0, Math.PI * 2);
            const gradient = ctx.createRadialGradient(
                predScreenX, predScreenY, 0,
                predScreenX, predScreenY, glowRadius
            );
            gradient.addColorStop(0, 'rgba(255, 60, 60, 0.8)');
            gradient.addColorStop(0.5, 'rgba(255, 60, 60, 0.3)');
            gradient.addColorStop(1, 'rgba(255, 60, 60, 0)');
            ctx.fillStyle = gradient;
            ctx.fill();

            // Predator core
            const coreRadius = 8 * this.zoom;
            ctx.beginPath();
            ctx.arc(predScreenX, predScreenY, coreRadius, 0, Math.PI * 2);
            ctx.fillStyle = '#ff3c3c';
            ctx.fill();
            ctx.strokeStyle = '#ffffff';
            ctx.lineWidth = 2;
            ctx.stroke();

            // Inner marker
            ctx.beginPath();
            ctx.arc(predScreenX, predScreenY, 3 * this.zoom, 0, Math.PI * 2);
            ctx.fillStyle = '#ffffff';
            ctx.fill();
        }

        // Draw origin crosshair when zoomed out far or tracking
        if (this.autoTrack || this.zoom < 0.5) {
            const originX = toScreenX(0);
            const originY = toScreenY(0);
            ctx.strokeStyle = 'rgba(88, 166, 255, 0.3)';
            ctx.lineWidth = 1;
            ctx.setLineDash([5, 5]);
            ctx.beginPath();
            ctx.moveTo(originX - 20, originY);
            ctx.lineTo(originX + 20, originY);
            ctx.moveTo(originX, originY - 20);
            ctx.lineTo(originX, originY + 20);
            ctx.stroke();
            ctx.setLineDash([]);
        }
    }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new SimulationController();
});
