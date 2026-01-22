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
}

// ============ UI Controller ============

class SimulationController {
    constructor() {
        this.canvas = document.getElementById('canvas');
        this.ctx = this.canvas.getContext('2d');
        this.isPlaying = false;
        this.stepCount = 0;
        this.animationId = null;

        // Initialize simulation with default params
        this.initSimulation();

        // Bind UI elements
        this.bindControls();

        // Initial render
        this.render();
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

        this.initSimulation();
        this.render();
    }

    animate() {
        if (!this.isPlaying) return;

        // Single step per frame for smoother animation
        this.sim.step();
        this.stepCount++;

        document.getElementById('stepCount').textContent = this.stepCount;
        this.render();

        this.animationId = requestAnimationFrame(() => this.animate());
    }

    render() {
        const ctx = this.ctx;
        const width = this.canvas.width;
        const height = this.canvas.height;

        // Clear with dark background
        ctx.fillStyle = '#0d1117';
        ctx.fillRect(0, 0, width, height);

        // Draw subtle grid
        ctx.strokeStyle = 'rgba(48, 54, 61, 0.5)';
        ctx.lineWidth = 1;
        const gridSize = 60;
        for (let x = 0; x <= width; x += gridSize) {
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, height);
            ctx.stroke();
        }
        for (let y = 0; y <= height; y += gridSize) {
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(width, y);
            ctx.stroke();
        }

        // Coordinate transform: map [-2.5, 2.5] to [0, width/height]
        const scale = width / 5;
        const offsetX = width / 2;
        const offsetY = height / 2;

        // Draw particles - simple circles without glow
        for (let i = 0; i < this.sim.N; i++) {
            const screenX = this.sim.x[i] * scale + offsetX;
            const screenY = -this.sim.y[i] * scale + offsetY;

            // Map phase to color (HSL color wheel)
            const hue = ((this.sim.theta[i] + Math.PI) / (2 * Math.PI)) * 360;

            ctx.beginPath();
            ctx.arc(screenX, screenY, 4, 0, Math.PI * 2);
            ctx.fillStyle = `hsl(${hue}, 80%, 55%)`;
            ctx.fill();
        }
    }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new SimulationController();
});
