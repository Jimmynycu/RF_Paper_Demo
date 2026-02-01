import re

file_path = "demo_dashboard.html"

with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# Define the DMA Animation content to insert into Slide 16
dma_content = """    <section class="slide" id="s16">
        <div style="display:flex;gap:8px;align-items:center;margin-bottom:16px">
            <span class="tag tag-6g">Paper 3</span>
            <h2 style="margin:0">How RIMSA Works</h2>
        </div>
        <p style="margin-bottom:12px;color:var(--muted)">Now that you understand RIS hardware, let's see why Paper 3's
            approach (RIMSA) is superior to traditional RIS (DMA).</p>

        <div class="grid grid-2">
            <div class="card">
                <h3>Old: Serial Feed (DMA)</h3>
                <p style="font-size:0.9rem;color:var(--muted)">Watch the circles do "the wave" (like fans in a stadium)
                </p>

                <div id="serial-container" style="display:flex;gap:10px;justify-content:center;margin:30px 0"></div>

                <div
                    style="margin-top:20px;padding:12px;background:#450a0a;border-radius:8px;border:1px solid var(--red)">
                    <p style="font-size:0.9rem;margin-bottom:8px"><strong style="color:var(--red)">⚠️ Three Critical
                            Problems:</strong></p>
                    <ul style="font-size:0.85rem;line-height:1.6;margin:0;padding-left:20px">
                        <li><strong>Beam Squint:</strong> Different frequencies arrive at different times → beam
                            spreads like a rainbow → user can't catch full signal</li>
                        <li><strong>Intersymbol Interference:</strong> First element sends new bit while last element
                            still sends old bit → air fills with mixed 1s and 0s → data corruption</li>
                        <li><strong>Narrowband Only:</strong> Forced to use slow signals to avoid problems #1 and #2
                        </li>
                    </ul>
                </div>
            </div>

            <div class="card">
                <h3>New: Parallel Feed (RIMSA)</h3>
                <p style="font-size:0.9rem;color:var(--muted)">Watch the circles "breathe together"</p>

                <div id="parallel-container"
                    style="display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin:30px auto;width:fit-content">
                </div>



                <div
                    style="margin-top:20px;padding:12px;background:#14532d;border-radius:8px;border:1px solid var(--green)">
                    <p style="font-size:0.9rem;margin-bottom:8px"><strong style="color:var(--green)">✓ What RIMSA
                            Solves:</strong></p>
                    <ul style="font-size:0.85rem;line-height:1.6;margin:0;padding-left:20px">
                        <li><strong>Independent Control:</strong> Phase and Amplitude adjust separately (no coupling)
                        </li>
                        <li><strong>Direct Transmission:</strong> Signal generated on surface → no double fading → high
                            gain</li>
                    </ul>
                </div>

                <div
                    style="margin-top:10px;padding:12px;background:#450a0a;border-radius:8px;border:1px solid var(--red)">
                    <p style="font-size:0.9rem;margin-bottom:8px"><strong style="color:var(--red)">⚠️ New Problem
                            Created:</strong></p>
                    <ul style="font-size:0.85rem;line-height:1.6;margin:0;padding-left:20px">
                        <li><strong>Control Complexity:</strong> 1000s of independent knobs → LLM solves this in
                            real-time</li>
                    </ul>
                </div>
            </div>
        </div>

        <div style="text-align:center;margin-top:20px">
            <div style="display:flex;align-items:center;justify-content:center;gap:20px;margin-bottom:12px">
                <label style="font-size:0.9rem;color:var(--muted)">Signal Frequency:</label>
                <label style="cursor:pointer">
                    <input type="radio" name="freq" value="low" checked style="margin-right:5px">
                    <span style="color:var(--text)">LOW</span>
                </label>
                <label style="cursor:pointer">
                    <input type="radio" name="freq" value="mid" style="margin-right:5px">
                    <span style="color:var(--text)">MID</span>
                </label>
                <label style="cursor:pointer">
                    <input type="radio" name="freq" value="high" style="margin-right:5px">
                    <span style="color:var(--text)">HIGH</span>
                </label>
            </div>
            <button class="btn" onclick="toggleRIMSAAnimation()" id="animBtn">▶ Start Stream</button>
        </div>

        <div class="slide-num">16/21</div>

        <style>
            .element {
                width: 20px;
                height: 20px;
                border-radius: 50%;
                transition: background-color 0.05s;
                border: 2px solid var(--border);
            }
        </style>

        <script>
            (function () {
                const serialContainer = document.getElementById('serial-container');
                const parallelContainer = document.getElementById('parallel-container');
                const count = 10;

                // Frequency settings: [speed multiplier, delay per element in seconds]
                const FREQ_SETTINGS = {
                    low: { speed: 0.5, delay: 0.3 },    // Slow, clear red/blue separation
                    mid: { speed: 2, delay: 0.15 },     // Faster, overlap creates purple
                    high: { speed: 8, delay: 0.08 }     // Very fast, all purple mess
                };

                let animationRunning = false;
                let animationFrameId = null;
                let startTime = 0;

                // Generate Elements (only once)
                if (serialContainer && serialContainer.children.length === 0) {
                    for (let i = 0; i < count; i++) {
                        let el = document.createElement('div');
                        el.className = 'element serial-el';
                        el.dataset.index = i;
                        serialContainer.appendChild(el);

                        let el2 = document.createElement('div');
                        el2.className = 'element parallel-el';
                        parallelContainer.appendChild(el2);
                    }
                }

                // Get selected frequency setting
                function getFreqSetting() {
                    const selected = document.querySelector('input[name="freq"]:checked');
                    return FREQ_SETTINGS[selected ? selected.value : 'low'];
                }

                // Color calculation based on time
                function getSignalColor(time, speed) {
                    const phase = (time * speed) % 2;
                    if (phase < 1) {
                        return '#ef4444'; // Red
                    } else {
                        return '#3b82f6'; // Blue
                    }
                }

                // Animation loop
                function animate() {
                    if (!animationRunning) return;

                    const currentTime = (Date.now() - startTime) / 1000;
                    const setting = getFreqSetting();

                    // Update serial elements (with delay - creates wave/interference)
                    const serialElements = document.querySelectorAll('.serial-el');
                    serialElements.forEach((el, index) => {
                        const delay = index * setting.delay;
                        const delayedTime = currentTime - delay;
                        el.style.backgroundColor = getSignalColor(delayedTime, setting.speed);
                    });

                    // Update parallel elements (no delay - perfect sync)
                    const parallelElements = document.querySelectorAll('.parallel-el');
                    const parallelColor = getSignalColor(currentTime, setting.speed);
                    parallelElements.forEach(el => {
                        el.style.backgroundColor = parallelColor;
                    });

                    animationFrameId = requestAnimationFrame(animate);
                }

                // Toggle animation on/off
                window.toggleRIMSAAnimation = function () {
                    const btn = document.getElementById('animBtn');

                    if (!animationRunning) {
                        animationRunning = true;
                        animationRunning = true;
                        startTime = Date.now();
                        btn.textContent = '⏸ Stop Stream';
                        btn.style.background = 'var(--red)';
                        animate();
                    } else {
                        animationRunning = false;
                        if (animationFrameId) {
                            cancelAnimationFrame(animationFrameId);
                        }
                        btn.textContent = '▶ Start Stream';
                        btn.style.background = 'var(--accent)';

                        // Reset colors
                        document.querySelectorAll('.element').forEach(el => {
                            el.style.backgroundColor = 'var(--card)';
                        });
                    }
                };
            })();
        </script>
    </section>"""

# Find markers for Slide 16
start_marker = '<!-- SLIDE 16: Paper 3 - DMA vs RIMSA -->'
end_marker = '<!-- SLIDE 17: Paper 3 Solution -->'

start_idx = content.find(start_marker)
end_idx = content.find(end_marker)

if start_idx == -1 or end_idx == -1:
    print(f"Could not find markers! Start: {start_idx}, End: {end_idx}")
    # Try fallback for Start if missing
    if start_idx == -1:
         # It might be there but my string is slightly off?
         # I verified it in View 130
         # "    <!-- SLIDE 16: Paper 3 - DMA vs RIMSA -->"
         # Maybe leading spaces?
         start_idx = content.find("<!-- SLIDE 16")
    
    if end_idx == -1:
         end_idx = content.find("<!-- SLIDE 17")

    if start_idx == -1 or end_idx == -1:
        print("Still failed to find markers.")
        exit(1)

# Find newline after start_marker
newline_pos = content.find('\n', start_idx)
insert_pos = newline_pos + 1
target_end = end_idx

# Perform replacement
new_content = content[:insert_pos] + dma_content + "\n\n    " + content[target_end:]

with open(file_path, "w", encoding="utf-8") as f:
    f.write(new_content)

print(f"Successfully replaced Slide 16 content. Start: {start_idx}, End: {end_idx}")
