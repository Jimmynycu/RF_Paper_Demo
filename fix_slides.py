import re

file_path = "demo_dashboard.html"

with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# Define the RF content to insert
# Note: Double backslashes for LaTeX formulas in Python string
rf_content = """    <section class="slide" id="s15">
        <div style="display:flex;gap:8px;align-items:center;margin-bottom:16px">
            <span class="tag tag-6g">Paper 3</span>
            <h2 style="margin:0">RF Fundamentals: RIS, Beamforming, SINR</h2>
        </div>
        <p style="margin-bottom:16px;color:var(--muted)">Before diving into the solution, let's understand the three key
            concepts behind RIMSA...</p>

        <div class="grid grid-3">
            <div class="card">
                <h3>RIS Structure (from Paper)</h3>
                <div style="text-align:center;margin-bottom:8px">
                    <img src="paper_images/p3_page3_img1.png" alt="RIMSA Architecture"
                        style="width:100%;height:auto;min-height:150px;object-fit:cover;border-radius:8px;border:1px solid var(--border)">
                    <p style="font-size:0.7rem;color:var(--muted);margin-top:2px">Fig. from Paper: RIMSA with Digital
                        Processor + Metasurface</p>
                </div>
                <ul style="margin-top:6px;font-size:0.8rem">
                    <li>Each element: patch + varactor</li>
                    <li>Varactor bias → phase shift (0-2π)</li>
                    <li>Low power vs. active relays</li>
                </ul>
            </div>
            <div class="card">
                <h3>Beamforming</h3>
                <p>Steer RF beam by controlling phase.</p>
                <div class="formula">\\(\\theta = \\arcsin\\left(\\frac{\\Delta\\phi \\lambda}{2\\pi d}\\right)\\)</div>
                <p class="key">Phase difference → beam direction</p>
                <p style="margin-top:8px">RIS adds reflective beamforming on top of transmit beamforming.</p>
            </div>
            <div class="card">
                <h3>Sum-Rate Objective</h3>
                <p>Maximize total data rate for all users:</p>
                <div class="formula">\\(R = \\sum_k \\log_2(1 + \\text{SINR}_k)\\)</div>
                <p>SINR = Signal / (Interference + Noise)</p>
                <p class="key">Balance: helping one user may interfere with others!</p>
            </div>
        </div>
        <p
            style="margin-top:16px;padding:12px;background:var(--card);border:1px solid var(--border);border-radius:8px;text-align:center">
            💡 <strong>Three pillars of 6G:</strong> RIS hardware + Beamforming physics + System optimization →
            Connected via LLM
        </p>
        <div class="slide-num">15/21</div>
    </section>"""

# Find start of slide 15
start_marker = '<!-- SLIDE 15: Paper 3 RF Background -->'
# Find start of slide 16
end_marker = '<!-- SLIDE 16: Paper 3 - DMA vs RIMSA -->'

start_idx = content.find(start_marker)
end_idx = content.find(end_marker)

if start_idx == -1:
    print(f"Could not find start marker: '{start_marker}'")
    # Try searching for a substring just in case
    start_marker_fallback = '<!-- SLIDE 15'
    start_idx = content.find(start_marker_fallback)
    if start_idx == -1:
        print("Could not find fallback start marker either.")
        exit(1)
    else:
        print(f"Found fallback start marker at {start_idx}")
        # Adjust start_marker length effectively for the slice logic
        # We need to find the newline after this marker
        newline_pos = content.find('\n', start_idx)
        start_marker_len = newline_pos - start_idx
        # But our logic below uses start_idx + len(start_marker)
        # So let's just use the fallback marker length? No, we need to skip the full comment line.
        # Let's assume the line is the marker.
        pass

if end_idx == -1:
    print(f"Could not find end marker: '{end_marker}'")
    # Try searching for fallback
    end_marker_fallback = '<!-- SLIDE 16'
    end_idx = content.find(end_marker_fallback)
    if end_idx == -1:
        print("Could not find fallback end marker either.")
        exit(1)
    else:
        print(f"Found fallback end marker at {end_idx}")

# Calculate insertion point
# Start replace AFTER the start marker line
# End replace BEFORE the end marker line

# To be safe, we just slice from start_idx + len(first_line) to end_idx
insert_pos = content.find('\n', start_idx) + 1
target_end = end_idx

if insert_pos > target_end:
    print("Error: Start position is after end position!")
    exit(1)

# Perform replacement
new_content = content[:insert_pos] + rf_content + "\n\n    " + content[target_end:]

with open(file_path, "w", encoding="utf-8") as f:
    f.write(new_content)

print("Successfully replaced Slide 15 content.")
