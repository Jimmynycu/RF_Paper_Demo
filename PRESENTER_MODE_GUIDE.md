# Presenter Mode - User Guide

## Overview

The presentation now includes a **dual-monitor presenter mode** similar to PowerPoint or Google Slides. This allows you to see speaker notes, next slide preview, and presentation controls on your monitor, while the audience sees only the clean slides on a second monitor/projector.

---

## How It Works

### Three Files
1. **`demo_dashboard.html`** - Main presentation (can be used standalone or in presenter mode)
2. **`presenter.html`** - Presenter view with notes, controls, and timer
3. **`presentation.html`** - Clean audience view for second monitor

### Architecture
```
┌─────────────────────────┐          ┌─────────────────────────┐
│   Presenter Window      │          │  Presentation Window    │
│  (Your Monitor)         │◄────────►│  (Second Monitor/       │
│                         │  Sync    │   Projector)            │
│  • Current Slide        │          │                         │
│  • Next Slide Preview   │          │  Full-Screen Slides     │
│  • Speaker Notes        │          │  (Clean, No Controls)   │
│  • Timer                │          │                         │
│  • Controls             │          │                         │
└─────────────────────────┘          └─────────────────────────┘
```

---

## Quick Start

### Option 1: Using Presenter Mode (Recommended for Presentations)

1. **Open** `demo_dashboard.html` in your browser
2. **Click** the green "🎤 Presenter Mode" button in the top-left corner
3. The **Presenter View** opens in a new window
4. **Click** "🖥️ Open Presentation Window" in the presenter view
5. **Drag** the presentation window to your second monitor/projector
6. **Press F11** on the presentation window for fullscreen

### Option 2: Manual Setup

1. Open `presenter.html` - This is your control panel
2. Click "🖥️ Open Presentation Window" 
3. Drag the new window to your external monitor
4. Use the presenter window to control everything

### Option 3: Direct Presentation (No Presenter View)

Just open `demo_dashboard.html` and present normally with navigation controls

---

## Features

### Presenter View Features

#### 📊 **Dual Slide Preview**
- **Left pane**: Current slide being shown to audience
- **Right pane**: Next slide preview (slightly dimmed)

#### 📝 **Speaker Notes**
Every slide has detailed speaker notes including:
- Key teaching points
- What to emphasize
- Demo talking points
- Time management tips

#### ⏱️ **Timer**
- Start/Pause/Reset controls
- Shows elapsed time (MM:SS format)
- Helps keep presentation on track

#### 🎮 **Navigation Controls**
- Previous/Next buttons
- Slide counter (e.g., "5 / 21")
- Current slide title display
- Keyboard shortcuts

#### 🔄 **Synchronization**
- Both windows stay in perfect sync
- Navigate from either window
- Real-time updates via BroadcastChannel API

---

## Keyboard Shortcuts

### In Presenter View
- `→` or `PageDown` - Next slide
- `←` or `PageUp` - Previous slide
- `Home` - Go to first slide
- `End` - Go to last slide

### In Presentation View (Audience Monitor)
- Same shortcuts work
- Click anywhere to advance
- `Space` or `↓` - Next slide
- `Backspace` or `↑` - Previous slide

---

## Speaker Notes Summary

### Slide 1: Title
- Introduce yourself
- Brief overview of 3 papers
- Total time: 20-25 minutes

### Slide 2: About Me
- Current role at Homee.AI
- Previous experience at MediaTek, Broadcom
- RF hardware + modern AI/ML skills

### Slide 3: Research Overview
- Paper 1: Physics-informed optimization
- Paper 2: Zero-data learning (PINN)
- Paper 3: LLM in 6G

### Slide 4-8: Paper 1 (ESA Design)
- Q factor fundamentals
- Chu limit as physics constraint
- MOEA/D optimization
- Live demo notes

### Slide 9-13: Paper 2 (FSS PINN)
- Inverse design problem
- Mode matching method
- PINN dual loss
- Zero-data advantage

### Slide 14-18: Paper 3 (LLM-RIMSA)
- 6G real-time control
- RIS beamforming
- Transformer architecture
- Performance benchmarks

### Slide 19-21: Closing
- Formulas reference
- Value proposition
- Q&A preparation

---

## Technical Details

### Window Communication

Uses **BroadcastChannel API** for cross-window communication:

```javascript
const channel = new BroadcastChannel('presentation-sync');

// Send slide change
channel.postMessage({
    type: 'goto-slide',
    slideIndex: 5
});

// Receive slide change
channel.onmessage = (event) => {
    if (event.data.type === 'goto-slide') {
        // Update slide
    }
};
```

### Browser Compatibility

✅ **Works in:**
- Chrome 54+
- Firefox 38+
- Edge 79+
- Safari 15.4+

⚠️ **Note:** BroadcastChannel requires same-origin windows (both windows must be from the same server/localhost)

---

## Troubleshooting

### Windows Not Syncing?

**Check:**
1. Both windows are from the same origin (e.g., both `http://localhost:8000`)
2. Browser supports BroadcastChannel (check console for errors)
3. Pop-up blocker isn't preventing window creation

**Fix:** 
- Allow pop-ups for this site
- Refresh both windows
- Try manually opening `presenter.html`

### Presentation Window Won't Go Fullscreen?

**Try:**
- Press `F11` while the window is focused
- Use browser's fullscreen option (usually under View menu)
- Check if external monitor is detected properly

### Timer Not Starting?

**Check:**
- Click "▶ Start" button
- Refresh the page if timer seems stuck

### Notes Not Showing?

**Verify:**
- You're in `presenter.html`, not `demo_dashboard.html`
- The slide navigation is working
- Check browser console for JavaScript errors

---

## Tips for Best Experience

### Setup Before Presentation
1. ✅ Test both monitors/projector ahead of time
2. ✅ Position windows correctly BEFORE starting
3. ✅ Familiarize yourself with keyboard shortcuts
4. ✅ Read through speaker notes for your slides
5. ✅ Test the timer functionality

### During Presentation
- 📍 Keep presenter window on your laptop
- 🖥️ Fullscreen the presentation window on projector
- 👀 Glance at speaker notes for key points
- ⏱️ Watch timer to stay on schedule
- 👁️ Check "Next Slide" preview to prepare transitions

### Multi-Monitor Setup
```
Your Laptop Screen:           External Monitor/Projector:
┌──────────────────┐         ┌──────────────────────────┐
│ Presenter View   │         │   Presentation           │
│ • Notes visible  │         │   • Clean slides only    │
│ • Timer running  │         │   • Fullscreen mode      │
│ • Next preview   │         │   • Audience view        │
└──────────────────┘         └──────────────────────────┘
```

---

## Advanced Features

### Customizing Speaker Notes

Edit `presenter.html` and modify the `slideNotes` array:

```javascript
const slideNotes = [
    {
        title: "Title",
        notes: "<strong>Your HTML notes here</strong><ul><li>Point 1</li></ul>"
    },
    // ... more slides
];
```

### Changing Timer Position

In `presenter.html`, modify the CSS grid layout in `.container`

### Adjusting Slide Previews

Modify iframe dimensions in `#currentSlide` and `#nextSlide` CSS

---

## File Structure

```
spectral-eclipse/
├── demo_dashboard.html      # Main presentation (21 slides)
├── presenter.html           # Presenter view with notes & controls
├── presentation.html        # Clean audience view wrapper
├── paper_images/           # Images used in slides
└── PRESENTER_MODE_GUIDE.md # This file
```

---

## Questions or Issues?

If you encounter problems:
1. Check browser console (F12) for errors
2. Ensure your server is running (`python demo_server.py`)
3. Verify all three HTML files are in the same directory
4. Test in latest Chrome/Firefox for best compatibility

---

**Ready to present? Click "🎤 Presenter Mode" and impress your audience! 🚀**
