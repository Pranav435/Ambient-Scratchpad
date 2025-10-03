# main.py (patched)
"""
Ambient Scratchpad — minimal, self-contained build focused on:
  1) System-wide hotkey Option+Space for Quick Capture
  2) Fix black text on black background for the bottom "Actionables" text view
  3) Shift header title slightly to the right

Requires:
  pip install rumps pyobjc

If the global shortcut doesn't fire, grant Accessibility:
  System Settings → Privacy & Security → Accessibility → allow Terminal/Python/your app.
"""

import threading
import time
import rumps

# --- PyObjC / AppKit ---
try:
    from AppKit import (
        NSApp, NSApplication, NSApplicationActivationPolicyAccessory,
        NSWindow, NSWindowStyleMaskTitled, NSWindowStyleMaskClosable, NSWindowStyleMaskResizable,
        NSViewWidthSizable, NSViewHeightSizable,
        NSScrollView, NSTextView, NSFont, NSColor,
        NSVisualEffectView, NSView, NSBezelBorder,
        NSEvent, NSEventMaskKeyDown, NSTextAlignmentLeft, NSBox,
        NSStackView, NSTextField
    )
    from Foundation import NSObject, NSMakeRect
    from PyObjCTools import AppHelper
    PYOBJC_AVAILABLE = True
except Exception:
    PYOBJC_AVAILABLE = False


# ---------- App state ----------
NOTES = []  # list of dicts for demo
GLOBAL_WINDOWS = []  # keep strong references to NSWindow instances


def _now_iso():
    import datetime
    return datetime.datetime.now().isoformat(timespec="seconds")


# ---------- Quick Capture pipeline (demo) ----------
def quick_capture_pipeline(text: str):
    """
    Replace this with your real pipeline. For now it just appends to NOTES.
    """
    NOTES.insert(0, {"text": text, "timestamp": _now_iso()})
    # In a larger build, you'd also recompute summaries/related/etc. here.


# ---------- Native UI: Show All Notes ----------
if PYOBJC_AVAILABLE:

    class ShowAllNotesWindowController(NSObject):
        def init(self):
            self = super().init()
            if self is None:
                return None

            self.window = None
            self.header_label = None
            self.detail_text = None
            self.actionable_text = None
            return self

        def open(self):
            # Window
            self.window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
                NSMakeRect(100, 100, 820, 560),
                NSWindowStyleMaskTitled | NSWindowStyleMaskClosable | NSWindowStyleMaskResizable,
                2,  # buffered
                False
            )
            self.window.setTitle_("Ambient Scratchpad — All Notes")
            self.window.setReleasedWhenClosed_(False)
            self.window.setIsVisible_(True)

            content = self.window.contentView()
            content.setAutoresizingMask_(NSViewWidthSizable | NSViewHeightSizable)

            root = NSStackView.alloc().init()
            root.setOrientation_(1)  # vertical
            root.setSpacing_(8.0)
            # Add subtle insets: top/right/bottom 12, left 12 (we'll add extra header offset below)
            root.setEdgeInsets_((12, 12, 12, 12))
            root.setAutoresizingMask_(NSViewWidthSizable | NSViewHeightSizable)
            root.setFrame_(content.bounds())
            content.addSubview_(root)

            # --- Header (shifted right by 12 px using a spacer view) ---
            header_stack = NSStackView.alloc().init()
            header_stack.setOrientation_(0)  # horizontal
            header_stack.setSpacing_(8.0)

            spacer = NSView.alloc().initWithFrame_(NSMakeRect(0, 0, 12, 4))  # the shift-to-right spacer
            spacer.setAutoresizingMask_(0)
            header_stack.addView_inGravity_(spacer, 1)

            self.header_label = NSTextField.alloc().initWithFrame_(NSMakeRect(0, 0, 600, 22))
            self.header_label.setStringValue_("0 notes — Highlights: none yet")
            self.header_label.setEditable_(False)
            self.header_label.setBordered_(False)
            self.header_label.setDrawsBackground_(False)
            self.header_label.setFont_(NSFont.boldSystemFontOfSize_(14.0))
            header_stack.addView_inGravity_(self.header_label, 1)

            root.addView_inGravity_(header_stack, 1)

            # --- Middle: details (scrollable text view) ---
            detail_scroll = NSScrollView.alloc().initWithFrame_(NSMakeRect(0, 0, 800, 360))
            detail_scroll.setHasVerticalScroller_(True)
            detail_scroll.setBorderType_(NSBezelBorder)
            self.detail_text = NSTextView.alloc().initWithFrame_(detail_scroll.contentView().bounds())
            self.detail_text.setEditable_(False)
            self.detail_text.setFont_(NSFont.systemFontOfSize_(13.0))
            # Safer colors for both light/dark
            self.detail_text.setTextColor_(NSColor.labelColor())
            self.detail_text.setBackgroundColor_(NSColor.textBackgroundColor())
            self.detail_text.setString_("Select or add a note…")
            detail_scroll.setDocumentView_(self.detail_text)

            root.addView_inGravity_(detail_scroll, 1)

            # --- Bottom: Actionables (fix black-on-black by forcing readable colors) ---
            actionable_scroll = NSScrollView.alloc().initWithFrame_(NSMakeRect(0, 0, 800, 140))
            actionable_scroll.setHasVerticalScroller_(True)
            actionable_scroll.setBorderType_(NSBezelBorder)

            self.actionable_text = NSTextView.alloc().initWithFrame_(actionable_scroll.contentView().bounds())
            self.actionable_text.setEditable_(False)
            self.actionable_text.setFont_(NSFont.monospacedSystemFontOfSize_weight_(12.0, 0))
            # Critical: readable in both modes
            self.actionable_text.setTextColor_(NSColor.labelColor())
            # Use controlBackgroundColor to avoid "black on black" when vibrancy is applied
            self.actionable_text.setBackgroundColor_(NSColor.controlBackgroundColor())
            self.actionable_text.setString_("Actionable summary, ideas, snippets will show here…")
            actionable_scroll.setDocumentView_(self.actionable_text)

            root.addView_inGravity_(actionable_scroll, 1)

            GLOBAL_WINDOWS.append(self.window)
            self.refresh()

        def refresh(self):
            # Update header
            count = len(NOTES)
            highlights = "recent" if count else "none yet"
            self.header_label.setStringValue_(f"{count} notes — Highlights: {highlights}")

            # Update detail to latest note
            if NOTES:
                latest = NOTES[0]
                self.detail_text.setString_(f"[{latest['timestamp']}]  {latest['text']}")
                self.actionable_text.setString_("• Example next steps for latest note\n• (wire your own summarizer here)")
            else:
                self.detail_text.setString_("No notes yet. Hit Option+Space to add one.")
                self.actionable_text.setString_("")

    # Keep a singleton controller for demo
    _WC = ShowAllNotesWindowController.alloc().init()


# ---------- Menubar app ----------
class AmbientScratchpadApp(rumps.App):
    def __init__(self):
        super().__init__("✳︎", quit_button=rumps.MenuItem("Quit"))
        self.menu = [
            rumps.MenuItem("Quick Capture", callback=self.quick_capture),
            rumps.MenuItem("Show All Notes", callback=self.show_all_notes),
        ]

        # Install global hotkey (Option+Space) via NSEvent monitors
        self._install_global_hotkey()

    # ---- Quick Capture ----
    def quick_capture(self, _=None):
        # Simple capture prompt (replace with your floating capture UI if you have one)
        result = rumps.Window(
            message="Type a quick note:",
            title="Quick Capture",
            default_text="",
            cancel=True,
            dimensions=(480, 80),
        ).run()

        if result and result.clicked:
            text = result.text.strip()
            if text:
                # Pipeline
                quick_capture_pipeline(text)
                # If window is open, refresh it
                if PYOBJC_AVAILABLE and _WC and _WC.window:
                    try:
                        _WC.refresh()
                    except Exception:
                        pass
                rumps.notification("Ambient Scratchpad", "Captured", text[:80])

    # ---- Window ----
    def show_all_notes(self, _=None):
        if not PYOBJC_AVAILABLE:
            rumps.alert("PyObjC not available", "Install pyobjc and restart.")
            return
        if not _WC.window:
            _WC.open()
        else:
            _WC.window.makeKeyAndOrderFront_(None)
            _WC.refresh()

    # ---- Hotkey (Option+Space) ----
    def _install_global_hotkey(self):
        if not PYOBJC_AVAILABLE:
            return

        # Try both global and local monitors: global triggers while unfocused;
        # local ensures it still works if our app has key focus.
        try:
            # Use modern modifier flag, fall back to legacy names when needed
            try:
                from AppKit import NSEventModifierFlagOption as _OPT_FLAG
            except Exception:
                from AppKit import NSAlternateKeyMask as _OPT_FLAG  # legacy

            def _check_opt_space(event):
                try:
                    # keyCode 49 == Space; characters can also be " "
                    keycode = int(event.keyCode())
                    flags = int(event.modifierFlags())
                    chars = str(event.characters())
                    is_opt = (flags & _OPT_FLAG) == _OPT_FLAG
                    if is_opt and (keycode == 49 or chars == " "):
                        # Fire on keyDown
                        AppHelper.callAfter(self.quick_capture, None)
                except Exception:
                    pass

            # Global monitor (can't stop propagation)
            self._global_monitor = NSEvent.addGlobalMonitorForEventsMatchingMask_handler_(
                NSEventMaskKeyDown, _check_opt_space
            )

            # Local monitor (when app is frontmost)
            def _local_handler(event):
                _check_opt_space(event)
                return event  # Don't consume; just observe

            self._local_monitor = NSEvent.addLocalMonitorForEventsMatchingMask_handler_(
                NSEventMaskKeyDown, _local_handler
            )
        except Exception as e:
            # If we fail to install, surface a one-time notification to guide user
            rumps.notification("Ambient Scratchpad", "Hotkey unavailable",
                               "Enable Accessibility for Python/Terminal, then restart.")

# ---------- Run ----------
def main():
    if PYOBJC_AVAILABLE:
        # Run as accessory (menubar-only) so it doesn't clutter the Dock
        app = NSApplication.sharedApplication()
        app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)
    AmbientScratchpadApp().run()


if __name__ == "__main__":
    main()
