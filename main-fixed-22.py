# main.py (patched)
"""
Ambient Scratchpad — minimal, self-contained build focused on:
  1) System-wide hotkey Option+Space for Quick Capture
  2) Fix black text on black background for the bottom "Actionables" text view
  3) Shift header title slightly to the right
  4) Proper PyObjC init using objc.super(...)
"""

import rumps

# --- PyObjC / AppKit ---
try:
    from AppKit import (
        NSApplication, NSApplicationActivationPolicyAccessory,
        NSWindow, NSWindowStyleMaskTitled, NSWindowStyleMaskClosable, NSWindowStyleMaskResizable,
        NSViewWidthSizable, NSViewHeightSizable,
        NSScrollView, NSTextView, NSFont, NSColor,
        NSView, NSBezelBorder,
        NSEvent, NSEventMaskKeyDown, NSStackView, NSTextField
    )
    from Foundation import NSObject, NSMakeRect
    from PyObjCTools import AppHelper
    import objc
    PYOBJC_AVAILABLE = True
except Exception:
    PYOBJC_AVAILABLE = False

NOTES = []
GLOBAL_WINDOWS = []

def _now_iso():
    import datetime
    return datetime.datetime.now().isoformat(timespec="seconds")

def quick_capture_pipeline(text: str):
    NOTES.insert(0, {"text": text, "timestamp": _now_iso()})

if PYOBJC_AVAILABLE:
    class ShowAllNotesWindowController(NSObject):
        def init(self):
            self = objc.super(ShowAllNotesWindowController, self).init()
            if self is None:
                return None

            self.window = None
            self.header_label = None
            self.detail_text = None
            self.actionable_text = None
            return self

        def open(self):
            self.window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
                NSMakeRect(100, 100, 820, 560),
                NSWindowStyleMaskTitled | NSWindowStyleMaskClosable | NSWindowStyleMaskResizable,
                2,
                False
            )
            self.window.setTitle_("Ambient Scratchpad — All Notes")
            self.window.setReleasedWhenClosed_(False)
            self.window.setIsVisible_(True)

            content = self.window.contentView()
            content.setAutoresizingMask_(NSViewWidthSizable | NSViewHeightSizable)

            root = NSStackView.alloc().init()
            root.setOrientation_(1)
            root.setSpacing_(8.0)
            root.setEdgeInsets_((12, 12, 12, 12))
            root.setAutoresizingMask_(NSViewWidthSizable | NSViewHeightSizable)
            root.setFrame_(content.bounds())
            content.addSubview_(root)

            # Header shifted right
            header_stack = NSStackView.alloc().init()
            header_stack.setOrientation_(0)
            header_stack.setSpacing_(8.0)

            spacer = NSView.alloc().initWithFrame_(NSMakeRect(0, 0, 12, 4))
            header_stack.addView_inGravity_(spacer, 1)

            self.header_label = NSTextField.alloc().initWithFrame_(NSMakeRect(0, 0, 600, 22))
            self.header_label.setStringValue_("0 notes — Highlights: none yet")
            self.header_label.setEditable_(False)
            self.header_label.setBordered_(False)
            self.header_label.setDrawsBackground_(False)
            self.header_label.setFont_(NSFont.boldSystemFontOfSize_(14.0))
            header_stack.addView_inGravity_(self.header_label, 1)
            root.addView_inGravity_(header_stack, 1)

            # Middle detail
            detail_scroll = NSScrollView.alloc().initWithFrame_(NSMakeRect(0, 0, 800, 360))
            detail_scroll.setHasVerticalScroller_(True)
            detail_scroll.setBorderType_(NSBezelBorder)
            self.detail_text = NSTextView.alloc().initWithFrame_(detail_scroll.contentView().bounds())
            self.detail_text.setEditable_(False)
            self.detail_text.setFont_(NSFont.systemFontOfSize_(13.0))
            self.detail_text.setTextColor_(NSColor.labelColor())
            self.detail_text.setBackgroundColor_(NSColor.textBackgroundColor())
            self.detail_text.setString_("Select or add a note…")
            detail_scroll.setDocumentView_(self.detail_text)
            root.addView_inGravity_(detail_scroll, 1)

            # Bottom actionables (fix colors)
            actionable_scroll = NSScrollView.alloc().initWithFrame_(NSMakeRect(0, 0, 800, 140))
            actionable_scroll.setHasVerticalScroller_(True)
            actionable_scroll.setBorderType_(NSBezelBorder)
            self.actionable_text = NSTextView.alloc().initWithFrame_(actionable_scroll.contentView().bounds())
            self.actionable_text.setEditable_(False)
            self.actionable_text.setFont_(NSFont.monospacedSystemFontOfSize_weight_(12.0, 0))
            self.actionable_text.setTextColor_(NSColor.labelColor())
            self.actionable_text.setBackgroundColor_(NSColor.controlBackgroundColor())
            self.actionable_text.setString_("Actionable summary, ideas, snippets will show here…")
            actionable_scroll.setDocumentView_(self.actionable_text)
            root.addView_inGravity_(actionable_scroll, 1)

            GLOBAL_WINDOWS.append(self.window)
            self.refresh()

        def refresh(self):
            count = len(NOTES)
            highlights = "recent" if count else "none yet"
            self.header_label.setStringValue_(f"{count} notes — Highlights: {highlights}")
            if NOTES:
                latest = NOTES[0]
                self.detail_text.setString_(f"[{latest['timestamp']}]  {latest['text']}")
                self.actionable_text.setString_("• Example next steps for latest note")
            else:
                self.detail_text.setString_("No notes yet. Hit Option+Space to add one.")
                self.actionable_text.setString_("")

    _WC = ShowAllNotesWindowController.alloc().init()

class AmbientScratchpadApp(rumps.App):
    def __init__(self):
        super().__init__("✳︎", quit_button=rumps.MenuItem("Quit"))
        self.menu = [
            rumps.MenuItem("Quick Capture", callback=self.quick_capture),
            rumps.MenuItem("Show All Notes", callback=self.show_all_notes),
        ]
        self._install_global_hotkey()

    def quick_capture(self, _=None):
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
                quick_capture_pipeline(text)
                if PYOBJC_AVAILABLE and _WC and _WC.window:
                    try: _WC.refresh()
                    except Exception: pass
                rumps.notification("Ambient Scratchpad", "Captured", text[:80])

    def show_all_notes(self, _=None):
        if not PYOBJC_AVAILABLE:
            rumps.alert("PyObjC not available", "Install pyobjc and restart.")
            return
        if not _WC.window:
            _WC.open()
        else:
            _WC.window.makeKeyAndOrderFront_(None)
            _WC.refresh()

    def _install_global_hotkey(self):
        if not PYOBJC_AVAILABLE:
            return
        try:
            from AppKit import NSEventModifierFlagOption as _OPT_FLAG
        except Exception:
            from AppKit import NSAlternateKeyMask as _OPT_FLAG

        def _check_opt_space(event):
            try:
                keycode = int(event.keyCode())
                flags = int(event.modifierFlags())
                chars = str(event.characters())
                is_opt = (flags & _OPT_FLAG) == _OPT_FLAG
                if is_opt and (keycode == 49 or chars == " "):
                    AppHelper.callAfter(self.quick_capture, None)
            except Exception:
                pass

        self._global_monitor = NSEvent.addGlobalMonitorForEventsMatchingMask_handler_(
            NSEventMaskKeyDown, _check_opt_space
        )
        def _local_handler(event):
            _check_opt_space(event)
            return event
        self._local_monitor = NSEvent.addLocalMonitorForEventsMatchingMask_handler_(
            NSEventMaskKeyDown, _local_handler
        )

def main():
    if PYOBJC_AVAILABLE:
        app = NSApplication.sharedApplication()
        app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)
    AmbientScratchpadApp().run()

if __name__ == "__main__":
    main()
