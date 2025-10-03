# main-fixed-12.py
# Updated with requested UI changes

import os
import sys
import json
import time
import uuid
import threading
import sqlite3
import struct
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

import rumps
from Cocoa import (
    NSObject,
    NSWindow,
    NSScrollView,
    NSTextView,
    NSFont,
    NSMakeRect,
    NSBackingStoreBuffered,
    NSTitledWindowMask,
    NSClosableWindowMask,
    NSResizableWindowMask,
    NSColor,
    NSTextField,
    NSBezelBorder,
    NSTextAlignment,
)

class ShowAllNotesWindowController(NSObject):
    def init(self):
        self = super(ShowAllNotesWindowController, self).init()
        if self is None:
            return None

        # Create the window
        self.window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(200, 200, 700, 500),
            NSTitledWindowMask | NSClosableWindowMask | NSResizableWindowMask,
            NSBackingStoreBuffered,
            False,
        )
        self.window.setTitle_("All Notes")

        # Title / header label (nudged right)
        self.titleField = NSTextField.alloc().initWithFrame_(NSMakeRect(20, 460, 660, 24))
        self.titleField.setBezeled_(False)
        self.titleField.setDrawsBackground_(False)
        self.titleField.setEditable_(False)
        self.titleField.setSelectable_(False)
        self.titleField.setFont_(NSFont.systemFontOfSize_(14))
        self.titleField.setStringValue_("All Notes Summary")
        self.titleField.setAlignment_(NSTextAlignment.Natural)
        self.window.contentView().addSubview_(self.titleField)

        # Scrollable text area for notes
        self.scrollView = NSScrollView.alloc().initWithFrame_(NSMakeRect(20, 100, 660, 350))
        self.textView = NSTextView.alloc().initWithFrame_(self.scrollView.bounds())
        self.textView.setFont_(NSFont.systemFontOfSize_(12))
        self.textView.setEditable_(False)
        self.scrollView.setDocumentView_(self.textView)
        self.scrollView.setHasVerticalScroller_(True)
        self.window.contentView().addSubview_(self.scrollView)

        # Bottom text box for full summary (fixed colors)
        self.summaryField = NSTextView.alloc().initWithFrame_(NSMakeRect(20, 20, 660, 70))
        self.summaryField.setFont_(NSFont.systemFontOfSize_(12))
        self.summaryField.setEditable_(False)
        # ✅ Fix: readable colors
        self.summaryField.setTextColor_(NSColor.textColor())
        self.summaryField.setBackgroundColor_(NSColor.windowBackgroundColor())
        self.window.contentView().addSubview_(self.summaryField)

        return self

    def showWindow(self):
        self.window.makeKeyAndOrderFront_(None)

    def updateNotes(self, notes_text: str, summary_text: str):
        self.textView.setString_(notes_text)
        self.summaryField.setString_(summary_text)
        self.titleField.setStringValue_(f"{len(notes_text.splitlines())} Notes — Highlights")


# Placeholder menu bar app
class ScratchpadApp(rumps.App):
    def __init__(self):
        super(ScratchpadApp, self).__init__("Scratchpad")
        self.notes_window_controller = ShowAllNotesWindowController.alloc().init()

    @rumps.clicked("Show All Notes")
    def show_notes(self, _):
        self.notes_window_controller.showWindow()
        self.notes_window_controller.updateNotes(
            "Example note 1\nExample note 2", "Actionable points and summary displayed here."
        )


if __name__ == "__main__":
    ScratchpadApp().run()
