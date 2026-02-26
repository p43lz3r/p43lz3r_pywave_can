# 2026-02-24 00:00 v1.1.0 - Full Textual component-class coverage: DataTable cursor/hover/header,
#                           Tabs underline bar, Input focus/cursor/placeholder, Button hover/focus,
#                           OptionList highlighted (fixes blue dropdown bar), ScrollBar, Select focus.
"""
Midnight theme – classic blue/cyan CAN Bus TUI colour scheme.

All colours for both themes are defined here as a single dict.
Keys are grouped by GUI area and commented so you know exactly
what each colour controls.  No colour is defined anywhere else.
"""

THEME: dict = {

    # -------------------------------------------------------------------------
    # Global screen background and default text colour
    # -------------------------------------------------------------------------
    "bg":  "#000080",   # Screen / panel background (main blue)
    "fg":  "#00ffff",   # Default foreground / text colour (cyan)

    # -------------------------------------------------------------------------
    # Panel borders  (ConnectionPanel, MonitorPanel, StatusPanel, etc.)
    # -------------------------------------------------------------------------
    "border": "#00ffff",

    # -------------------------------------------------------------------------
    # Panel title bar  (.panel-title, #disc-control-title, .fs-title, etc.)
    # -------------------------------------------------------------------------
    "title_bg": "#00aaaa",   # Title bar background
    "title_fg": "#000000",   # Title bar text

    # -------------------------------------------------------------------------
    # Header and Footer bars (top / bottom app chrome)
    # -------------------------------------------------------------------------
    "header_bg": "#00aaaa",
    "header_fg": "#000000",
    "footer_bg": "#00aaaa",
    "footer_fg": "#000000",

    # -------------------------------------------------------------------------
    # Form labels  (Port:, Speed:, Mode:, ID:, Data:, …)
    # -------------------------------------------------------------------------
    "accent": "#ffffff",   # .form-label, .send-label, .send-label-sm/md

    # -------------------------------------------------------------------------
    # Status indicators
    # -------------------------------------------------------------------------
    "ok":  "#00ff00",   # Connected status, send-status, trace-scroll-on
    "err": "#ff5555",   # Disconnected status, stale-frame highlight

    # -------------------------------------------------------------------------
    # Buttons
    # -------------------------------------------------------------------------
    "btn_bg":       "#004488",
    "btn_fg":       "#ffffff",
    "btn_hover_bg": "#0066cc",   # Button:hover background
    "btn_hover_fg": "#ffffff",   # Button:hover text
    "btn_focus_bg": "#0055aa",   # Button:focus background
    "btn_border":   "#00ffff",   # Button:focus border colour

    # -------------------------------------------------------------------------
    # Data tables  (#monitor-table, #event-log, #stats-table, #trace-table, …)
    # -------------------------------------------------------------------------
    "table_bg":             "#000060",
    "log_bg":               "#000060",
    # Column header row
    "table_header_bg":      "#004466",
    "table_header_fg":      "#00ffff",
    # Cursor bar – table has keyboard focus
    "table_cursor_bg":      "#007799",
    "table_cursor_fg":      "#ffffff",
    # Cursor bar – table visible but not focused (blurred)
    "table_cursor_blur_bg": "#004455",
    "table_cursor_blur_fg": "#aaaaaa",
    # Mouse-hover row
    "table_hover_bg":       "#003366",
    "table_hover_fg":       "#ffffff",
    # Zebra stripes (only active when zebra_stripes=True on a DataTable)
    "table_even_bg":        "#000070",
    "table_odd_bg":         "#000060",

    # -------------------------------------------------------------------------
    # Input fields
    # -------------------------------------------------------------------------
    "input_bg":           "#001a4d",
    "input_border":       "#006688",   # Normal border
    "input_focus_bg":     "#001a4d",   # Background when focused
    "input_focus_border": "#00ffff",   # Border colour when focused
    "input_cursor_bg":    "#00ffff",   # Text-cursor block background
    "input_cursor_fg":    "#000000",   # Text-cursor block foreground
    "input_placeholder":  "#446688",   # Placeholder / hint text colour

    # -------------------------------------------------------------------------
    # Panels with slightly darker background
    # -------------------------------------------------------------------------
    "panel_dark_bg": "#000060",

    # -------------------------------------------------------------------------
    # Select / dropdown widgets
    # -------------------------------------------------------------------------
    "select_bg":           "#0000aa",   # Outer Select widget background
    "select_fg":           "#ffff55",   # Outer Select widget text
    "select_current_bg":   "#000060",   # Closed state (shows selected value)
    "select_current_fg":   "#ffff55",
    "select_focus_border": "#00ffff",   # Border when closed Select is focused
    "select_arrow":        "#00ffff",   # ▼ arrow colour
    "select_overlay_bg":   "#00A8AF",   # Open dropdown background
    "select_overlay_fg":   "#ffffff",   # Open dropdown text
    # OptionList .option-list--option-highlighted  (the bar that was stuck blue)
    "select_highlight_bg": "#ffff55",
    "select_highlight_fg": "#0000aa",

    # -------------------------------------------------------------------------
    # Tabs  (Details screen: Event Log / Statistics / DBC / Trace)
    # -------------------------------------------------------------------------
    "tab_active_bg":   "#00aaaa",   # Active tab label background
    "tab_active_fg":   "#000000",   # Active tab label text
    "tab_inactive_bg": "#000060",   # Inactive tab label background
    "tab_inactive_fg": "#00ffff",   # Inactive tab label text
    "tab_hover_bg":    "#006688",   # Hovered (not yet active) tab
    "tab_hover_fg":    "#ffffff",
    "tab_underline":   "#00ffff",   # Underline bar below the active tab
    "tab_bar_bg":      "#000060",   # Background of the tab-bar strip

    # -------------------------------------------------------------------------
    # ScrollBar
    # -------------------------------------------------------------------------
    "scrollbar_bg":             "#001a4d",   # Track colour
    "scrollbar_thumb_bg":       "#006688",   # Thumb colour
    "scrollbar_thumb_hover_bg": "#00aaaa",   # Thumb colour when hovered

    # -------------------------------------------------------------------------
    # Trace recorder state labels
    # -------------------------------------------------------------------------
    "trace_idle":      "#888888",
    "trace_recording": "#ff4444",
    "trace_paused":    "#ffaa00",
    "trace_warning":   "#ff4444",
    "trace_scroll_on": "#00ff00",
    "trace_scroll_off":"#ff8800",

    # -------------------------------------------------------------------------
    # Byte-change highlight in monitor table
    # -------------------------------------------------------------------------
    "highlight": "#ffff00",

    # -------------------------------------------------------------------------
    # Monitor PAUSED banner
    # -------------------------------------------------------------------------
    "paused":    "#ff8800",
    "paused_fg": "#000000",

    # -------------------------------------------------------------------------
    # Bus-load bar  (three-zone colour scale)
    # -------------------------------------------------------------------------
    "load_low":  "#00ff00",   # 0 – 60 %
    "load_mid":  "#ffaa00",   # 60 – 85 %
    "load_high": "#ff5555",   # 85 – 100 %

    # -------------------------------------------------------------------------
    # Shortcuts modal dialog  (F1 overlay)
    # -------------------------------------------------------------------------
    "modal_bg":     "#001a4d",
    "modal_border": "#00ffff",

    # -------------------------------------------------------------------------
    # Hint / secondary text
    # -------------------------------------------------------------------------
    "hint": "#888888",

    # -------------------------------------------------------------------------
    # DBC placeholder text
    # -------------------------------------------------------------------------
    "dbc_placeholder": "#00aaaa",

    # -------------------------------------------------------------------------
    # DBC status line  (#dbc-status)
    # -------------------------------------------------------------------------
    "dbc_status": "#00ffff",
}
