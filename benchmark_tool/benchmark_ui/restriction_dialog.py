"""
CreateRestrictionDialog — popup for creating Road Closure (RC) or
Turn Restriction (TR) records that are saved as JSON files.
"""
from __future__ import annotations

from PyQt6.QtCore    import Qt, QDate, QTime
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QLineEdit, QPushButton, QComboBox, QMessageBox, QFrame,
    QDateEdit, QTimeEdit,
)


_STYLE = """
QDialog { background:#1e1e2e; color:#cdd6f4; }
QLabel  { color:#cdd6f4; }

QLineEdit {
    background:#313244; color:#cdd6f4;
    border:1px solid #45475a; border-radius:4px;
    padding:5px 8px;
}
QLineEdit:focus { border-color:#cba6f7; }
QLineEdit[readOnly="true"] {
    background:#181825; color:#a6adc8;
    border:1px solid #313244;
}

QDateEdit, QTimeEdit {
    background:#313244; color:#cdd6f4;
    border:1px solid #45475a; border-radius:4px;
    padding:4px 8px;
}
QDateEdit:focus, QTimeEdit:focus { border-color:#cba6f7; }
QDateEdit::drop-down, QTimeEdit::drop-down { border:none; width:20px; }
QDateEdit::down-arrow, QTimeEdit::down-arrow { image:none; }

/* Calendar popup */
QCalendarWidget QWidget { background:#313244; color:#cdd6f4; }
QCalendarWidget QAbstractItemView {
    background:#1e1e2e; color:#cdd6f4;
    selection-background-color:#cba6f7; selection-color:#1e1e2e;
}
QCalendarWidget QToolButton { color:#cdd6f4; background:#45475a; border-radius:4px; }
QCalendarWidget QToolButton:hover { background:#585b70; }
QCalendarWidget #qt_calendar_navigationbar { background:#181825; }
QCalendarWidget QSpinBox {
    background:#313244; color:#cdd6f4;
    border:1px solid #45475a; border-radius:4px;
}

QComboBox {
    background:#313244; color:#cdd6f4;
    border:1px solid #45475a; border-radius:4px;
    padding:4px 8px;
}
QComboBox::drop-down { border:none; }
QComboBox QAbstractItemView {
    background:#313244; color:#cdd6f4;
    selection-background-color:#45475a;
}
"""


def _hint(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet("color:#585b70; font-size:10px;")
    return lbl


def _row_label(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet("color:#a6adc8; font-size:11px;")
    return lbl


def _make_date_edit(q_date: QDate, read_only: bool = False) -> QDateEdit:
    w = QDateEdit(q_date)
    w.setDisplayFormat("dd/MM/yyyy")
    w.setCalendarPopup(True)
    w.setReadOnly(read_only)
    if read_only:
        w.setStyleSheet(
            "QDateEdit{background:#181825;color:#a6adc8;"
            "border:1px solid #313244;border-radius:4px;padding:4px 8px;}"
        )
    return w


def _make_time_edit(q_time: QTime) -> QTimeEdit:
    w = QTimeEdit(q_time)
    w.setDisplayFormat("HH:mm")   # no seconds
    return w


class CreateRestrictionDialog(QDialog):
    """
    Modal dialog for creating or editing a restriction record.

    Pass `prefill` (existing record dict) to open in edit mode.
    After exec() == Accepted, read `self.result_data` for the dict.
    Geometry fields ('geometry' for RC, 'node_coords' for TR) are left
    empty here — the caller fills them after fetching from Overpass.
    """

    def __init__(self, restriction_type: str, parent=None, prefill: dict | None = None):
        super().__init__(parent)
        self.restriction_type = restriction_type  # "RC" or "TR"
        self.result_data: dict | None = None
        self._prefill = prefill or {}

        editing = bool(prefill)
        type_label = "Road Closure" if restriction_type == "RC" else "Turn Restriction"
        icon       = "🚧" if restriction_type == "RC" else "↩"
        verb       = "Edit" if editing else "New"
        self.setWindowTitle(f"{verb} {type_label}")
        self.setFixedWidth(460)
        self.setStyleSheet(_STYLE)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(12)

        # ── Title ──────────────────────────────────────────────────────
        title = QLabel(f"{icon} {verb} {type_label}"
                       + (f"  [{prefill.get('id','')}]" if editing else ""))
        title.setStyleSheet(
            "font-size:14px; font-weight:700; color:#cba6f7; padding-bottom:2px;"
        )
        layout.addWidget(title)

        div = QFrame(); div.setFrameShape(QFrame.Shape.HLine)
        div.setStyleSheet("color:#313244;")
        layout.addWidget(div)

        # ── Form ───────────────────────────────────────────────────────
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form.setSpacing(10)

        # Name
        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("e.g. No Entry Võ Văn Kiệt…")
        form.addRow(_row_label("Name *"), self._name_edit)

        # Date create (read-only, today)
        today = QDate.currentDate()
        self._date_create = _make_date_edit(today, read_only=True)
        form.addRow(_row_label("Date create"), self._date_create)

        # Date end
        self._date_end = _make_date_edit(QDate(9999, 1, 1))
        self._date_end.setMaximumDate(QDate(9999, 1, 1))
        self._date_end.dateChanged.connect(self._on_date_end_changed)
        form.addRow(_row_label("Date end *"), self._date_end)

        # ── Time range ─────────────────────────────────────────────────
        # Two rows: "Bắt đầu" (Start) and "Kết thúc" (End) each with date + time
        form.addRow(_row_label(""), QLabel())   # spacer row

        # Start datetime row
        self._dt_start_date = _make_date_edit(today)
        self._dt_start_time = _make_time_edit(QTime(0, 0))
        self._dt_start_date.dateChanged.connect(self._on_start_date_changed)
        self._dt_start_time.timeChanged.connect(self._on_start_time_changed)
        start_row = QHBoxLayout()
        start_row.addWidget(self._dt_start_date, 1)
        start_row.addWidget(self._dt_start_time)
        form.addRow(_row_label("Start *"), start_row)

        # End datetime row
        self._dt_end_date = _make_date_edit(today)
        self._dt_end_time = _make_time_edit(QTime(23, 59))
        self._dt_end_date.dateChanged.connect(self._on_end_date_changed)
        end_row = QHBoxLayout()
        end_row.addWidget(self._dt_end_date, 1)
        end_row.addWidget(self._dt_end_time)
        form.addRow(_row_label("End *"), end_row)

        form.addRow(_hint("  End date/time cannot be earlier than start"), QLabel())

        # Vehicle type
        self._vehicle_combo = QComboBox()
        self._vehicle_combo.addItems(["2W/4W", "2W", "4W"])
        form.addRow(_row_label("Vehicle type"), self._vehicle_combo)

        # ── Type-specific fields ───────────────────────────────────────
        if restriction_type == "RC":
            self._way_ids_edit = QLineEdit()
            self._way_ids_edit.setPlaceholderText("719473072, 234567, …")
            form.addRow(_row_label("Way IDs *"), self._way_ids_edit)
            form.addRow("", _hint("Comma-separated OSM way IDs to block"))

            # Direction of closure — based on OSM way node order
            self._direction_combo = QComboBox()
            self._direction_combo.addItems([
                "↔  Both directions  (block all traffic)",
                "→  Forward  (block in OSM way direction)",
                "←  Reverse  (block against OSM way direction)",
            ])
            form.addRow(_row_label("Direction *"), self._direction_combo)
            form.addRow("", _hint("OSM way direction = from first node → last node"))
        else:  # TR
            self._node_from = QLineEdit()
            self._node_from.setPlaceholderText("OSM node ID (from)")
            self._node_via  = QLineEdit()
            self._node_via.setPlaceholderText("OSM node ID (via / intersection)")
            self._node_to   = QLineEdit()
            self._node_to.setPlaceholderText("OSM node ID (to)")
            form.addRow(_row_label("From node *"), self._node_from)
            form.addRow(_row_label("Via node *"),  self._node_via)
            form.addRow(_row_label("To node *"),   self._node_to)
            form.addRow("", _hint("3 consecutive OSM node IDs defining the forbidden turn"))

        layout.addLayout(form)

        # ── Pre-fill fields when editing ───────────────────────────────
        if editing:
            self._apply_prefill()

        div2 = QFrame(); div2.setFrameShape(QFrame.Shape.HLine)
        div2.setStyleSheet("color:#313244;")
        layout.addWidget(div2)

        # ── Buttons ────────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_cancel = QPushButton("Cancel")
        btn_cancel.setStyleSheet(
            "QPushButton{background:#313244;color:#cdd6f4;border-radius:6px;"
            "padding:6px 18px;}"
            "QPushButton:hover{background:#45475a;}"
        )
        btn_cancel.clicked.connect(self.reject)

        btn_save = QPushButton("Save")
        btn_save.setStyleSheet(
            "QPushButton{background:#cba6f7;color:#1e1e2e;border-radius:6px;"
            "padding:6px 18px;font-weight:700;}"
            "QPushButton:hover{background:#d4b8fb;}"
        )
        btn_save.clicked.connect(self._on_save)
        btn_save.setDefault(True)

        btn_row.addStretch()
        btn_row.addWidget(btn_cancel)
        btn_row.addWidget(btn_save)
        layout.addLayout(btn_row)

    # ── Pre-fill for edit mode ─────────────────────────────────────────
    def _apply_prefill(self):
        p = self._prefill

        def _parse_date(s: str, fallback: QDate) -> QDate:
            from PyQt6.QtCore import QDate as _QD
            d = _QD.fromString(s, "yyyy-MM-dd")
            return d if d.isValid() else fallback

        def _parse_time(s: str, fallback: QTime) -> QTime:
            from PyQt6.QtCore import QTime as _QT
            t = _QT.fromString(s, "HH:mm")
            return t if t.isValid() else fallback

        today = QDate.currentDate()

        if p.get("name"):
            self._name_edit.setText(p["name"])

        if p.get("date_end"):
            self._date_end.setDate(_parse_date(p["date_end"], QDate(9999, 1, 1)))

        # dt_start / dt_end stored as "yyyy-MM-dd HH:mm"
        if p.get("dt_start"):
            parts = p["dt_start"].split(" ")
            self._dt_start_date.setDate(_parse_date(parts[0], today))
            if len(parts) > 1:
                self._dt_start_time.setTime(_parse_time(parts[1], QTime(0, 0)))
        elif p.get("time_start"):
            self._dt_start_time.setTime(_parse_time(p["time_start"], QTime(0, 0)))

        if p.get("dt_end"):
            parts = p["dt_end"].split(" ")
            self._dt_end_date.setDate(_parse_date(parts[0], today))
            if len(parts) > 1:
                self._dt_end_time.setTime(_parse_time(parts[1], QTime(23, 59)))
        elif p.get("time_end"):
            self._dt_end_time.setTime(_parse_time(p["time_end"], QTime(23, 59)))

        vmap = {"2W/4W": 0, "2W": 1, "4W": 2}
        if p.get("vehicle_type") in vmap:
            self._vehicle_combo.setCurrentIndex(vmap[p["vehicle_type"]])

        if self.restriction_type == "RC":
            way_ids = p.get("way_ids", [])
            if way_ids:
                self._way_ids_edit.setText(", ".join(str(w) for w in way_ids))
            dmap = {"2way": 0, "1way": 1, "1way_reverse": 2}
            self._direction_combo.setCurrentIndex(dmap.get(p.get("direction", "2way"), 0))
        else:  # TR
            node_ids = p.get("node_ids", [])
            if len(node_ids) >= 3:
                self._node_from.setText(str(node_ids[0]))
                self._node_via.setText(str(node_ids[1]))
                self._node_to.setText(str(node_ids[2]))

    # ── Date / time constraint handlers ───────────────────────────────
    def _on_date_end_changed(self, d: QDate):
        """Date end cannot be before date create."""
        if d < self._date_create.date():
            self._date_end.setDate(self._date_create.date())

    def _on_start_date_changed(self, d: QDate):
        """End date must be >= start date."""
        if self._dt_end_date.date() < d:
            self._dt_end_date.setDate(d)

    def _on_end_date_changed(self, d: QDate):
        """End date cannot be before start date."""
        if d < self._dt_start_date.date():
            self._dt_end_date.setDate(self._dt_start_date.date())
        # If same day, enforce time order
        self._enforce_time_order()

    def _on_start_time_changed(self, _t: QTime):
        self._enforce_time_order()

    def _enforce_time_order(self):
        """If start and end are on the same day, end time must be >= start time."""
        if self._dt_start_date.date() == self._dt_end_date.date():
            if self._dt_end_time.time() < self._dt_start_time.time():
                self._dt_end_time.setTime(self._dt_start_time.time())

    # ── Save ───────────────────────────────────────────────────────────
    def _on_save(self):
        name = self._name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "Validation", "Name is required.")
            return

        # Date validation
        date_create = self._date_create.date()
        date_end    = self._date_end.date()
        if date_end < date_create:
            QMessageBox.warning(self, "Validation",
                                "Date end cannot be earlier than date create.")
            return

        # Datetime range validation
        start_date = self._dt_start_date.date()
        end_date   = self._dt_end_date.date()
        start_time = self._dt_start_time.time()
        end_time   = self._dt_end_time.time()

        if end_date < start_date or (
            end_date == start_date and end_time < start_time
        ):
            QMessageBox.warning(self, "Validation",
                                "End date/time cannot be earlier than start date/time.")
            return

        vehicle = self._vehicle_combo.currentText()

        base = {
            "name":         name,
            "date_create":  date_create.toString("yyyy-MM-dd"),
            "date_end":     date_end.toString("yyyy-MM-dd"),
            "time_start":   start_time.toString("HH:mm"),
            "time_end":     end_time.toString("HH:mm"),
            "dt_start":     f"{start_date.toString('yyyy-MM-dd')} {start_time.toString('HH:mm')}",
            "dt_end":       f"{end_date.toString('yyyy-MM-dd')} {end_time.toString('HH:mm')}",
            "vehicle_type": vehicle,
        }

        if self.restriction_type == "RC":
            raw = self._way_ids_edit.text().strip()
            if not raw:
                QMessageBox.warning(self, "Validation", "Enter at least one Way ID.")
                return
            try:
                way_ids = [int(x.strip()) for x in raw.split(",") if x.strip()]
            except ValueError:
                QMessageBox.warning(self, "Validation", "Way IDs must be integers.")
                return
            if not way_ids:
                QMessageBox.warning(self, "Validation", "Enter at least one valid Way ID.")
                return

            # 0=2way, 1=1way (forward), 2=1way_reverse
            _dir_map = {0: "2way", 1: "1way", 2: "1way_reverse"}
            direction = _dir_map.get(self._direction_combo.currentIndex(), "2way")

            self.result_data = {
                "type":      "RC",
                **base,
                "direction": direction,
                "way_ids":   way_ids,
                "geometry":  {},
            }

        else:  # TR
            try:
                node_from = int(self._node_from.text().strip())
                node_via  = int(self._node_via.text().strip())
                node_to   = int(self._node_to.text().strip())
            except ValueError:
                QMessageBox.warning(self, "Validation", "Node IDs must be integers.")
                return

            self.result_data = {
                "type":        "TR",
                **base,
                "node_ids":    [node_from, node_via, node_to],
                "node_coords": {},
            }

        self.accept()
