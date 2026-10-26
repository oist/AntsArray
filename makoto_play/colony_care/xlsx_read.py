"""Read one sheet of an .xlsx workbook with the standard library only (no openpyxl).

Cells come back as stripped strings (None when empty). Cells with a date format become
"YYYY-MM-DD", or "YYYY-MM-DD HH:MM" when they carry a time of day.
"""
import re
import zipfile
import xml.etree.ElementTree as ET
from datetime import date, timedelta
from pathlib import Path

import sheet_model as M

NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
      "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships"}
DATE_FORMAT_IDS = {14, 15, 16, 17, 22, 27, 28, 29, 30, 31, 36} | set(range(50, 59))   # incl. Japanese
EXCEL_EPOCH = date(1899, 12, 30)
NUMBER = re.compile(r"^[0-9]+(?:\.[0-9]+)?$")


def _column(ref):
    n = 0
    for ch in re.match(r"[A-Z]+", ref).group(0):
        n = n * 26 + ord(ch) - 64
    return n - 1


def _text(node):
    """Text of a shared or inline string: its own <t> and rich-text runs, not phonetic (rPh) runs."""
    parts = node.findall("m:t", NS) + node.findall("m:r/m:t", NS)
    return "".join(t.text or "" for t in parts)


def _shared_strings(z):
    if "xl/sharedStrings.xml" not in z.namelist():
        return []
    return [_text(si) for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", NS)]


def _date_styles(z):
    """Indices of the cell styles that show a date (built-in date formats, or custom ones with 'yy')."""
    styles = ET.fromstring(z.read("xl/styles.xml"))
    custom = {int(n.attrib["numFmtId"]): n.attrib.get("formatCode", "")
              for n in styles.iter("{%s}numFmt" % NS["m"])}
    xfs = styles.find("m:cellXfs", NS)
    out = set()
    for i, xf in enumerate(xfs.findall("m:xf", NS) if xfs is not None else []):
        fmt = int(xf.attrib.get("numFmtId", 0))
        if fmt in DATE_FORMAT_IDS or "yy" in custom.get(fmt, "").lower():
            out.add(i)
    return out


def _as_date(value):
    serial = float(value)
    day = EXCEL_EPOCH + timedelta(days=int(serial))
    minutes = round((serial % 1) * 1440)
    if minutes == 1440:
        day, minutes = day + timedelta(days=1), 0
    return day.isoformat() + (" %02d:%02d" % divmod(minutes, 60) if minutes else "")


def _sheet_path(z, sheet_name):
    wb = ET.fromstring(z.read("xl/workbook.xml"))
    rels = {r.attrib["Id"]: r.attrib["Target"] for r in ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))}
    names = []
    for sheet in wb.find("m:sheets", NS):
        names.append(sheet.attrib["name"])
        if sheet.attrib["name"] == sheet_name:
            target = rels[sheet.attrib["{%s}id" % NS["r"]]].lstrip("/")
            return target if target.startswith("xl/") else "xl/" + target
    raise M.InputError("no sheet %r in the workbook (sheets: %s)" % (sheet_name, ", ".join(names)))


def read_sheet(path, sheet_name):
    """Rows of one sheet as lists of cell strings, in sheet order (row gaps are not kept)."""
    with zipfile.ZipFile(Path(path)) as z:
        shared = _shared_strings(z)
        dates = _date_styles(z)
        root = ET.fromstring(z.read(_sheet_path(z, sheet_name)))
    rows = []
    for row in root.iter("{%s}row" % NS["m"]):
        cells = {}
        for c in row.findall("m:c", NS):
            kind, value = c.attrib.get("t"), None
            if kind == "inlineStr":
                value = _text(c.find("m:is", NS)) if c.find("m:is", NS) is not None else ""
            else:
                v = c.find("m:v", NS)
                if v is None or v.text is None:
                    continue
                value = shared[int(v.text)] if kind == "s" else v.text
                if kind not in ("s", "str") and int(c.attrib.get("s", 0)) in dates and NUMBER.match(value):
                    value = _as_date(value)
            value = value.strip()
            cells[_column(c.attrib["r"])] = value if value else None
        width = max(cells) + 1 if cells else 0
        rows.append([cells.get(i) for i in range(width)])
    return rows
