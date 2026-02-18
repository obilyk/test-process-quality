import sys
import csv
import numpy as np
from typing import Optional, Tuple

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout,
    QGroupBox, QLabel, QPushButton, QSpinBox,
    QTableWidget, QTableWidgetItem,
    QMessageBox, QFileDialog, QSizePolicy,
    QAbstractItemView, QHeaderView
)


# ---------------- Parsing / formatting ----------------

def parse_cell_value(text: str) -> Optional[float]:
    t = (text or "").strip()
    if not t:
        return None

    unicode_fracs = {"½": 0.5, "¼": 0.25, "¾": 0.75}
    if t in unicode_fracs:
        return unicode_fracs[t]

    t = t.replace(",", ".")
    t = t.replace("⁄", "/").replace("∕", "/").replace("／", "/")
    t = t.replace(" / ", "/").replace("/ ", "/").replace(" /", "/")

    try:
        if "/" in t:
            a, b = t.split("/", 1)
            a = float(a.strip())
            b = float(b.strip())
            if b == 0:
                return None
            return a / b
        return float(t)
    except Exception:
        return None


def format_saaty_value(x: float) -> str:
    best_label = None
    best_err = 1e9
    for k in range(1, 10):
        for label, v in ((str(k), float(k)), (f"1/{k}", 1.0 / k)):
            err = abs(x - v)
            if err < best_err:
                best_err = err
                best_label = label

    if best_label and best_err < 1e-6:
        return best_label
    if abs(x - round(x)) < 1e-6:
        return str(int(round(x)))
    return f"{x:.6f}"


# ---------------- AHP computations ----------------

def ahp_pipeline_with_bi(A: np.ndarray):
    n = A.shape[0]

    col_sums = A.sum(axis=0)
    col_sums[col_sums == 0] = 1.0
    BI = (A / col_sums).mean(axis=1)

    E = np.exp(np.mean(np.log(A), axis=1))
    En = E / E.sum()

    En1 = A @ En
    En2 = En1 / En
    lambda_max = float(np.mean(En2))

    if n <= 2:
        CI = CR = 0.0
    else:
        CI = (lambda_max - n) / (n - 1)
        RI = {
            3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24,
            7: 1.32, 8: 1.41, 9: 1.45,
            10: 1.49, 11: 1.51, 12: 1.48,
            13: 1.56, 14: 1.57, 15: 1.59
        }.get(n, 1.59)
        CR = CI / RI if RI else 0.0

    return BI, E, En, En1, En2, lambda_max, CI, CR


# ---------------- GUI ----------------

class AHPWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Метод бінарних (парних) порівнянь")
        self.resize(1320, 780)

        self.n = 8
        self._ignore = False

        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)

        # ---------- top ----------
        top = QHBoxLayout()
        outer.addLayout(top)

        top.addStretch()

        top.addWidget(QLabel("К-сть критеріїв:"))
        self.n_spin = QSpinBox()
        self.n_spin.setRange(2, 15)
        self.n_spin.setValue(self.n)
        self.n_spin.valueChanged.connect(self.on_n_changed)
        top.addWidget(self.n_spin)

        btn_clear = QPushButton("Очистити")
        btn_clear.clicked.connect(self.reset_matrix)
        top.addWidget(btn_clear)

        btn_calc = QPushButton("Обчислити")
        btn_calc.setDefault(True)
        btn_calc.clicked.connect(self.recalculate)
        top.addWidget(btn_calc)

        # ---------- main ----------
        main = QHBoxLayout()
        outer.addLayout(main, 1)

        # matrix
        g_mat = QGroupBox("Матриця ескпертних оцінок")
        main.addWidget(g_mat, 3)
        vm = QVBoxLayout(g_mat)

        self.table = QTableWidget()
        self.table.setEditTriggers(QAbstractItemView.AllEditTriggers)
        self.table.cellChanged.connect(self.on_cell_changed)
        self.table.setStyleSheet("QTableWidget { font-size: 16px; }")

        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Fixed)
        self.table.verticalHeader().setSectionResizeMode(QHeaderView.Fixed)
        self.table.horizontalHeader().setDefaultSectionSize(50)
        self.table.verticalHeader().setDefaultSectionSize(30)

        vm.addWidget(self.table)

        # results
        g_res = QGroupBox("Проміжні значення")
        main.addWidget(g_res, 2)
        vr = QVBoxLayout(g_res)

        self.mid = QTableWidget()
        self.mid.setColumnCount(5)
        self.mid.setHorizontalHeaderLabels(["BI", "E", "En", "En1", "En2"])
        self.mid.setEditTriggers(QTableWidget.NoEditTriggers)
        self.mid.setStyleSheet("QTableWidget { font-size: 14px; }")

        self.mid.horizontalHeader().setSectionResizeMode(QHeaderView.Fixed)
        self.mid.verticalHeader().setSectionResizeMode(QHeaderView.Fixed)
        self.mid.horizontalHeader().setDefaultSectionSize(55)
        self.mid.verticalHeader().setDefaultSectionSize(16)

        vr.addWidget(self.mid)

        self.lbl_lambda = QLabel("λmax: —")
        self.lbl_ci = QLabel("ІП: —")
        self.lbl_cr = QLabel("ВП: —")
        vr.addWidget(self.lbl_lambda)
        vr.addWidget(self.lbl_ci)
        vr.addWidget(self.lbl_cr)

        self.build_table()
        self.reset_matrix()

    # ---------- table logic ----------

    def on_n_changed(self, v):
        self.n = v
        self.build_table()
        self.reset_matrix()

    def build_table(self):
        self._ignore = True
        self.table.setRowCount(self.n)
        self.table.setColumnCount(self.n)
        labels = [str(i + 1) for i in range(self.n)]
        self.table.setHorizontalHeaderLabels(labels)
        self.table.setVerticalHeaderLabels(labels)

        for i in range(self.n):
            for j in range(self.n):
                it = QTableWidgetItem("1" if i == j else "")
                it.setTextAlignment(Qt.AlignCenter)
                if i == j:
                    it.setFlags(Qt.ItemIsEnabled)
                    it.setBackground(Qt.lightGray)
                self.table.setItem(i, j, it)

        self.mid.setRowCount(self.n)
        self.mid.setVerticalHeaderLabels(labels)
        self._ignore = False

    def reset_matrix(self):
        self._ignore = True
        for i in range(self.n):
            for j in range(self.n):
                self.table.item(i, j).setText("1" if i == j else "")
        self._ignore = False

    def on_cell_changed(self, r, c):
        if self._ignore or r == c:
            return

        txt = self.table.item(r, c).text().strip()
        if txt == "":
            return

        v = parse_cell_value(txt)
        if v is None or v <= 0:
            return

        txt = txt.replace("⁄", "/").replace("∕", "/").replace("／", "/")
        self._ignore = True
        self.table.item(r, c).setText(txt)
        self.table.item(c, r).setText(format_saaty_value(1 / v))
        self._ignore = False
        self.recalculate()

    def recalculate(self):
        A = np.ones((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    v = parse_cell_value(self.table.item(i, j).text())
                    if v is None:
                        return
                    A[i, j] = v

        BI, E, En, En1, En2, lm, CI, CR = ahp_pipeline_with_bi(A)

        for i in range(self.n):
            for c, v in enumerate([BI[i], E[i], En[i], En1[i], En2[i]]):
                it = QTableWidgetItem(f"{v:.12f}")
                it.setTextAlignment(Qt.AlignCenter)
                self.mid.setItem(i, c, it)

        self.lbl_lambda.setText(f"λmax: {lm:.12f}")
        self.lbl_ci.setText(f"ІП: {CI:.12f}")
        self.lbl_cr.setText(f"ВП: {CR:.12f}")


def main():
    app = QApplication(sys.argv)
    w = AHPWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
