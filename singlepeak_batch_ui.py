import json
import os
import queue
import re
import subprocess
import sys
import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk


APP_DIR = Path(__file__).resolve().parent
RUNNER = APP_DIR / "run_singlepeak_batch.py"


class SinglePeakBatchUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("单峰时间戳批处理")
        self.root.geometry("920x720")
        self.root.minsize(820, 620)

        self.process: subprocess.Popen[str] | None = None
        self.messages: queue.Queue[tuple[str, object]] = queue.Queue()
        self.channel_files: dict[str, Path] = {}

        self.data_dir = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.pair1_signal = tk.StringVar(value="3")
        self.pair1_idler = tk.StringVar(value="1")
        self.pair2_signal = tk.StringVar(value="4")
        self.pair2_idler = tk.StringVar(value="2")
        self.total_time_s = tk.StringVar(value="10000")
        self.split_step_s = tk.StringVar(value="10")
        self.correlation_window_s = tk.StringVar(value="4")
        self.correlation_frames = tk.StringVar(value="40000000")
        self.bin_width_ps = tk.StringVar(value="20")
        self.bin_num = tk.StringVar(value="10000")
        self.hist_bin_width_ps = tk.StringVar(value="1")
        self.hist_center_ps = tk.StringVar(value="100000")
        self.hist_points = tk.StringVar(value="200000")
        self.fit_half_window_bins = tk.StringVar(value="800")
        self.read_chunk_size = tk.StringVar(value="2000000")
        self.workers = tk.StringVar(value="4")
        self.pair1_time_diff = tk.StringVar()
        self.pair2_time_diff = tk.StringVar()
        self.save_histograms = tk.BooleanVar(value=True)
        self.status = tk.StringVar(value="请选择数据文件夹")

        self._build_ui()
        self._update_output_suggestion()
        self.root.after(100, self._drain_messages)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, padding=14)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(3, weight=1)

        paths = ttk.LabelFrame(outer, text="文件夹", padding=10)
        paths.grid(row=0, column=0, sticky="ew")
        paths.columnconfigure(1, weight=1)

        ttk.Label(paths, text="数据文件夹").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(paths, textvariable=self.data_dir).grid(row=0, column=1, sticky="ew", pady=4)
        ttk.Button(paths, text="选择...", command=self._choose_data_dir).grid(row=0, column=2, padx=(8, 0), pady=4)
        ttk.Button(paths, text="扫描", command=self._scan_channels).grid(row=0, column=3, padx=(8, 0), pady=4)

        ttk.Label(paths, text="输出文件夹").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(paths, textvariable=self.output_dir).grid(row=1, column=1, sticky="ew", pady=4)
        ttk.Button(paths, text="选择...", command=self._choose_output_dir).grid(row=1, column=2, padx=(8, 0), pady=4)

        pair_frame = ttk.LabelFrame(outer, text="通道配对", padding=10)
        pair_frame.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        for col in range(8):
            pair_frame.columnconfigure(col, weight=1 if col in (1, 3, 5, 7) else 0)

        channels = ("1", "2", "3", "4")
        ttk.Label(pair_frame, text="配对 1").grid(row=0, column=0, sticky="w")
        ttk.Combobox(pair_frame, textvariable=self.pair1_signal, values=channels, state="readonly", width=5).grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Label(pair_frame, text="-").grid(row=0, column=2)
        ttk.Combobox(pair_frame, textvariable=self.pair1_idler, values=channels, state="readonly", width=5).grid(row=0, column=3, sticky="ew", padx=6)
        ttk.Label(pair_frame, text="固定粗时延(ps)").grid(row=0, column=4, sticky="e", padx=(14, 6))
        ttk.Entry(pair_frame, textvariable=self.pair1_time_diff, width=14).grid(row=0, column=5, sticky="ew")

        ttk.Label(pair_frame, text="配对 2").grid(row=0, column=6, sticky="e", padx=(14, 6))
        second_pair = ttk.Frame(pair_frame)
        second_pair.grid(row=0, column=7, sticky="ew")
        second_pair.columnconfigure(0, weight=1)
        second_pair.columnconfigure(2, weight=1)
        ttk.Combobox(second_pair, textvariable=self.pair2_signal, values=channels, state="readonly", width=5).grid(row=0, column=0, sticky="ew")
        ttk.Label(second_pair, text="-").grid(row=0, column=1, padx=4)
        ttk.Combobox(second_pair, textvariable=self.pair2_idler, values=channels, state="readonly", width=5).grid(row=0, column=2, sticky="ew")

        ttk.Label(pair_frame, text="固定粗时延 2(ps)").grid(row=1, column=4, sticky="e", padx=(14, 6), pady=(8, 0))
        ttk.Entry(pair_frame, textvariable=self.pair2_time_diff, width=14).grid(row=1, column=5, sticky="ew", pady=(8, 0))
        self.scan_result = ttk.Label(pair_frame, text="尚未扫描", foreground="#555555")
        self.scan_result.grid(row=1, column=0, columnspan=4, sticky="w", pady=(8, 0))

        params = ttk.LabelFrame(outer, text="处理参数", padding=10)
        params.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        for col in (1, 3, 5, 7):
            params.columnconfigure(col, weight=1)

        fields = [
            ("总时长(s)", self.total_time_s),
            ("分段时长(s)", self.split_step_s),
            ("粗相关窗口(s)", self.correlation_window_s),
            ("粗相关事件数", self.correlation_frames),
            ("计算分箱(ps)", self.bin_width_ps),
            ("计算分箱数", self.bin_num),
            ("保存分辨率(ps)", self.hist_bin_width_ps),
            ("横坐标中心(ps)", self.hist_center_ps),
            ("横坐标点数", self.hist_points),
            ("高斯拟合半窗", self.fit_half_window_bins),
            ("读取块大小", self.read_chunk_size),
            ("并行进程数", self.workers),
        ]
        for index, (label, variable) in enumerate(fields):
            row = index // 4
            pair_col = (index % 4) * 2
            ttk.Label(params, text=label).grid(row=row, column=pair_col, sticky="e", padx=(0, 6), pady=4)
            ttk.Entry(params, textvariable=variable, width=12).grid(row=row, column=pair_col + 1, sticky="ew", padx=(0, 12), pady=4)

        ttk.Checkbutton(params, text="保存每段直方图", variable=self.save_histograms).grid(
            row=3, column=0, columnspan=2, sticky="w", pady=(6, 0)
        )

        log_frame = ttk.LabelFrame(outer, text="运行日志", padding=8)
        log_frame.grid(row=3, column=0, sticky="nsew", pady=(10, 0))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        self.log = tk.Text(log_frame, height=12, wrap="word", state="disabled", font=("Consolas", 9))
        self.log.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.log.configure(yscrollcommand=scrollbar.set)

        actions = ttk.Frame(outer)
        actions.grid(row=4, column=0, sticky="ew", pady=(10, 0))
        actions.columnconfigure(0, weight=1)
        ttk.Label(actions, textvariable=self.status).grid(row=0, column=0, sticky="w")
        ttk.Button(actions, text="打开输出", command=self._open_output).grid(row=0, column=1, padx=(8, 0))
        self.stop_button = ttk.Button(actions, text="停止", command=self._stop, state="disabled")
        self.stop_button.grid(row=0, column=2, padx=(8, 0))
        self.start_button = ttk.Button(actions, text="开始处理", command=self._start)
        self.start_button.grid(row=0, column=3, padx=(8, 0))

        for variable in (self.pair1_signal, self.pair1_idler, self.pair2_signal, self.pair2_idler):
            variable.trace_add("write", lambda *_: self._update_output_suggestion())

    def _choose_data_dir(self) -> None:
        selected = filedialog.askdirectory(title="选择时间戳数据文件夹")
        if not selected:
            return
        self.data_dir.set(selected)
        self._update_output_suggestion(force=True)
        self._scan_channels()

    def _choose_output_dir(self) -> None:
        initial = self.output_dir.get() or self.data_dir.get()
        selected = filedialog.askdirectory(title="选择输出文件夹", initialdir=initial or None)
        if selected:
            self.output_dir.set(selected)

    def _pair_suffix(self) -> str:
        return f"{self.pair1_signal.get()}{self.pair1_idler.get()}_{self.pair2_signal.get()}{self.pair2_idler.get()}"

    def _update_output_suggestion(self, force: bool = False) -> None:
        data_text = self.data_dir.get().strip()
        if not data_text:
            return
        current = self.output_dir.get().strip()
        suggested_name = f"单峰全程_{self._pair_suffix()}_1ps完整横坐标"
        if force or not current or Path(current).parent == Path(data_text):
            self.output_dir.set(str(Path(data_text) / suggested_name))

    @staticmethod
    def _channel_number(path: Path) -> str | None:
        match = re.match(r"^(?:ch(?:annel)?[_ -]*)?([1-4])(?:[-_ ]|$)", path.name, re.IGNORECASE)
        return match.group(1) if match else None

    def _scan_channels(self) -> bool:
        folder = Path(self.data_dir.get().strip())
        if not folder.is_dir():
            messagebox.showerror("文件夹无效", "请选择包含 ttbin 文件的数据文件夹。")
            return False

        candidates: dict[str, list[Path]] = {str(i): [] for i in range(1, 5)}
        for path in folder.glob("*.ttbin"):
            channel = self._channel_number(path)
            if channel:
                candidates[channel].append(path)

        found: dict[str, Path] = {}
        for channel, paths in candidates.items():
            if not paths:
                continue
            base_files = [path for path in paths if not re.search(r"\.\d+\.ttbin$", path.name, re.IGNORECASE)]
            found[channel] = sorted(base_files or paths, key=lambda path: (len(path.name), path.name))[0]

        self.channel_files = found
        if len(found) == 4:
            self.scan_result.configure(text="已识别通道 1、2、3、4", foreground="#176b37")
            self.status.set("已识别 4 个通道")
            return True

        missing = ", ".join(str(i) for i in range(1, 5) if str(i) not in found)
        self.scan_result.configure(text=f"缺少通道: {missing}", foreground="#a33a2b")
        self.status.set("通道文件不完整")
        return False

    @staticmethod
    def _positive_number(name: str, value: str, integer: bool = False) -> int | float:
        try:
            parsed = int(value) if integer else float(value)
        except ValueError as exc:
            raise ValueError(f"{name} 必须是数字。") from exc
        if parsed <= 0:
            raise ValueError(f"{name} 必须大于 0。")
        return parsed

    @staticmethod
    def _optional_integer(name: str, value: str) -> int | None:
        text = value.strip()
        if not text:
            return None
        try:
            return int(text)
        except ValueError as exc:
            raise ValueError(f"{name} 必须是整数或留空。") from exc

    def _build_config(self) -> tuple[Path, Path]:
        if not self._scan_channels():
            raise ValueError("没有识别到完整的 1/2/3/4 通道主文件。")

        if not self.output_dir.get().strip():
            raise ValueError("请选择输出文件夹。")
        data_dir = Path(self.data_dir.get().strip()).resolve()
        output_dir = Path(self.output_dir.get().strip()).resolve()

        pair_channels = [
            (self.pair1_signal.get(), self.pair1_idler.get(), self.pair1_time_diff.get()),
            (self.pair2_signal.get(), self.pair2_idler.get(), self.pair2_time_diff.get()),
        ]
        for signal, idler, _ in pair_channels:
            if signal == idler:
                raise ValueError("同一配对的两个通道不能相同。")

        total_time = self._positive_number("总时长", self.total_time_s.get())
        split_step = self._positive_number("分段时长", self.split_step_s.get())
        if split_step > total_time:
            raise ValueError("分段时长不能大于总时长。")

        defaults = {
            "output_root": str(output_dir),
            "split_step_s": split_step,
            "total_time_s": total_time,
            "correlation_window_s": self._positive_number("粗相关窗口", self.correlation_window_s.get()),
            "correlation_frames": self._positive_number("粗相关事件数", self.correlation_frames.get(), True),
            "bin_width_ps": self._positive_number("计算分箱", self.bin_width_ps.get(), True),
            "bin_num": self._positive_number("计算分箱数", self.bin_num.get(), True),
            "read_chunk_size": self._positive_number("读取块大小", self.read_chunk_size.get(), True),
            "save_hist_bin_width_ps": self._positive_number("保存分辨率", self.hist_bin_width_ps.get(), True),
            "save_hist_center_ps": self._optional_integer("横坐标中心", self.hist_center_ps.get()),
            "save_hist_points": self._positive_number("横坐标点数", self.hist_points.get(), True),
            "fit_half_window_bins": self._positive_number("高斯拟合半窗", self.fit_half_window_bins.get(), True),
            "workers": self._positive_number("并行进程数", self.workers.get(), True),
            "max_slices": int(total_time // split_step),
            "no_hist": not self.save_histograms.get(),
        }

        pairs = []
        for index, (signal, idler, time_diff_text) in enumerate(pair_channels, start=1):
            pair = {
                "label": f"ch{signal}_ch{idler}",
                "signal": self.channel_files[signal].name,
                "idler": self.channel_files[idler].name,
            }
            time_diff = self._optional_integer(f"固定粗时延 {index}", time_diff_text)
            if time_diff is not None:
                pair["time_diff_ps"] = time_diff
            pairs.append(pair)

        job_name = f"{data_dir.name}_{self._pair_suffix()}_singlepeak"
        config = {
            "defaults": defaults,
            "jobs": [{"name": job_name, "root": str(data_dir), "pairs": pairs}],
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        config_path = output_dir / "singlepeak_ui_job.json"
        with config_path.open("w", encoding="utf-8") as handle:
            json.dump(config, handle, ensure_ascii=False, indent=2)
        return config_path, output_dir

    def _start(self) -> None:
        if self.process is not None and self.process.poll() is None:
            return
        try:
            output_dir = Path(self.output_dir.get().strip())
            summary = output_dir / "batch_singlepeak_summary.csv"
            if summary.exists() and not messagebox.askyesno("输出已存在", "该输出目录已有处理结果，继续会覆盖同名文件。是否继续？"):
                return
            config_path, output_dir = self._build_config()
        except (OSError, ValueError) as exc:
            messagebox.showerror("无法开始", str(exc))
            return

        self._clear_log()
        self._append_log(f"配置: {config_path}\n输出: {output_dir}\n\n")
        self.status.set("正在处理")
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")

        command = [sys.executable, "-u", str(RUNNER), "--job-file", str(config_path)]
        creationflags = 0
        if os.name == "nt":
            creationflags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
        try:
            self.process = subprocess.Popen(
                command,
                cwd=str(APP_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=creationflags,
            )
        except OSError as exc:
            self.process = None
            self.start_button.configure(state="normal")
            self.stop_button.configure(state="disabled")
            self.status.set("启动失败")
            messagebox.showerror("启动失败", str(exc))
            return

        threading.Thread(target=self._read_process_output, args=(self.process, output_dir), daemon=True).start()

    def _read_process_output(self, process: subprocess.Popen[str], output_dir: Path) -> None:
        log_path = output_dir / "singlepeak_ui_run.log"
        try:
            with log_path.open("w", encoding="utf-8") as log_file:
                assert process.stdout is not None
                for line in process.stdout:
                    log_file.write(line)
                    log_file.flush()
                    self.messages.put(("log", line))
            return_code = process.wait()
            self.messages.put(("done", return_code))
        except OSError as exc:
            self.messages.put(("error", str(exc)))

    def _stop(self) -> None:
        process = self.process
        if process is None or process.poll() is not None:
            return
        if not messagebox.askyesno("停止处理", "确定停止当前处理任务？已写出的文件会保留。"):
            return
        self.status.set("正在停止")
        self.stop_button.configure(state="disabled")
        self._terminate_process_tree(process)

    @staticmethod
    def _terminate_process_tree(process: subprocess.Popen[str]) -> None:
        if process.poll() is not None:
            return
        if os.name == "nt":
            try:
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=10,
                    check=False,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
                return
            except (OSError, subprocess.TimeoutExpired):
                pass
        process.terminate()

    def _open_output(self) -> None:
        path = Path(self.output_dir.get().strip())
        if not path.is_dir():
            messagebox.showerror("输出不存在", "当前输出文件夹还没有创建。")
            return
        if os.name == "nt":
            os.startfile(path)
        else:
            subprocess.Popen(["xdg-open", str(path)])

    def _drain_messages(self) -> None:
        try:
            while True:
                kind, payload = self.messages.get_nowait()
                if kind == "log":
                    self._append_log(str(payload))
                elif kind == "done":
                    return_code = int(payload)
                    self.process = None
                    self.start_button.configure(state="normal")
                    self.stop_button.configure(state="disabled")
                    if return_code == 0:
                        self.status.set("处理完成")
                        messagebox.showinfo("处理完成", "直方图和峰值 CSV 已生成。")
                    else:
                        self.status.set(f"处理失败（代码 {return_code}）")
                        messagebox.showerror("处理失败", "请查看运行日志中的最后一段错误信息。")
                elif kind == "error":
                    self.process = None
                    self.start_button.configure(state="normal")
                    self.stop_button.configure(state="disabled")
                    self.status.set("日志写入失败")
                    messagebox.showerror("运行错误", str(payload))
        except queue.Empty:
            pass
        self.root.after(100, self._drain_messages)

    def _append_log(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text)
        self.log.see("end")
        self.log.configure(state="disabled")

    def _clear_log(self) -> None:
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    def _on_close(self) -> None:
        if self.process is not None and self.process.poll() is None:
            if not messagebox.askyesno("退出", "处理仍在运行，退出会停止任务。确定退出？"):
                return
            self._terminate_process_tree(self.process)
        self.root.destroy()


def main() -> None:
    if not RUNNER.is_file():
        raise FileNotFoundError(f"找不到批处理脚本: {RUNNER}")
    root = tk.Tk()
    SinglePeakBatchUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
