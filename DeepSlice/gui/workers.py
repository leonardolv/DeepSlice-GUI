import traceback

from PySide6.QtCore import QObject, QRunnable, Signal, Slot

from ..error_logging import get_logger


LOGGER = get_logger("gui.worker")


import threading


class WorkerSignals(QObject):
    finished = Signal(object)
    # error emits the full traceback text. Callers may inspect the exception
    # class name (it appears in the traceback) for typed dispatch.
    error = Signal(str)
    progress = Signal(int, int, str)
    log = Signal(str)


class FunctionWorker(QRunnable):
    def __init__(self, fn, *args, inject_callbacks: bool = False, **kwargs):
        super().__init__()
        self.fn = fn
        self.task_name = getattr(fn, "__name__", str(fn))
        self.args = args
        self.kwargs = kwargs
        self.inject_callbacks = inject_callbacks
        self.signals = WorkerSignals()
        self._cancel_event = threading.Event()

    def request_cancel(self) -> None:
        """Set a cooperative cancellation flag tasks can poll."""
        self._cancel_event.set()

    def is_cancel_requested(self) -> bool:
        return self._cancel_event.is_set()

    @Slot()
    def run(self):
        try:
            kwargs = dict(self.kwargs)
            if self.inject_callbacks:
                kwargs["progress_callback"] = self._emit_progress
                kwargs["log_callback"] = self.signals.log.emit
                if "cancel_check" not in kwargs:
                    kwargs["cancel_check"] = self.is_cancel_requested
            result = self.fn(*self.args, **kwargs)
        except Exception as exc:
            error_text = traceback.format_exc()
            LOGGER.error(
                "Background task '%s' failed: %s",
                self.task_name,
                exc,
                exc_info=(type(exc), exc, exc.__traceback__),
            )
            self.signals.error.emit(error_text)
        else:
            self.signals.finished.emit(result)

    def _emit_progress(self, completed: int, total: int, phase: str):
        self.signals.progress.emit(int(completed), int(total), str(phase))
