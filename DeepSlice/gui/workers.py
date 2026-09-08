import inspect
import traceback

from PySide6.QtCore import QObject, QRunnable, Signal, Slot

from ..error_logging import get_logger


LOGGER = get_logger("gui.worker")


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

    def _accepts_kwarg(self, name: str) -> bool:
        """Whether calling `self.fn` with keyword `name` would not raise.

        `inject_callbacks` used to add progress_callback/log_callback
        unconditionally - safe as long as every target declares both, which
        held until `_atlas_preview_task` and `_load_quint_task` were added.
        Both declare `progress_callback`/`log_callback` but not a third
        callback this module used to also inject unconditionally
        (`cancel_check` - see the deleted `request_cancel`/
        `is_cancel_requested`, which had no caller anywhere in the app and
        were removed rather than fixed). Neither has a `**kwargs`
        catch-all, so a function accepting fewer injected callbacks than
        this class assumes raises `TypeError` on every single call - which
        is exactly what happened, silently caught by `run`'s own `except
        Exception` and surfaced to the user as a generic "Failed to load
        QuickNII file"/atlas-preview error. Only inject a callback a
        function actually declared (or can absorb via `**kwargs`).
        """
        try:
            params = inspect.signature(self.fn).parameters.values()
        except (TypeError, ValueError):
            return True  # can't introspect (e.g. some builtins) - keep the old behavior
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params):
            return True
        return any(p.name == name for p in params)

    @Slot()
    def run(self):
        try:
            kwargs = dict(self.kwargs)
            if self.inject_callbacks:
                if self._accepts_kwarg("progress_callback"):
                    kwargs["progress_callback"] = self._emit_progress
                if self._accepts_kwarg("log_callback"):
                    kwargs["log_callback"] = self.signals.log.emit
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
