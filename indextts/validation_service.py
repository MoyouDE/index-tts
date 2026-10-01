"""Compatibility facade over independent, lazily constructed validation services."""
import threading
from .validation_common import SessionFiles, audio_details, context_rows


class ValidationService:
    def __init__(self, output_dir, *, cpu_threads=4):
        self.files = SessionFiles(output_dir)
        self.output_dir = self.files.output_dir
        self.cpu_threads = cpu_threads
        self._producer_service = self._emotion_service = None
        self._service_lock = threading.RLock()

    @property
    def producer(self):
        with self._service_lock:
            if self._producer_service is None:
                from .producer_service import ProducerService
                self._producer_service = ProducerService(self.output_dir, cpu_threads=self.cpu_threads)
            return self._producer_service

    @property
    def emotion(self):
        with self._service_lock:
            if self._emotion_service is None:
                from .emotion_service import EmotionService
                self._emotion_service = EmotionService(self.output_dir)
            return self._emotion_service

    def session_dir(self, session_id):
        return self.files.session_dir(session_id)

    def build(self, reference, voice_id, name, gender, profile, device, model_dir, session_id, progress=None):
        return self.producer.build(reference, voice_id, name, gender, profile, device, model_dir, session_id, progress)

    def inspect(self, uploaded, model_dir, session_id):
        return self.producer.inspect(uploaded, model_dir, session_id)

    def emotion_default(self, model_dir):
        return self.emotion.emotion_default(model_dir)

    def analyze(self, model_dir, mode, text, kind, rows, target, use_default, threshold, session_id):
        return self.emotion.analyze(model_dir, mode, text, kind, rows, target, use_default, threshold, session_id)

    def unload_producer(self):
        with self._service_lock:
            if self._producer_service is not None:
                return self._producer_service.close()
        return "制包模型已卸载"

    def unload_emotion(self):
        with self._service_lock:
            if self._emotion_service is not None:
                return self._emotion_service.close()
        return "情感模型已卸载"

    def close(self):
        self.unload_producer()
        self.unload_emotion()
