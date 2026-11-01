import os
import json
import time
import uuid
import queue
import platform
import threading
from datetime import datetime, timezone

try:
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False


class AsyncTelemetryLogger:
    """
    Asynchronous Edge-to-Cloud Telemetry Engine (Section 7.1.6)
    Streams operational metrics to AWS CloudWatch & S3 via Boto3 on a non-blocking
    background thread (< 1 ms main-loop overhead), with automatic local JSONL fallback.
    """
    def __init__(self,
                 s3_bucket: str = "fyp-gesture-controller-telemetry",
                 cloudwatch_namespace: str = "FYP/GestureController",
                 region_name: str = "ap-southeast-1"):
        self.session_id = str(uuid.uuid4())
        self.s3_bucket = os.getenv("AWS_S3_BUCKET", s3_bucket)
        self.cw_namespace = cloudwatch_namespace
        self.region_name = os.getenv("AWS_REGION", region_name)
        self.local_log_dir = "logs"
        os.makedirs(self.local_log_dir, exist_ok=True)
        self.local_log_file = os.path.join(self.local_log_dir, f"session_{self.session_id[:8]}.jsonl")

        self.event_queue = queue.Queue(maxsize=2000)
        self.running = True
        self.cloud_enabled = False

        if BOTO3_AVAILABLE and os.getenv("AWS_ACCESS_KEY_ID"):
            try:
                self.session = boto3.Session(region_name=self.region_name)
                self.cw_client = self.session.client("cloudwatch")
                self.s3_client = self.session.client("s3")
                self.cloud_enabled = True
                print(f"[Telemetry] AWS Boto3 active (Region: {self.region_name}).")
            except Exception as e:
                print(f"[Telemetry] AWS fallback to local disk: {e}")
        else:
            print(f"[Telemetry] Local offline logging active -> {self.local_log_file}")

        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

        # Log initial Hardware Profile (matches ERD Section 7.1)
        self.log_event("HARDWARE_PROFILE", {
            "os_version": f"{platform.system()} {platform.release()}",
            "cpu_architecture": platform.machine(),
            "python_version": platform.python_version()
        })

    def log_event(self, event_type: str, payload: dict):
        record = {
            "session_id": self.session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "payload": payload
        }
        try:
            self.event_queue.put_nowait(record)
        except queue.Full:
            pass

    def _worker_loop(self):
        batch = []
        last_cw_push = time.time()

        while self.running or not self.event_queue.empty():
            try:
                record = self.event_queue.get(timeout=0.5)
                batch.append(record)

                # Always write to local JSONL log
                with open(self.local_log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record) + "\n")

                # Push real-time performance metrics to CloudWatch every 5 seconds
                if self.cloud_enabled and record["event_type"] == "TRACKING_METRIC":
                    now = time.time()
                    if now - last_cw_push >= 5.0:
                        self._push_cloudwatch_metrics(record["payload"])
                        last_cw_push = now

                self.event_queue.task_done()
            except queue.Empty:
                continue
            except Exception:
                continue

        # Flush final session archive to S3 on exit
        if self.cloud_enabled and batch:
            self._upload_session_to_s3()

    def _push_cloudwatch_metrics(self, metrics: dict):
        try:
            self.cw_client.put_metric_data(
                Namespace=self.cw_namespace,
                MetricData=[
                    {
                        "MetricName": "PipelineFPS",
                        "Dimensions": [{"Name": "SessionID", "Value": self.session_id[:8]}],
                        "Value": float(metrics.get("fps_measured", 0.0)),
                        "Unit": "Count/Second"
                    },
                    {
                        "MetricName": "TrackingLatencyMs",
                        "Dimensions": [{"Name": "SessionID", "Value": self.session_id[:8]}],
                        "Value": float(metrics.get("tracking_latency_ms", 0.0)),
                        "Unit": "Milliseconds"
                    }
                ]
            )
        except Exception:
            pass

    def _upload_session_to_s3(self):
        try:
            s3_key = f"sessions/{os.path.basename(self.local_log_file)}"
            self.s3_client.upload_file(self.local_log_file, self.s3_bucket, s3_key)
            print(f"[Telemetry] Uploaded session archive to s3://{self.s3_bucket}/{s3_key}")
        except Exception:
            pass

    def shutdown(self):
        self.running = False
        self.worker_thread.join(timeout=2.0)
