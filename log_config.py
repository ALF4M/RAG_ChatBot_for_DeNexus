import logging
import json
import socket

class LogstashFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "pathname": record.pathname,
            "lineno": record.lineno,
        }
        return json.dumps(log_record)


class LogstashTCPHandler(logging.Handler):
    def __init__(self, host, port):
        super().__init__()
        self.host = host
        self.port = port
        self.sock = socket.create_connection((host, port))

    def emit(self, record):
        try:
            msg = self.format(record) + "\n"
            self.sock.sendall(msg.encode("utf-8"))
        except Exception:
            self.handleError(record)


def get_logstash_logger(name: str, host: str = "localhost", port: int = 5001) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = LogstashTCPHandler(host, port)
        handler.setFormatter(LogstashFormatter())
        logger.addHandler(handler)

    return logger
