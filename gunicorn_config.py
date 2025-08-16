#!/usr/bin/env python3
"""
Gunicorn configuration for Chatty AI production deployment
"""

import multiprocessing
import os

# Server socket
bind = "0.0.0.0:5000"
backlog = 2048

# Worker processes
workers = 1  # Single worker for Raspberry Pi 5 to avoid resource conflicts
worker_class = "eventlet"  # Required for Socket.IO
worker_connections = 1000

# Restart workers after serving this many requests (helps with memory leaks)
max_requests = 1000
max_requests_jitter = 100

# Timeout settings
timeout = 300  # 5 minutes for AI processing
keepalive = 2
graceful_timeout = 30

# Logging
accesslog = "/home/nickspi5/Chatty_AI/logs/access.log"
errorlog = "/home/nickspi5/Chatty_AI/logs/error.log"
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = 'chatty_ai_server'

# Server mechanics
daemon = False  # Keep False for systemd service
pidfile = '/tmp/chatty_ai.pid'
user = None  # Let systemd handle user switching
group = None
tmp_upload_dir = '/tmp'

# SSL (if you want to enable HTTPS later)
# keyfile = '/path/to/keyfile'
# certfile = '/path/to/certfile'

# Application specific
# Preload app for memory efficiency (be careful with this on single worker)
preload_app = False  # Set to False to avoid issues with camera/audio resources

# Environment variables
raw_env = [
    'CHATTY_AI_ENV=production',
    'PYTHONPATH=/home/nickspi5/Chatty_AI'
]

# Memory and resource limits
limit_request_line = 4094
limit_request_fields = 100
limit_request_field_size = 8190

# Security
forwarded_allow_ips = '*'  # Configure based on your reverse proxy setup
secure_scheme_headers = {
    'X-FORWARDED-PROTOCOL': 'ssl',
    'X-FORWARDED-PROTO': 'https',
    'X-FORWARDED-SSL': 'on'
}

def post_fork(server, worker):
    """Called just after a worker has been forked."""
    server.log.info("Worker spawned (pid: %s)", worker.pid)

def pre_fork(server, worker):
    """Called just before a worker is forked."""
    server.log.info("About to fork worker (pid: %s)", worker.pid)

def worker_int(worker):
    """Called when a worker receives the SIGINT or SIGQUIT signal."""
    worker.log.info("Worker received SIGINT/SIGQUIT")

def on_starting(server):
    """Called just before the master process is initialized."""
    server.log.info("Chatty AI Server starting...")

def on_reload(server):
    """Called to recycle workers during a reload via SIGHUP."""
    server.log.info("Reloading Chatty AI Server...")

def when_ready(server):
    """Called just after the server is started."""
    server.log.info("Chatty AI Server is ready. PID: %s", os.getpid())

def on_exit(server):
    """Called just before exiting."""
    server.log.info("Chatty AI Server shutting down...")

def pre_request(worker, req):
    """Called just before a worker processes the request."""
    worker.log.debug("%s %s" % (req.method, req.path))

def post_request(worker, req, environ, resp):
    """Called after a worker processes the request."""
    pass