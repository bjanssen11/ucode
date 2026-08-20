#!/usr/bin/env python3
"""Arch B probe: MITM proxy between TUI and app-server, rewriting model in turn/start.

This demonstrates feasibility of interposing on the real TUI without modifying it.
The proxy:
1. Listens on a unix socket that the TUI connects to (via --remote)
2. Forwards all messages to a real app-server
3. Rewrites turn/start.model to a fixed value (proving router capability)
4. Passes everything else through unchanged
"""
import json
import os
import socket
import subprocess
import sys
import threading
import time
import argparse

class CodexInterposer:
    """MITM proxy for Codex messages."""
    
    def __init__(self, listen_sock_path, app_server_sock_path, target_model):
        self.listen_sock_path = listen_sock_path
        self.app_server_sock_path = app_server_sock_path
        self.target_model = target_model
        self.listener = None
        self.running = True
        
    def cleanup(self):
        """Clean up listener socket."""
        if os.path.exists(self.listen_sock_path):
            try:
                os.remove(self.listen_sock_path)
            except:
                pass
        if self.listener:
            try:
                self.listener.close()
            except:
                pass
    
    def rewrite_message(self, msg):
        """Rewrite turn/start to force target model."""
        if not isinstance(msg, dict):
            return msg
        
        method = msg.get("method")
        if method == "turn/start":
            params = msg.get("params", {})
            if isinstance(params, dict):
                old_model = params.get("model")
                if old_model != self.target_model:
                    print(f"[REWRITE] turn/start: {old_model!r} -> {self.target_model!r}")
                    params["model"] = self.target_model
                    msg["params"] = params
        
        return msg
    
    def relay_messages(self, client_sock, as_sock):
        """Relay messages bidirectionally, rewriting turn/start."""
        
        def tui_to_as():
            """TUI -> app-server (with rewriting)"""
            buffer = ""
            while self.running:
                try:
                    data = client_sock.recv(1024)
                    if not data:
                        print("[TUI->AS] Connection closed by TUI")
                        break
                    
                    buffer += data.decode('utf-8', errors='replace')
                    
                    # Process complete lines
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        line = line.strip()
                        if not line:
                            continue
                        
                        try:
                            msg = json.loads(line)
                            msg = self.rewrite_message(msg)
                            rewritten = json.dumps(msg)
                            as_sock.sendall((rewritten + '\n').encode('utf-8'))
                            print(f"[TUI->AS] {msg.get('method', msg.get('type', '?'))}")
                        except Exception as e:
                            print(f"[TUI->AS] Error: {e}")
                            as_sock.sendall((line + '\n').encode('utf-8'))
                except Exception as e:
                    print(f"[TUI->AS] Exception: {e}")
                    break
        
        def as_to_tui():
            """app-server -> TUI (pass-through)"""
            buffer = ""
            while self.running:
                try:
                    data = as_sock.recv(1024)
                    if not data:
                        print("[AS->TUI] Connection closed by app-server")
                        break
                    
                    buffer += data.decode('utf-8', errors='replace')
                    
                    # Process complete lines
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        line = line.strip()
                        if not line:
                            continue
                        
                        try:
                            msg = json.loads(line)
                            method = msg.get('method')
                            if method in ('turn/start', 'turn/completed', 'item/completed'):
                                print(f"[AS->TUI] {method}")
                        except:
                            pass
                        
                        client_sock.sendall((line + '\n').encode('utf-8'))
                except Exception as e:
                    print(f"[AS->TUI] Exception: {e}")
                    break
        
        t1 = threading.Thread(target=tui_to_as, daemon=True)
        t2 = threading.Thread(target=as_to_tui, daemon=True)
        t1.start()
        t2.start()
        
        # Wait for either thread to finish
        t1.join(timeout=300)
        t2.join(timeout=300)
        
        self.running = False
    
    def handle_client(self, client_sock, addr):
        """Handle a single TUI connection."""
        print(f"[CLIENT] Connected from {addr}")
        
        try:
            # Connect to real app-server
            as_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            as_sock.connect(self.app_server_sock_path)
            print(f"[RELAY] Connected to app-server at {self.app_server_sock_path}")
            
            # Relay bidirectionally
            self.relay_messages(client_sock, as_sock)
            
            as_sock.close()
        except Exception as e:
            print(f"[ERROR] Failed to relay: {e}")
        finally:
            client_sock.close()
            print(f"[CLIENT] Disconnected")
    
    def run(self):
        """Start the interposer listening for TUI connections."""
        self.cleanup()
        
        print(f"[STARTUP] Creating listener at {self.listen_sock_path}")
        
        self.listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.listener.bind(self.listen_sock_path)
        self.listener.listen(1)
        
        print(f"[STARTUP] Listening for TUI connections")
        print(f"[INFO] Target model: {self.target_model!r}")
        print(f"[INFO] To connect TUI: codex --remote unix://{self.listen_sock_path}")
        
        try:
            while self.running:
                try:
                    self.listener.settimeout(1.0)
                    client_sock, addr = self.listener.accept()
                    
                    # Handle in a thread
                    t = threading.Thread(
                        target=self.handle_client,
                        args=(client_sock, addr),
                        daemon=True
                    )
                    t.start()
                except socket.timeout:
                    continue
                except KeyboardInterrupt:
                    print("\n[SHUTDOWN] Interrupted")
                    break
        finally:
            self.cleanup()


def main():
    parser = argparse.ArgumentParser(
        description="Codex model interposer: MITM proxy to rewrite turn/start.model"
    )
    parser.add_argument(
        "--listen",
        default="/home/lilly.luo/.cache/codex-b/tui-remote.sock",
        help="Socket for TUI to connect to"
    )
    parser.add_argument(
        "--app-server",
        default="/home/lilly.luo/.cache/codex-b/as.sock",
        help="Real app-server socket"
    )
    parser.add_argument(
        "--model",
        default="gpt-5.5",
        help="Model to force for all turns"
    )
    
    args = parser.parse_args()
    
    interposer = CodexInterposer(args.listen, args.app_server, args.model)
    interposer.run()


if __name__ == "__main__":
    sys.exit(main() or 0)
