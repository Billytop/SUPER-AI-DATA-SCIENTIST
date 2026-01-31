
import socket

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

if is_port_in_use(8001):
    print("Port 8001 is IN USE (Server likely running)")
else:
    print("Port 8001 is FREE (Server not running)")
